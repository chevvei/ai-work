"""夜间流水线（cron 每晚 21:30 触发，07:30 前必须收工）。

  1. 增量入库：/api/admin/rebuild —— 切块 + 增量向量化 + 检索评测卡口，过卡口才热替换
  2. 代码概要：/api/admin/summarize —— 用显卡给还没概要的文件生成“概要 + 预测问法”，到点自动停，完成后自动再入库
  3. 端到端自测：/api/admin/selftest —— 引用命中率、首遍引用率、拒答率、延迟
  4. 昨日问答体检：读 kb.db，列点踩 / 没引用 / 没找到的问题和待审核队列
  5. 写报告 logs/nightly/YYYYMMDD.{json,md}，和上一晚逐项对比

套路：所有步骤都走线上服务的 HTTP 接口——模型和索引已在显存里，不另起进程抢显存；线上服务也是唯一的“真相”。
运行：/home/sti/venvs/gpu/bin/python scripts/nightly.py [--deadline 07:30] [--summarize-limit 600]
"""
import argparse
import json
import sqlite3
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "nightly"
TOKEN = (ROOT / "data" / "admin_token").read_text().strip()
KEYS = ["cite_hit", "first_cited", "cited_any", "refused", "latency_p50", "ttft_p50"]


def api(base, method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data, {"X-Admin-Token": TOKEN, "Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_job(base, get, deadline=None, on_late=None, every=15):
    late = False
    while (j := get())["state"] == "running":
        if deadline and not late and datetime.now() > deadline and on_late:
            on_late()
            late = True
        time.sleep(every)
    return j


def rebuild(base):
    t = time.time()
    api(base, "POST", "/api/admin/rebuild")
    j = wait_job(base, lambda: api(base, "GET", "/api/admin/overview")["job"])
    return {"state": j["state"], "seconds": round(time.time() - t), "log": j["log"][-8:]}


def summarize(base, deadline, limit):
    pending = api(base, "GET", "/api/admin/summarize")["pending"]
    if not sum(pending.values()):
        return {"state": "skip", "pending": pending}
    t = time.time()
    api(base, "POST", "/api/admin/summarize", {"projects": [], "limit": limit})
    # 到点再 POST 一次 = 请求停止；已生成的概要按文件落盘，下一晚接着做
    j = wait_job(base, lambda: api(base, "GET", "/api/admin/summarize")["job"], deadline,
                 lambda: api(base, "POST", "/api/admin/summarize", {"projects": [], "limit": None}))
    time.sleep(5)  # 概要完成后服务会自动触发一次 rebuild，等它结束
    rb = wait_job(base, lambda: api(base, "GET", "/api/admin/overview")["job"])
    left = api(base, "GET", "/api/admin/summarize")["pending"]
    return {"state": j["state"], "seconds": round(time.time() - t), "before": sum(pending.values()), "after": sum(left.values()),
            "log": j["log"][-5:], "rebuild": rb["state"]}


def day_health(db_path, since):
    """线上问答体检。套路：坏例子比平均分有用——报告直接列出问题原文，早上点开就知道该补什么资料。"""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    one = lambda sql: db.execute(sql, (since,)).fetchone()[0]
    h = {"asks": one("SELECT COUNT(*) FROM qa WHERE ts>=?"), "users": one("SELECT COUNT(DISTINCT client) FROM qa WHERE ts>=?"),
         "dislikes": one("SELECT COALESCE(SUM(dislikes),0) FROM qa WHERE ts>=?"),
         "likes": one("SELECT COALESCE(SUM(likes),0) FROM qa WHERE ts>=?"),
         "no_cite": one("SELECT COUNT(*) FROM qa WHERE ts>=? AND cited=0"),
         "refused": one("SELECT COUNT(*) FROM qa WHERE ts>=? AND answer LIKE '%没有找到%'"),
         "avg_seconds": round(one("SELECT COALESCE(AVG(seconds),0) FROM qa WHERE ts>=?"), 2)}
    rows = db.execute("""SELECT id, question, cited, dislikes, answer LIKE '%没有找到%' AS refused FROM qa
                         WHERE ts>=? AND (cited=0 OR dislikes>0 OR answer LIKE '%没有找到%') ORDER BY id DESC LIMIT 30""", (since,))
    h["bad"] = [{"id": r["id"], "question": r["question"][:80],
                 "why": "、".join(w for w, c in (("点踩", r["dislikes"]), ("没找到", r["refused"]), ("无引用", not r["cited"])) if c)}
                for r in rows]
    h["feedback_pending"] = {f"{r[0]}/{r[1] or '-'}": r[2] for r in db.execute(
        "SELECT origin, category, COUNT(*) FROM feedback WHERE status='pending' GROUP BY origin, category")}
    return h


def synth(n, seed, api_base):
    """合成训练/评测草稿：线上 vLLM 出题 + 真实检索回放 + 程序硬校验，只追加到 drafts.jsonl，等人审。
    套路：夜里只“造”不“用”——没人审过的数据不进训练、不进卡口题库，防止错误自我放大。"""
    import subprocess
    t = time.time()
    p = subprocess.run([str(Path.home() / "venvs/gpu/bin/python"), "-m", "finetune.make_data", "--n", str(n), "--seed", str(seed),
                        "--append", "--review", "0", "--api-base", api_base, "--api-model", "qwen2.5-coder-7b", "--embed-device", "cpu"],
                       cwd=ROOT, capture_output=True, text=True)
    last = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        stats = json.loads(last)
    except json.JSONDecodeError:
        stats = {"error": (p.stderr or last)[-300:]}
    return {"seconds": round(time.time() - t), "seed": seed, **stats}


def delta(cur, prev, key):
    if prev is None or key not in prev or cur.get(key) is None or prev.get(key) is None:
        return ""
    d = cur[key] - prev[key]
    return f"（{'+' if d >= 0 else ''}{d:.3f}）" if abs(d) > 1e-9 else "（持平）"


def render(r, prev):
    st, ps = r.get("selftest", {}).get("summary", {}), (prev or {}).get("selftest", {}).get("summary")
    h = r["health"]
    lines = [f"# 夜间报告 {r['date']}", "", f"开始 {r['started']}，结束 {r['finished']}，总用时 {r['seconds'] // 60} 分钟", "",
             "## 1. 增量入库", f"- 状态：{r['rebuild']['state']}，用时 {r['rebuild']['seconds']}s"]
    lines += [f"  - {x}" for x in r["rebuild"]["log"]]
    s = r["summarize"]
    lines += ["", "## 2. 代码概要"]
    lines += ([f"- 无待处理文件"] if s["state"] == "skip" else
              [f"- 状态：{s['state']}，用时 {s['seconds']}s，待处理 {s['before']} → {s['after']}，随后入库：{s['rebuild']}"])
    lines += ["", "## 3. 端到端自测（线上模型 + 线上索引）"]
    if st:
        lines += ["| 指标 | 今晚 | 对比上一晚 |", "|---|---|---|"]
        lines += [f"| {k} | {st.get(k)} | {delta(st, ps, k)} |" for k in KEYS]
        miss = [i["question"] for i in r["selftest"]["items"] if not i["cite_hit"]]
        lines += ["", f"未命中 {len(miss)} 题："] + [f"- {q}" for q in miss[:20]]
    else:
        lines += [f"- 失败：{r['selftest']['error']}" if r.get("selftest") else "- 跳过"]
    lines += ["", "## 4. 线上问答体检（过去 24 小时）",
              f"- 提问 {h['asks']}，用户 {h['users']}，点赞 {h['likes']}，点踩 {h['dislikes']}，"
              f"无引用 {h['no_cite']}，没找到 {h['refused']}，平均 {h['avg_seconds']}s",
              f"- 待审核队列：{h['feedback_pending'] or '空'}", "", "需要关注的问题："]
    lines += [f"- #{b['id']} [{b['why']}] {b['question']}" for b in h["bad"]] or ["- 无"]
    if r.get("synth"):
        s = r["synth"]
        lines += ["", "## 5. 合成训练草稿（待人审，见 finetune/data/drafts.jsonl）",
                  f"- 用时 {s['seconds']}s，seed {s['seed']}：自动校验通过 {s.get('auto_pass', 0)}，不通过 {s.get('auto_fail', 0)}，"
                  f"检索找不回丢弃 {s.get('drop_not_retrieved', 0)}，已造过跳过 {s.get('skip_seen', 0)}，草稿总数 {s.get('kept', '?')}" + (f"，错误 {s['error']}" if "error" in s else "")]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8770")
    ap.add_argument("--deadline", default="07:30", help="到点停止概要生成，保证白天服务不被占")
    ap.add_argument("--summarize-limit", type=int, default=None)
    ap.add_argument("--selftest-limit", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["rebuild", "summarize", "selftest", "synth"])
    ap.add_argument("--synth", type=int, default=100, help="每晚合成多少条训练草稿（0 关闭）")
    ap.add_argument("--vllm", default="http://127.0.0.1:8001/v1")
    a = ap.parse_args()
    now = datetime.now()
    hh, mm = map(int, a.deadline.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    OUT.mkdir(parents=True, exist_ok=True)
    prev_files = sorted(OUT.glob("*.json"))
    prev = json.loads(prev_files[-1].read_text(encoding="utf-8")) if prev_files else None
    t0 = time.time()
    r = {"date": now.strftime("%Y-%m-%d"), "started": now.strftime("%H:%M:%S"), "deadline": deadline.isoformat()}
    step = lambda name, fn, empty: (print(f"[{datetime.now():%H:%M:%S}] {name}", flush=True), fn() if name not in a.skip else empty)[1]
    r["rebuild"] = step("rebuild", lambda: rebuild(a.base), {"state": "skip", "seconds": 0, "log": []})
    r["summarize"] = step("summarize", lambda: summarize(a.base, deadline, a.summarize_limit), {"state": "skip"})
    try:
        r["selftest"] = step("selftest", lambda: api(a.base, "POST", "/api/admin/selftest"
                                                     + (f"?limit={a.selftest_limit}" if a.selftest_limit else ""), timeout=7200), {})
    except Exception as e:  # 自测失败不影响报告产出
        r["selftest"] = {"error": f"{type(e).__name__}: {e}"}
    r["health"] = day_health(ROOT / "data" / "kb.db", time.time() - 86400)
    if "synth" not in a.skip and a.synth and datetime.now() < deadline:
        print(f"[{datetime.now():%H:%M:%S}] synth", flush=True)
        r["synth"] = synth(a.synth, int(now.strftime("%Y%m%d")), a.vllm)
    r["finished"], r["seconds"] = datetime.now().strftime("%H:%M:%S"), round(time.time() - t0)
    stem = OUT / now.strftime("%Y%m%d")
    stem.with_suffix(".json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(r, prev), encoding="utf-8")
    print(f"报告：{stem}.md", flush=True)


if __name__ == "__main__":
    main()
