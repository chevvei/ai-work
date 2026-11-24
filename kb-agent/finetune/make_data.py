"""QLoRA 第 1 步：从知识库自动造“证据 -> 带引用回答”样本，再交给人工校对。

流程（self-instruct + 检索回放）：
  1. 按项目分层抽 chunk，让底座模型“扮演新同事”针对这块资料提一个具体问题；
  2. 用真实检索器回放这个问题——源 chunk 没被检索回来就丢弃（训练分布必须和线上一致）；
  3. 用线上同一套 prompt 生成草稿回答，程序硬校验：有引用、引用了源 chunk、无越界编号、没自写行号；
  4. 每 10 条造 1 条“拒答”样本：把源文件的块从证据里拿掉，目标答案是“资料中没有找到”。
产物（含公司代码，已 gitignore）：
  data/drafts.jsonl   全部草稿 + 自动校验结果
  data/review.md      抽样给人工校对的清单（改“状态/回答”两处即可）

运行：~/venvs/gpu/bin/python -m finetune.make_data --n 40 --review 30
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from kb.agent import ANSWER_TAIL, SYSTEM, check_citations, format_evidence
from kb.cli import DEFAULT_LLM
from kb.llm import TransformersLLM
from kb.retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "finetune" / "data"
REFUSAL = "资料中没有找到与该问题直接相关的内容。"
SELF_LINENO = re.compile(r"第\s*\d+\s*行|L\d+-L?\d+|\.(cpp|h|hpp|md|py):\d+")

ASK_PROMPT = """下面是 {project} 项目的一段资料。假设你是刚加入团队的开发者，针对这段资料提出一个具体、可回答的问题。
要求：问题里带上项目名 {project}；不要出现“这段代码”“上面”之类指代；只输出问题本身，不超过 40 字。

【资料】{path}
```
{text}
```"""


def sample_chunks(chunks, n, seed):
    """分层抽样：每个项目配额相同，避免大项目（LidarAssistant 近 3000 块）淹没小项目。"""
    rng = random.Random(seed)
    by_proj = defaultdict(list)
    for c in chunks:
        if len(c.text) >= 300 and (c.title or c.kind != "code"):  # 太短/无标题的代码块提不出好问题
            by_proj[c.project].append(c)
    per = max(1, n // len(by_proj))
    picked = []
    for items in by_proj.values():
        picked += rng.sample(items, min(per, len(items)))
    rng.shuffle(picked)
    return picked[:n]


def build_messages(question, evidence):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"【资料】\n{format_evidence(evidence)}\n\n【问题】{question}{ANSWER_TAIL}"}]


def auto_check(answer, evidence, src_id):
    cites = check_citations(answer, len(evidence))
    src_n = next((i for i, c in enumerate(evidence, 1) if c.id == src_id), None)
    return {"has_cite": bool(cites["cited"]), "invalid": cites["invalid"], "cites_source": src_n in cites["cited"],
            "self_lineno": bool(SELF_LINENO.search(answer))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽多少个 chunk 造题")
    ap.add_argument("--review", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="追加到已有 drafts.jsonl（换 seed 扩量，已造过题的源块跳过）")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(ROOT / "index")
    llm = TransformersLLM(str(DEFAULT_LLM))
    rows, stats, t0 = [], defaultdict(int), time.time()
    if args.append and (DATA / "drafts.jsonl").exists():
        rows = [json.loads(line) for line in open(DATA / "drafts.jsonl", encoding="utf-8")]
    seen = {r.get("source_id") for r in rows}
    for i, src in enumerate(sample_chunks(retriever.chunks, args.n, args.seed)):
        if src.id in seen:
            stats["skip_seen"] += 1
            continue
        q = llm.chat([{"role": "user", "content": ASK_PROMPT.format(project=src.project, path=src.path, text=src.text[:1500])}],
                     max_new_tokens=60).splitlines()[0].strip()
        evidence = retriever.search(q, k=8)
        if src.id not in {c.id for c in evidence}:
            stats["drop_not_retrieved"] += 1  # 套路：检索都找不回的题，训练了也用不上
            continue
        refusal = i % 10 == 9
        if refusal:
            evidence = [c for c in evidence if c.path != src.path]
            answer, check = REFUSAL, {"refusal": True}
        else:
            answer = llm.chat(build_messages(q, evidence))
            check = auto_check(answer, evidence, src.id)
        ok = refusal or (check["has_cite"] and check["cites_source"] and not check["invalid"] and not check["self_lineno"])
        stats["auto_pass" if ok else "auto_fail"] += 1
        rows.append({"id": f"s{args.seed}_{i:04d}", "kind": "refusal" if refusal else "answer", "question": q,
                     "source": f"{src.project}/{src.path}:{src.start}-{src.end}", "source_id": src.id, "evidence": [c.id for c in evidence],
                     "messages": build_messages(q, evidence), "answer": answer, "check": check, "auto_ok": ok, "status": "todo"})
        print(f"[{i + 1}/{args.n}] {'OK ' if ok else 'BAD'} {q}", flush=True)

    with open(DATA / "drafts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_review(rows[:args.review], retriever)
    stats.update(kept=len(rows), seconds=round(time.time() - t0, 1))
    print(json.dumps(stats, ensure_ascii=False))


def write_review(rows, retriever):
    """校对清单：每条只需改“状态”（ok/fix/drop），fix 时直接改“回答”段落。"""
    lines = ["# QLoRA 样本人工校对", "", "规则：状态填 ok（直接用）/ fix（已改回答）/ drop（丢弃）。回答必须只依据下方证据、每句带 [n]、不写行号。", ""]
    for r in rows:
        ev = [retriever.by_id[cid] for cid in r["evidence"]]
        lines += [f"## {r['id']}", f"状态: {'ok' if r['auto_ok'] else 'todo'}", f"问题: {r['question']}", f"源块: {r['source']}",
                  f"自动校验: {json.dumps(r['check'], ensure_ascii=False)}", "", "### 回答", r["answer"], "", "### 证据（只读）"]
        lines += [f"- [{n}] {c.project}/{c.path}:{c.start}-{c.end} {c.title[:50]}" for n, c in enumerate(ev, 1)]
        lines.append("")
    (DATA / "review.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
