"""命令行入口。

  python -m kb.cli ingest                 # 扫描 sources.toml -> index/chunks.jsonl
  python -m kb.cli index                  # bge-m3 向量化 -> index/vectors.faiss
  python -m kb.cli search "UDS 超时"       # 只看检索结果（不用 LLM，调检索时最常用）
  python -m kb.cli ask "LidarAssistant 怎么编译" [--mode agent] [--project udslib]
  python -m kb.cli eval                   # 用 eval/questions.jsonl 量化检索命中率
  python -m kb.cli serve                  # 127.0.0.1:8770 网页问答
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("KB_MODELS", Path.home() / "models"))
DEFAULT_LLM = MODELS / "Qwen2.5-Coder-7B-Instruct"
DEFAULT_EMBED = MODELS / "bge-m3"
DEFAULT_RERANK = MODELS / "bge-reranker-v2-m3"


def make_llm(args):
    from .llm import OpenAICompatLLM, TransformersLLM
    if args.api_base:
        return OpenAICompatLLM(args.api_base, args.api_model)
    return TransformersLLM(str(args.llm), load_4bit=args.load_4bit, lora=args.lora and str(args.lora))


def make_retriever(args, vectors=True):
    from .retrieve import Reranker, Retriever
    use_rr = args.rerank and not args.no_rerank and (args.rerank / "model.safetensors").exists()
    r = Retriever(args.index_dir, use_vectors=vectors and not args.bm25_only,
                  reranker=Reranker(args.rerank) if use_rr else None, rerank_top=args.rerank_top)
    r.route = not args.no_route
    return r


SUMMARIES = ROOT / "data" / "summaries.jsonl"  # 代码概要缓存：按文件内容 hash，换索引/重建都复用


def cmd_ingest(args):
    from .ingest import ingest
    stats = ingest(args.config, args.index_dir / "chunks.jsonl", SUMMARIES)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def cmd_summarize(args):
    """离线生成代码概要（需要 GPU；线上服务运行时请在后台“知识更新”页触发，共用已加载的模型）。"""
    from .summarize import summarize
    files = json.loads((args.index_dir / "files.json").read_text(encoding="utf-8"))
    t0 = time.time()
    n = summarize(make_llm(args), files, SUMMARIES, set(args.project) if args.project else None, args.limit)
    print(f"生成 {n} 个文件概要，用时 {time.time() - t0:.0f}s；接着运行 ingest + index 生效")


def cmd_index(args):
    from .retrieve import build_index
    t0 = time.time()
    s = build_index(args.index_dir, args.embed)
    print(f"向量化 {s['chunks']} 块（新编码 {s['encoded']}，复用 {s['reused']}），用时 {time.time() - t0:.1f}s")


def cmd_search(args):
    r = make_retriever(args)
    for n, c in enumerate(r.search(args.query, k=args.k, project=args.project), 1):
        print(f"[{n}] {c.project}/{c.path}:{c.start}-{c.end} {c.title[:60]}")


def cmd_ask(args):
    from .agent import KnowledgeAgent
    agent = KnowledgeAgent(make_retriever(args), make_llm(args), k=args.k)
    res = agent.ask(args.question, project=args.project, mode=args.mode)
    print(res["answer"])
    print("\n来源：")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['project']}/{s['path']}:{s['start']}-{s['end']}")
    if res["invalid_citations"]:
        print(f"  警告：引用了不存在的编号 {res['invalid_citations']}")
    print(f"\n用时 {res['seconds']}s，token {res['usage']}，trace {res['trace']}")


def evaluate(r, rows, k=8, use_project=False, mode="hybrid_route"):
    hits = {1: 0, 3: 0, k: 0}
    mrr, report = 0.0, []
    for row in rows:
        results = r.search(row["question"], k=k, project=row.get("project") if use_project else None)
        rank = next((i for i, c in enumerate(results, 1) if any(e in f"{c.project}/{c.path}" for e in row["expect"])), None)
        for kk in hits:
            hits[kk] += bool(rank and rank <= kk)
        mrr += 1 / rank if rank else 0
        report.append({"question": row["question"], "rank": rank, "top1": f"{results[0].project}/{results[0].path}" if results else None})
    n = len(rows)
    summary = {"n": n, **{f"recall@{kk}": round(v / n, 3) for kk, v in hits.items()}, "mrr": round(mrr / n, 3), "mode": mode}
    return {"summary": summary, "items": report}


def load_questions(path=ROOT / "eval" / "questions.jsonl", config=ROOT / "sources.toml"):
    """人工题 + 每张知识卡片自动生成的回归题。"""
    from .cards import eval_rows
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    d = cards_dir(config)
    return rows + (eval_rows(d) if d else [])


def cards_dir(config=ROOT / "sources.toml"):
    import tomllib
    with open(config, "rb") as f:
        return next((Path(s["root"]) for s in tomllib.load(f).get("source", []) if s.get("split_by_subdir")), None)


def cmd_eval(args):
    """检索评测：答案在不在前 k 个证据里。LLM 答得好的前提是证据先被找回来。"""
    mode = ("bm25" if args.bm25_only else "hybrid") + ("" if args.no_route else "_route")
    r = make_retriever(args)
    mode += "_rerank" if r.reranker else ""
    rows = load_questions(args.questions, args.config)
    t0 = time.time()
    result = evaluate(r, rows, args.k, args.use_project, mode)
    result["summary"]["ms_per_q"] = round((time.time() - t0) * 1000 / max(1, len(rows)))
    for item in result["items"]:
        print(f"{'OK ' if item['rank'] else 'MISS'} rank={item['rank']} {item['question']}  -> {item['top1']}")
    print(json.dumps(result["summary"], ensure_ascii=False))
    out = ROOT / "logs" / f"eval_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.gate or args.accept:
        ok, msg = gate(result, accept=args.accept)
        print(msg)
        if not ok:
            raise SystemExit(2)


BASELINE = ROOT / "logs" / "eval_baseline.json"  # 每台部署机自己的基线，不入 git


def gate(result, accept=False, tol=0.02):
    """评测卡口：和上次通过的基线比，任何一道老题从命中变成 MISS、或 recall@8 掉超过 tol，就拒绝上线。

    套路：只看总分会被“新题拉高、老题掉了”掩盖，所以按题对比；通过后基线自动前移（棘轮）。
    新加了难题导致总分下降时，人工确认后用 --accept 重置基线。返回 (是否通过, 说明)。
    """
    if BASELINE.exists() and not accept:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        was_ok = {i["question"] for i in base["items"] if i["rank"]}
        lost = [i["question"] for i in result["items"] if i["question"] in was_ok and not i["rank"]]
        key = next(k for k in result["summary"] if k.startswith("recall@") and k not in ("recall@1", "recall@3"))
        drop = base["summary"].get(key, 0) - result["summary"][key]
        if lost or drop > tol:
            return False, f"GATE FAIL：{key} 下降 {drop:.3f}；回退题 {len(lost)} 道" + "".join(f"\n  - {q}" for q in lost)
    BASELINE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return True, f"GATE PASS：基线已更新 {BASELINE.name}"


def cmd_serve(args):
    from .server import serve
    serve(args, make_retriever(args), make_llm(args))


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2) if xs else None


def bench_one(agent, row):
    """单题端到端打分。first_cited：模型自己写了 [n]（没靠程序补），反映“模型本身守不守规矩”，是后续微调要拉高的指标。"""
    t0, first = time.time(), []
    res = agent.ask(row["question"], on_event=lambda e: e["type"] == "token" and not first and first.append(time.time()))
    cited = [f"{s['project']}/{s['path']}" for s in res["sources"]]
    return {"question": row["question"], "ttft": round(first[0] - t0, 2) if first else None, "seconds": round(time.time() - t0, 2),
            "out_tokens": res["usage"].get("completion_tokens", 0), "cited_any": bool(cited),
            "first_cited": not any(t["step"] == "attach_cite" for t in res["trace"]),
            "cite_hit": any(e in c for c in cited for e in row["expect"]), "refused": "没有找到" in res["answer"],
            "answer": res["answer"]}


def bench_rows(agent, rows, concurrency=1, log=print):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock, items = threading.Lock(), []

    def one(row):
        it = bench_one(agent, row)
        with lock:
            items.append(it)
            log(f"{'HIT ' if it['cite_hit'] else 'MISS'} ttft={it['ttft']}s total={it['seconds']}s {row['question'][:40]}")

    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(one, rows))
    return bench_summary(items, time.time() - t0, concurrency), items


def bench_summary(items, wall, concurrency):
    n = max(1, len(items))
    rate = lambda key: round(sum(bool(i.get(key)) for i in items) / n, 3)
    return {"concurrency": concurrency, "n": len(items), "cite_hit": rate("cite_hit"), "cited_any": rate("cited_any"),
            "first_cited": rate("first_cited"), "refused": rate("refused"),
            "ttft_p50": pct([i["ttft"] for i in items if i["ttft"]], .5), "ttft_p90": pct([i["ttft"] for i in items if i["ttft"]], .9),
            "latency_p50": pct([i["seconds"] for i in items], .5), "latency_p90": pct([i["seconds"] for i in items], .9),
            "wall_s": round(wall, 1), "qps": round(len(items) / max(wall, 1e-6), 3),
            "out_tok_per_s": round(sum(i["out_tokens"] for i in items) / max(wall, 1e-6), 1)}


def cmd_bench(args):
    """端到端评测：检索评测只回答“证据找没找到”，这里回答“答案对不对、多快、能扛几个人”。

    每题记录：首字延迟 TTFT、总耗时、输出 token、引用命中（答案里 [n] 指向的文件是否含标准出处）、是否拒答。
    同一套题分别跑 transformers 和 vLLM、不同并发，结果写 logs/bench_*.json，直接可比。
    套路：质量和性能必须同一次跑出来——只看速度会选出“快但乱答”的配置。
    """
    from .agent import KnowledgeAgent
    if not args.api_base and args.concurrency > 1:
        raise SystemExit("本机 transformers 后端不支持并发生成，--concurrency 只能为 1（并发请用 --api-base 接 vLLM）")
    agent = KnowledgeAgent(make_retriever(args), make_llm(args), k=args.k)
    rows = load_questions(args.questions, args.config)[:args.limit]
    summary, items = bench_rows(agent, rows, args.concurrency, log=lambda s: print(s, flush=True))
    summary = {"backend": args.api_base or f"transformers{'+lora' if args.lora else ''}", **summary}
    print(json.dumps(summary, ensure_ascii=False))
    out = ROOT / "logs" / f"bench_{time.strftime('%Y%m%d_%H%M%S')}_c{args.concurrency}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细：{out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "sources.toml")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "index")
    ap.add_argument("--embed", type=Path, default=DEFAULT_EMBED)
    ap.add_argument("--llm", type=Path, default=DEFAULT_LLM)
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4 加载，显存约 5.5GB")
    ap.add_argument("--lora", type=Path, help="加载 QLoRA 适配器目录，如 finetune/out/lora")
    ap.add_argument("--api-base", help="改用 OpenAI 兼容服务，如 http://127.0.0.1:8000/v1 (vLLM)")
    ap.add_argument("--api-model", default="qwen2.5-coder-7b")
    ap.add_argument("--bm25-only", action="store_true", help="只用关键词检索（对比实验/无 GPU）")
    ap.add_argument("--no-route", action="store_true", help="关闭“问题点名项目 -> 只搜该项目”的路由（对比实验）")
    ap.add_argument("--rerank", type=Path, default=DEFAULT_RERANK, help="交叉编码器重排模型目录（不存在则跳过）")
    ap.add_argument("--no-rerank", action="store_true", help="关闭重排（对比实验）")
    ap.add_argument("--rerank-top", type=int, default=30, help="融合后取前 N 个送去重排")
    ap.add_argument("-k", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    p = sub.add_parser("summarize")
    p.add_argument("--project", nargs="*", help="只处理这些项目，默认全部代码文件")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_summarize)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--project")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--project")
    p.add_argument("--mode", choices=["rag", "agent"], default="rag")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("eval")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    p.add_argument("--use-project", action="store_true")
    p.add_argument("--gate", action="store_true", help="和 logs/eval_baseline.json 比：老题变 MISS 或 recall@8 降超 0.02 则退出码 2")
    p.add_argument("--accept", action="store_true", help="人工确认后把本次结果设为新基线")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("bench", help="端到端评测：引用命中率 + TTFT/延迟/吞吐，可对比后端与并发")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1", help="局域网上线用 0.0.0.0")
    p.add_argument("--concurrency", type=int, help="同时生成的请求数；默认本机模型 1、--api-base(vLLM) 8")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data", help="后台开关 site.json、问答库 kb.db、管理员口令")
    p.set_defaults(fn=cmd_serve)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
