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


def make_llm(args):
    from .llm import OpenAICompatLLM, TransformersLLM
    if args.api_base:
        return OpenAICompatLLM(args.api_base, args.api_model)
    return TransformersLLM(str(args.llm), load_4bit=args.load_4bit)


def make_retriever(args, vectors=True):
    from .retrieve import Retriever
    r = Retriever(args.index_dir, use_vectors=vectors and not args.bm25_only)
    r.route = not args.no_route
    return r


def cmd_ingest(args):
    from .ingest import ingest
    stats = ingest(args.config, args.index_dir / "chunks.jsonl")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def cmd_index(args):
    from .retrieve import build_index
    t0 = time.time()
    n = build_index(args.index_dir, args.embed)
    print(f"向量化 {n} 块，用时 {time.time() - t0:.1f}s")


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


def cmd_eval(args):
    """检索评测：答案在不在前 k 个证据里。LLM 答得好的前提是证据先被找回来。"""
    r = make_retriever(args)
    rows = [json.loads(x) for x in args.questions.read_text(encoding="utf-8").splitlines() if x.strip()]
    hits = {1: 0, 3: 0, args.k: 0}
    mrr, report = 0.0, []
    for row in rows:
        results = r.search(row["question"], k=args.k, project=row.get("project") if args.use_project else None)
        rank = next((i for i, c in enumerate(results, 1) if any(e in f"{c.project}/{c.path}" for e in row["expect"])), None)
        for k in hits:
            hits[k] += bool(rank and rank <= k)
        mrr += 1 / rank if rank else 0
        report.append({"question": row["question"], "rank": rank, "top1": f"{results[0].project}/{results[0].path}" if results else None})
    n = len(rows)
    summary = {"n": n, **{f"recall@{k}": round(v / n, 3) for k, v in hits.items()}, "mrr": round(mrr / n, 3),
               "mode": ("bm25" if args.bm25_only else "hybrid") + ("" if args.no_route else "_route")}
    for item in report:
        print(f"{'OK ' if item['rank'] else 'MISS'} rank={item['rank']} {item['question']}  -> {item['top1']}")
    print(json.dumps(summary, ensure_ascii=False))
    out = ROOT / "logs" / f"eval_{summary['mode']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "items": report}, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_serve(args):
    from .server import serve
    serve(args, make_retriever(args), make_llm(args))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "sources.toml")
    ap.add_argument("--index-dir", type=Path, default=ROOT / "index")
    ap.add_argument("--embed", type=Path, default=DEFAULT_EMBED)
    ap.add_argument("--llm", type=Path, default=DEFAULT_LLM)
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4 加载，显存约 5.5GB")
    ap.add_argument("--api-base", help="改用 OpenAI 兼容服务，如 http://127.0.0.1:8000/v1 (vLLM)")
    ap.add_argument("--api-model", default="qwen2.5-coder-7b")
    ap.add_argument("--bm25-only", action="store_true", help="只用关键词检索（对比实验/无 GPU）")
    ap.add_argument("--no-route", action="store_true", help="关闭“问题点名项目 -> 只搜该项目”的路由（对比实验）")
    ap.add_argument("-k", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    sub.add_parser("index").set_defaults(fn=cmd_index)
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
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8770)
    p.set_defaults(fn=cmd_serve)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
