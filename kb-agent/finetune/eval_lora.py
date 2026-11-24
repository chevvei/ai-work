"""QLoRA 第 3 步：同一批 val 样本，底座 vs 底座+LoRA，量化“格式问题”有没有被修好。

指标（都由程序判定，不靠人眼）：
- has_cite        回答至少带一个合法 [n]
- cites_source    引用了出题的源块（答对出处）
- self_lineno     自己写了行号/路径:行号（应为 0，位置由系统给）
- code_faithful   回答里的代码块逐行都能在证据里找到（防“凭记忆重写代码”）
- over_refusal    正常题（证据里有源块）却回答“没有找到”——误拒率，越低越好
- refusal_ok      拒答样本是否回答“资料中没有找到”
两边用同样的 4bit 底座，差异只来自 LoRA，对比才公平。

套路：拒答和误拒必须成对看。只看 refusal_ok 会鼓励“一律拒答”，只看 has_cite 会鼓励“硬答”。

运行：
  ~/venvs/gpu/bin/python -m finetune.eval_lora                      # 原 val（68 条，含 5 条拒答）
  ~/venvs/gpu/bin/python -m finetune.eval_lora --extra-refusal      # 再从正常题派生拒答题，扩大拒答样本
  ~/venvs/gpu/bin/python -m finetune.eval_lora --extra-refusal --dry-run   # 只看派生了多少条，不占 GPU
  ~/venvs/gpu/bin/python -m finetune.eval_lora --rescore logs/eval_lora_X.json  # 改了指标后离线重算，不占 GPU
"""
import argparse
import json
import re
import time
from pathlib import Path

from kb.agent import check_citations
from kb.cli import DEFAULT_LLM

from .make_data import SELF_LINENO, build_messages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune" / "out"
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
REFUSED = "没有找到"


def code_faithful(answer, evidence_text):
    lines = [x.strip() for block in FENCE.findall(answer) for x in block.splitlines() if len(x.strip()) > 3]
    return all(x in evidence_text for x in lines)


def score(row, answer):
    evidence_text = row["messages"][1]["content"]
    if row["kind"] == "refusal":
        return {"refusal_ok": REFUSED in answer}
    n_ev = len(row["evidence"])
    cites = check_citations(answer, n_ev)
    src_n = 1 + row["evidence"].index(row["source_id"]) if row.get("source_id") in row["evidence"] else None
    return {"has_cite": bool(cites["cited"]), "cites_source": src_n in cites["cited"] if src_n else None,
            "self_lineno": bool(SELF_LINENO.search(answer)), "code_faithful": code_faithful(answer, evidence_text),
            "over_refusal": REFUSED in answer}


def summarize(scores):
    keys = sorted({k for s in scores for k in s})
    return {k: round(sum(bool(s[k]) for s in scores if s.get(k) is not None) / max(1, sum(s.get(k) is not None for s in scores)), 3)
            for k in keys}


def derive_refusals(rows, by_id):
    """从正常题派生拒答题：把源文件的所有块从证据里拿掉，问题不变。

    套路：和 make_data 造拒答训练样本的做法一致（分布一致），但只用 val 里的题，
    不会和训练集泄漏。代价是证据少了几块，模型看到的“干扰证据”是真实检索结果，比随便塞无关块更难。
    已知噪声：同名 .h/.cpp、讲同一符号的文档仍可能留在证据里含答案（实测 14 个“没拒”里 13 个属此类），
    所以 refusal_ok 在派生集上是保守下界，失败样本要抽检。
    """
    out = []
    for r in rows:
        src = by_id.get(r.get("source_id"))
        if r["kind"] != "answer" or src is None:
            continue
        ev = [by_id[cid] for cid in r["evidence"] if cid in by_id and by_id[cid].path != src.path]
        if not ev:
            continue
        out.append({**r, "id": r["id"] + "_ref", "kind": "refusal", "evidence": [c.id for c in ev],
                    "messages": build_messages(r["question"], ev)})
    return out


def load_val(extra_refusal):
    rows = [json.loads(x) for x in (OUT / "val.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if extra_refusal:
        from kb.ingest import load_chunks
        by_id = {c.id: c for c in load_chunks(ROOT / "index" / "chunks.jsonl")}
        rows += derive_refusals(rows, by_id)
    return rows


def rescore(log_path):
    """改了打分规则后，用已保存的回答重算，不必再花 10 分钟跑生成。"""
    log = json.loads(Path(log_path).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in load_val(extra_refusal=any(it["id"].endswith("_ref") for it in log["items"]))}
    res = {"base": [], "lora": []}
    for it in log["items"]:
        row = rows[it["id"]]
        for side in res:
            it[f"score_{side}"] = score(row, it[side])
            res[side].append(it[f"score_{side}"])
    log["summary"].update(base=summarize(res["base"]), lora=summarize(res["lora"]), rescored=time.strftime("%Y%m%d_%H%M%S"))
    Path(log_path).write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(log["summary"], ensure_ascii=False, indent=1))


def generate(model, tok, messages):
    import torch
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=512, do_sample=False, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-refusal", action="store_true", help="从正常题派生拒答题，扩大拒答样本")
    ap.add_argument("--dry-run", action="store_true", help="只统计样本构成，不加载模型")
    ap.add_argument("--rescore", metavar="LOG", help="用已有 eval_lora_*.json 的回答离线重算指标")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.rescore)

    rows = load_val(args.extra_refusal)
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("answer", "refusal")}
    print("[val]", json.dumps(kinds), flush=True)
    if args.dry_run:
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(DEFAULT_LLM)
    base = AutoModelForCausalLM.from_pretrained(
        DEFAULT_LLM, device_map="cuda", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    model = PeftModel.from_pretrained(base, OUT / "lora").eval()

    items, res = [], {"base": [], "lora": []}
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        with model.disable_adapter():  # 同一份权重关掉 LoRA 就是底座，省一次加载和 5GB 显存
            a_base = generate(model, tok, row["messages"])
        a_lora = generate(model, tok, row["messages"])
        s_base, s_lora = score(row, a_base), score(row, a_lora)
        res["base"].append(s_base)
        res["lora"].append(s_lora)
        items.append({"id": row["id"], "question": row["question"], "base": a_base, "lora": a_lora, "score_base": s_base, "score_lora": s_lora})
        print(f"[{i}/{len(rows)}] base {s_base} | lora {s_lora}", flush=True)

    summary = {"n": len(rows), **kinds, "base": summarize(res["base"]), "lora": summarize(res["lora"]), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = ROOT / "logs" / f"eval_lora_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
