"""QLoRA 第 3 步：同一批 val 样本，底座 vs 底座+LoRA，量化“格式问题”有没有被修好。

指标（都由程序判定，不靠人眼）：
- has_cite        回答至少带一个合法 [n]
- cites_source    引用了出题的源块（答对出处）
- self_lineno     自己写了行号/路径:行号（应为 0，位置由系统给）
- code_faithful   回答里的代码块逐行都能在证据里找到（防“凭记忆重写代码”）
- refusal_ok      拒答样本是否回答“资料中没有找到”
两边用同样的 4bit 底座，差异只来自 LoRA，对比才公平。

运行：~/venvs/gpu/bin/python -m finetune.eval_lora
"""
import json
import re
import time
from pathlib import Path

import torch

from kb.agent import check_citations
from kb.cli import DEFAULT_LLM

from .make_data import SELF_LINENO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune" / "out"
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def code_faithful(answer, evidence_text):
    lines = [x.strip() for block in FENCE.findall(answer) for x in block.splitlines() if len(x.strip()) > 3]
    return all(x in evidence_text for x in lines)


def score(row, answer):
    evidence_text = row["messages"][1]["content"]
    if row["kind"] == "refusal":
        return {"refusal_ok": "没有找到" in answer}
    n_ev = len(row["evidence"])
    cites = check_citations(answer, n_ev)
    src_n = 1 + row["evidence"].index(row["source_id"]) if row.get("source_id") in row["evidence"] else None
    return {"has_cite": bool(cites["cited"]), "cites_source": src_n in cites["cited"] if src_n else None,
            "self_lineno": bool(SELF_LINENO.search(answer)), "code_faithful": code_faithful(answer, evidence_text)}


def summarize(scores):
    keys = sorted({k for s in scores for k in s})
    return {k: round(sum(bool(s[k]) for s in scores if s.get(k) is not None) / max(1, sum(s.get(k) is not None for s in scores)), 3)
            for k in keys}


def generate(model, tok, messages):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=512, do_sample=False, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(x) for x in (OUT / "val.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
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

    summary = {"n": len(rows), "base": summarize(res["base"]), "lora": summarize(res["lora"]), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = ROOT / "logs" / f"eval_lora_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
