"""QLoRA 第 2 步：用校对后的样本微调 Qwen2.5-Coder-7B 的“看资料、带引用回答”格式。

工程套路（每条都对应一个真实踩坑点）：
- 底座 4bit NF4 冻结 + LoRA(q/k/v/o, r=16)：7B 训练显存从 >60GB 压到 ~14GB，3090 单卡可跑。
- 只对 assistant 段算 loss：prompt 里 2~3k token 的证据不是要模型“背”的，算进 loss 会让它学会复读代码。
- 超长样本整条丢弃而不是截断：截断会把答案砍掉或把被引用的 [n] 证据砍掉，等于教模型乱引用。
- 按项目分层切 train/val：val 用来和底座对比（eval_lora.py），不能混进训练。
- gradient checkpointing + paged_adamw_8bit：用算力换显存，长序列必开。

运行：~/venvs/gpu/bin/python -m finetune.train_qlora [--epochs 2] [--max-len 5120]
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from kb.cli import DEFAULT_LLM

from .eval_lora import code_faithful
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "finetune" / "data"
OUT = ROOT / "finetune" / "out"


def review_overrides():
    """读 review.md：状态 ok/fix/drop，fix 时取人工改过的“### 回答”段。"""
    path = DATA / "review.md"
    if not path.exists():
        return {}
    res = {}
    for block in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
        sid = block.split("\n", 1)[0].strip()
        status = re.search(r"^状态:\s*(\w+)", block, re.M)
        answer = re.search(r"^### 回答\n(.*?)\n### 证据", block, re.S | re.M)
        res[sid] = {"status": status.group(1) if status else "todo", "answer": answer.group(1).strip() if answer else None}
    return res


def load_samples(rows):
    over = review_overrides()
    kept, stats = [], defaultdict(int)
    for r in rows:
        o = over.get(r["id"])
        if o and o["status"] == "drop":
            stats["human_drop"] += 1
            continue
        if o and o["status"] == "fix" and o["answer"]:
            r["answer"], r["auto_ok"] = o["answer"], True
            stats["human_fix"] += 1
        elif o and o["status"] == "ok":
            r["auto_ok"] = True
            stats["human_ok"] += 1
        if not r["auto_ok"]:
            stats["auto_fail_drop"] += 1  # 没过自动校验、人也没救回来的，不进训练
            continue
        if not (o and o["status"] == "fix") and not code_faithful(r["answer"], r["messages"][1]["content"]):
            # 套路：训练目标来自底座自己的回答（拒绝采样），过滤器漏掉的毛病会被“训进去”，所以要和评测用同一把尺子
            stats["code_unfaithful_drop"] += 1
            continue
        kept.append(r)
    return kept, dict(stats)


def split(rows, val_ratio, seed):
    rng = random.Random(seed)
    by_proj = defaultdict(list)
    for r in rows:
        by_proj[r["source"].split("/")[0]].append(r)
    train, val = [], []
    for items in by_proj.values():
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) >= 4 else 0
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def encode(tok, row, max_len):
    """prompt 部分 label 置 -100，只让 assistant 回答参与 loss。"""
    prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(row["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L], labels[i, :L], mask[i, :L] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    # 套路：max_len 按 token 分布的 P97 定，而不是拍脑袋；8 块证据的 RAG 样本 P50≈3.4k，3072 会丢掉六成
    ap.add_argument("--max-len", type=int, default=5120)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    rows = [json.loads(x) for x in (DATA / "drafts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    # 套路：先切 val 再过滤。val 只用问题+证据，自动校验失败的“难题”恰恰最该留在 val 里考微调效果
    train_all, val_rows = split(rows, args.val_ratio, args.seed)
    train_rows, stats = load_samples(train_all)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows), encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(DEFAULT_LLM)
    train_ds = [e for e in (encode(tok, r, args.max_len) for r in train_rows) if e]
    stats.update(train=len(train_ds), train_too_long=len(train_rows) - len(train_ds), val=len(val_rows))
    print("[data]", json.dumps(stats, ensure_ascii=False), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_LLM, device_map="cuda", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = Trainer(
        model=model, train_dataset=train_ds, data_collator=lambda b: collate(b, tok.pad_token_id),
        args=TrainingArguments(output_dir=str(OUT / "ckpt"), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
                               num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                               logging_steps=5, save_strategy="no", bf16=True, optim="paged_adamw_8bit", report_to=[],
                               gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                               remove_unused_columns=False, seed=args.seed))
    result = trainer.train()
    model.save_pretrained(OUT / "lora")
    report = {"time": time.strftime("%Y%m%d_%H%M%S"), "args": vars(args), "data": stats, "train_loss": round(result.training_loss, 4),
              "seconds": round(time.time() - t0, 1), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
              "loss_curve": [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / f"train_{report['time']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done]", json.dumps({k: report[k] for k in ("train_loss", "seconds", "peak_mem_gb")}), flush=True)


if __name__ == "__main__":
    main()
