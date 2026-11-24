"""finetune 的纯函数单测：不加载模型、不读真实数据，几毫秒跑完。

套路：训练/评测脚本里最容易悄悄出错的是“数据怎么切、loss 算在哪、指标怎么判”，
这些都是纯函数，单测成本极低；而一次训练 20 分钟，靠跑训练发现这类 bug 代价太高。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import train_qlora  # noqa: E402
from finetune.eval_lora import code_faithful, derive_refusals, score, summarize  # noqa: E402
from kb.ingest import Chunk  # noqa: E402


def _chunk(cid, path):
    return Chunk(id=cid, project="p", path=path, start=1, end=5, kind="code", title="t", text=f"text of {cid}")


def _row(rid, kind="answer", project="p", source_id="a", evidence=("a", "b")):
    return {"id": rid, "kind": kind, "question": "q?", "source": f"{project}/x.cpp:1-5", "source_id": source_id,
            "evidence": list(evidence), "messages": [{}, {"content": "int x = 1;\nreturn x;"}], "answer": "x [1]", "auto_ok": True}


def test_code_faithful_rejects_invented_lines():
    ev = "int x = 1;\nreturn x;"
    assert code_faithful("看 [1]\n```cpp\nint x = 1;\n```", ev)
    assert not code_faithful("```cpp\nint x = 2;\n```", ev)
    assert code_faithful("没有代码块 [1]", ev)  # 没贴代码不算违规


def test_score_over_refusal_and_source_citation():
    row = _row("r1")
    good = score(row, "x 等于 1 [1]")
    assert good["cites_source"] and not good["over_refusal"]
    refused = score(row, "资料中没有找到与该问题直接相关的内容。")
    assert refused["over_refusal"] and not refused["has_cite"]
    assert score(_row("r2", kind="refusal"), "资料中没有找到")["refusal_ok"]


def test_summarize_ignores_none():
    s = summarize([{"cites_source": True}, {"cites_source": None}, {"cites_source": False}])
    assert s["cites_source"] == 0.5


def test_derive_refusals_removes_whole_source_file():
    by_id = {"a": _chunk("a", "x.cpp"), "a2": _chunk("a2", "x.cpp"), "b": _chunk("b", "y.cpp")}
    out = derive_refusals([_row("r1", evidence=("a", "a2", "b")), _row("r2", kind="refusal")], by_id)
    assert [r["id"] for r in out] == ["r1_ref"]  # 原本就是拒答题的不再派生
    assert out[0]["kind"] == "refusal" and out[0]["evidence"] == ["b"]  # 同文件的其他块也要拿掉，否则答案还在
    assert "x.cpp" not in out[0]["messages"][1]["content"]


def test_split_is_per_project_and_keeps_small_projects_in_train():
    rows = [_row(f"a{i}", project="big") for i in range(20)] + [_row(f"s{i}", project="tiny") for i in range(3)]
    train, val = train_qlora.split(rows, 0.15, seed=0)
    assert len(train) + len(val) == 23 and not {r["id"] for r in train} & {r["id"] for r in val}
    assert all(r["source"].startswith("big/") for r in val)  # 少于 4 条的项目不切 val，否则训练里一条都没有
    assert train_qlora.split(rows, 0.15, seed=0)[1] == val  # 同 seed 可复现


class _FakeTok:
    eos_token = "<e>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "P" * 5

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_masks_prompt_and_drops_too_long():
    row = {"messages": [], "answer": "ab"}
    enc = train_qlora.encode(_FakeTok(), row, max_len=100)
    assert enc["labels"][:5] == [-100] * 5  # prompt（证据）不算 loss
    assert enc["labels"][5:] == enc["input_ids"][5:] and len(enc["input_ids"]) == 5 + 2 + 3
    assert train_qlora.encode(_FakeTok(), row, max_len=8) is None  # 超长整条丢，不截断


def test_review_overrides_parses_status_and_fixed_answer(tmp_path, monkeypatch):
    (tmp_path / "review.md").write_text(
        "# 校对\n\n## s1\n状态: fix\n问题: q\n\n### 回答\n改过的回答 [1]\n\n### 证据（只读）\n- [1] x\n\n"
        "## s2\n状态: drop\n\n### 回答\n原回答\n\n### 证据（只读）\n", encoding="utf-8")
    monkeypatch.setattr(train_qlora, "DATA", tmp_path)
    over = train_qlora.review_overrides()
    assert over["s1"] == {"status": "fix", "answer": "改过的回答 [1]"}
    assert over["s2"]["status"] == "drop"
    kept, stats = train_qlora.load_samples([_row("s1"), _row("s2"), _row("s3")])
    assert [r["id"] for r in kept] == ["s1", "s3"] and stats == {"human_fix": 1, "human_drop": 1}
