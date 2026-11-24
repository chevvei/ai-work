from kb import cards, projects
from kb.ingest import Chunk
from kb.retrieve import dedupe
from kb.summarize import excerpt, parse, summary_text


def test_card_roundtrip_and_stale(tmp_path):
    src = tmp_path / "a.cpp"
    src.write_text("int a;\nint b;\nint c;\n", encoding="utf-8")
    files = {"proj/a.cpp": str(src)}
    p = cards.write_card(tmp_path / "cards", "proj", "b 是什么？", "b 是整数（参考1）", ["proj/a.cpp:2-2"], files)
    meta, body = cards.load(p)
    assert meta["question"] == "b 是什么？" and meta["refs"]["proj/a.cpp:2-2"]
    assert cards.card_answer(body) == "b 是整数（参考1）"
    assert cards.check_stale(tmp_path / "cards", files) == []
    src.write_text("int a;\nlong b;\nint c;\n", encoding="utf-8")
    assert cards.check_stale(tmp_path / "cards", files)[0]["changed"] == ["proj/a.cpp:2-2"]
    assert cards.eval_rows(tmp_path / "cards")[0]["expect"] == [f"proj-知识卡片/{p.name}"]


def test_renumber():
    assert cards.renumber("先 A [3]，再 B [1][9]", [{"n": 1}, {"n": 3}]) == "先 A （参考2），再 B （参考1）"


def test_card_alts(tmp_path):
    p = cards.write_card(tmp_path, "proj", "校验和怎么算？", "用 CRC32（参考1）", [], {},
                         alts=["CRC 怎么算", " ", "校验和怎么算？", "刷写校验失败"])
    meta, body = cards.load(p)
    assert meta["alts"] == ["CRC 怎么算", "刷写校验失败"] and "- CRC 怎么算" in body
    assert cards.card_answer(body) == "用 CRC32（参考1）"
    assert [r["question"] for r in cards.eval_rows(tmp_path)] == ["校验和怎么算？", "CRC 怎么算", "刷写校验失败"]


def test_store_auto_queue(tmp_path):
    from kb.store import Store
    s = Store(tmp_path)
    s.add_feedback("miss", "q1", "p", "", ["p/a:1-2"], alts=["q1b"])
    assert s.has_pending(None, "q1") and not s.has_pending("dislike", "q1")
    assert s.feedback()[0]["alts"] == ["q1b"]


def test_add_source(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text('[[source]]\nproject = "x"\nroot = "/tmp"\ninclude = ["."]\n', encoding="utf-8")
    projects.add_source(cfg, "newp", str(tmp_path), ["src"], ["新项目"], link_to=["monitor"])
    assert projects.existing(cfg) == {"x", "newp"}
    try:
        projects.add_source(cfg, "newp", str(tmp_path), ["src"])
        assert False
    except ValueError:
        pass


def test_summary_parse():
    d = parse('```json\n{"summary": "实现点云缺失检测 PointCloudMissingDetector。", "questions": ["点云缺失怎么报警？", "这个文件是做什么的？", "阈值在哪改？"]}\n```')
    assert d["summary"].startswith("实现点云") and d["questions"] == ["点云缺失怎么报警？", "阈值在哪改？"]
    d = parse("- 这个文件定义了 RegInfo 结构体。\n- 如何比较两个 RegInfo？\n")
    assert d["summary"].startswith("这个文件定义") and d["questions"] == ["如何比较两个 RegInfo？"]
    assert "可能的问法" in summary_text(d)
    assert "以下仅列出符号" in excerpt("x" * 4000 + "\nclass Foo {\n")


def test_dedupe_keeps_code_under_summary():
    s = Chunk("s", "p", "a.cpp", 1, 100, "summary", "", "")
    c = Chunk("c", "p", "a.cpp", 10, 40, "code", "", "")
    c2 = Chunk("c2", "p", "a.cpp", 30, 50, "code", "", "")
    assert [x.id for x in dedupe([s, c, c2, s])] == ["s", "c"]
