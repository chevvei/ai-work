import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.agent import check_citations, format_evidence  # noqa: E402
from kb.ingest import Chunk, chunk_code, chunk_markdown  # noqa: E402
from kb.retrieve import dedupe, rrf, tokenize  # noqa: E402


def test_secret_redacted():
    from kb.ingest import redact
    text = "\n".join(['data.passWord = "123456";', 'std::string password = "abc123";', 'comm->setPassWord("654321");',
                      'std::string psw = "111222";', 'pc_password = 778899', 'std::string token = getToken();', 'int n = 0;',
                      '// 恢复默认密码为 "334455"', 'RLogInfo("Restoring default password to 556677");'])
    body = redact(text)
    assert not any(v in body for v in ("123456", "abc123", "654321", "111222", "778899", "334455", "556677"))
    assert 'passWord = "***"' in body and 'setPassWord("***")' in body and "pc_password = ***" in body
    assert "getToken();" in body  # 非字面量的赋值不动，避免把代码逻辑抹掉


def test_tokenize_splits_identifiers_and_keeps_hex():
    toks = tokenize("UdsClient::sendRequest(0x22) read_data_by_id 读取数据")
    assert {"udsclient", "uds", "client", "sendrequest", "send", "request"} <= set(toks)
    assert "0x22" in toks and "read_data_by_id" in toks and "data" in toks
    assert {"读取", "取数", "数据"} <= set(toks)


def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["b", "c", "a"]])[0] == "b"
    assert rrf([["x"], []]) == ["x"]


def test_chunk_code_breaks_at_boundaries_with_line_numbers():
    text = "\n".join(["int f() {"] + ["  x();"] * 45 + ["}", ""] + ["void g() {", "}"])
    chunks = chunk_code("p", "a.cpp", text)
    assert chunks[0].start == 1 and chunks[0].end == 47
    assert chunks[0].title.startswith("int f()")
    assert chunks[-1].text.endswith("}") and chunks[-1].end == len(text.splitlines())


def test_chunk_code_forced_split_overlaps():
    text = "\n".join(f"x{i}();" for i in range(300))
    chunks = chunk_code("p", "a.cpp", text)
    assert all(c.end - c.start + 1 <= 120 for c in chunks)
    assert chunks[1].start <= chunks[0].end  # 强制断开保留重叠


def test_chunk_markdown_heading_path_and_fence():
    text = "# 编译\n说明\n## Linux\n```bash\n# 不是标题\nmake\n```\n## Windows\nvs"
    chunks = chunk_markdown("p", "README.md", text)
    titles = [c.title for c in chunks]
    assert titles == ["编译", "编译 > Linux", "编译 > Windows"]
    assert "# 不是标题" in chunks[1].text


def _c(path, s, e):
    return Chunk(f"{path}{s}", "p", path, s, e, "code", "", "t")


def test_dedupe_overlapping_ranges():
    kept = dedupe([_c("a", 1, 40), _c("a", 30, 60), _c("a", 61, 80), _c("b", 1, 40)])
    assert [(c.path, c.start) for c in kept] == [("a", 1), ("a", 61), ("b", 1)]


def test_check_citations_flags_invalid():
    r = check_citations("结论 [1][3]，另见 [9]", 3)
    assert r == {"cited": [1, 3], "invalid": [9]}
    assert format_evidence([_c("a.cpp", 1, 2)]).startswith("[1] p/a.cpp L1-L2")


def test_attach_citations_by_overlap():
    from kb.agent import attach_citations
    a, b = _c("a.md", 1, 2), _c("b.md", 1, 2)
    a.text, b.text = "memory_map.xlsx 需要一个名为 flash 的 sheet，包含区域名和起始地址", "点击 Erase 按钮会发送擦除例程并二次确认"
    ans = "**结论**：需要名为 flash 的 sheet，包含区域名和起始地址。\n```\n点击 Erase 按钮\n```\n| 列 | 含义 |\n点击 Erase 按钮会发送擦除例程\n天气不错今天出门散步"
    out = attach_citations(ans, [a, b]).split("\n")
    assert out[0].endswith("起始地址 [1]。") and out[5].endswith("[2]")
    assert out[2] == "点击 Erase 按钮" and out[4] == "| 列 | 含义 |" and not out[6].endswith("]")


def test_bench_summary_first_cited_and_nightly_render():
    import importlib.util
    from kb.cli import bench_summary
    items = [{"cite_hit": True, "cited_any": True, "first_cited": True, "refused": False, "ttft": 1.0, "seconds": 5.0, "out_tokens": 100},
             {"cite_hit": False, "cited_any": True, "first_cited": False, "refused": False, "ttft": 2.0, "seconds": 9.0, "out_tokens": 50}]
    s = bench_summary(items, 10.0, 2)
    assert s["first_cited"] == 0.5 and s["cited_any"] == 1.0 and s["out_tok_per_s"] == 15.0
    spec = importlib.util.spec_from_file_location("nightly", Path(__file__).resolve().parents[1] / "scripts" / "nightly.py")
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    r = {"date": "d", "started": "21:30", "finished": "22:00", "seconds": 1800, "rebuild": {"state": "done", "seconds": 3, "log": []},
         "summarize": {"state": "skip"}, "selftest": {"summary": s, "items": [dict(i, question=f"q{n}") for n, i in enumerate(items)]},
         "health": {"asks": 1, "users": 1, "likes": 0, "dislikes": 0, "no_cite": 0, "refused": 0, "avg_seconds": 1, "bad": [], "feedback_pending": {}}}
    md = nightly.render(r, {"selftest": {"summary": dict(s, first_cited=0.25)}})
    assert "| first_cited | 0.5 | （+0.250） |" in md and "- q1" in md
