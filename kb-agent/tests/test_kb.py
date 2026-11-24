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
