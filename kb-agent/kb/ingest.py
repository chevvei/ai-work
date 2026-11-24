"""第 1 步：扫描知识源 -> 过滤 -> 切块。只依赖标准库（PDF 可选 pypdf）。

工程套路：
- 切块决定检索上限。块太大，向量被稀释、上下文塞不下；块太小，丢失函数完整语义。
- 代码优先在“函数结束 }”或空行处断开；Markdown 按标题断开并带上标题路径。
- 每块必须带 project/path/行号，回答才能给出可点击、可核对的引用。
"""
import fnmatch
import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

CODE_TARGET_LINES = 40
CODE_MAX_LINES = 120
MAX_CHARS = 3000
DOC_MAX_CHARS = 1800
HEADING = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass
class Chunk:
    id: str
    project: str
    path: str
    start: int
    end: int
    kind: str
    title: str
    text: str

    def embed_text(self):
        """嵌入时把项目/路径/标题拼进去：用户常问“udslib 里的 xxx”，路径本身就是强信号。"""
        return f"[{self.project}] {self.path} {self.title}\n{self.text}"


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    defaults = cfg.get("defaults", {})
    sources = []
    for src in cfg.get("source", []):
        merged = dict(defaults)
        merged.update(src)
        sources.append(merged)
    return sources


def _excluded(rel_parts, name, cfg):
    for part in rel_parts[:-1]:
        if any(fnmatch.fnmatch(part, pat) for pat in cfg["exclude_dirs"]):
            return True
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in cfg["exclude_globs"])


def _wanted(path, cfg):
    return path.suffix.lower() in cfg["extensions"] or path.name in cfg.get("filenames", [])


def iter_files(src):
    """按 include 白名单遍历，再用黑名单剔除。返回 (绝对路径, 相对路径)。"""
    root = Path(src["root"])
    seen = set()
    for inc in src["include"]:
        target = root / inc
        candidates = [target] if target.is_file() else sorted(target.rglob("*")) if target.is_dir() else []
        for path in candidates:
            if not path.is_file() or path in seen:
                continue
            rel = path.relative_to(root)
            if _excluded(rel.parts, path.name, src) or not _wanted(path, src):
                continue
            try:
                if path.stat().st_size > src["max_bytes"]:
                    continue
            except OSError:
                continue
            seen.add(path)
            yield path, rel.as_posix()


def read_text(path):
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return None  # 二进制
    for enc in ("utf-8", "gb18030"):  # 国内 Windows 工程常见 GBK 注释
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# 套路：文件级黑名单挡不住“代码里写死的默认密码”，入库前再做一次行内脱敏，保留变量名、只抹掉值
# 覆盖三种写法：x.passWord = "v"、setPassWord("v")、ini 里的 pc_password = v
_KEY = r"(?:pass_?word|passwd|pwd|psw|secret|api_?key|access_?key|token)"
SECRET = re.compile(rf"""(\w*{_KEY}\w*\s*(?:[:=]|\()\s*)(["'])[^"'\n]{{3,}}\2""", re.I)
SECRET_INI = re.compile(rf"""^(\s*\w*{_KEY}\w*\s*[:=]\s*)([^\s"'][^\n]*)$""", re.I | re.M)
# 注释/日志里的自然语言写法：“默认密码为 "123456"”“password to 123456”
SECRET_PROSE = re.compile(r"""((?:密码|password)\s*(?:为|是|to|is)\s*\\?["']?)[^\s"'\\)]{3,}""", re.I)


def redact(text):
    return SECRET_PROSE.sub(r"\1***", SECRET_INI.sub(r"\1***", SECRET.sub(r"\1\2***\2", text)))


def _make(project, rel, start, end, kind, title, text):
    cid = hashlib.sha1(f"{project}:{rel}:{start}:{end}".encode()).hexdigest()[:16]
    return Chunk(cid, project, rel, start, end, kind, title, redact(text.strip("\n")))


def _is_code_boundary(line):
    s = line.rstrip()
    return s in ("}", "};") or s == "" or s.startswith("} //") or s.startswith("}  //")


def chunk_code(project, rel, text):
    """贪心累积行；够长后遇到顶层 } 或空行就断；超长强制断开并保留 5 行重叠。"""
    lines = text.splitlines()
    chunks, start, size = [], 0, 0
    i = 0
    while i < len(lines):
        size += len(lines[i]) + 1
        count = i - start + 1
        natural = count >= CODE_TARGET_LINES and _is_code_boundary(lines[i])
        forced = count >= CODE_MAX_LINES or size >= MAX_CHARS
        if natural or forced or i == len(lines) - 1:
            body = "\n".join(lines[start:i + 1])
            if body.strip():
                chunks.append(_make(project, rel, start + 1, i + 1, "code", _guess_symbol(lines[start:i + 1]), body))
            next_start = i + 1 if natural or i == len(lines) - 1 else max(start + 1, i - 4)
            start, size = next_start, sum(len(x) + 1 for x in lines[next_start:i + 1])
            i = max(i + 1, next_start)
            continue
        i += 1
    return chunks


SYMBOL = re.compile(r"^(?:class|struct|namespace|enum)\s+\w+|^[\w:<>,\s\*&~]+\b(\w+::)?~?\w+\s*\([^;]*$")


def _guess_symbol(lines):
    """取块内第一个像定义的行做标题，方便结果列表一眼看出是哪个函数/类。"""
    for line in lines:
        s = line.strip()
        if s and not s.startswith(("//", "/*", "*", "#")) and SYMBOL.match(s):
            return s[:120]
    return ""


def chunk_markdown(project, rel, text):
    """按标题切；记住标题层级路径，块过大再按行窗口拆。"""
    lines = text.splitlines()
    chunks, stack = [], []
    sec_start, buf = 0, []

    def flush(end):
        body = "\n".join(buf)
        if not body.strip():
            return
        title = " > ".join(t for _, t in stack)
        if len(body) <= DOC_MAX_CHARS:
            chunks.append(_make(project, rel, sec_start + 1, end, "doc", title, body))
            return
        part, part_start, size = [], sec_start, 0
        for offset, line in enumerate(buf):
            part.append(line)
            size += len(line) + 1
            if size >= DOC_MAX_CHARS:
                chunks.append(_make(project, rel, part_start + 1, sec_start + offset + 1, "doc", title, "\n".join(part)))
                part, part_start, size = [], sec_start + offset + 1, 0
        if "\n".join(part).strip():
            chunks.append(_make(project, rel, part_start + 1, end, "doc", title, "\n".join(part)))

    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush(i)
            level = len(m.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2).strip())]
            sec_start, buf = i, [line]
        else:
            buf.append(line)
    flush(len(lines))
    return chunks


def chunk_pdf(project, rel, path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    chunks = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
        text = (page.extract_text() or "").strip()
        for k in range(0, len(text), DOC_MAX_CHARS):
            # PDF 没有行号，start/end 记页码，引用显示为 “p.页码”
            chunks.append(_make(project, rel, page_no, page_no, "pdf", f"第 {page_no} 页", text[k:k + DOC_MAX_CHARS]))
    return chunks


def chunk_file(project, rel, path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return chunk_pdf(project, rel, path)
    text = read_text(path)
    if not text or not text.strip():
        return []
    if suffix in (".md", ".txt"):
        return chunk_markdown(project, rel, text)
    return chunk_code(project, rel, text)


def ingest(config_path, out_path):
    """扫描所有知识源，写 chunks.jsonl；返回统计，方便判断是否漏扫/多扫。"""
    stats, files = {}, {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for src in load_config(config_path):
            n_files = n_chunks = 0
            for path, rel in iter_files(src):
                chunks = chunk_file(src["project"], rel, path)
                if not chunks:
                    continue
                n_files += 1
                files[f"{src['project']}/{rel}"] = str(path)
                for c in chunks:
                    out.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
                n_chunks += len(chunks)
            stats[src["project"]] = {"files": n_files, "chunks": n_chunks}
    # files.json 是 read 工具的白名单：Agent 只能读入库过的文件，不能任意读盘
    (out_path.parent / "files.json").write_text(json.dumps(files, ensure_ascii=False, indent=1), encoding="utf-8")
    aliases = {s["project"]: s.get("aliases", []) for s in load_config(config_path)}
    (out_path.parent / "projects.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def load_chunks(path):
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
