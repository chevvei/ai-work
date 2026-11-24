"""后台“新增项目”：探测目录结构给出默认配置，确认后追加写回 sources.toml。

套路：只追加、不重写 sources.toml——手写的注释和顺序都保留，git diff 一眼能看出后台加了什么。
"""
import re
import subprocess
import tomllib
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{2,40}$")
CANDIDATES = ["src", "include", "inc", "lib", "doc", "docs", "Docs", "demo", "examples", "test", "tests", "tool", "tools",
              "scripts", "config", "cmake", "README.md", "README_CN.md", "CHANGELOG.md", "AGENTS.md", "CLAUDE.md", "CMakeLists.txt"]


def _git(root, *a):
    try:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def suggest(root):
    """给出 include / 别名 / 当前分支建议：README 第一行标题常常就是大家口头叫的名字。"""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f"目录不存在：{root}")
    include = [c for c in CANDIDATES if (root / c).exists()]
    title = ""
    for readme in ("README.md", "README_CN.md"):
        if (root / readme).exists():
            title = next((ln.lstrip("# ").strip() for ln in (root / readme).read_text(encoding="utf-8", errors="replace").splitlines()
                          if ln.startswith("#")), "")
            break
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", root.name)
    aliases = sorted({a for a in (title[:30], root.name.replace("_", " "), root.name.replace("-", " ")) if a and a.lower() != name.lower()})
    return {"name": name, "root": str(root), "include": include or ["."], "aliases": aliases,
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"), "is_git": bool(_git(root, "rev-parse", "--git-dir"))}


def existing(config):
    with open(config, "rb") as f:
        return {s["project"] for s in tomllib.load(f).get("source", [])}


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def add_source(config, name, root, include, aliases=(), ref="", link_to=()):
    """link_to：把本项目挂到已有产品下（例如新 SDK 被 monitor 使用），问那个产品时也会顺带检索本项目。"""
    if not NAME.match(name):
        raise ValueError("项目名只能用字母/数字/下划线/横线/中文，2-40 字符")
    if name in existing(config):
        raise ValueError(f"项目 {name} 已存在")
    if not Path(root).expanduser().is_dir():
        raise ValueError(f"目录不存在：{root}")
    if ref and not _git(root, "rev-parse", "--verify", ref):
        raise ValueError(f"分支/提交 {ref} 不存在")
    al = [a for a in [*aliases, *link_to] if a.strip()]
    lines = ["", "[[source]]", f"project = {_q(name)}"]
    if al:
        lines.append(f"aliases = [{', '.join(_q(a) for a in al)}]" + ("  # 含关联产品：" + "、".join(link_to) if link_to else ""))
    lines.append(f"root = {_q(Path(root).expanduser())}")
    if ref:
        lines.append(f"ref = {_q(ref)}")
    lines.append(f"include = [{', '.join(_q(i) for i in include)}]  # 后台新增")
    text = Path(config).read_text(encoding="utf-8").rstrip("\n") + "\n" + "\n".join(lines) + "\n"
    tomllib.loads(text)  # 先校验再落盘：拼坏了直接报错，不会把线上配置写坏
    Path(config).write_text(text, encoding="utf-8")
    return lines
