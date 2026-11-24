"""知识卡片：把“纠错 / 起草 + 管理员审核”的结果沉淀成 Markdown 文件，作为普通知识源入库。

为什么是文件而不是数据库：
- 卡片要能被 git 管理、被人直接改、被 ingest 当普通文档切块，和 FAQ 走同一条链路，不需要第二套检索。
- 每张卡片记录它依据的代码位置和内容 hash；代码一改，hash 对不上就标“过期”，回到审核队列。
  套路：知识不怕错一次，怕“对过一次之后悄悄过期”。

目录：<cards_dir>/<产品>/<id>.md，sources.toml 里一个 split_by_subdir 源自动展开成 <产品>-知识卡片。
"""
import hashlib
import json
import re
import time
from pathlib import Path

REF = re.compile(r"^(?P<file>[^:]+?)(?::(?P<start>\d+)-(?P<end>\d+))?$")
SAFE = re.compile(r"[^\w\u4e00-\u9fff-]+")


def parse_ref(ref):
    m = REF.match(ref.strip())
    if not m:
        return None
    return m["file"], int(m["start"] or 0), int(m["end"] or 0)


def ref_hash(ref, files):
    """引用片段的内容 hash；没写行号就 hash 整个文件。文件已不在知识库里返回 None。"""
    p = parse_ref(ref)
    if not p or p[0] not in files or not Path(files[p[0]]).exists():
        return None
    lines = Path(files[p[0]]).read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(lines[p[1] - 1:p[2]] if p[1] else lines)
    return hashlib.sha1(body.encode()).hexdigest()[:12]


def dump(meta, body):
    # 值统一用 JSON 编码：不依赖 yaml 库，中文/引号/列表都不会转义出错
    head = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    return f"---\n{head}\n---\n{body.strip()}\n"


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text
    head, _, body = text[4:].partition("\n---\n")
    meta = {}
    for line in head.splitlines():
        k, _, v = line.partition(":")
        try:
            meta[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError:
            meta[k.strip()] = v.strip()
    return meta, body


def write_card(cards_dir, product, question, answer, refs, files, kind="", author="admin", card_id=None, alts=()):
    """写卡片。正文开头重复问题：检索时“问题原话”是最强的匹配信号。

    alts：其他问法（客户说法 / 开发说法 / 报错原文）。和主问题放在同一块里一起被检索，且每条都变成回归题。
    套路：一张卡片覆盖一个“意图”，而不是一句话；多写两种问法，比多写两张卡片更不容易互相打架。
    """
    card_id = card_id or time.strftime("%Y%m%d%H%M%S") + "_" + SAFE.sub("_", question)[:24].strip("_")
    path = Path(cards_dir) / SAFE.sub("_", product) / f"{card_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    alts = [a.strip() for a in alts if a.strip() and a.strip() != question]
    meta = {"id": card_id, "product": product, "question": question, "alts": alts, "kind": kind, "author": author,
            "updated": time.strftime("%Y-%m-%d %H:%M"), "refs": {r: ref_hash(r, files) for r in refs if r.strip()}}
    ref_lines = "".join(f"\n- 参考{i}：`{r}`" for i, r in enumerate(meta["refs"], 1))
    alt_lines = "".join(f"\n- {a}" for a in alts)
    body = (f"# {question}\n" + (f"\n其他问法：{alt_lines}\n" if alts else "") + f"\n{answer.strip()}\n"
            + (f"\n相关代码/文档：{ref_lines}\n" if ref_lines else ""))
    path.write_text(dump(meta, body), encoding="utf-8")
    return path


def all_cards(cards_dir):
    root = Path(cards_dir)
    return [(p, *load(p)) for p in sorted(root.glob("*/*.md"))] if root.exists() else []


def card_answer(body):
    """从卡片正文还原“答案”部分（去掉标题行和参考列表），用于重新审核时回填编辑框。"""
    body = re.sub(r"^# .*\n+", "", body.lstrip(), count=1)
    body = re.sub(r"^其他问法：\n(- .*\n)*\n?", "", body, count=1)
    return body.split("\n相关代码/文档：")[0].strip()


CITE = re.compile(r"\[(\d+)\]")


def renumber(answer, sources):
    """模型回答里的 [n] 是“本次证据编号”，写进卡片后就失效了；改成卡片自己的 参考i，与 refs 顺序对应。"""
    order = {s["n"]: i for i, s in enumerate(sources, 1)}
    return CITE.sub(lambda m: f"（参考{order[int(m[1])]}）" if int(m[1]) in order else "", answer)


def check_stale(cards_dir, files):
    """返回过期卡片：引用的代码片段内容变了，或文件已不在知识库里。"""
    out = []
    for path, meta, _ in all_cards(cards_dir):
        changed = [r for r, h in (meta.get("refs") or {}).items() if ref_hash(r, files) != h]
        if changed:
            out.append({"path": str(path), "id": meta.get("id"), "product": meta.get("product"),
                        "question": meta.get("question"), "changed": changed})
    return out


def eval_rows(cards_dir):
    """每张卡片自动变成一道回归题：以后任何改动让它检索不到，评测卡口就会拦下。"""
    rows = []
    for path, meta, _ in all_cards(cards_dir):
        for q in [meta.get("question"), *(meta.get("alts") or [])]:
            if q:
                rows.append({"question": q, "expect": [f"{path.parent.name}-知识卡片/{path.name}"], "source": "card"})
    return rows
