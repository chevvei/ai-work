"""B 方案：让知识库“读懂代码”——离线为每个代码文件生成中文概要 + 预测问法，作为额外的块入库。

为什么有效：用户问“点云缺了怎么报警”，代码里只有 PointCloudMissingDetector / kMissThreshold，
中英文、口语和标识符之间没有字面重叠，BM25 和向量都容易漏。先让模型把代码“翻译”成人话和问法，
检索就变成“问题对问题”的匹配（业内叫 doc2query / HyDE 的离线版）。

套路：
- 以文件内容 sha1 为键缓存，代码没变就不重算；新提交只算改动的文件，可以放夜间跑。
- 概要块和原代码块指向同一个文件/行号：命中概要后，引用仍落到真实代码位置。
- 只喂文件头部 + 符号行，控制 prompt 长度；7B 写概要比写答案容易得多，错了也只影响召回、不会直接给用户看。
"""
import hashlib
import json
import re
from pathlib import Path

CODE_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".py"}
SYMBOL = re.compile(r"^\s*(class|struct|enum|namespace|def|#define|template|typedef|using)\b|^\s*[\w:<>,\*&\s]+\([^;]*\)\s*(const)?\s*\{?\s*$")

PROMPT = """下面是项目 {project} 的源文件 {path}。
请输出一个 JSON 对象，不要输出其他任何内容：
{{"summary": "一到两句中文，说明这个文件实现了什么功能、在什么场景下用到，提到关键类名/函数名",
 "questions": ["开发或客户可能怎么问到这个文件，4 个口语化中文问题，要带具体功能词或类名/函数名，不要写“这个文件是做什么的”这类空泛问题"]}}

```
{code}
```"""


def file_key(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def excerpt(text, head_chars=3500, max_symbols=60):
    """头部（include/注释/类声明）+ 后面所有符号行：用 1/3 的 token 保住文件的“目录”。"""
    head = text[:head_chars]
    rest = [ln.rstrip() for ln in text[head_chars:].splitlines() if SYMBOL.match(ln) and len(ln) < 200][:max_symbols]
    return head + ("\n// ...（以下仅列出符号）\n" + "\n".join(rest) if rest else "")


def parse(reply):
    """优先解析 JSON；小模型不守格式时退化为：问号结尾的行当问法，其余第一行当概要。

    套路：让小模型输出结构化内容，JSON 比“标签: 内容”可靠得多，但仍要有宽松兜底解析，别让一个文件的格式问题卡住整批。
    """
    m = re.search(r"\{.*\}", reply, re.S)
    d = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = {}
    if isinstance(d, dict) and d.get("summary"):
        summary, qs = str(d["summary"]), [str(q) for q in d.get("questions") or []]
    else:
        lines = [re.sub(r"^[-*•\d.、\s]+|\*\*|概要[:：]|问法[:：]", "", ln).strip() for ln in reply.splitlines()]
        lines = [ln for ln in lines if ln]
        qs = [ln for ln in lines if ln.endswith(("？", "?"))]
        rest = [ln for ln in lines if ln not in qs]
        summary = rest[0] if rest else ""
    qs = [q.strip() for q in qs if len(q.strip()) > 3 and "这个文件" not in q and "该文件" not in q]
    return {"summary": summary.strip(), "questions": qs[:6]}


def load_cache(path):
    path = Path(path)
    if not path.exists():
        return {}
    return {d["key"]: d for d in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if d.get("summary")}


def todo(files, cache, projects=None):
    out = []
    for ref, abs_path in files.items():
        project = ref.split("/", 1)[0]
        if Path(ref).suffix.lower() not in CODE_EXT or (projects and project not in projects) or not Path(abs_path).exists():
            continue
        if file_key(abs_path) not in cache:
            out.append((ref, abs_path))
    return out


def summarize(llm, files, cache_path, projects=None, limit=None, lock=None, log=print, should_stop=lambda: False):
    """逐个文件生成并追加写缓存（断了可以续跑）。lock：和线上问答共用的 GPU 锁，每个文件单独拿，前台不会被饿死。"""
    import contextlib
    cache = load_cache(cache_path)
    jobs = todo(files, cache, projects)[:limit]
    log(f"待生成 {len(jobs)} 个文件（已缓存 {len(cache)}）")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for ref, abs_path in jobs:
            if should_stop():
                break
            project, rel = ref.split("/", 1)
            text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            with lock or contextlib.nullcontext():
                reply = llm.chat([{"role": "user", "content": PROMPT.format(project=project, path=rel, code=excerpt(text))}], max_new_tokens=260)
            d = parse(reply)
            if d["summary"]:
                f.write(json.dumps({"key": file_key(abs_path), "ref": ref, **d}, ensure_ascii=False) + "\n")
                f.flush()
            else:
                log(f"解析失败 {ref}：{reply[:200]!r}")
            done += 1
            if done % 10 == 0 or done == len(jobs):
                log(f"{done}/{len(jobs)} {ref}")
    return done


def summary_text(d):
    return f"文件概要：{d['summary']}\n可能的问法：\n" + "\n".join(f"- {q}" for q in d["questions"])
