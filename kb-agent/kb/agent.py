"""第 4 步：问答编排。

两种模式：
- rag：检索一次 -> 把证据编号塞进提示 -> 要求逐句引用 [n]。稳定、快，MVP 首选。
- agent：在 rag 基础上允许模型最多调用 N 次工具（search / read）补证据，再作答。
  适合“这个函数在哪被调用”“这个宏的值是多少”这种一次检索拿不全的问题。

防幻觉三板斧：只许用证据回答；每个结论带 [n]；证据不足就明说“资料中没有找到”。
回答后再做一次引用校验：引用了不存在的编号，直接标记出来，而不是相信模型。
"""
import re
import time

SYSTEM = """你是公司内部软件知识库助手，负责回答 LidarAssistant、monitor、udslib、rs_driver 等项目的开发与使用问题。
规则：
1. 只依据【资料】回答，不要编造资料里没有的函数名、参数、数值。
2. 每个关键结论后用 [编号] 标注出处，例如 “在 UdsClient::sendRequest 中设置超时 [2]”。
3. 资料不足以回答时，明确说“资料中没有找到”，并指出最可能相关的文件。
4. 用中文回答，先给结论，再给步骤；代码标识符保持原样。不要自己写行号或文件路径，系统会根据 [编号] 自动附上精确位置。
5. 不要改写或凭记忆重写代码；需要展示代码时只原样摘抄资料中的几行。"""

AGENT_RULE = """
你可以先调用工具补充资料（每次只输出一行，不要输出其他内容）：
ACTION: search | <检索词，可带函数名/宏名/中文描述>
ACTION: read | <项目/相对路径> | <起始行> | <结束行>
资料已足够时，只输出 DONE。"""

# 套路：7B 小模型隔着上千 token 证据会“忘记” system 里的格式要求，把关键约束在问题后再说一遍（近因效应）
ANSWER_TAIL = "\n\n回答要求：资料里与问题相关的步骤不要遗漏；资料里没有的内容不要写、不要推测；每个结论句末必须带 [编号]，如“先安装依赖 [3]”。"

ACTION = re.compile(r"^\s*ACTION:\s*(search|read)\s*\|\s*(.+)$", re.M)
CITE = re.compile(r"\[(\d+)\]")


def format_evidence(chunks, max_chars=2400):
    parts = []
    for n, c in enumerate(chunks, 1):
        loc = f"p.{c.start}" if c.kind == "pdf" else f"L{c.start}-L{c.end}"
        body = c.text if len(c.text) <= max_chars else c.text[:max_chars] + "\n...(截断)"
        parts.append(f"[{n}] {c.project}/{c.path} {loc} {c.title}\n```\n{body}\n```")
    return "\n\n".join(parts)


def check_citations(answer, n_evidence):
    cited = sorted({int(x) for x in CITE.findall(answer)})
    return {"cited": [i for i in cited if 1 <= i <= n_evidence], "invalid": [i for i in cited if not 1 <= i <= n_evidence]}


def source_list(chunks, cited):
    out = []
    for i in cited:
        c = chunks[i - 1]
        out.append({"n": i, "project": c.project, "path": c.path, "start": c.start, "end": c.end, "kind": c.kind})
    return out


class KnowledgeAgent:
    def __init__(self, retriever, llm, k=8, max_steps=3):
        self.retriever, self.llm, self.k, self.max_steps = retriever, llm, k, max_steps

    def ask(self, question, project=None, mode="rag"):
        t0 = time.time()
        evidence = self.retriever.search(question, k=self.k, project=project)
        trace = [{"step": "search", "query": question, "hits": len(evidence)}]
        if mode == "agent":
            evidence, trace = self._agent_loop(question, evidence, project, trace)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"【资料】\n{format_evidence(evidence)}\n\n【问题】{question}{ANSWER_TAIL}"}]
        answer = self.llm.chat(messages)
        cites = check_citations(answer, len(evidence))
        if evidence and not cites["cited"]:
            # 套路：提示词约束是“软”的，程序校验是“硬”的。没引用就把原回答发回去要求补引用，只重试一次
            trace.append({"step": "recite"})
            answer = self.llm.chat(messages + [{"role": "assistant", "content": answer}, {"role": "user", "content":
                                   "上面的回答缺少出处。请保持内容不变，在每个结论句末补上对应的资料编号 [n]；找不到出处的句子删掉。"}])
            cites = check_citations(answer, len(evidence))
        return {"question": question, "answer": answer, "sources": source_list(evidence, cites["cited"]),
                "invalid_citations": cites["invalid"], "evidence": [f"{c.project}/{c.path}:{c.start}-{c.end}" for c in evidence],
                "trace": trace, "seconds": round(time.time() - t0, 2), "usage": getattr(self.llm, "last_usage", {})}

    def _agent_loop(self, question, evidence, project, trace):
        for _ in range(self.max_steps):
            messages = [{"role": "system", "content": SYSTEM + AGENT_RULE},
                        {"role": "user", "content": f"【资料】\n{format_evidence(evidence, 1200)}\n\n【问题】{question}\n"
                                                    "如果资料不足，输出一行 ACTION；否则输出 DONE。"}]
            reply = self.llm.chat(messages, max_new_tokens=80)
            m = ACTION.search(reply)
            if not m:
                trace.append({"step": "done"})
                break
            tool, arg = m.group(1), m.group(2).strip()
            new = []
            if tool == "search":
                new = self.retriever.search(arg, k=4, project=project)
            else:
                fields = [x.strip() for x in arg.split("|")]
                if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
                    hit = self.retriever.read(fields[0], fields[1], fields[2])
                    new = [hit] if hit else []
            known = {(c.project, c.path, c.start) for c in evidence}
            new = [c for c in new if (c.project, c.path, c.start) not in known]
            trace.append({"step": tool, "arg": arg, "added": len(new)})
            evidence = evidence + new
            if len(evidence) >= self.k + 6:  # 证据上限：上下文越长，小模型越容易“看漏”
                break
        return evidence, trace
