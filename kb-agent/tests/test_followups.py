import threading
from types import SimpleNamespace

from kb.agent import STOPPED, KnowledgeAgent, pick_followups, similar


def test_similar_and_pick():
    assert similar("如何计算CRC校验", "如何计算CRC校验？") > .9
    out = pick_followups("CRC 怎么算", ["- CRC 怎么算", "1. 超时时间在哪配置", "超时时间在哪里配置", "短", "安全访问的种子怎么生成"])
    assert out == ["超时时间在哪配置？", "安全访问的种子怎么生成？"]


class FakeLLM:
    def __init__(self, ev=None):
        self.ev, self.calls = ev, 0

    def chat(self, messages, max_new_tokens=None, on_token=None, stop_event=None):
        self.calls += 1
        if self.ev:
            self.ev.set()
        return "答案在这里 [1]" if max_new_tokens is None else "配置文件在哪\n怎么调试"


def chunk(i):
    return SimpleNamespace(id=i, project="p", path=f"f{i}.cpp", start=1, end=9, kind="code", title="", text="x")


def retriever():
    s = SimpleNamespace(id=99, text="文件概要：x\n可能的问法：\n- 超时怎么配置\n- 重试几次")
    return SimpleNamespace(search=lambda *a, **k: [chunk(1)], summary_of={("p", "f1.cpp"): s})


def test_followups_from_summary_first():
    llm = FakeLLM()
    events = []
    res = KnowledgeAgent(retriever(), llm).ask("怎么发请求", on_event=events.append, followups=True)
    assert res["followups"] == ["超时怎么配置？", "重试几次？"] and llm.calls == 1
    assert {"type": "followups", "items": res["followups"]} in events


def test_stop_skips_followups():
    ev = threading.Event()
    res = KnowledgeAgent(retriever(), FakeLLM(ev)).ask("怎么发请求", stop_event=ev, followups=True)
    assert res["stopped"] and res["answer"].endswith(STOPPED.strip()) and res["followups"] == []
