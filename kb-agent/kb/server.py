"""第 5 步：本机网页问答。只监听 127.0.0.1（源码不出本机）；GPU 生成串行化，避免并发请求把显存打爆。"""
import threading
from pathlib import Path

PAGE = Path(__file__).with_name("chat.html")


def serve(args, retriever, llm):
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    from .agent import KnowledgeAgent

    agent = KnowledgeAgent(retriever, llm, k=args.k)
    lock = threading.Lock()
    app = FastAPI(title="本地知识库 Agent")
    projects = sorted({c.project for c in retriever.chunks})

    class Ask(BaseModel):
        question: str = Field(min_length=1, max_length=2000)
        project: str | None = None
        mode: str = Field(default="rag", pattern="^(rag|agent)$")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/projects")
    def list_projects():
        return {"projects": projects, "chunks": len(retriever.chunks)}

    @app.post("/api/ask")
    def ask(req: Ask):
        if req.project and req.project not in projects:
            raise HTTPException(400, "未知项目")
        if not lock.acquire(timeout=300):
            raise HTTPException(503, "GPU 忙，请稍后再试")
        try:
            return agent.ask(req.question.strip(), project=req.project or None, mode=req.mode)
        finally:
            lock.release()

    uvicorn.run(app, host="127.0.0.1", port=args.port)
