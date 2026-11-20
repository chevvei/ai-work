import logging
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.llm_client import ModelClient, ModelError, Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI(title="第一课：单轮模型问答")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空白")
        return value


@app.get("/health")
def health():
    # 存活检查不调用收费模型，也不代表上游已经联通。
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest):
    # 同步路由由 FastAPI 在线程池运行，阻塞 HTTP 请求不会占住事件循环。
    request_id = str(uuid4())
    try:
        return ModelClient(Settings.from_env()).ask(payload.question, request_id)
    except ModelError as exc:
        return JSONResponse(status_code=exc.status_code,
                            content={"request_id": request_id, "error": exc.kind})
