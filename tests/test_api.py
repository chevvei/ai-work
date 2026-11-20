from fastapi.testclient import TestClient

from app.main import app
from app.llm_client import ModelClient

client = TestClient(app)


def test_health_does_not_need_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert client.get("/health").json() == {"status": "ok"}


def test_blank_question_is_rejected():
    assert client.post("/chat", json={"question": "  "}).status_code == 422


def test_missing_config_is_reported(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    response = client.post("/chat", json={"question": "你好"})
    assert response.status_code == 503
    assert "request_id" in response.json()


def test_chat_success(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def ask(self, question, request_id):
        return {"answer": "模拟回答", "request_id": request_id,
                "model": self.settings.model, "latency_ms": 1, "usage": None}

    monkeypatch.setattr(ModelClient, "ask", ask)
    response = client.post("/chat", json={"question": "你好"})
    assert response.status_code == 200
    assert response.json()["answer"] == "模拟回答"
