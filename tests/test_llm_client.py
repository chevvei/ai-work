import json

import httpx
import pytest

from app.llm_client import ModelClient, ModelError, Settings


def settings(model):
    return Settings(model=model, api_key="test-key")


def test_openai_compatible_request_uses_model_id_and_v1():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["authorization"] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "你好"}}],
                                         "usage": {"prompt_tokens": 2, "completion_tokens": 3}})

    result = ModelClient(settings("openai/gpt-5.4"), httpx.MockTransport(handler)).ask("你好", "r1")
    assert seen["url"] == "https://tokenhub.robosense.cn/v1/chat/completions"
    assert seen["body"]["model"] == "openai/gpt-5.4"
    assert seen["authorization"] == "Bearer test-key"
    assert result["answer"] == "你好"
    assert result["usage"]["prompt_tokens"] == 2


def test_claude_uses_root_url_and_anthropic_messages_path():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-api-key"]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "回答"}],
                                         "usage": {"input_tokens": 4, "output_tokens": 2}})

    result = ModelClient(settings("anthropic/claude-sonnet-5"), httpx.MockTransport(handler)).ask("问题")
    assert seen["url"] == "https://tokenhub.robosense.cn/v1/messages"
    assert seen["body"]["model"] == "anthropic/claude-sonnet-5"
    assert seen["key"] == "test-key"
    assert result["answer"] == "回答"


def test_invalid_response_is_reported():
    def handler(request):
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(ModelError, match="invalid_upstream_response"):
        ModelClient(settings("gpt-5.4"), httpx.MockTransport(handler)).ask("问题")
