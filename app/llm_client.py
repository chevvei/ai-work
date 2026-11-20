import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class ModelError(Exception):
    def __init__(self, kind: str, status_code: int = 502):
        super().__init__(kind)
        self.kind = kind
        self.status_code = status_code


@dataclass
class Settings:
    model: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls):
        model = os.environ.get("LLM_MODEL", "").strip()
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not model or not api_key:
            raise ModelError("请在本机设置 LLM_MODEL 和 LLM_API_KEY", 503)
        return cls(model=model, api_key=api_key)

    @property
    def is_claude(self):
        return self.model.startswith(("claude-", "anthropic/claude-"))

    @property
    def base_url(self):
        if self.is_claude:
            return "https://tokenhub.robosense.cn/"
        return "https://tokenhub.robosense.cn/v1"


class ModelClient:
    def __init__(self, settings: Settings, transport: Optional[httpx.BaseTransport] = None):
        self.settings = settings
        self.transport = transport

    def ask(self, question: str, request_id: Optional[str] = None):
        request_id = request_id or str(uuid4())
        started = time.perf_counter()
        status = "error"
        error_type = None
        usage = None
        try:
            # 模型 ID 原样传给网关：带供应商前缀的 ID 不擅自改名。
            body = {"model": self.settings.model,
                    "messages": [{"role": "user", "content": question}]}
            if self.settings.is_claude:
                url = self.settings.base_url + "v1/messages"
                headers = {"x-api-key": self.settings.api_key,
                           "anthropic-version": "2023-06-01"}
                body["max_tokens"] = 1024
            else:
                url = self.settings.base_url + "/chat/completions"
                headers = {"Authorization": "Bearer " + self.settings.api_key}
                body["max_completion_tokens"] = 1024

            # 第一版不自动重试，避免入门练习重复计费；超时先显式报告。
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0),
                              transport=self.transport) as client:
                response = client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                code = response.status_code
                kind = {400: "upstream_bad_request", 401: "upstream_authentication",
                        403: "upstream_permission", 404: "upstream_not_found",
                        429: "upstream_rate_limit"}.get(code, "upstream_error")
                raise ModelError(kind, 429 if code == 429 else 502)
            try:
                data = response.json()
                if self.settings.is_claude:
                    answer = "".join(block["text"] for block in data["content"]
                                     if block["type"] == "text")
                else:
                    answer = data["choices"][0]["message"]["content"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("无有效文本回答")
                # 保留各协议的原始 usage 字段；缺失用量不伪造为零。
                usage = data.get("usage")
                if usage is not None and not isinstance(usage, dict):
                    raise ValueError("usage 格式错误")
            except (ValueError, KeyError, IndexError, TypeError):
                raise ModelError("invalid_upstream_response") from None
            status = "ok"
            return {"request_id": request_id, "answer": answer,
                    "model": self.settings.model,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "usage": usage}
        except httpx.TimeoutException:
            error_type = "upstream_timeout"
            raise ModelError(error_type, 504) from None
        except httpx.RequestError:
            error_type = "upstream_network"
            raise ModelError(error_type) from None
        except ModelError as exc:
            error_type = exc.kind
            raise
        finally:
            # 不记录 Key、请求头、提问正文或上游原始错误正文。
            token_fields = {}
            if usage:
                token_fields = {key: usage[key] for key in
                                ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")
                                if isinstance(usage.get(key), int)}
            logger.info("request_id=%s model=%s latency_ms=%.2f status=%s error_type=%s tokens=%s",
                        request_id, self.settings.model,
                        (time.perf_counter() - started) * 1000,
                        status, error_type, token_fields)
