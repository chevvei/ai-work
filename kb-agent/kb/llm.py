"""本地大模型后端。

- TransformersLLM：直接在本机 3090 上加载 HF 权重（默认 BF16，7B≈15GB 显存；--4bit 用 bitsandbytes≈5.5GB）。
- OpenAICompatLLM：对接 vLLM / Ollama / llama.cpp server 等 OpenAI 兼容接口，换后端不改业务代码。
套路：业务层只依赖 chat(messages) 一个接口，推理引擎可插拔——这正是部署岗天天做的事。
"""
import json
import threading
import urllib.request


class TransformersLLM:
    def __init__(self, model_path, load_4bit=False, max_new_tokens=768, lora=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if lora:
            # 套路：底座和适配器分开发布，换版本只换几十 MB 的 adapter；训练时是 4bit 底座，部署用 BF16 也能加载
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages, max_new_tokens=None, stop=None, on_token=None, stop_event=None):
        """on_token：流式回调，每解码出一段文本调用一次。
        套路：generate 放到子线程，主线程从 TextIteratorStreamer 里边取边推——首字延迟从“整段生成完”降到“prefill 完”。"""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        kwargs = dict(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                      repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id,
                      stop_strings=stop, tokenizer=self.tokenizer if stop else None)
        if stop_event is not None:
            # 套路：停止生成靠 StoppingCriteria——每解码一个 token 问一次“要不要停”，GPU 立刻释放，而不是只断开前端连接
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):
                    return stop_event.is_set()
            kwargs["stopping_criteria"] = StoppingCriteriaList([_Stop()])
        if on_token is None:
            with self.torch.no_grad():
                out = self.model.generate(**kwargs)
            n_out = int(out.shape[1] - inputs["input_ids"].shape[1])
            text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            import threading
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            box = {}

            def run():
                with self.torch.no_grad():  # no_grad 是线程局部的，必须在子线程里再开一次
                    box["out"] = self.model.generate(**kwargs, streamer=streamer)
            th = threading.Thread(target=run, daemon=True)
            th.start()
            text = ""
            for piece in streamer:
                text += piece
                if piece:
                    on_token(piece)
            th.join()
            n_out = int(box["out"].shape[1] - inputs["input_ids"].shape[1])
        self.last_usage = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": n_out}
        return text.strip()


class OpenAICompatLLM:
    def __init__(self, base_url, model, api_key="local", max_new_tokens=768, timeout=300):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.max_new_tokens, self.timeout = model, api_key, max_new_tokens, timeout
        self._tls = threading.local()  # 并发请求各在自己的线程里，用量统计放线程局部，避免互相覆盖

    @property
    def last_usage(self):
        return getattr(self._tls, "usage", {})

    def chat(self, messages, max_new_tokens=None, stop=None, on_token=None, stop_event=None):
        """总是用 stream=True：既能边收边推给前端，又能在 stop_event 置位时直接关连接——vLLM 发现客户端断开会 abort 该请求，
        KV cache 立即回收。套路：远端引擎的“停止”就是断开连接，不需要额外的取消接口。"""
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_new_tokens or self.max_new_tokens,
                "repetition_penalty": 1.05, "stream": True, "stream_options": {"include_usage": True}}
        if stop:
            body["stop"] = stop
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                           "Authorization": f"Bearer {self.api_key}"})
        text, self._tls.usage = "", {}
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                if stop_event is not None and stop_event.is_set():
                    break
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                data = json.loads(line[5:])
                self._tls.usage = data.get("usage") or self._tls.usage
                for ch in data.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    text += piece
                    if piece and on_token:
                        on_token(piece)
        return text.strip()
