"""本地大模型后端。

- TransformersLLM：直接在本机 3090 上加载 HF 权重（默认 BF16，7B≈15GB 显存；--4bit 用 bitsandbytes≈5.5GB）。
- OpenAICompatLLM：对接 vLLM / Ollama / llama.cpp server 等 OpenAI 兼容接口，换后端不改业务代码。
套路：业务层只依赖 chat(messages) 一个接口，推理引擎可插拔——这正是部署岗天天做的事。
"""
import json
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

    def chat(self, messages, max_new_tokens=None, stop=None):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id,
                                      stop_strings=stop, tokenizer=self.tokenizer if stop else None)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_usage = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(out.shape[1] - inputs["input_ids"].shape[1])}
        return text.strip()


class OpenAICompatLLM:
    def __init__(self, base_url, model, api_key="local", max_new_tokens=768, timeout=300):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.max_new_tokens, self.timeout = model, api_key, max_new_tokens, timeout

    def chat(self, messages, max_new_tokens=None, stop=None):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_new_tokens or self.max_new_tokens}
        if stop:
            body["stop"] = stop
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                           "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        self.last_usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"].strip()
