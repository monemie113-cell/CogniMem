import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
from .base import LLMClient

# Hugging Face 镜像源（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class HuggingFaceClient(LLMClient):
    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 256, temperature: float = 0.7):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """加载模型和分词器，使用镜像源加速"""
        try:
            print(f"Loading LLM model {self.model_name} on {self.device} from mirror...")
            # 增加超时时间，避免过早失败
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                timeout=60.0  # 60秒超时
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                timeout=60.0
            )
            if self.device == "cpu":
                self.model.to("cpu")
            self.model.eval()
            print("LLM model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model {self.model_name}: {e}")

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        full_prompt = prompt
        if context and context.get("memory"):
            memory_text = context["memory"]
            full_prompt = f"相关记忆：\n{memory_text}\n\n用户问题：{prompt}"

        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去除输入部分
        if response.startswith(full_prompt):
            response = response[len(full_prompt):].lstrip()
        return response