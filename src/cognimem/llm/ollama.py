import requests
import json
from typing import Dict, Any, Optional
from .base import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, model_name: str = "qwen2.5:0.5b", base_url: str = "http://localhost:11434", timeout: int = 60):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self):
        """检查 Ollama 服务是否可用"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama service not responding at {self.base_url}")
            models = resp.json().get("models", [])
            if not any(m.get("name") == self.model_name for m in models):
                raise RuntimeError(f"Model '{self.model_name}' not found in Ollama. Please pull it first.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")

    def load_model(self):
        """Ollama 无需显式加载，服务端管理"""
        pass

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """调用 Ollama 生成回复"""
        full_prompt = prompt
        if context and context.get("memory"):
            memory_text = context["memory"]
            full_prompt = f"相关记忆：\n{memory_text}\n\n用户问题：{prompt}"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 256
            }
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API error: {resp.status_code} - {resp.text}")
            result = resp.json()
            return result.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")