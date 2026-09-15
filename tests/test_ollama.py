# test_ollama.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cognimem.llm import OllamaClient

client = OllamaClient(model_name="qwen2.5:0.5b")
response = client.generate("请用中文简单介绍你自己")
print(response)