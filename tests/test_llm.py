# test_llm.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cognimem.llm import HuggingFaceClient

client = HuggingFaceClient(model_name="Qwen/Qwen2.5-0.5B", device="cpu")
response = client.generate("你好，请简单介绍一下你自己")
print(response)