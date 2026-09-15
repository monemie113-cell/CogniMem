# test_ollama_api.py
import requests

try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    if resp.status_code == 200:
        print("Ollama API 正常！模型列表:", resp.json().get("models", []))
    else:
        print(f"API 返回状态码 {resp.status_code}")
except Exception as e:
    print(f"API 连接失败: {e}")