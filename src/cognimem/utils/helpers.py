import hashlib
import re

def hash_text(text: str) -> str:
    """返回字符串的 SHA-256 哈希值（十六进制）"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def normalize_text(text: str) -> str:
    """去除多余空白，转换为小写（英文），中文保留原样但去除两端空格"""
    text = text.strip()
    # 如果包含中文，不强制小写
    if re.search(r'[\u4e00-\u9fff]', text):
        return text
    return text.lower()

def truncate_text(text: str, max_len: int = 100) -> str:
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

def safe_json_loads(data, default=None):
    import json
    try:
        return json.loads(data)
    except:
        return default