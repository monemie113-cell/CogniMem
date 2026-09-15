import numpy as np
import hashlib
import json
from typing import Optional, Dict

class EngramTable:
    """
    基于哈希键值对的快速知识缓存表。
    键：输入文本的 MD5 哈希（十六进制字符串）
    值：对应的语义向量（numpy 数组）
    """
    def __init__(self, embed_dim: int = 384, max_size: int = 10000):
        self.embed_dim = embed_dim
        self.max_size = max_size
        self._data: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        根据原始文本键查找向量。
        若存在则增加命中计数，否则增加未命中计数。
        """
        hash_key = self._hash(key)
        vec = self._data.get(hash_key)
        if vec is not None:
            self.hits += 1
            return vec
        else:
            self.misses += 1
            return None

    def put(self, key: str, vector: np.ndarray):
        """存储键值对，若超出最大容量则按 LRU 策略淘汰（此处简化为丢弃最早添加的）"""
        hash_key = self._hash(key)
        if len(self._data) >= self.max_size:
            # 简单淘汰：删除第一个插入的（可用 collections.OrderedDict 优化）
            first_key = next(iter(self._data))
            del self._data[first_key]
        self._data[hash_key] = vector

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def save(self, filepath: str):
        """将数据保存为 JSON 文件（向量转为列表）"""
        data_serializable = {k: v.tolist() for k, v in self._data.items()}
        with open(filepath, 'w') as f:
            json.dump(data_serializable, f)

    def load(self, filepath: str):
        """从 JSON 文件加载数据"""
        with open(filepath, 'r') as f:
            data_serializable = json.load(f)
        for k, v_list in data_serializable.items():
            self._data[k] = np.array(v_list)