# CogniMem/test_engram.py
import sys
import os
import numpy as np  # 确保导入 numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognimem.layers.engram import EngramLayer

config = {'engram': {'embed_dim': 384, 'table_size': 100}}
engram = EngramLayer(config)

texts = [
    "Python是谁发明的？",
    "Python是谁发明的？",  # 重复
    "什么是机器学习？"
]

for t in texts:
    vec, meta = engram.process(t)
    print(f"文本: {t}")
    print(f"  缓存命中: {meta['cached']}, 命中率: {meta['hit_rate']:.2f}")
    print(f"  向量范数: {np.linalg.norm(vec):.4f}\n")