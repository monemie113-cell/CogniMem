# CogniMem/test_hysparse.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
from cognimem.layers.hybrid_encoder import HybridEncoder

config = {'window_size': 3, 'global_samples': 2, 'dim': 384}
encoder = HybridEncoder(config)

seq = np.random.randn(5, 384)
encoded, meta = encoder.process(seq)
print("向量序列输出形状:", encoded.shape)
print("元信息:", meta)

text = "这是一个测试句子"
encoded2, meta2 = encoder.process(text)
print("字符串输出形状:", encoded2.shape)
print("字符串元信息:", meta2)