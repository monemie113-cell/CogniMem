# CogniMem/quality_check.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from cognimem.layers.hybrid_encoder import HybridEncoder

def project_to_dim(seq, target_dim=384):
    """将序列投影到目标维度（使用与HybridEncoder相同的随机投影）"""
    n, d = seq.shape
    if d == target_dim:
        return seq
    np.random.seed(123)
    proj = np.random.randn(target_dim, d) * 0.1
    np.random.seed()
    return np.dot(seq, proj.T)  # (n, target_dim)

def full_attention(seq):
    n, d = seq.shape
    Q, K, V = seq, seq, seq
    scores = np.dot(Q, K.T) / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
    out = np.dot(attn_weights, V)
    return out

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.mean(np.sum(a_norm * b_norm, axis=1))

def test_config(window_size, global_samples, seq_len=100, dim=384):
    seq = np.random.randn(seq_len, dim).astype(np.float32)
    config = {'window_size': window_size, 'global_samples': global_samples, 'dim': dim}
    encoder = HybridEncoder(config)
    hysparse_out, _ = encoder.process(seq)
    full_out = full_attention(seq)
    sim = cosine_similarity(hysparse_out, full_out)
    return sim

def main():
    # 真实文本测试
    print("\n=== Real Text Test ===")
    from cognimem.utils.text_processor import embed_text
    texts = [
        "Python is a programming language.",
        "It is widely used for data science.",
        "NumPy is a key library for numerical computing.",
        "PyTorch is popular for deep learning.",
        "The ecosystem is rich and diverse."
    ]
    # 生成嵌入序列 (5, 96) 或 (5, 384) 取决于 fallback
    seq_raw = np.array([embed_text(t) for t in texts])
    print(f"Raw embedding shape: {seq_raw.shape}")
    # 统一投影到 384 维
    seq = project_to_dim(seq_raw, target_dim=384)
    print(f"Projected shape: {seq.shape}")
    config = {'window_size': 5, 'global_samples': 3, 'dim': 384}
    encoder = HybridEncoder(config)
    hysparse_out, _ = encoder.process(seq)
    full_out = full_attention(seq)
    sim = cosine_similarity(hysparse_out, full_out)
    print(f"Real text similarity: {sim:.4f}")

    print("\n=== Quality Check: HySparse vs Full Attention ===")
    # 基准配置
    base_window = 5
    base_samples = 3
    sim = test_config(base_window, base_samples)
    print(f"Baseline (w={base_window}, g={base_samples}): similarity = {sim:.4f}")

    # 变化窗口大小
    print("\n--- Varying window size (g=3) ---")
    for w in [3, 5, 7, 10]:
        sim = test_config(w, 3)
        print(f"w={w}: similarity = {sim:.4f}")

    # 变化全局采样点数
    print("\n--- Varying global samples (w=5) ---")
    for g in [1, 3, 5, 8]:
        sim = test_config(5, g)
        print(f"g={g}: similarity = {sim:.4f}")

    # 极端配置
    print("\n--- Extreme configurations ---")
    sim_large = test_config(15, 10)
    print(f"w=15, g=10: similarity = {sim_large:.4f}")
    sim_small = test_config(1, 0)
    print(f"w=1, g=0: similarity = {sim_small:.4f}")

if __name__ == "__main__":
    main()