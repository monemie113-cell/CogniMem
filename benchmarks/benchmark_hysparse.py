# CogniMem/benchmark.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

# 导入两种编码器
from cognimem.layers.hybrid_encoder import HybridEncoder          # NumPy 版本
from cognimem.layers.hybrid_encoder_torch import HybridEncoderTorch  # PyTorch 版本
import torch

def full_attention(seq):
    n, d = seq.shape
    Q = seq
    K = seq
    V = seq
    scores = np.dot(Q, K.T) / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
    out = np.dot(attn_weights, V)
    return out

# ----- NumPy 版本 (原有) -----
def benchmark_hysparse_numpy(seq_len, dim=384, repeats=10):
    config = {'window_size': 5, 'global_samples': 3, 'dim': dim}
    encoder = HybridEncoder(config)
    seq = np.random.randn(seq_len, dim).astype(np.float32)
    # 预热
    encoder.process(seq)
    # 计时
    start = time.perf_counter()
    for _ in range(repeats):
        encoder.process(seq)
    elapsed = (time.perf_counter() - start) / repeats * 1000  # ms
    # 内存
    tracemalloc.start()
    encoder.process(seq)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return elapsed, peak_mb

# ----- PyTorch CPU 版本 -----
def benchmark_hysparse_torch_cpu(seq_len, dim=384, repeats=10):
    config = {'window_size': 5, 'global_samples': 3, 'dim': dim, 'device': 'cpu'}
    encoder = HybridEncoderTorch(config)
    seq = np.random.randn(seq_len, dim).astype(np.float32)
    # 预热
    encoder.process(seq)
    # 计时
    start = time.perf_counter()
    for _ in range(repeats):
        encoder.process(seq)
    elapsed = (time.perf_counter() - start) / repeats * 1000
    # 内存（粗略，PyTorch 内存较难精确，用 tracemalloc 近似）
    tracemalloc.start()
    encoder.process(seq)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return elapsed, peak_mb

# ----- PyTorch GPU 版本（如果可用）-----
def benchmark_hysparse_torch_gpu(seq_len, dim=384, repeats=10):
    if not torch.cuda.is_available():
        return float('nan'), float('nan')
    config = {'window_size': 5, 'global_samples': 3, 'dim': dim, 'device': 'cuda'}
    encoder = HybridEncoderTorch(config)
    seq = np.random.randn(seq_len, dim).astype(np.float32)
    # 预热
    encoder.process(seq)
    # 计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        encoder.process(seq)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000
    # 内存：GPU 内存较难获取，返回 NaN
    peak_mb = float('nan')
    return elapsed, peak_mb

# ----- 全注意力基准 (仅 NumPy) -----
def benchmark_full(seq_len, dim=384, repeats=10):
    seq = np.random.randn(seq_len, dim).astype(np.float32)
    full_attention(seq)  # 预热
    start = time.perf_counter()
    for _ in range(repeats):
        full_attention(seq)
    elapsed = (time.perf_counter() - start) / repeats * 1000
    tracemalloc.start()
    full_attention(seq)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return elapsed, peak_mb

def main():
    lengths = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # 数据存储
    results = {
        'numpy_time': [],
        'torch_cpu_time': [],
        'torch_gpu_time': [],
        'full_time': [],
        'numpy_mem': [],
        'torch_cpu_mem': [],
        'torch_gpu_mem': [],
        'full_mem': []
    }

    for n in lengths:
        print(f"Testing length {n}...")
        # NumPy
        t1, m1 = benchmark_hysparse_numpy(n, repeats=5)
        results['numpy_time'].append(t1)
        results['numpy_mem'].append(m1)
        # Torch CPU
        t2, m2 = benchmark_hysparse_torch_cpu(n, repeats=5)
        results['torch_cpu_time'].append(t2)
        results['torch_cpu_mem'].append(m2)
        # Torch GPU
        t3, m3 = benchmark_hysparse_torch_gpu(n, repeats=5)
        results['torch_gpu_time'].append(t3)
        results['torch_gpu_mem'].append(m3)
        # Full attention
        t4, m4 = benchmark_full(n, repeats=5)
        results['full_time'].append(t4)
        results['full_mem'].append(m4)

    # 打印表格
    print("\n=== Benchmark Results (Extended) ===")
    print("SeqLen\tNumPy(ms)\tTorchCPU(ms)\tTorchGPU(ms)\tFull(ms)\tNumPy(MB)\tTorchCPU(MB)\tTorchGPU(MB)\tFull(MB)")
    for i, n in enumerate(lengths):
        print(f"{n}\t{results['numpy_time'][i]:.2f}\t\t{results['torch_cpu_time'][i]:.2f}\t\t{results['torch_gpu_time'][i]:.2f}\t\t{results['full_time'][i]:.2f}\t\t{results['numpy_mem'][i]:.2f}\t\t{results['torch_cpu_mem'][i]:.2f}\t\t{results['torch_gpu_mem'][i]:.2f}\t\t{results['full_mem'][i]:.2f}")

    # 绘图（只绘制时间）
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(lengths, results['numpy_time'], 'o-', label='NumPy HySparse')
        plt.plot(lengths, results['torch_cpu_time'], 's-', label='Torch CPU HySparse')
        if not all(np.isnan(x) for x in results['torch_gpu_time']):
            plt.plot(lengths, results['torch_gpu_time'], 'd-', label='Torch GPU HySparse')
        plt.plot(lengths, results['full_time'], '^-', label='Full Attention')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.title('Time Complexity Comparison')
        plt.grid(True)
        plt.savefig('benchmark_comparison.png')
        print("Plot saved as benchmark_comparison.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()