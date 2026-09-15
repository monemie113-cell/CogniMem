# CogniMem/benchmark_hybrid.py
import time
import numpy as np
import matplotlib.pyplot as plt
from src.cognimem.layers.hybrid_encoder import HybridEncoder


def run_benchmark():
    lengths = [50, 100, 200, 500, 1000, 2000, 5000]
    times = []

    for n in lengths:
        config = {'dim': 384, 'num_blocks': 2, 'sparse_layers_per_block': 4}
        encoder = HybridEncoder(config)
        seq = np.random.randn(n, 384).astype(np.float32)

        # 预热
        encoder.process(seq)

        start = time.perf_counter()
        for _ in range(5):
            encoder.process(seq)
        elapsed = (time.perf_counter() - start) / 5 * 1000
        times.append(elapsed)
        print(f"Length {n}: {elapsed:.2f} ms")

    # 验证线性增长：O(n) 意味着 time/n 应近似常数
    ratios = [t / n for t, n in zip(times, lengths)]
    print(f"\nTime/Length ratios: {[f'{r:.4f}' for r in ratios]}")
    print("如果 ratios 近似恒定 → 线性复杂度 O(n) 验证通过 ✓")

    plt.plot(lengths, times, 'o-')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('HySparse Hybrid Block Complexity')
    plt.show()


if __name__ == "__main__":
    run_benchmark()