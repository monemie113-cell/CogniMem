# CogniMem/test_liquid.py
import sys
import os

# 将 src 目录添加到 sys.path（优先级最高）
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 现在可以正常导入
import numpy as np
from cognimem.models import LiquidCell

def main():
    # 初始化液态细胞
    cell = LiquidCell(hidden_dim=4)
    state = None

    # 模拟连续的输入流
    inputs = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
    ]

    print("=== 液态推理状态演化 (CfC) ===")
    for i, inp in enumerate(inputs):
        output, state = cell.forward(inp, state)
        print(f"Step {i+1} -> 输出状态(前4维): {np.round(output[:4], 4)}")
        print(f"       状态变化幅度: {np.linalg.norm(state):.4f}")

if __name__ == "__main__":
    main()