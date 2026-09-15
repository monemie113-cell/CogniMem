# CogniMem/test_gate.py
import sys
import os

# 设置 Hugging Face 镜像源（放在最前面）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 将 src 目录加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cognimem.models.small_gate import SmallGateModel

def main():
    gate = SmallGateModel({})

    # 测试1：短文本
    print("=== 输入：'你好' ===")
    scores1 = gate.compute_scores("你好")
    print(scores1)

    # 测试2：长指令
    print("\n=== 输入：'请帮我分析Q3的销售数据，并生成报表' ===")
    scores2 = gate.compute_scores("请帮我分析Q3的销售数据，并生成报表")
    print(scores2)

    # 测试3：与上下文无关
    print("\n=== 输入：'天气不错'（提供上下文：销售数据） ===")
    scores3 = gate.compute_scores("天气不错", context="销售数据")
    print(scores3)

if __name__ == "__main__":
    main()