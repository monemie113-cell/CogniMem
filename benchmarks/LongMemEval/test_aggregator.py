import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from cognimem.memory.aggregator import Aggregator

agg = Aggregator()

# 用之前 10 条样本的真实数据测试
test_cases = [
    {
        'query': 'How many items of clothing do I need to pick up or return from a store?',
        'retrieved': ['I need to pick up 2 shirts from the tailor', 'I have to return 1 dress to the store'],
        'expected_type': 'COUNT',
    },
    {
        'query': 'How many hours in total did I spend driving?',
        'retrieved': ['It took me 5 hours to drive there', 'The drive back was 10 hours'],
        'expected_type': 'SUM',
        'expected_answer': '15 hours',
    },
    {
        'query': 'How much total money have I spent on bike-related expenses?',
        'retrieved': ['I bought a new tire for $50', 'The repair cost $80', 'A helmet for $55'],
        'expected_type': 'SUM',
        'expected_answer': '$185',
    },
    {
        'query': 'How many projects have I led or am currently leading?',
        'retrieved': ['I led a marketing project last year', 'I am currently leading a data migration project'],
        'expected_type': 'COUNT',
    },
]

print("=== 聚合器单元测试 ===\n")
for i, tc in enumerate(test_cases):
    result = agg.aggregate(tc['query'], tc['retrieved'])
    print(f"用例 {i+1}: {tc['query'][:60]}")
    print(f"  期望类型: {tc['expected_type']}")
    if result:
        print(f"  实际类型: {result['type']}")
        print(f"  输出答案: {result['answer']}")
        print(f"  置信度: {result['confidence']:.2f}")
        if 'expected_answer' in tc:
            match = "✅" if result['answer'] == tc['expected_answer'] else f"❌ (期望 {tc['expected_answer']})"
            print(f"  匹配: {match}")
    else:
        print(f"  ❌ 未产生聚合结果")
    print()