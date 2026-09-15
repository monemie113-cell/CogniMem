# CogniMem/benchmarks/LongMemEval/check_predictions.py
import json

with open('cognimem_predictions.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"总行数: {len(lines)}\n")

for i, line in enumerate(lines[:3]):
    data = json.loads(line)
    print(f"=== 第 {i+1} 条 ===")
    print(f"question_id: {data.get('question_id')}")
    print(f"question: {data.get('question', '')[:120]}")
    retrieved = data.get('retrieved', [])
    print(f"retrieved 条数: {len(retrieved)}")
    for j, r in enumerate(retrieved[:3]):
        print(f"  [{j+1}] {r[:180]}")
    print()