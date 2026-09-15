# CogniMem/benchmarks/diagnose_bm25_effect.py
import os
import sys
import json
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cognimem.core.pipeline import CogniMemPipeline

config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

pipeline = CogniMemPipeline(config)
mem = pipeline.memory

# 存储少量对话
test_dialogues = [
    {'content': 'I went to a LGBTQ support group yesterday and it was so powerful.',
     'metadata': {'speaker': 'Caroline', 'session_time': '1:56 pm on 8 May, 2023'}},
    {'content': 'I painted that lake sunrise last year! It is special to me.',
     'metadata': {'speaker': 'Melanie', 'session_time': '1:14 pm on 25 May, 2023'}},
    {'content': 'I ran a charity race for mental health last Saturday.',
     'metadata': {'speaker': 'Melanie', 'session_time': '1:14 pm on 25 May, 2023'}},
]
for d in test_dialogues:
    mem.store({'content': d['content'], 'timestamp': 0, 'metadata': d['metadata']})

# 手动调用 BM25
print("=== BM25 原始结果 ===")
mem._ensure_bm25_indexed()
print(f"BM25 索引大小: {mem.bm25.size}")
for query in ["When did Caroline go to the LGBTQ support group?", "charity race"]:
    print(f"\n查询: {query}")
    bm25_res = mem.bm25.search(query, top_k=5)
    print(f"  BM25 返回 {len(bm25_res)} 条:")
    for r in bm25_res:
        print(f"    score={r['bm25_score']:.3f}, {r['content'][:80]}")

# 完整 recall
print("\n=== 完整 recall 结果 ===")
for query in ["When did Caroline go to the LGBTQ support group?", "charity race"]:
    print(f"\n查询: {query}")
    results = mem.recall(query, k=5)
    for i, r in enumerate(results):
        print(f"  [{i+1}] src={r.get('source')}, conf={r.get('confidence', 0):.3f}")
        print(f"      {r.get('content', '')[:100]}")