# CogniMem/benchmarks/diagnose_bm25.py
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cognimem.core.pipeline import CogniMemPipeline

config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 打印配置
print("=== 配置检查 ===")
print(f"memory.recall 内容: {config.get('memory', {}).get('recall', {})}")

# 初始化 pipeline
pipeline = CogniMemPipeline(config)
mem = pipeline.memory

print(f"\n=== BM25 状态 ===")
print(f"use_bm25: {getattr(mem, 'use_bm25', 'NOT SET')}")
print(f"bm25 object: {mem.bm25}")
print(f"_bm25_indexed: {getattr(mem, '_bm25_indexed', 'NOT SET')}")

# 检查 recall 方法是否有 BM25 调用代码
import inspect
source = inspect.getsource(mem.recall)
has_bm25_call = 'bm25' in source.lower()
print(f"\nrecall 方法包含 'bm25' 关键字: {has_bm25_call}")

# 打印 recall 方法中和 bm25 相关的行
print("\n=== recall 方法中的 BM25 相关代码 ===")
for i, line in enumerate(source.split('\n')):
    if 'bm25' in line.lower():
        print(f"  行{i}: {line.strip()}")