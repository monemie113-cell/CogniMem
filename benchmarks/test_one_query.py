# test_one_query.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import json, yaml
from cognimem.core.pipeline import CogniMemPipeline

config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

pipeline = CogniMemPipeline(config)

data_path = os.path.join(os.path.dirname(__file__), 'locomo', 'data', 'locomo10.json')
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

sample = data[0]

# 收集会话时间戳
session_timestamps = {}
for key, val in sample['conversation'].items():
    if key.endswith('_date_time'):
        session_name = key.replace('_date_time', '')
        session_timestamps[session_name] = val

# 注入对话
for session_key, session_content in sample['conversation'].items():
    if session_key.startswith('session_') and not session_key.endswith('_date_time'):
        session_ts = session_timestamps.get(session_key, '')
        if not isinstance(session_content, list):
            continue
        for turn in session_content:
            if isinstance(turn, dict):
                text = turn.get('text', '')
                speaker = turn.get('speaker', '')
                if text:
                    pipeline.memory.store({
                        'content': text,
                        'timestamp': 0,
                        'metadata': {
                            'speaker': speaker,
                            'session': session_key,
                            'session_time': session_ts
                        }
                    })

# ===== 直接测试扩散激活 =====
print("\n=== 直接测试扩散激活 ===")
entities = ['Caroline', 'LGBTQ']
print(f"种子实体: {entities}")
act_results = pipeline.memory._spreading_activation_search(entities, limit=10)
print(f"扩散激活返回 {len(act_results)} 条结果")
for i, r in enumerate(act_results[:5]):
    print(f"  [{i+1}] conf={r.get('confidence', 0):.3f}, content={r.get('content', '')[:120]}")

# ===== 测试完整 recall =====
print("\n=== 完整 recall 结果 ===")
query = "When did Caroline go to the LGBTQ support group?"
results = pipeline.memory.recall(query, k=5)
for i, r in enumerate(results):
    print(f"  [{i+1}] src={r.get('source')}, conf={r.get('confidence', 0):.3f}")
    print(f"      {r.get('content', '')[:150]}")