# check_recall.py
import json
import os
import sys
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cognimem.core.pipeline import CogniMemPipeline


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_path = os.path.join(os.path.dirname(__file__), 'locomo', 'data', 'locomo10.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        locomo_data = json.load(f)

    samples = locomo_data
    pipeline = CogniMemPipeline(config)

    total_qa = 0
    exact_hits = 0
    token_hits = 0

    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        # ===== 收集会话时间戳 =====
        session_timestamps = {}
        for key, val in sample['conversation'].items():
            if key.endswith('_date_time'):
                session_name = key.replace('_date_time', '')
                session_timestamps[session_name] = val

        # 注入对话历史
        for session_key, session_content in sample['conversation'].items():
            if session_key.startswith('session_') and not session_key.endswith('_date_time'):
                session_ts = session_timestamps.get(session_key, '')
                if not isinstance(session_content, list):
                    continue
                for turn in session_content:
                    if isinstance(turn, str):
                        text = turn
                        speaker = ''
                    elif isinstance(turn, dict):
                        text = turn.get('text', '')
                        speaker = turn.get('speaker', '')
                    else:
                        continue
                    if not text:
                        continue
                    item = {
                        'content': text,
                        'timestamp': 0,
                        'metadata': {
                            'speaker': speaker,
                            'session': session_key,
                            'session_time': session_ts
                        }
                    }
                    pipeline.memory.store(item)

        # 处理每个 QA
        for qa in sample.get('qa', []):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')
            if not question or not ground_truth:
                continue
            total_qa += 1

            retrieved = pipeline.memory.recall(question, k=5)
            retrieved_texts = [item.get('content', '') for item in retrieved if item.get('content')]

            gt_str = str(ground_truth).lower().strip()
            exact_found = any(gt_str in text.lower() for text in retrieved_texts)
            if exact_found:
                exact_hits += 1

            words = [w for w in gt_str.split() if len(w) >= 2]
            if words:
                token_found = any(any(w in text.lower() for w in words) for text in retrieved_texts)
                if token_found:
                    token_hits += 1

    print(f"\n=== 召回率统计 (k=5) ===")
    print(f"总 QA 数: {total_qa}")
    print(f"精确包含命中: {exact_hits} ({exact_hits / total_qa * 100:.1f}%)")
    print(f"关键词命中: {token_hits} ({token_hits / total_qa * 100:.1f}%)")


if __name__ == "__main__":
    main()