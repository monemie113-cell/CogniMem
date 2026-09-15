# evaluate_locomo.py
import json
import os
import sys
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cognimem.core.pipeline import CogniMemPipeline

def evaluate_conversation(sample, config):
    pipeline = CogniMemPipeline(config)

    # 注入对话历史
    for session_key, session_content in sample['conversation'].items():
        if session_key.startswith('session_'):
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
                    'metadata': {'speaker': speaker}
                }
                pipeline.memory.store(item)

    predictions = []
    for qa in sample.get('qa', []):
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')
        # 确保转换为字符串
        question = str(question)
        ground_truth = str(ground_truth)
        if not question or not ground_truth:
            continue
        try:
            result = pipeline.run(question)
            retrieved = result.get('response', '')
            if retrieved is None:
                retrieved = ''
            retrieved = str(retrieved)
        except Exception as e:
            print(f"Error on question '{question}': {e}")
            retrieved = ""
        predictions.append({
            'question': question,
            'retrieved': retrieved,
            'ground_truth': ground_truth
        })
    return predictions

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_path = os.path.join(os.path.dirname(__file__), 'locomo', 'data', 'locomo10.json')
    with open(data_path, 'r') as f:
        locomo_data = json.load(f)

    print(f"Loaded {len(locomo_data)} conversations. Evaluating hit rate...")

    all_predictions = []
    for sample in tqdm(locomo_data, desc="Processing"):
        preds = evaluate_conversation(sample, config)
        all_predictions.extend(preds)

    # 计算命中率
    hits = 0
    total = len(all_predictions)
    for item in all_predictions:
        retrieved = item['retrieved'].strip().lower()
        gt = item['ground_truth'].strip().lower()
        if not retrieved or not gt:
            continue
        if gt in retrieved:
            hits += 1

    if total > 0:
        hit_rate = hits / total
        print(f"\n✅ Hit Rate (exact answer inclusion): {hit_rate:.4f} ({hits}/{total})")
    else:
        print("No valid QA pairs.")

if __name__ == "__main__":
    main()