# evaluate_agent.py
import json
import os
import sys
import yaml
from tqdm import tqdm
from rouge_score import rouge_scorer
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线，避免连接 huggingface.co

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cognimem.core.pipeline import CogniMemPipeline

PREDICTIONS_FILE = "predictions.json"

def load_existing_predictions():
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_predictions(predictions):
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)

def evaluate_conversation(sample, config):
    pipeline = CogniMemPipeline(config)
    # 注入对话历史
    for session_key, session_content in sample['conversation'].items():
        if session_key.startswith('session_'):
            for turn in session_content:
                if isinstance(turn, str):
                    text = turn
                elif isinstance(turn, dict):
                    text = turn.get('text', '')
                else:
                    continue
                if not text:
                    continue
                item = {'content': text, 'timestamp': 0, 'metadata': {'speaker': turn.get('speaker', '') if isinstance(turn, dict) else ''}}
                pipeline.memory.store(item)
    # 回答QA
    results = []
    for qa in sample.get('qa', []):
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')
        if not question or ground_truth is None:
            continue
        try:
            result = pipeline.run(question)
            predicted = result.get('response', '')
        except Exception as e:
            print(f"Error answering: {e}")
            predicted = ""
        results.append({
            'question': question,
            'predicted': predicted,
            'ground_truth': ground_truth
        })
    return results

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not config.get('llm', {}).get('enabled', False):
            print("⚠️  LLM is disabled. Enable it in config.")
            return

    data_path = os.path.join(os.path.dirname(__file__), 'locomo', 'data', 'locomo10.json')
    with open(data_path, 'r') as f:
        locomo_data = json.load(f)

    # 加载已有预测结果
    existing = load_existing_predictions()
    # 确定已处理的对话索引（基于已保存的预测数量，每个对话约若干个QA）
    # 简单做法：按顺序处理，但每处理完一个就保存，用已有结果数量来判断跳过
    # 这里采用更稳健的方式：直接处理所有，但保存时追加，避免重复
    print(f"Loaded {len(locomo_data)} conversations. Existing predictions: {len(existing)}")

    all_results = existing
    for idx, sample in enumerate(tqdm(locomo_data[:2], desc="Processing（快速测试）")):
        # 如果已有该对话的预测结果，跳过（简化：假定每个对话的QA数量固定，但为安全，我们记录已处理的样本ID）
        # 更精确：检查sample_id是否已在existing中
        sample_id = sample.get('sample_id', idx)
        # 检查是否已经处理过（根据sample_id）
        if any(r.get('sample_id') == sample_id for r in existing):
            continue
        results = evaluate_conversation(sample, config)
        # 为每条结果添加sample_id
        for r in results:
            r['sample_id'] = sample_id
        all_results.extend(results)
        save_predictions(all_results)  # 每处理一个对话保存一次

    # 最后计算ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for item in all_results:
        pred = str(item['predicted']) if item['predicted'] is not None else ""
        ref = str(item['ground_truth']) if item['ground_truth'] is not None else ""
        if pred and ref:
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
                scores.append(score)
            except Exception as e:
                print(f"Scoring error: {e}")
                continue

    if scores:
        avg_rouge = sum(scores) / len(scores)
        print(f"\n✅ Average ROUGE-L F1: {avg_rouge:.4f} (based on {len(scores)} pairs)")
    else:
        print("No valid predictions.")

if __name__ == "__main__":
    main()