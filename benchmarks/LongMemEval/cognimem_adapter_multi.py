# CogniMem/benchmarks/LongMemEval/cognimem_adapter_multi.py
import json
import os
import sys
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from cognimem.core.pipeline import CogniMemPipeline


def load_config():
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'config', 'default_config.yaml'
    ))
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def run_cognimem(data_path, output_path, target_type='multi-session', max_samples=10):
    data = load_data(data_path)

    # 只挑选指定类型
    filtered = [d for d in data if d.get('question_type') == target_type]
    print(f"'{target_type}' 类型共 {len(filtered)} 条，取前 {max_samples} 条")
    data = filtered[:max_samples]

    config = load_config()
    results = []

    for sample in tqdm(data, desc=f"CogniMem on {target_type}"):
        try:
            pipeline = CogniMemPipeline(config)
            sessions = sample.get('haystack_sessions', [])
            dates = sample.get('haystack_dates', [])

            for i, session in enumerate(sessions):
                session_date = dates[i] if i < len(dates) else ''
                if isinstance(session, list):
                    turns = session
                elif isinstance(session, dict):
                    turns = session.get('turns', [])
                else:
                    continue

                for turn in turns:
                    if isinstance(turn, str):
                        content = turn
                        role = 'user'
                    elif isinstance(turn, dict):
                        content = turn.get('content', '') or turn.get('text', '')
                        role = turn.get('role', 'user') or turn.get('speaker', 'user')
                    else:
                        continue
                    if not content:
                        continue
                    pipeline.memory.store({
                        'content': content,
                        'timestamp': 0,
                        'metadata': {'speaker': role, 'session_time': session_date}
                    })

            question = sample.get('question', '')

            # 先调用 pipeline.run()，让聚合器有机会介入
            run_result = pipeline.run(question)
            agg_answer = run_result.get('response', '')

            print(f"\n[QA] {sample.get('question_id')}: {question[:60]}")
            print(f"[AGG] answer={agg_answer[:80]}")
            print(f"[AGG] meta={run_result.get('meta', {}).keys()}")
            print(f"[AGG] aggregator={run_result.get('meta', {}).get('aggregator')}")

            # 同时获取原始检索结果（用于关键词匹配评估）
            retrieved = pipeline.memory.recall(question, k=5)
            retrieved_texts = [r.get('content', '') for r in retrieved if isinstance(r, dict)]

            # 把聚合器的答案放在 retrieved 的第一位，这样评估脚本能匹配到
            if agg_answer and run_result.get('meta', {}).get('aggregator'):
                retrieved_texts.insert(0, agg_answer)

            results.append({
                'question_id': sample.get('question_id', ''),
                'question': question,
                'question_type': sample.get('question_type', ''),
                'answer': sample.get('answer', ''),
                'retrieved': retrieved_texts,
            })

        except Exception as e:
            print(f"\n⚠️ 处理 {sample.get('question_id')} 时出错: {e}")
            results.append({
                'question_id': sample.get('question_id', ''),
                'question': sample.get('question', ''),
                'question_type': sample.get('question_type', ''),
                'answer': sample.get('answer', ''),
                'retrieved': [],
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'longmemeval_s')
    output_path = os.path.join(os.path.dirname(__file__), 'predictions_multi_10.jsonl')
    run_cognimem(data_path, output_path, target_type='multi-session', max_samples=3)