# CogniMem/benchmarks/LongMemEval/cognimem_adapter.py
import json
import os
import sys
import traceback
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from cognimem.core.pipeline import CogniMemPipeline


def load_config():
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'config', 'default_config.yaml'
    ))
    print(f"配置文件路径: {config_path}")
    print(f"文件存在: {os.path.exists(config_path)}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                print("⚠️ 配置文件解析为 None，使用默认空配置")
                return {}
            print(f"配置文件键: {list(config.keys())}")
            return config
    except Exception as e:
        print(f"⚠️ 加载配置失败: {e}，使用默认空配置")
        traceback.print_exc()
        return {}


def find_data_file():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        return None
    for fname in os.listdir(data_dir):
        if fname.startswith('longmemeval') and '_s' in fname and 'oracle' not in fname:
            return os.path.join(data_dir, fname)
    for fname in os.listdir(data_dir):
        if fname.startswith('longmemeval'):
            return os.path.join(data_dir, fname)
    return None


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def run_cognimem(data_path, output_path, max_samples=None):
    print(f"数据文件: {data_path}")
    data = load_data(data_path)
    print(f"数据总条数: {len(data)}")

    if max_samples:
        data = data[:max_samples]
    print(f"本次评估条数: {len(data)}")

    config = load_config()
    if not config:
        print("⚠️ 警告：配置为空，系统将使用默认参数")

    results = []
    first_error_printed = False

    for sample in tqdm(data, desc="CogniMem on LongMemEval"):
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
                        'metadata': {
                            'speaker': role,
                            'session_time': session_date
                        }
                    })

            question = sample.get('question', '')
            retrieved = pipeline.memory.recall(question, k=5)

            results.append({
                'question_id': sample.get('question_id', ''),
                'question': question,
                'retrieved': [r.get('content', '') for r in retrieved if isinstance(r, dict)],
            })

        except Exception as e:
            # 只对第一条错误打印完整堆栈
            if not first_error_printed:
                traceback.print_exc()
                first_error_printed = True
            print(f"\n⚠️ 处理 {sample.get('question_id')} 时出错: {e}")
            results.append({
                'question_id': sample.get('question_id', ''),
                'question': sample.get('question', ''),
                'retrieved': [],
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    data_path = find_data_file()
    if data_path is None:
        print("❌ 在 data/ 目录下未找到 LongMemEval 数据文件")
    else:
        output_path = os.path.join(os.path.dirname(__file__), 'cognimem_predictions.jsonl')
        run_cognimem(data_path, output_path, max_samples=5)