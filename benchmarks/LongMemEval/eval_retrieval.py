# CogniMem/benchmarks/LongMemEval/eval_retrieval.py
"""
LongMemEval 检索层评估（零 LLM 成本）
计算：
1. 整体精确包含命中率和关键词命中率
2. 按问题类别细分的命中率
"""
import json
import os
import re


def normalize(text):
    """标准化文本用于匹配"""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_data_file():
    """自动定位 LongMemEval 数据文件"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        return None
    for fname in os.listdir(data_dir):
        if fname.startswith('longmemeval') and '_s' in fname and 'oracle' not in fname:
            return os.path.join(data_dir, fname)
    # 回退：选任意 longmemeval 文件
    for fname in os.listdir(data_dir):
        if fname.startswith('longmemeval'):
            return os.path.join(data_dir, fname)
    return None


def load_data(path):
    """加载 JSON 或 JSONL 格式的数据"""
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def extract_answer(item):
    """兼容多种答案字段名"""
    for key in ('answer', 'gold_answer', 'correct_answer', 'target'):
        if key in item and item[key]:
            return item[key]
    return ''


def extract_category(item):
    """兼容多种类别字段名"""
    for key in ('question_type', 'category', 'type', 'task_type'):
        if key in item and item[key]:
            return item[key]
    return 'unknown'


def extract_qid(item):
    """兼容多种 ID 字段名"""
    for key in ('question_id', 'id', 'qid'):
        if key in item and item[key]:
            return item[key]
    return ''


def main():
    # ===== 加载原始数据 =====
    data_path = find_data_file()
    if data_path is None:
        print("❌ 未找到 LongMemEval 数据文件")
        return

    print(f"数据文件: {data_path}")
    raw_data = load_data(data_path)
    print(f"原始数据条数: {len(raw_data)}")

    # ===== 建立映射 =====
    gt_map = {}
    category_map = {}
    for item in raw_data:
        qid = extract_qid(item)
        if not qid:
            continue
        gt_map[qid] = extract_answer(item)
        category_map[qid] = extract_category(item)

    # 打印发现的类别
    categories_found = set(category_map.values())
    print(f"发现的问题类别: {categories_found}")
    print()

    # ===== 加载 CogniMem 预测 =====
    pred_path = os.path.join(os.path.dirname(__file__), 'cognimem_predictions.jsonl')
    if not os.path.exists(pred_path):
        print(f"❌ 未找到预测文件: {pred_path}")
        return

    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    print(f"CogniMem 预测条数: {len(predictions)}")
    print()

    # ===== 统计 =====
    total = 0
    exact_hits = 0
    token_hits = 0
    category_stats = {}

    for pred in predictions:
        qid = pred.get('question_id', '')
        gt = gt_map.get(qid, '')
        if not gt:
            continue

        qtype = category_map.get(qid, 'unknown')
        if qtype not in category_stats:
            category_stats[qtype] = {'total': 0, 'exact': 0, 'token': 0}

        retrieved_texts = [normalize(r) for r in pred.get('retrieved', [])]
        combined = ' '.join(retrieved_texts)

        total += 1
        category_stats[qtype]['total'] += 1
        gt_norm = normalize(gt)

        # 精确包含命中
        exact_found = gt_norm in combined
        if exact_found:
            exact_hits += 1
            category_stats[qtype]['exact'] += 1

        # 关键词命中（至少一个非停用词出现）
        stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'and', 'or',
                      'to', 'for', 'with', 'by', 'from', 'as', 'is', 'was',
                      'are', 'were', 'be', 'been', 'being'}
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', gt_norm)
                 if w not in stop_words]
        token_found = False
        if words:
            token_found = any(w in combined for w in words)
        if token_found:
            token_hits += 1
            category_stats[qtype]['token'] += 1

    # ===== 输出结果 =====
    print(f"=== LongMemEval 检索层评估 ===")
    print(f"总 QA 数: {total}")
    if total > 0:
        print(f"精确包含命中: {exact_hits} ({exact_hits / total * 100:.1f}%)")
        print(f"关键词命中: {token_hits} ({token_hits / total * 100:.1f}%)")

    print(f"\n=== 按问题类别细分 ===")
    for qtype in sorted(category_stats.keys()):
        stats = category_stats[qtype]
        if stats['total'] > 0:
            exact_pct = stats['exact'] / stats['total'] * 100
            token_pct = stats['token'] / stats['total'] * 100
            print(f"  {qtype}:")
            print(f"    总数: {stats['total']}")
            print(f"    精确包含: {stats['exact']} ({exact_pct:.1f}%)")
            print(f"    关键词命中: {stats['token']} ({token_pct:.1f}%)")


if __name__ == "__main__":
    main()