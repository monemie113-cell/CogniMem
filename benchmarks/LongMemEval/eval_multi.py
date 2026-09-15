import json
import os

# 加载原始数据，建立 qid -> answer_session_ids 映射
with open('data/longmemeval_s', 'r', encoding='utf-8') as f:
    raw = json.load(f)

# 建立映射：qid -> 会话ID集合
qid_to_sessions = {}
qid_to_session_ids = {}
for item in raw:
    qid = item.get('question_id', '')
    answer_session_ids = set(item.get('answer_session_ids', []))
    haystack_session_ids = item.get('haystack_session_ids', [])
    qid_to_session_ids[qid] = (answer_session_ids, haystack_session_ids)

# 加载预测
with open('predictions_multi_10.jsonl', 'r', encoding='utf-8') as f:
    preds = [json.loads(line) for line in f if line.strip()]

# 需要知道 retrieved 每条来自哪个 session——这需要修改 adapter 保存 session 信息
# 暂时无法追踪，我们只能看 retrieved 是否包含答案关键词

exact = 0
token = 0
total = 0
for p in preds:
    ans = str(p.get('answer', '')).lower().strip()
    if not ans:
        continue
    total += 1
    combined = ' '.join(str(r).lower() for r in p.get('retrieved', []))
    if ans in combined:
        exact += 1
    # 对数字答案，用更宽松的匹配
    num_match = None
    for token_candidate in ans.split():
        if token_candidate.isdigit() or any(c.isdigit() for c in token_candidate):
            if token_candidate in combined:
                num_match = token_candidate
                break
    if num_match or (ans and any(w in combined for w in ans.split() if len(w) >= 3)):
        token += 1

print(f"multi-session 样本数: {total}")
print(f"精确包含: {exact}/{total} ({exact/total*100:.1f}%)")
print(f"关键词命中: {token}/{total} ({token/total*100:.1f}%)")