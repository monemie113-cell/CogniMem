# CogniMem/benchmarks/diagnose_query.py
import sqlite3

conn = sqlite3.connect('semantic.db')
c = conn.cursor()

# 1. 查找包含 "LGBTQ support group" 的三元组
print("=== 包含 'LGBTQ support group' 的三元组 ===")
c.execute("SELECT subject, predicate, object, evidence, episodic_id FROM triples WHERE evidence LIKE '%LGBTQ support group%'")
rows = c.fetchall()
for row in rows:
    print(f"  subj={row[0]}, pred={row[1]}, obj={row[2]}")
    print(f"  evidence={row[3][:150]}")
    print(f"  episodic_id={row[4]}")
    print()

# 2. 查找包含 "charity" 或 "race" 的三元组
print("=== 包含 'charity' 或 'race' 的三元组 ===")
c.execute("SELECT subject, predicate, object, evidence, episodic_id FROM triples WHERE evidence LIKE '%charity%' OR evidence LIKE '%race%'")
rows = c.fetchall()
for row in rows[:5]:
    print(f"  subj={row[0]}, pred={row[1]}, obj={row[2]}")
    print(f"  evidence={row[3][:150]}")
    print(f"  episodic_id={row[4]}")
    print()

# 3. 查看这些 episodic_id 对应的 chunk
print("=== 检查相关 chunk ===")
c.execute("SELECT episodic_id FROM triples WHERE evidence LIKE '%LGBTQ support group%' LIMIT 1")
row = c.fetchone()
if row and row[0]:
    ep_id = row[0]
    c.execute("SELECT id, start_turn_id, end_turn_id, content FROM chunks WHERE start_turn_id <= ? AND end_turn_id >= ? LIMIT 1", (ep_id, ep_id))
    chunk = c.fetchone()
    if chunk:
        print(f"  ep_id={ep_id} → chunk_id={chunk[0]}, range=({chunk[1]},{chunk[2]})")
        print(f"  content={chunk[3][:300]}")
    else:
        print(f"  ep_id={ep_id} 没有对应的 chunk")
print()

# 4. 统计
c.execute("SELECT COUNT(*) FROM triples")
print(f"三元组总数: {c.fetchone()[0]}")

c.execute("SELECT COUNT(*) FROM episodes")
print(f"对话轮次总数: {c.fetchone()[0]}")

c.execute("SELECT COUNT(*) FROM chunks")
print(f"Chunk 总数: {c.fetchone()[0]}")

# 5. 查看 subject 分布 Top 10
print("\n=== subject 分布 Top 10 ===")
c.execute("SELECT subject, COUNT(*) as cnt FROM triples GROUP BY subject ORDER BY cnt DESC LIMIT 10")
for row in c.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()