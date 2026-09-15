# CogniMem/benchmarks/diagnose_chunks.py
import sqlite3

conn = sqlite3.connect('episodic.db')
c = conn.cursor()

# 1. 检查 chunks 表是否存在
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in c.fetchall()]
print(f"=== episodic.db 中的表: {tables} ===\n")

# 2. chunks 表内容
if 'chunks' in tables:
    c.execute("SELECT COUNT(*) FROM chunks")
    print(f"chunks 总数: {c.fetchone()[0]}")

    c.execute("SELECT id, start_turn_id, end_turn_id, LENGTH(content) FROM chunks LIMIT 5")
    print("\n前 5 个 chunk:")
    for row in c.fetchall():
        print(f"  chunk_id={row[0]}, range=({row[1]}-{row[2]}), length={row[3]}")

# 3. 关键：episodic_id=3 和 19 是否被 chunk 覆盖
print("\n=== 检查关键 turn_id 是否被 chunk 覆盖 ===")
for turn_id in [3, 19, 20]:
    if 'chunks' in tables:
        c.execute(
            "SELECT id, start_turn_id, end_turn_id, content FROM chunks "
            "WHERE start_turn_id <= ? AND end_turn_id >= ? LIMIT 1",
            (turn_id, turn_id)
        )
        chunk = c.fetchone()
        if chunk:
            print(f"\nturn_id={turn_id} → chunk_id={chunk[0]}, range=({chunk[1]}-{chunk[2]})")
            print(f"  content 前200字: {chunk[3][:200]}")
        else:
            print(f"\nturn_id={turn_id} → 没有覆盖的 chunk ❌")

# 4. 检查 episodes 表本身
c.execute("SELECT COUNT(*) FROM episodes")
print(f"\nepisodes 总数: {c.fetchone()[0]}")

# 5. 确认 episodic_id=3 对应的是什么内容
c.execute("SELECT id, content FROM episodes WHERE id = 3")
row = c.fetchone()
if row:
    print(f"\nepisodes.id=3: {row[1][:200]}")

conn.close()