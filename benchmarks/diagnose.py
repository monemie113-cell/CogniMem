# CogniMem/benchmarks/diagnose.py
import sqlite3

conn = sqlite3.connect('semantic.db')
c = conn.cursor()

# 查看前 20 条三元组
c.execute("SELECT subject, predicate, object FROM triples LIMIT 20")
print("=== 前 20 条三元组 ===")
for row in c.fetchall():
    print(row)

print("\n=== 统计 ===")
c.execute("SELECT COUNT(*) FROM triples WHERE subject LIKE '%Caroline%'")
print(f"subject 含 Caroline: {c.fetchone()[0]}")

c.execute("SELECT COUNT(*) FROM triples WHERE subject = 'I' OR subject = 'i'")
print(f"subject = 'I': {c.fetchone()[0]}")

c.execute("SELECT COUNT(*) FROM triples WHERE subject = 'user'")
print(f"subject = 'user': {c.fetchone()[0]}")

c.execute("SELECT COUNT(*) FROM triples")
print(f"三元组总数: {c.fetchone()[0]}")

# 查看 subject 分布 top 10
c.execute("SELECT subject, COUNT(*) as cnt FROM triples GROUP BY subject ORDER BY cnt DESC LIMIT 10")
print("\n=== subject 分布 Top 10 ===")
for row in c.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()