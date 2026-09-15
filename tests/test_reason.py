# CogniMem/test_reason.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognimem.memory import PersistentMemory

def print_results(title, results):
    print(f"\n=== {title} ===")
    if not results:
        print("  (无结果)")
        return
    for r in results:
        depth = r.get('depth', 0)
        print(f"  {r['subject']} {r['predicate']} {r['object']} (置信度:{r['confidence']:.2f}, depth:{depth})")

def main():
    config = {'memory': {}}
    pm = PersistentMemory(config)

    # 1. 清空旧数据（可选）
    # 实际生产环境不要清空，这里为了测试干净，我们直接删除旧数据库文件
    import os
    for db in ['episodic.db', 'semantic.db']:
        if os.path.exists(db):
            os.remove(db)
            print(f"已删除旧数据库: {db}")

    # 重新初始化
    pm = PersistentMemory(config)

    # 2. 存储事实
    print("\n--- 存储事实 ---")
    facts = [
        {'subject': 'Alice', 'predicate': 'is_friend_of', 'object': 'Bob', 'confidence': 0.9, 'source': 'human'},
        {'subject': 'Bob', 'predicate': 'is_friend_of', 'object': 'Charlie', 'confidence': 0.8, 'source': 'human'},
    ]
    for fact in facts:
        pm.store(fact)
        print(f"存储: {fact['subject']} {fact['predicate']} {fact['object']}")

    # 3. 验证存储是否成功（查询所有三元组）
    print("\n--- 验证存储 ---")
    # 直接查询 subject=Alice 的所有三元组
    all_alice = pm.semantic.exact_match('Alice')
    print_results("Alice 的直接关系", all_alice)

    # 4. 测试图推理
    print("\n--- 图推理测试 (max_depth=2) ---")
    results = pm.reason('Alice', 'is_friend_of', max_depth=2)
    print_results("Alice 的朋友 (含传递)", results)

    # 5. 测试冲突解决
    print("\n--- 冲突解决测试 ---")
    pm.store({'subject': 'Alice', 'predicate': 'is_friend_of', 'object': 'David', 'confidence': 0.95, 'source': 'human'})
    results_after = pm.reason('Alice', 'is_friend_of', max_depth=2)
    print_results("冲突后 Alice 的朋友", results_after)

    # 6. 测试向量检索（附加）
    print("\n--- 向量相似度检索 (查询 'friend') ---")
    vec_results = pm.semantic.vector_similarity('friend', top_k=3)
    for r in vec_results:
        print(f"  {r['subject']} {r['predicate']} {r['object']} (score:{r['score']:.2f})")

if __name__ == "__main__":
    main()