import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cognimem.memory import PersistentMemory

pm = PersistentMemory({})
curiosity = pm.get_curiosity()

# 存储少量信息
pm.store({'subject': 'Alice', 'predicate': 'is_friend_of', 'object': 'Bob', 'confidence': 0.5})

# 生成问题
questions = curiosity.generate_questions('Alice', 'is_friend_of')
print("问题:", questions)

# 从文本生成问题
text = "Alice and Charlie are working on a project."
questions2 = curiosity.generate_questions_from_text(text)
print("从文本生成的问题:", questions2)