# 显式导入各个记忆组件
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working_memory import WorkingMemory
from .routing import RoutingManager
from .persistent_memory import PersistentMemory

# 定义公开接口
__all__ = [
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "RoutingManager",
    "PersistentMemory",
]

# 可选：导入后立即进行简单验证，便于调试
try:
    # 尝试实例化一个最小的对象，检查类是否可构造
    _test_episodic = EpisodicMemory
    _test_semantic = SemanticMemory
except NameError as e:
    raise ImportError(
        "CogniMem memory module import failed. "
        "Please ensure 'episodic.py' and 'semantic.py' contain the class definitions "
        "for 'EpisodicMemory' and 'SemanticMemory' respectively."
    ) from e