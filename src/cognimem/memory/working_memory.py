from typing import List, Dict, Any
from collections import deque
import time

class WorkingMemory:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, item: Dict[str, Any]):
        item['_timestamp'] = time.time()
        self.buffer.append(item)

    def get_recent(self, n: int = 5) -> List[Dict]:
        return list(self.buffer)[-n:]

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)