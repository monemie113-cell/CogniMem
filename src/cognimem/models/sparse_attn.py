import numpy as np
from typing import Tuple, Dict, Any

class SparseAttention:
    def __init__(self, sparse_ratio: float = 0.3):
        self.ratio = sparse_ratio

    def forward(self, seq: np.ndarray, kv_cache: Dict = None) -> Tuple[np.ndarray, Dict]:
        if kv_cache is None:
            kv_cache = {}
        kv_cache['last_sparse'] = seq.shape
        return seq, kv_cache