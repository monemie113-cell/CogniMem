import numpy as np
from typing import Any, Dict
from ..models.engram_table import EngramTable
from ..utils.text_processor import embed_text

class EngramLayer:
    """
    Engram 知识查找层。
    使用输入文本的完整字符串作为键，从缓存表中快速检索预计算向量。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.embed_dim = config.get('embed_dim', 384)
        self.table_size = config.get('table_size', 10000)
        self.table = EngramTable(embed_dim=self.embed_dim, max_size=self.table_size)
        # 保留 gate_weights 仅作兼容（实际不使用）
        self.gate_weights = np.random.randn(self.embed_dim)

    def process(self, inputs: Any, context: Dict = None) -> tuple:
        """
        输入可以是字符串或字符串列表。
        返回：向量表示和元信息。
        """
        # 确保输入是字符串
        if isinstance(inputs, list):
            # 如果是 token 列表，合并为字符串
            text = ' '.join(inputs)
        elif isinstance(inputs, str):
            text = inputs
        else:
            text = str(inputs)

        # 尝试从缓存中获取
        cached_vec = self.table.get(text)
        if cached_vec is not None:
            # 命中缓存，直接返回
            return cached_vec, {
                'cached': True,
                'hit_rate': self.table.hit_rate(),
                'gate_score': 0.0  # 占位
            }

        # 未命中：调用 embed_text 计算向量
        vec = embed_text(text)
        # 存入缓存
        self.table.put(text, vec)
        return vec, {
            'cached': False,
            'hit_rate': self.table.hit_rate(),
            'gate_score': 0.0
        }