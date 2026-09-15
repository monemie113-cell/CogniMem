import numpy as np
from typing import Any, Dict, List
from ..models.liquid_cell import LiquidCell
from ..utils.text_processor import embed_text

class LiquidEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 512)
        self.time_constants = config.get('time_constants', [0.1, 1.0, 10.0])
        self.cell = None          # 延迟初始化
        self.state = None
        self.proj = None          # 投影矩阵，用于将任意维度映射到 hidden_dim

    def _ensure_vector_list(self, data: Any) -> List[np.ndarray]:
        """
        将输入转换为向量列表，并确保每个向量维度与 hidden_dim 匹配。
        若维度不匹配，则使用随机投影矩阵（保持确定性）。
        """
        # 1. 获取原始向量列表
        raw_vectors = self._to_vector_list(data)
        if not raw_vectors:
            return [np.zeros(self.hidden_dim)]

        # 2. 检查维度，若需要则创建投影矩阵
        sample_vec = raw_vectors[0]
        actual_dim = sample_vec.shape[0]
        if actual_dim != self.hidden_dim:
            if self.proj is None:
                # 使用固定种子保证投影矩阵的一致性
                np.random.seed(42)
                self.proj = np.random.randn(self.hidden_dim, actual_dim).astype(np.float32) * 0.1
                np.random.seed()
            # 应用投影
            projected = [np.dot(self.proj, vec) for vec in raw_vectors]
            return projected
        else:
            return raw_vectors

    def _to_vector_list(self, data: Any) -> List[np.ndarray]:
        """将各种输入类型转换为原始向量列表（未经投影）"""
        if data is None:
            return []
        if isinstance(data, str):
            return [embed_text(data)]
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return [data]
            else:
                return [data[i] for i in range(data.shape[0])]
        if isinstance(data, list):
            if not data:
                return []
            first = data[0]
            if isinstance(first, (int, float)):
                return [np.array(data, dtype=np.float32)]
            if isinstance(first, np.ndarray):
                return data
            if isinstance(first, str):
                return [embed_text(x) for x in data]
            return [embed_text(str(data))]
        return [embed_text(str(data))]

    def process(self, encoded_context: Any, memory_ctx: Dict = None, **kwargs) -> tuple:
        # 1. 确保输入为向量列表（已投影到 hidden_dim）
        vectors = self._ensure_vector_list(encoded_context)

        # 2. 延迟初始化 LiquidCell（确保输入维度已知）
        if self.cell is None and vectors:
            self.cell = LiquidCell(self.hidden_dim, self.time_constants)

        # 3. 按时间步推进
        outputs = []
        for vec in vectors:
            output, self.state = self.cell.forward(vec, self.state)
            outputs.append(output)
        final_output = outputs[-1] if outputs else np.zeros(self.hidden_dim)
        return final_output, {'state': self.state}

    def reset_state(self):
        self.state = None
        self.cell = None