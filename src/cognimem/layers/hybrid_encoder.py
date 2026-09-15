import numpy as np
from typing import Any, Dict, Tuple


class HybridEncoder:
    """
    HySparse: Hybrid Block 架构
    每个 block = 1层 Full Attention + N层 Sparse Attention
    Full Attention 负责 token 选择和 KV Cache 生成，Sparse 层直接复用
    """

    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config.get('window_size', None)
        self.global_samples = config.get('global_samples', None)
        self.dim = config.get('dim', 384)
        # 【关键】每个 block 中 Full Attention 后的 Sparse 层数
        self.sparse_layers_per_block = config.get('sparse_layers_per_block', 4)
        # 【关键】总 block 数
        self.num_blocks = config.get('num_blocks', 2)
        self.kv_cache = {}

    def process(self, inputs: Any, memory_ctx: np.ndarray = None, **kwargs) -> Tuple[Any, Dict]:
        # 1. 转换为序列
        seq = self._to_sequence(inputs)
        if seq is None or seq.shape[0] == 0:
            return np.array([]), {'kv_cache_size': 0}

        seq_len, dim = seq.shape
        # 2. 维度适配
        if dim != self.dim:
            if not hasattr(self, 'proj'):
                np.random.seed(123)
                self.proj = np.random.randn(self.dim, dim) * 0.1
                np.random.seed()
            seq = np.dot(seq, self.proj.T)

        # 3. 自适应参数
        w, g = self._get_adaptive_params(seq_len)
        # 4. 执行 Hybrid Block 编码
        encoded = self._hybrid_block_forward(seq, w, g)
        return encoded, {
            'kv_cache_size': seq_len,
            'window': w,
            'global_samples': g,
            'num_blocks': self.num_blocks,
            'sparse_layers_per_block': self.sparse_layers_per_block
        }

    def _to_sequence(self, inputs):
        """将各种输入转换为 (seq_len, dim) 的 numpy 数组"""
        if isinstance(inputs, str):
            from ..utils.text_processor import embed_text
            vec = embed_text(inputs)
            return vec.reshape(1, -1)
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 1:
                return inputs.reshape(1, -1)
            return inputs
        if isinstance(inputs, list):
            if not inputs:
                return np.array([])
            if isinstance(inputs[0], np.ndarray):
                return np.stack(inputs)
            arr = np.array(inputs)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            return arr
        return np.array(inputs).reshape(1, -1) if inputs is not None else np.array([])

    def _hybrid_block_forward(self, seq: np.ndarray, window: int, global_samples: int) -> np.ndarray:
        """
        Hybrid Block 前向传播：
        每个 block: 1层 Full Attention → N层 Sparse Attention（复用 Full 的 KV 和 token 索引）
        """
        n, d = seq.shape
        if n <= 1:
            return seq

        current = seq
        for block_idx in range(self.num_blocks):
            # -------- Step 1: Full Attention（生成 token 索引 + KV Cache） --------
            full_out, top_k_indices, kv_cache = self._full_attention_with_cache(current)

            # -------- Step 2: N 层 Sparse Attention（复用 Full 的 token 索引和 KV） --------
            sparse_out = full_out
            for _ in range(self.sparse_layers_per_block):
                sparse_out = self._sparse_attention_with_shared_cache(
                    sparse_out, top_k_indices, kv_cache, window, global_samples
                )

            # 残差连接：保留部分原始信息
            current = 0.6 * sparse_out + 0.4 * current

        return current

    def _full_attention_with_cache(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Full Attention：计算完整注意力，同时生成 top-k token 索引和 KV Cache"""
        n, d = seq.shape
        # 计算注意力分数
        scores = np.dot(seq, seq.T) / np.sqrt(d)
        # 数值稳定
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
        out = np.dot(weights, seq)

        # 【关键】选择 top-k 重要 token（基于注意力权重总和）
        importance = np.sum(weights, axis=0)  # 每个 token 被关注的总权重
        k = max(3, int(n * 0.1))  # 选择前 10% 的 token
        top_k_indices = np.argsort(importance)[-k:]

        # KV Cache（简化：存储当前序列的键值对）
        kv_cache = {'k': seq.copy(), 'v': seq.copy(), 'indices': top_k_indices}
        return out, top_k_indices, kv_cache

    def _sparse_attention_with_shared_cache(self, seq: np.ndarray, top_k_indices: np.ndarray,
                                            kv_cache: Dict, window: int, global_samples: int) -> np.ndarray:
        """
        Sparse Attention：复用 Full Attention 的 token 索引和 KV Cache
        只对 top-k token 和局部窗口内的 token 计算注意力
        """
        n, d = seq.shape

        # 1. 从 KV Cache 中取出 top-k token 的键值
        k_tokens = kv_cache['k'][top_k_indices]  # (k, d)
        v_tokens = kv_cache['v'][top_k_indices]  # (k, d)

        # 2. 对 top-k token 计算注意力
        scores_topk = np.dot(seq, k_tokens.T) / np.sqrt(d)  # (n, k)
        weights_topk = np.exp(scores_topk - np.max(scores_topk, axis=1, keepdims=True))
        weights_topk = weights_topk / (np.sum(weights_topk, axis=1, keepdims=True) + 1e-8)
        out_topk = np.dot(weights_topk, v_tokens)  # (n, d)

        # 3. 对局部窗口内的 token 计算注意力（与现有逻辑一致）
        from numpy.lib.stride_tricks import sliding_window_view
        pad = window
        padded = np.pad(seq, ((pad, pad), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, (2 * window + 1, d))[:, 0, :, :]
        scores_local = np.einsum('ijk,ik->ij', windows, seq)
        weights_local = np.exp(scores_local - np.max(scores_local, axis=1, keepdims=True))
        weights_local = weights_local / (np.sum(weights_local, axis=1, keepdims=True) + 1e-8)
        out_local = np.einsum('ijk,ij->ik', windows, weights_local)

        # 4. 融合：top-k 注意力 + 局部窗口注意力
        alpha = 0.5  # 平衡权重
        return alpha * out_topk + (1 - alpha) * out_local

    def _get_adaptive_params(self, n: int):
        if self.window_size is not None and self.global_samples is not None:
            return self.window_size, self.global_samples
        # 自适应策略：长序列用更激进的稀疏参数
        if n <= 100:
            return 10, 8
        elif n <= 500:
            return 7, 5
        elif n <= 2000:
            return 5, 3
        else:
            return 3, 2