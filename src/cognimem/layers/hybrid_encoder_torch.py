import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

class HybridEncoderTorch:
    """
    PyTorch 实现的 HySparse，支持 GPU 加速。
    接口与 NumPy 版本完全一致，可无缝替换。
    """
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config.get('window_size', None)
        self.global_samples = config.get('global_samples', None)
        self.dim = config.get('dim', 384)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.kv_cache = {}
        self.proj = None

    def process(self, inputs: Any, memory_ctx: Optional[torch.Tensor] = None, **kwargs) -> tuple:
        seq = self._to_sequence(inputs)
        if seq is None or seq.shape[0] == 0:
            return torch.empty(0, self.dim, device=self.device), {'kv_cache_size': 0}

        seq_len, dim = seq.shape
        if dim != self.dim:
            if self.proj is None:
                np.random.seed(123)
                proj_np = np.random.randn(self.dim, dim).astype(np.float32) * 0.1
                self.proj = torch.tensor(proj_np, device=self.device)
            seq = torch.mm(seq, self.proj.T)

        w, g = self._get_adaptive_params(seq_len)
        encoded = self._hybrid_sparse_attention_vectorized(seq, w, g)
        return encoded, {'kv_cache_size': seq_len, 'window': w, 'global_samples': g}

    def _get_adaptive_params(self, n: int):
        if self.window_size is not None and self.global_samples is not None:
            return self.window_size, self.global_samples
        if n <= 50:
            return 15, 10
        elif n >= 500:
            return 3, 2
        else:
            w = int(np.interp(n, [50, 500], [15, 3]))
            g = int(np.interp(n, [50, 500], [10, 2]))
            return w, g

    def _to_sequence(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        if isinstance(inputs, np.ndarray):
            return torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if isinstance(inputs, str):
            from ..utils.text_processor import embed_text
            vec = embed_text(inputs)
            return torch.tensor(vec, dtype=torch.float32, device=self.device).reshape(1, -1)
        if isinstance(inputs, list):
            if not inputs:
                return torch.empty(0, self.dim, device=self.device)
            if isinstance(inputs[0], torch.Tensor):
                return torch.stack(inputs).to(self.device)
            if isinstance(inputs[0], np.ndarray):
                return torch.tensor(np.stack(inputs), dtype=torch.float32, device=self.device)
            arr = np.array(inputs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return torch.tensor(arr, device=self.device)
        return torch.tensor(np.array(inputs), dtype=torch.float32, device=self.device).reshape(1, -1)

    def _hybrid_sparse_attention_vectorized(self, seq: torch.Tensor, window: int, global_samples: int) -> torch.Tensor:
        n, d = seq.shape
        if n <= 1:
            return seq

        w = window
        # 修复：增加批次维度，使得 padding 参数 (0,0,w,w) 填充第 1 维（序列长度）
        # seq 形状 (n, d) -> (1, n, d)
        padded = F.pad(seq.unsqueeze(0), (0, 0, w, w), mode='replicate').squeeze(0)  # (n+2w, d)

        # unfold 沿着第 0 维滑动窗口
        windows = padded.unfold(0, 2*w+1, 1).transpose(1, 2).contiguous()  # (n, 2w+1, d)
        scores = torch.einsum('ijk,ik->ij', windows, seq)
        weights = F.softmax(scores, dim=-1)
        local_out = torch.einsum('ijk,ij->ik', windows, weights)

        g = global_samples
        if g > 0 and n > g:
            indices = torch.linspace(0, n-1, g, dtype=torch.long, device=seq.device)
            sampled = seq[indices]
            scores_global = torch.mm(seq, sampled.T)
            weights_global = F.softmax(scores_global, dim=-1)
            global_out = torch.mm(weights_global, sampled)
        else:
            global_out = seq.clone()

        alpha = 0.6
        return alpha * local_out + (1 - alpha) * global_out

    def reset_cache(self):
        self.kv_cache.clear()