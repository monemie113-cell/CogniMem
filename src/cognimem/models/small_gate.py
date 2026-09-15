import numpy as np
from ..utils.text_processor import embed_text
from .base import BaseModel

class SmallGateModel(BaseModel):
    def __init__(self, config):
        self.config = config

    def compute_scores(self, inputs, context=None):
        if not inputs:
            return {k: 0.0 for k in ['importance', 'confidence', 'priority', 'timeliness', 'contradiction', 'relevance']}

        # 转为字符串
        if isinstance(inputs, str):
            text = inputs
        else:
            text = str(inputs)

        # 计算基础特征
        text_len = len(text)
        # 独特字符数（衡量多样性）
        unique_chars = len(set(text))
        # 文本复杂度 = 长度 * (独特字符比例)
        complexity = text_len * (unique_chars / max(1, text_len))

        # 获取向量范数（若embedding失败，则fallback为随机但基于哈希的向量，范数接近1）
        try:
            vec = embed_text(text)
            norm = np.linalg.norm(vec)
        except Exception:
            norm = 1.0  # fallback

        # 计算 importance：综合范数和复杂度，且给予最小基准 0.3
        # 将范数从 [0, ~10] 映射到 [0.3, 0.9]
        norm_score = min(0.9, 0.3 + 0.6 * (norm / 5.0))
        # 复杂度因子：短文本但复杂（如“你好”）也能得到一些分数
        complexity_factor = min(1.0, complexity / 20)
        # 最终 importance = max(0.3, 0.5 * norm_score + 0.5 * complexity_factor)
        importance = max(0.3, 0.5 * norm_score + 0.5 * complexity_factor)

        # 置信度：长度越长，置信度越高
        confidence = min(0.9, 0.4 + 0.4 * (text_len / 100))

        # 优先级：与 importance 正相关
        priority = 0.3 + 0.7 * importance

        # 时效性、矛盾、相关性（沿用之前的逻辑）
        timeliness = 0.8
        contradiction = 0.5
        relevance = 0.5
        if context:
            try:
                ctx_vec = embed_text(str(context))
                sim = np.dot(vec, ctx_vec) / (np.linalg.norm(vec) * np.linalg.norm(ctx_vec) + 1e-8)
                relevance = max(0.0, min(1.0, (sim + 1) / 2))
            except:
                pass

        return {
            'importance': float(importance),
            'confidence': float(confidence),
            'priority': float(priority),
            'timeliness': float(timeliness),
            'contradiction': float(contradiction),
            'relevance': float(relevance)
        }

    def forward(self, inputs, **kwargs):
        return self.compute_scores(inputs, kwargs.get('context'))

    def load(self, path): pass
    def save(self, path): pass