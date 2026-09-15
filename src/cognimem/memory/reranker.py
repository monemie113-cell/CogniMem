"""
交叉编码器重排序器。
使用 cross-encoder/ms-marco-MiniLM-L-6-v2 对候选结果进行精排。
支持优雅降级：如果模型加载失败，回退到轻量级词重叠评分。
"""

from typing import List, Dict, Any, Optional
import re


class CrossEncoderReranker:
    """
    交叉编码器重排序器。
    优先使用 sentence-transformers 的 CrossEncoder；
    如果加载失败，回退到基于词重叠的启发式评分。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512)
            self._available = True
            print(f"CrossEncoder loaded: {self.model_name}")
        except Exception as e:
            print(f"CrossEncoder load failed: {e}. Using heuristic fallback.")
            self._available = False

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 5, blend_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        对候选文档重新排序。
        blend_weight: 重排序分数与原始置信度的融合权重。
        """
        if not candidates:
            return []

        if self._available:
            return self._rerank_with_model(query, candidates, top_k, blend_weight)
        else:
            return self._rerank_heuristic(query, candidates, top_k)

    def _rerank_with_model(self, query: str, candidates: List[Dict],
                           top_k: int, blend_weight: float) -> List[Dict]:
        """使用交叉编码器重排序"""
        pairs = [(query, c.get('content', '')[:512]) for c in candidates]

        try:
            scores = self.model.predict(pairs)
            # scores 是 numpy 数组，归一化到 0~1
            import numpy as np
            scores = np.array(scores)
            if scores.max() != scores.min():
                norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                norm_scores = np.ones_like(scores) * 0.5

            for i, cand in enumerate(candidates):
                orig_conf = cand.get('confidence', 0.5)
                cand['rerank_score'] = float(norm_scores[i])
                cand['confidence'] = (
                    blend_weight * float(norm_scores[i]) +
                    (1 - blend_weight) * orig_conf
                )
                cand['source'] = cand.get('source', '') + '+rerank'

            candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            print(f"Rerank predict failed: {e}. Using heuristic.")
            return self._rerank_heuristic(query, candidates, top_k)

    def _rerank_heuristic(self, query: str, candidates: List[Dict],
                          top_k: int) -> List[Dict]:
        """启发式重排序：基于查询词与文档的词重叠率"""
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        query_words.update(re.findall(r'[\u4e00-\u9fa5]{2,}', query))

        for cand in candidates:
            content = cand.get('content', '').lower()
            overlap = sum(1 for w in query_words if w.lower() in content)
            overlap_ratio = overlap / max(len(query_words), 1)
            orig_conf = cand.get('confidence', 0.5)
            cand['rerank_score'] = overlap_ratio
            cand['confidence'] = 0.7 * orig_conf + 0.3 * overlap_ratio
            cand['source'] = cand.get('source', '') + '+heuristic'

        candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return candidates[:top_k]

    @property
    def is_available(self) -> bool:
        return self._available