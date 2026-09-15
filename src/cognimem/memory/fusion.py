"""
倒数排名融合（Reciprocal Rank Fusion）。
基于排名而非分数进行融合，鲁棒性极强。
"""

from typing import List, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    weights: List[float] = None
) -> List[Dict[str, Any]]:
    """
    融合多个排序列表。

    Args:
        ranked_lists: 多个检索器返回的排序列表，每个列表元素须包含 'id' 和 'content'。
        k: RRF 平滑参数，默认 60。
        weights: 每个列表的权重，默认等权。

    Returns:
        融合后的排序列表。
    """
    if not ranked_lists:
        return []

    n_lists = len(ranked_lists)
    if weights is None:
        weights = [1.0] * n_lists

    # 收集所有文档
    doc_scores: Dict[str, float] = {}
    doc_contents: Dict[str, str] = {}
    doc_metadata: Dict[str, Dict] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = str(item.get('id', item.get('content', '')[:50]))
            rrf_score = w / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_contents:
                doc_contents[doc_id] = item.get('content', '')
                doc_metadata[doc_id] = {
                    'source': item.get('source', 'unknown'),
                    'original_confidence': item.get('confidence', 0.5),
                }

    # 构建结果
    results = []
    for doc_id, score in doc_scores.items():
        meta = doc_metadata.get(doc_id, {})
        results.append({
            'id': doc_id,
            'content': doc_contents.get(doc_id, ''),
            'confidence': score,
            'source': f"rrf({meta.get('source', '')})",
            'rrf_score': score,
        })

    results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
    return results