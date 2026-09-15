"""
BM25 检索器：基于 Okapi BM25 算法，零依赖实现。
支持中英文混合文本的 tokenization。
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional


def _tokenize(text: str) -> List[str]:
    """中英文混合 tokenization"""
    tokens = []
    # 英文单词
    tokens.extend(re.findall(r'[a-zA-Z]+', text.lower()))
    # 中文单字（简化的分词策略，生产环境可用 jieba）
    tokens.extend(re.findall(r'[\u4e00-\u9fa5]', text))
    # 数字
    tokens.extend(re.findall(r'\d+', text))
    return tokens


class BM25Retriever:
    """
    Okapi BM25 检索器。
    k1=1.2, b=0.75 为行业标准参数。
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.doc_ids: List[int] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avg_dl: float = 0.0
        self._indexed = False

    def index(self, documents: List[Dict[str, Any]]):
        """构建索引。documents 格式: [{'id': 1, 'content': '...'}]"""
        self.corpus = []
        self.doc_ids = []
        self.doc_freqs = []
        self.doc_lengths = []
        self.idf = {}

        for doc in documents:
            content = doc.get('content', '')
            doc_id = doc.get('id', 0)
            tokens = _tokenize(content)
            self.corpus.append(content)
            self.doc_ids.append(doc_id)
            self.doc_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))

        N = len(self.corpus)
        if N == 0:
            self._indexed = True
            return

        self.avg_dl = sum(self.doc_lengths) / N

        # 计算 IDF
        df = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1

        for term, freq in df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

        self._indexed = True

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """检索，返回 top_k 结果"""
        if not self._indexed or not self.corpus:
            return []

        query_tokens = _tokenize(query)
        scores = []

        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_lengths[i]
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = doc_freq.get(token, 0)
                if tf == 0:
                    continue
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            results.append({
                'id': self.doc_ids[idx],
                'content': self.corpus[idx],
                'bm25_score': score,
                'source': 'bm25'
            })
        return results

    @property
    def size(self) -> int:
        return len(self.corpus)