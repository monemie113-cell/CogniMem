"""
查询理解模块。
包含查询分解（Query Decomposition）、查询重写（Query Rewriting）
和查询扩展（Query Expansion）。
支持 LLM 驱动和规则驱动两种模式，LLM 不可用时自动降级。
"""

import re
from typing import List, Dict, Any, Optional


class QueryAnalyzer:
    """
    查询理解分析器。

    功能：
    1. 查询分解：将复杂查询拆分为子查询
    2. 查询重写：去除问题词，提取核心检索意图
    3. 查询扩展：生成同义变体，扩大检索覆盖面
    4. 时间信号提取：识别查询中的时间约束
    """

    # 问题词（用于查询重写）
    QUESTION_WORDS = {
        'what', 'when', 'where', 'who', 'which', 'how', 'why',
        '什么', '什么时候', '哪里', '谁', '哪个', '怎么', '为什么',
    }

    # 时间信号词
    TEMPORAL_SIGNALS = {
        'when', 'before', 'after', 'during', 'last', 'first', 'recently',
        'yesterday', 'today', 'tomorrow', 'ago',
        '什么时候', '之前', '之后', '最近', '昨天', '今天', '上次',
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        全面分析查询，返回分析结果字典。
        """
        result = {
            'original': query,
            'sub_queries': [],
            'rewritten': '',
            'expanded': [],
            'temporal_signal': None,
            'entities': [],
            'intent': 'factual',
        }

        # 1. 实体提取
        result['entities'] = self._extract_entities(query)

        # 2. 时间信号提取
        result['temporal_signal'] = self._extract_temporal(query)

        # 3. 查询重写
        result['rewritten'] = self._rewrite(query)

        # 4. 查询分解（多意图检测）
        result['sub_queries'] = self._decompose(query)

        # 5. 查询扩展
        result['expanded'] = self._expand(query, result['rewritten'])

        # 6. 意图识别
        result['intent'] = self._detect_intent(query)

        return result

    def _extract_entities(self, query: str) -> List[str]:
        """提取专有名词和关键实体"""
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        if not entities:
            entities = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
        # 去重并过滤停用词
        stop = {'什么', '怎么', '为什么', '哪个', '哪里', '是否', '能否'}
        return list(dict.fromkeys([e for e in entities if e not in stop]))

    def _extract_temporal(self, query: str) -> Optional[str]:
        """提取时间信号"""
        query_lower = query.lower()
        for signal in self.TEMPORAL_SIGNALS:
            if signal in query_lower:
                return signal
        # 提取具体日期
        date_match = re.search(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}', query)
        if date_match:
            return date_match.group()
        year_match = re.search(r'\d{4}年?', query)
        if year_match:
            return year_match.group()
        return None

    def _rewrite(self, query: str) -> str:
        """查询重写：去除问题词，保留核心检索意图"""
        words = query.split()
        filtered = [w for w in words
                    if w.lower() not in self.QUESTION_WORDS]
        if filtered:
            return ' '.join(filtered)
        return query

    def _decompose(self, query: str) -> List[str]:
        """查询分解：检测多意图查询并拆分"""
        sub_queries = []

        # 检测 "A 和 B" / "A and B" 型复合查询
        parts = re.split(r'\s+(?:和|与|及|以及|and|&)\s+', query)
        if len(parts) > 1:
            sub_queries.extend([p.strip() for p in parts if p.strip()])

        # 检测 "比较 A 和 B" / "compare A and B"
        compare_match = re.search(
            r'(?:compare|比较)\s+(.+?)\s+(?:和|与|and|&)\s+(.+)',
            query, re.IGNORECASE
        )
        if compare_match:
            sub_queries.extend([compare_match.group(1).strip(),
                                compare_match.group(2).strip()])

        # 如果没有检测到复合查询，返回原查询
        if not sub_queries:
            sub_queries = [query]

        return list(dict.fromkeys(sub_queries))

    def _expand(self, query: str, rewritten: str) -> List[str]:
        """查询扩展：生成查询的语义变体"""
        expanded = [query]

        if rewritten and rewritten != query:
            expanded.append(rewritten)

        if self.llm_client is not None:
            try:
                prompt = (
                    f"请为以下查询生成2个语义相同但表述不同的变体，"
                    f"每行一个，不要编号：\n{query}"
                )
                response = self.llm_client.generate(prompt)
                variants = [line.strip() for line in response.split('\n')
                            if line.strip() and len(line.strip()) > 3]
                expanded.extend(variants[:2])
            except Exception:
                pass

        return list(dict.fromkeys(expanded))

    def _detect_intent(self, query: str) -> str:
        """意图识别"""
        query_lower = query.lower()
        if any(w in query_lower for w in ['compare', '比较', '区别', '差异']):
            return 'comparative'
        if any(w in query_lower for w in ['how to', '怎么', '如何']):
            return 'procedural'
        if any(w in query_lower for w in ['why', '为什么', '原因']):
            return 'analytical'
        if any(w in query_lower for w in ['list', '列出', '有哪些']):
            return 'listing'
        return 'factual'