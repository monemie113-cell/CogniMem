"""
确定性查询分解模块。
纯规则实现，零LLM依赖，低成本设计。

支持：
- 并列连词拆分（and/or/和/与/以及）
- 复合问句拆分（多问号）
- 比较查询拆分（compare A and B）
- 时间条件分离
- 主谓宾结构化提取
"""

import re
from typing import List, Dict, Any, Optional


class QueryDecomposer:
    """
    查询分解器。
    将复杂查询拆分为多个可独立检索的子查询。
    """

    # 并列连词
    CONJUNCTIONS = [
        r'\s+and\s+', r'\s+or\s+', r'\s*和\s*', r'\s*与\s*',
        r'\s*以及\s*', r'\s*及\s*', r'\s*、\s*',
    ]

    # 比较查询模式
    COMPARISON_PATTERNS = [
        (r'(?:compare|比较)\s+(.+?)\s+(?:和|与|and|&)\s+(.+)', 'comparison'),
        (r'(.+?)\s+(?:和|与|and|&)\s+(.+?)\s+(?:有什么区别|有什么不同|区别|差异)', 'difference'),
        (r'(?:what|什么)\s+(?:is|是)\s+(?:the\s+)?(?:difference|区别)\s+(?:between|between)\s+(.+?)\s+(?:and|和)\s+(.+)', 'difference'),
    ]

    # 时间条件模式
    TEMPORAL_PATTERNS = [
        (r'\b(before|after|during|until|since)\s+(.+?)(?:\s|$)', 'temporal_relation'),
        (r'(?:在|从)\s*(.+?)\s*(?:之前|之后|期间|以来)', 'temporal_relation_zh'),
    ]

    # 问题词
    QUESTION_WORDS = {
        'what', 'when', 'where', 'who', 'which', 'how', 'why', 'whose', 'whom',
        '什么', '什么时候', '哪里', '谁', '哪个', '怎么', '为什么', '多少', '是否',
    }

    def decompose(self, query: str) -> Dict[str, Any]:
        """
        执行完整的查询分解。

        Returns:
            {
                'original': 原始查询,
                'sub_queries': 子查询列表（可独立检索）,
                'primary_query': 主查询（最核心的检索意图）,
                'temporal_constraint': 时间约束（如有）,
                'entities': 提取的实体,
                'query_type': 查询类型
            }
        """
        result = {
            'original': query,
            'sub_queries': [],
            'primary_query': query,
            'temporal_constraint': None,
            'entities': [],
            'query_type': 'simple',
        }

        # 1. 提取实体
        result['entities'] = self._extract_entities(query)

        # 2. 提取时间约束
        result['temporal_constraint'] = self._extract_temporal(query)

        # 3. 检测比较查询
        comparison = self._detect_comparison(query)
        if comparison:
            result['query_type'] = 'comparison'
            result['sub_queries'] = comparison
            result['primary_query'] = comparison[0] if comparison else query
            return result

        # 4. 拆分并列连词
        conjunction_parts = self._split_conjunctions(query)
        if len(conjunction_parts) > 1:
            result['query_type'] = 'conjunction'
            result['sub_queries'] = conjunction_parts
            result['primary_query'] = conjunction_parts[0]
            return result

        # 5. 拆分复合问句（多问号）
        question_parts = self._split_questions(query)
        if len(question_parts) > 1:
            result['query_type'] = 'multi_question'
            result['sub_queries'] = question_parts
            result['primary_query'] = question_parts[0]
            return result

        # 6. 简单查询
        result['sub_queries'] = [query]
        result['primary_query'] = self._extract_core_intent(query)

        return result

    def _extract_entities(self, query: str) -> List[str]:
        """提取专有名词和关键实体"""
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        if not entities:
            entities = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
        stop = {'什么', '怎么', '为什么', '哪个', '哪里', '是否', '能否', '如何'}
        return list(dict.fromkeys([e for e in entities if e not in stop]))

    def _extract_temporal(self, query: str) -> Optional[Dict[str, str]]:
        """提取时间约束"""
        for pattern, label in self.TEMPORAL_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    'type': label,
                    'relation': groups[0] if len(groups) > 0 else '',
                    'value': groups[1] if len(groups) > 1 else ''
                }
        # 提取具体日期
        date_match = re.search(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}', query)
        if date_match:
            return {'type': 'absolute_date', 'value': date_match.group()}
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            return {'type': 'year', 'value': year_match.group()}
        return None

    def _detect_comparison(self, query: str) -> Optional[List[str]]:
        """检测比较查询并返回子查询"""
        for pattern, _ in self.COMPARISON_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [g.strip() for g in match.groups() if g.strip()]
        return None

    def _split_conjunctions(self, query: str) -> List[str]:
        """按并列连词拆分查询"""
        parts = [query]
        for conj_pattern in self.CONJUNCTIONS:
            new_parts = []
            for p in parts:
                split_result = re.split(conj_pattern, p)
                # 过滤空字符串和过短的片段
                cleaned = [s.strip() for s in split_result if s.strip() and len(s.strip()) > 2]
                if len(cleaned) > 1:
                    new_parts.extend(cleaned)
                else:
                    new_parts.append(p)
            parts = new_parts
        return list(dict.fromkeys(parts))

    def _split_questions(self, query: str) -> List[str]:
        """按问号拆分复合问句"""
        if query.count('?') <= 1 and query.count('？') <= 1:
            return [query]
        parts = re.split(r'[?？]', query)
        return [p.strip() + '?' for p in parts if p.strip() and len(p.strip()) > 3]

    def _extract_core_intent(self, query: str) -> str:
        """提取核心检索意图（去除问题词）"""
        words = query.split()
        filtered = [w for w in words if w.lower() not in self.QUESTION_WORDS]
        core = ' '.join(filtered).strip()
        return core if core else query


# 全局单例（避免重复初始化）
_default_decomposer = None


def get_decomposer() -> QueryDecomposer:
    global _default_decomposer
    if _default_decomposer is None:
        _default_decomposer = QueryDecomposer()
    return _default_decomposer