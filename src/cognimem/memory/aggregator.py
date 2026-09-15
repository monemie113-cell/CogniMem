"""
轻量级聚合器：从多个检索片段中提取并聚合数值答案。
纯规则实现，零 LLM 依赖。
"""
import re
from typing import List, Dict, Any, Optional
from collections import Counter


class Aggregator:
    COUNT_PATTERNS = [r'how many', r'number of', r'count of', r'how often']
    SUM_PATTERNS = [
        r'how much (?:total|money|time|hours|days)',
        r'total (?:amount|money|hours|days|weeks|cost|spent)',
        r'in total', r'combined',
    ]
    MAX_PATTERNS = [r'\b(?:latest|most recent|newest|most expensive|longest)\b']
    MIN_PATTERNS = [r'\b(?:earliest|first|oldest|cheapest|shortest)\b']
    TEMPORAL_PATTERNS = [r'what time', r'\bwhen\b']

    def classify(self, query: str) -> Optional[str]:
        q = query.lower()
        # 修复 1：先处理 "how many <unit>" 歧义
        if re.search(r'how many\s+(hours|days|weeks|months|years|minutes|dollars)', q):
            return 'SUM'
        if any(re.search(p, q) for p in self.SUM_PATTERNS):
            return 'SUM'
        if any(re.search(p, q) for p in self.COUNT_PATTERNS):
            return 'COUNT'
        if any(re.search(p, q) for p in self.MAX_PATTERNS):
            return 'MAX'
        if any(re.search(p, q) for p in self.MIN_PATTERNS):
            return 'MIN'
        if any(re.search(p, q) for p in self.TEMPORAL_PATTERNS):
            return 'TEMPORAL'
        return None

    def aggregate(self, query: str, retrieved: List[str]) -> Optional[Dict[str, Any]]:
        if not retrieved:
            return None
        qtype = self.classify(query)
        if qtype is None:
            return None
        if qtype == 'SUM':
            return self._aggregate_sum(query, retrieved)
        if qtype == 'COUNT':
            return None  # COUNT 在真实数据上不可靠，暂时关闭
        if qtype in ('MAX', 'MIN'):
            return self._aggregate_extremum(query, retrieved, qtype)
        if qtype == 'TEMPORAL':
            return self._extract_temporal(query, retrieved)
        return None

    def _aggregate_sum(self, query, retrieved):
        # 修复 2：必须有单位约束
        unit = self._infer_unit(query)
        if unit is None:
            return None
        numbers = []
        for text in retrieved:
            numbers.extend(self._extract_numbers(text, unit))
        # 去重
        seen, unique = set(), []
        for n in numbers:
            k = round(n, 2)
            if k not in seen:
                seen.add(k); unique.append(n)
        if not unique:
            return None
        total = sum(unique)
        if unit == '$':
            answer = f"${total:.0f}" if total == int(total) else f"${total:.2f}"
        else:
            answer = f"{total:.0f} {unit}" if total == int(total) else f"{total:.1f} {unit}"
        return {'answer': answer, 'type': 'SUM', 'evidence': unique,
                'confidence': min(0.8, 0.4 + 0.1 * len(unique))}

    def _aggregate_count(self, query: str, retrieved: List[str]) -> Optional[Dict]:
        """
        计数策略（双层）：
        1. 优先：句子中若出现"小数字 + 名词"组合（如 "2 shirts"），直接累加
        2. 回退：用句子指纹去重，统计不同事件的句子数
        """
        query_nouns = self._extract_query_nouns(query)
        if not query_nouns:
            return None

        # ---------- 第一层：数字计数 ----------
        numbers = []
        for text in retrieved:
            for sent in re.split(r'[.!?]\s+', text):
                sent_lower = sent.lower()
                # 句子必须包含至少一个查询名词
                if not any(n in sent_lower for n in query_nouns):
                    continue
                # 匹配 "数字 + 名词" 模式（排除年份、编号等）
                for m in re.finditer(r'\b(\d{1,2})\s+[a-z]+s?\b', sent_lower):
                    num = int(m.group(1))
                    if 1 <= num <= 20:  # 合理计数范围
                        numbers.append(num)

        if numbers:
            total = sum(numbers)
            return {
                'answer': str(total),
                'type': 'COUNT',
                'evidence': numbers,
                'confidence': 0.65,
            }

        # ---------- 第二层：句子级去重 ----------
        sentences = []
        for text in retrieved:
            for sent in re.split(r'[.!?]\s+', text):
                sent_lower = sent.lower()
                if any(n in sent_lower for n in query_nouns):
                    # 用前 6 个词作为句子指纹
                    fingerprint = ' '.join(sent_lower.split()[:6])
                    sentences.append(fingerprint)

        unique = list(dict.fromkeys(sentences))
        if unique:
            return {
                'answer': str(len(unique)),
                'type': 'COUNT',
                'evidence': unique,
                'confidence': 0.5,
            }
        return None

    def _aggregate_extremum(self, query, retrieved, mode):
        unit = self._infer_unit(query)
        numbers = []
        for text in retrieved:
            numbers.extend(self._extract_numbers(text, unit))
        if not numbers:
            return None
        value = max(numbers) if mode == 'MAX' else min(numbers)
        if unit == '$':
            answer = f"${value:.0f}"
        elif unit:
            answer = f"{value:.0f} {unit}"
        else:
            answer = str(value)
        return {'answer': answer, 'type': mode, 'evidence': numbers, 'confidence': 0.6}

    def _extract_temporal(self, query, retrieved):
        patterns = [
            r'\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\b',
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
        ]
        candidates = []
        for text in retrieved:
            for p in patterns:
                for m in re.findall(p, text):
                    candidates.append(m.strip())
        if not candidates:
            return None
        top = Counter(c.lower() for c in candidates).most_common(1)[0][0]
        return {'answer': top, 'type': 'TEMPORAL', 'evidence': candidates, 'confidence': 0.5}

    def _infer_unit(self, query):
        q = query.lower()
        if any(w in q for w in ['money', 'cost', 'spent', 'paid', 'dollars', '$']):
            return '$'
        for u in ['hours', 'days', 'weeks', 'months', 'years', 'minutes']:
            if u in q or u[:-1] in q:
                return u
        return None

    def _extract_numbers(self, text, unit):
        text_l = text.lower()
        nums = []
        if unit == '$':
            for m in re.findall(r'\$\s*(\d+(?:\.\d+)?)', text_l):
                nums.append(float(m))
            for m in re.findall(r'(\d+(?:\.\d+)?)\s*(?:dollars|usd)', text_l):
                nums.append(float(m))
        elif unit:
            u = unit.rstrip('s')
            for m in re.findall(rf'(\d+(?:\.\d+)?)\s*(?:{u}s?)', text_l):
                nums.append(float(m))
        return [n for n in nums if not (1900 <= n <= 2100)]

    def _extract_query_nouns(self, query):
        stop = {'how', 'many', 'much', 'total', 'the', 'a', 'an', 'i', 'my', 'me',
                'did', 'do', 'have', 'has', 'is', 'are', 'was', 'were', 'been',
                'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
                'what', 'when', 'where', 'which', 'who', 'that', 'this'}
        return [w for w in re.findall(r'\b[a-z]{3,}\b', query.lower()) if w not in stop]