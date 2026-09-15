import time
from typing import Any, Dict, List, Optional

class RoutingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 60)

    def retrieve(self, query, episodic, semantic, working, k):
        results = []
        cache_key = hash(str(query))
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['ts'] < self.cache_ttl:
                return cached['data'][:k]

        # L0: working memory
        for item in working:
            if query.lower() in str(item).lower():
                results.append(item)

        # L1: exact triple match
        triples = semantic.exact_match(query)
        results.extend(triples)

        # L1.5: BM25 (fallback to exact for now)
        if len(results) < k:
            fuzzy = semantic.bm25_match(query)
            results.extend(fuzzy)

        # L2: vector similarity
        if len(results) < k:
            vec_results = semantic.vector_similarity(query, top_k=k)
            results.extend(vec_results)

        # L3: raw episodic search
        if len(results) < k:
            raw = episodic.raw_search(query, limit=k)
            results.extend(raw)

        results = self._dedup_and_rank(results)[:k]
        self.cache[cache_key] = {'ts': time.time(), 'data': results}
        return results

    def _dedup_and_rank(self, items):
        seen = set()
        unique = []
        for item in items:
            key = f"{item.get('subject','')}{item.get('predicate','')}{item.get('object','')}"
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return sorted(unique, key=lambda x: x.get('confidence', 0), reverse=True)