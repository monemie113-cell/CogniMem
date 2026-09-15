import time
import json
import sqlite3
import numpy as np
import re
from typing import Any, Dict, List, Optional
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working_memory import WorkingMemory
from .routing import RoutingManager
from ..utils.text_processor import get_nlp

try:
    from .spreading_activation import SpreadingActivation
    from .query_decomposition import get_decomposer
    from .bm25_retriever import BM25Retriever
    from .reranker import CrossEncoderReranker
    from .fusion import reciprocal_rank_fusion
    from .query_understanding import QueryAnalyzer
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class PersistentMemory:
    def __init__(self, config: Dict):
        self.config = config
        self.episodic = EpisodicMemory(
            config.get('episodic_db', 'episodic.db'),
            window_size=config.get('chunk_window_size', 3),
            parent_window_size=config.get('parent_window_size', 8)
        )
        self.semantic = SemanticMemory(config.get('semantic_db', 'semantic.db'))
        self.routing = RoutingManager(config.get('routing', {}))
        self.working_memory = WorkingMemory(capacity=config.get('working_memory_capacity', 10))
        self.working_memory_list = self.working_memory.buffer

        recall_config = config.get('recall', {})
        self.use_bm25 = recall_config.get('use_bm25', False)
        self.use_reranker = recall_config.get('use_reranker', False)
        self.use_query_understanding = recall_config.get('use_query_understanding', False)
        self.use_query_decomposition = recall_config.get('use_query_decomposition', False)
        self.use_spreading_activation = recall_config.get('use_spreading_activation', False)
        self.use_parent_chunks = recall_config.get('use_parent_chunks', False)

        self.bm25 = None
        self.reranker = None
        self.query_analyzer = None
        self.spreading = None
        self.decomposer = None
        self._bm25_indexed = False

        if ENHANCEMENTS_AVAILABLE:
            if self.use_bm25:
                try:
                    self.bm25 = BM25Retriever(k1=1.2, b=0.75)
                    print("BM25 retriever enabled.")
                except Exception as e:
                    print(f"BM25 init failed: {e}")
                    self.use_bm25 = False

            if self.use_reranker:
                try:
                    self.reranker = CrossEncoderReranker(
                        model_name=recall_config.get('reranker_model',
                                                     'cross-encoder/ms-marco-MiniLM-L-6-v2')
                    )
                except Exception as e:
                    print(f"Reranker init failed: {e}")
                    self.use_reranker = False

            if self.use_query_understanding:
                try:
                    self.query_analyzer = QueryAnalyzer(llm_client=None)
                    print("Query analyzer enabled.")
                except Exception as e:
                    print(f"Query analyzer init failed: {e}")
                    self.use_query_understanding = False

            if self.use_query_decomposition:
                try:
                    self.decomposer = get_decomposer()
                    print("Query decomposer enabled.")
                except Exception as e:
                    print(f"Query decomposer init failed: {e}")
                    self.decomposer = None

        if self.use_spreading_activation:
            try:
                self.spreading = SpreadingActivation(
                    db_path=self.semantic.db_path,
                    decay_factor=config.get('spreading_decay', 0.5),
                    max_hops=config.get('spreading_max_hops', 2),
                    activation_threshold=config.get('spreading_threshold', 0.1),
                    energy_budget=config.get('spreading_budget', 10.0)
                )
                print("Spreading activation enabled.")
            except Exception as e:
                print(f"Spreading activation init failed: {e}")
                self.use_spreading_activation = False

    # ==================== 存储 ====================
    def store(self, item: Dict):
        item = _json_safe(item)
        episodic_id = self.episodic.store(item)
        triples = self._extract_triples(item, episodic_id)
        for triple in triples:
            self.semantic.store(triple)
        self.working_memory.push(item)
        if self.spreading is not None:
            self.spreading.invalidate_cache()
        if self.use_bm25:
            self._bm25_indexed = False
        return True

    def _extract_triples(self, item: Dict, episodic_id: int) -> List[Dict]:
        """生产级三元组提取：spaCy + 代词消解 + 时间归一化 + 知识层 + 智能回退"""
        text = item.get('content', '') or item.get('input', '')
        if not text or len(text.strip()) < 5:
            return []

        speaker = ''
        session_time = ''
        metadata = item.get('metadata', {})
        if isinstance(metadata, dict):
            speaker = metadata.get('speaker', '') or metadata.get('name', '')
            session_time = metadata.get('session_time', '')

        triples = []
        text_stripped = text.strip()

        # ---------- 时间归一化 ----------
        time_triples = self._extract_time_triples(
            text_stripped, session_time, speaker, episodic_id
        )
        triples.extend(time_triples)

        # ---------- 策略 1：spaCy 依存分析 ----------
        nlp = get_nlp()
        if nlp is not None and re.search(r'[a-zA-Z]{2,}', text):
            try:
                doc = nlp(text[:512])
                for token in doc:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                        subj = token.text.strip()
                        pred = token.head.text.strip()
                        if not subj or not pred or len(subj) < 2 or len(pred) < 2:
                            continue
                        if subj.lower() in ('i', 'me', 'my', 'myself') and speaker:
                            subj = speaker
                        obj = None
                        for child in token.head.children:
                            if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                                candidate = child.text.strip()
                                if len(candidate) >= 2:
                                    obj = candidate
                                    break
                            if child.dep_ == "prep":
                                for grandchild in child.children:
                                    if grandchild.dep_ == "pobj" and len(grandchild.text.strip()) >= 2:
                                        obj = f"{child.text} {grandchild.text}".strip()
                                        break
                                if obj:
                                    break
                        if obj:
                            if obj.lower() in ('i', 'me', 'my', 'myself') and speaker:
                                obj = speaker
                            triples.append({
                                'subject': subj,
                                'predicate': pred,
                                'object': obj,
                                'confidence': item.get('confidence', 0.7),
                                'evidence': text_stripped,
                                'episodic_id': episodic_id
                            })
            except Exception:
                pass

        # ---------- 策略 2：中文规则 ----------
        if re.search(r'[\u4e00-\u9fa5]{2,}', text):
            patterns = [
                (r'([\u4e00-\u9fa5]{2,})\s*是\s*([\u4e00-\u9fa5]{2,})', '是'),
                (r'([\u4e00-\u9fa5]{2,})\s*有\s*([\u4e00-\u9fa5]{2,})', '有'),
                (r'([\u4e00-\u9fa5]{2,})\s*喜欢\s*([\u4e00-\u9fa5]{2,})', '喜欢'),
                (r'([\u4e00-\u9fa5]{2,})\s*在\s*([\u4e00-\u9fa5]{2,})', '在'),
                (r'([\u4e00-\u9fa5]{2,})\s*去了\s*([\u4e00-\u9fa5]{2,})', '去了'),
            ]
            for pattern, pred in patterns:
                matches = re.findall(pattern, text_stripped)
                for subj, obj in matches:
                    triples.append({
                        'subject': subj.strip(),
                        'predicate': pred,
                        'object': obj.strip(),
                        'confidence': item.get('confidence', 0.7),
                        'evidence': text_stripped,
                        'episodic_id': episodic_id
                    })

        # ---------- 策略 3：智能回退 ----------
        # 关键：即使策略 1/2 命中，也保留整句事件层（因为全文检索依赖 evidence）
        has_event_layer = any(
            t.get('predicate') == 'said' and t.get('object', '').startswith(text_stripped[:30])
            for t in triples
        )
        if not has_event_layer and len(text_stripped) >= 20:
            words = re.findall(r'[a-zA-Z]{3,}|[\u4e00-\u9fa5]{2,}', text_stripped)
            if len(words) >= 3:
                subj = speaker if speaker else 'user'
                triples.append({
                    'subject': subj,
                    'predicate': 'said',
                    'object': text_stripped[:150],
                    'confidence': item.get('confidence', 0.4),
                    'evidence': text_stripped,
                    'episodic_id': episodic_id
                })

        # ===== 追加知识层三元组（不替换事件层） =====
        knowledge_triples = self._extract_knowledge_triples(
            text_stripped, speaker, episodic_id
        )
        triples.extend(knowledge_triples)

        return triples

    def _extract_knowledge_triples(self, text: str, speaker: str,
                                    episodic_id: int) -> List[Dict]:
        """
        提取知识层三元组（实体-关系-实体），用于扩散激活的图遍历。
        与事件层（speaker, said, 整句）并存，不替换。
        关键：evidence 字段仍然存储完整原文，保证全文检索不受影响。
        """
        triples = []
        nlp = get_nlp()
        if nlp is None:
            return triples
        if not re.search(r'[a-zA-Z]{2,}', text):
            return triples
        if len(text.strip()) < 15:
            return triples

        try:
            doc = nlp(text[:512])
        except Exception:
            return triples

        for token in doc:
            if token.pos_ != "VERB":
                continue

            # 找主语（含代词消解）
            subj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child.text.strip()
                    if subj.lower() in ('i', 'me', 'my', 'myself') and speaker:
                        subj = speaker
                    break
            if not subj or len(subj) < 2:
                continue

            verb_lemma = token.lemma_.lower()
            if len(verb_lemma) < 2:
                continue

            # 模式 1：动词 + 直接宾语
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    obj = child.text.strip()
                    if len(obj) >= 3:
                        triples.append({
                            'subject': subj,
                            'predicate': verb_lemma,
                            'object': obj,
                            'confidence': 0.7,
                            'evidence': text.strip(),
                            'episodic_id': episodic_id
                        })

                # 模式 2：动词 + 介词 + 名词短语
                if child.dep_ == "prep":
                    prep = child.text.lower()
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            np_tokens = []
                            for t in grandchild.subtree:
                                if t.dep_ in ("compound", "amod", "det", "pobj"):
                                    np_tokens.append(t.text)
                            obj = " ".join(np_tokens) if np_tokens else grandchild.text
                            obj = obj.strip()
                            if 3 <= len(obj) <= 60:
                                triples.append({
                                    'subject': subj,
                                    'predicate': f"{verb_lemma}_{prep}",
                                    'object': obj,
                                    'confidence': 0.65,
                                    'evidence': text.strip(),
                                    'episodic_id': episodic_id
                                })

        # 去重
        seen = set()
        unique = []
        for t in triples:
            key = (t['subject'].lower(), t['predicate'], t['object'].lower())
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique

    def _extract_time_triples(self, text: str, session_time: str,
                               speaker: str, episodic_id: int) -> List[Dict]:
        """检测相对时间表达，基于 session_time 转换为绝对日期。"""
        from datetime import datetime, timedelta
        import re as _re

        triples = []
        if not session_time:
            return triples

        session_dt = self._parse_session_time(session_time)
        if session_dt is None:
            return triples

        time_patterns = [
            (r'\byesterday\b', timedelta(days=-1)),
            (r'\btoday\b', timedelta(days=0)),
            (r'\btomorrow\b', timedelta(days=1)),
            (r'\blast night\b', timedelta(days=-1)),
            (r'\blast week\b', timedelta(weeks=-1)),
            (r'\blast month\b', timedelta(days=-30)),
            (r'\blast year\b', timedelta(days=-365)),
            (r'\blast monday\b', timedelta(days=-1)),
            (r'\blast tuesday\b', timedelta(days=-2)),
            (r'\blast wednesday\b', timedelta(days=-3)),
            (r'\blast thursday\b', timedelta(days=-4)),
            (r'\blast friday\b', timedelta(days=-5)),
            (r'\blast saturday\b', timedelta(days=-6)),
            (r'\blast sunday\b', timedelta(days=-7)),
        ]

        text_lower = text.lower()
        for pattern, delta in time_patterns:
            if _re.search(pattern, text_lower):
                try:
                    event_dt = session_dt + delta
                    absolute_date = event_dt.strftime('%d %B %Y').lstrip('0')
                    triples.append({
                        'subject': speaker or 'session',
                        'predicate': 'event_date',
                        'object': absolute_date,
                        'confidence': 0.85,
                        'evidence': text.strip(),
                        'episodic_id': episodic_id
                    })
                except Exception:
                    pass
        return triples

    def _parse_session_time(self, session_time: str):
        """解析 '1:56 pm on 8 May, 2023' 格式为 datetime"""
        from datetime import datetime
        import re as _re
        m = _re.search(r'(\d{1,2}\s+\w+,?\s+\d{4})', session_time)
        if not m:
            return None
        date_str = m.group(1).replace(',', '')
        for fmt in ('%d %B %Y', '%d %b %Y'):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _get_turn_date(self, turn_id: int) -> Optional[str]:
        """获取与 turn 关联的绝对日期（如果可用）"""
        if turn_id is None:
            return None
        try:
            conn = sqlite3.connect(self.semantic.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT object FROM triples WHERE episodic_id = ? AND predicate = 'event_date' LIMIT 1",
                (turn_id,)
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    # ==================== 检索 ====================
    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        sub_queries = [query]
        primary_query = query
        entities = []

        if self.use_query_decomposition and self.decomposer is not None:
            try:
                decomposition = self.decomposer.decompose(query)
                sub_queries = decomposition['sub_queries']
                primary_query = decomposition['primary_query']
                entities = decomposition['entities']
            except Exception:
                pass

        if not entities:
            entities = re.findall(r'\b[A-Z][a-z]+\b', query)
            if not entities:
                entities = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
            entities = list(dict.fromkeys(entities))

        ranked_lists = []

        entity_results = self._entity_graph_search(entities, query=query, limit=20)
        if entity_results:
            ranked_lists.append(entity_results)

        if self.use_spreading_activation and entities:
            activation_results = self._spreading_activation_search(entities, limit=15)
            if activation_results:
                ranked_lists.append(activation_results)

        chunk_semantic_results = self._chunk_semantic_search(primary_query, entity_results, limit=15)
        if chunk_semantic_results:
            ranked_lists.append(chunk_semantic_results)

        for sub_q in sub_queries[:2]:
            vector_results = self._vector_search_episodes(sub_q, top_k=10)
            if vector_results:
                ranked_lists.append(vector_results)

        bm25_results = []
        if self.use_bm25 and self.bm25 is not None:
            self._ensure_bm25_indexed()
            for q in sub_queries[:2]:
                bm25_results.extend(self.bm25.search(q, top_k=15))
            if bm25_results:
                ranked_lists.append(bm25_results)

        if not ranked_lists:
            return []

        if ENHANCEMENTS_AVAILABLE:
            weights = []
            for rl in ranked_lists:
                src = rl[0].get('source', '') if rl else ''
                if 'entity' in src:
                    weights.append(1.2)
                elif 'spreading' in src:
                    weights.append(1.1)
                elif 'bm25' in src:
                    weights.append(1.0)
                elif 'vector' in src:
                    weights.append(1.0)
                else:
                    weights.append(0.9)
            fused = reciprocal_rank_fusion(ranked_lists, k=60, weights=weights)
        else:
            fused = self._simple_merge(ranked_lists)

        if self.use_reranker and self.reranker is not None:
            reranked = self.reranker.rerank(query, fused[:20], top_k=k * 2, blend_weight=0.3)
        else:
            for item in fused:
                if 'confidence' not in item:
                    item['confidence'] = 0.5
            fused.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)
            reranked = fused[:k * 2]

        if self.use_parent_chunks:
            final = self._replace_with_parents(reranked, k=k)
        else:
            final = reranked[:k]

        return final

    def _entity_graph_search(self, entities: List[str], query: str = '', limit: int = 20) -> List[Dict]:
        """实体+关键词驱动的图检索：IDF 加权，按相关性排序"""
        if not entities and not query:
            return []

        stop_words = {
            'when', 'what', 'where', 'who', 'which', 'how', 'why', 'whose',
            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'the', 'a', 'an',
            'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at', 'by',
            'this', 'that', 'these', 'those', 'with', 'from', 'as', 'be',
        }
        temporal_hints = {
            'yesterday', 'today', 'tomorrow', 'last', 'next', 'ago',
            'year', 'month', 'week', 'day', 'recently',
        }
        keywords = list(entities)
        if query:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            for w in words:
                if w in temporal_hints or w not in stop_words:
                    keywords.append(w)
            zh_words = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
            keywords.extend(zh_words)

        seen_lower = set()
        unique_kw = []
        for kw in keywords:
            low = kw.lower()
            if low not in seen_lower and len(kw) >= 2:
                seen_lower.add(low)
                unique_kw.append(kw)
        keywords = unique_kw[:5]

        if not keywords:
            return []

        # ===== IDF 加权 =====
        import math
        conn = sqlite3.connect(self.semantic.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM triples")
        total_docs = max(1, cursor.fetchone()[0])

        keyword_idf = {}
        for kw in keywords:
            cursor.execute(
                "SELECT COUNT(*) FROM triples WHERE subject LIKE ? OR object LIKE ? OR evidence LIKE ?",
                (f'%{kw}%', f'%{kw}%', f'%{kw}%')
            )
            df = cursor.fetchone()[0]
            keyword_idf[kw] = math.log((total_docs + 1) / (df + 1)) + 0.1

        # ===== 按 IDF 加权累加分数（DISTINCT 避免重复计数） =====
        turn_scores: Dict[int, float] = {}
        for kw in keywords:
            idf_weight = keyword_idf[kw]
            cursor.execute(
                "SELECT DISTINCT episodic_id FROM triples "
                "WHERE subject LIKE ? OR object LIKE ? OR evidence LIKE ?",
                (f'%{kw}%', f'%{kw}%', f'%{kw}%')
            )
            for row in cursor.fetchall():
                ep_id = row[0]
                if ep_id is None:
                    continue
                turn_scores[ep_id] = turn_scores.get(ep_id, 0.0) + idf_weight
        conn.close()

        # ===== 归一化 =====
        if turn_scores:
            max_score = max(turn_scores.values())
            if max_score > 0:
                turn_scores = {k: v / max_score for k, v in turn_scores.items()}

        # ===== 时间类查询优先返回含 event_date 的 turn =====
        if query and re.search(r'\bwhen\b', query.lower()):
            conn2 = sqlite3.connect(self.semantic.db_path)
            cursor2 = conn2.cursor()
            cursor2.execute(
                "SELECT DISTINCT episodic_id FROM triples WHERE predicate = 'event_date'"
            )
            event_turn_ids = {row[0] for row in cursor2.fetchall() if row[0]}
            conn2.close()
            for turn_id in event_turn_ids:
                if turn_id in turn_scores:
                    turn_scores[turn_id] = min(1.0, turn_scores[turn_id] * 2.0)

        sorted_turns = sorted(turn_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for turn_id, score in sorted_turns[:limit]:
            chunk = self.episodic.get_chunk_by_turn(turn_id)
            date_info = self._get_turn_date(turn_id)
            date_prefix = f"[Date: {date_info}] " if date_info else ""

            if chunk:
                content = date_prefix + chunk['content']
                if content not in seen:
                    seen.add(content)
                    results.append({
                        'id': f'entity_chunk_{chunk["id"]}',
                        'content': content,
                        'confidence': min(0.9, 0.5 + 0.4 * score),
                        'source': 'entity_graph'
                    })
            else:
                content = self.episodic.get_turn_content(turn_id)
                if content:
                    content = date_prefix + content
                    if content not in seen:
                        seen.add(content)
                        results.append({
                            'id': f'entity_turn_{turn_id}',
                            'content': content,
                            'confidence': min(0.9, 0.4 + 0.4 * score),
                            'source': 'entity_turn'
                        })
        return results

    def _spreading_activation_search(self, entities: List[str], limit: int = 15) -> List[Dict]:
        if self.spreading is None:
            return []
        try:
            activated = self.spreading.get_activated_turn_ids(entities, max_results=limit)
            results = []
            seen = set()
            for act in activated:
                turn_id = act['turn_id']
                content = self.episodic.get_turn_content(turn_id)
                if content and content not in seen:
                    seen.add(content)
                    results.append({
                        'id': f'spread_{turn_id}',
                        'content': content,
                        'confidence': act['activation_score'] * 0.7,
                        'source': 'spreading_activation'
                    })
            return results
        except Exception:
            return []

    def _chunk_semantic_search(self, query: str, candidates: List[Dict], limit: int = 15) -> List[Dict]:
        if not candidates:
            return []
        try:
            from ..utils.text_processor import embed_text
            query_vec = embed_text(query)
            scored = []
            for cand in candidates:
                try:
                    cand_vec = embed_text(cand['content'][:512])
                    sim = float(np.dot(query_vec, cand_vec))
                    scored.append({
                        'id': cand.get('id', ''),
                        'content': cand['content'],
                        'confidence': 0.7 * sim + 0.3 * cand.get('confidence', 0.5),
                        'source': 'chunk_semantic'
                    })
                except Exception:
                    scored.append(cand)
            scored.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return scored[:limit]
        except Exception:
            return candidates[:limit]

    def _vector_search_episodes(self, query: str, top_k: int = 15) -> List[Dict]:
        try:
            from ..utils.text_processor import embed_text
            query_vec = embed_text(query)
        except Exception:
            return []

        conn = sqlite3.connect(self.episodic.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM episodes ORDER BY id DESC LIMIT 200")
        rows = cursor.fetchall()
        conn.close()

        scored = []
        for row in rows:
            try:
                content = json.loads(row[1])
            except:
                content = row[1]
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                vec = embed_text(content[:256])
                sim = float(np.dot(query_vec, vec))
                scored.append((sim, row[0], content))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, doc_id, content in scored[:top_k]:
            results.append({
                'id': f'vec_{doc_id}',
                'content': content,
                'confidence': max(0.0, min(1.0, (sim + 1) / 2)),
                'source': 'vector'
            })
        return results

    def _simple_merge(self, lists: List[List[Dict]]) -> List[Dict]:
        merged = []
        seen = set()
        for lst in lists:
            for item in lst:
                key = item.get('content', '')[:100]
                if key and key not in seen:
                    seen.add(key)
                    merged.append(item)
        merged.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)
        return merged

    def _replace_with_parents(self, items: List[Dict], k: int) -> List[Dict]:
        final = []
        seen_contents = set()
        for item in items:
            content = item.get('content', '')
            source = item.get('source', '')
            turn_id = None
            item_id = item.get('id', '')
            for prefix in ('entity_turn_', 'spread_', 'vec_'):
                if item_id.startswith(prefix):
                    try:
                        turn_id = int(item_id.replace(prefix, ''))
                    except ValueError:
                        pass
                    break
            if turn_id is not None and self.use_parent_chunks:
                try:
                    chunk_with_parent = self.episodic.get_chunk_with_parent(turn_id, use_parent=True)
                    if chunk_with_parent and chunk_with_parent.get('content'):
                        content = chunk_with_parent['content']
                        source = source + '+parent'
                except Exception:
                    pass
            if content and content not in seen_contents:
                seen_contents.add(content)
                final.append({
                    'content': content,
                    'confidence': item.get('confidence', 0.5),
                    'source': source
                })
            if len(final) >= k:
                break
        return final

    def _ensure_bm25_indexed(self):
        if self._bm25_indexed or self.bm25 is None:
            return
        conn = sqlite3.connect(self.episodic.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM episodes")
        rows = cursor.fetchall()
        conn.close()
        if rows:
            docs = []
            for row in rows:
                try:
                    content = json.loads(row[1])
                except:
                    content = row[1]
                if isinstance(content, str) and content.strip():
                    docs.append({'id': row[0], 'content': content})
            self.bm25.index(docs)
        self._bm25_indexed = True

    def reason(self, subject: str, predicate: str, max_depth: int = 2) -> List[Dict]:
        return self.semantic.reason_with_path(subject, predicate, max_depth)

    def get_curiosity(self):
        from .curiosity import CuriosityEngine
        return CuriosityEngine(self)

    def _apply_forgetting(self):
        pass