import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from ..utils.text_processor import embed_text

class SemanticMemory:
    def __init__(self, db_path: str = 'semantic.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence REAL,
                evidence TEXT,
                embedding BLOB,
                is_controversial INTEGER DEFAULT 0,
                source TEXT DEFAULT '',
                episodic_id INTEGER
            )
        ''')
        # 迁移：检查并添加缺失的列
        cursor.execute("PRAGMA table_info(triples)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'is_controversial' not in columns:
            cursor.execute("ALTER TABLE triples ADD COLUMN is_controversial INTEGER DEFAULT 0")
        if 'source' not in columns:
            cursor.execute("ALTER TABLE triples ADD COLUMN source TEXT DEFAULT ''")
        if 'episodic_id' not in columns:
            cursor.execute("ALTER TABLE triples ADD COLUMN episodic_id INTEGER")
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject_predicate ON triples (subject, predicate)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_object ON triples (object)')
        conn.commit()
        conn.close()

    def store(self, triple: Dict[str, Any]):
        """存储三元组，自动处理冲突"""
        conflict = self.detect_conflict(triple)
        if conflict:
            resolved = self._resolve_conflict(triple, conflict)
            if resolved == 'update':
                self._update_existing(triple, conflict)
                return
            elif resolved == 'keep_both':
                triple['is_controversial'] = 1
                self._insert_new(triple)
                return
            else:
                return
        self._insert_new(triple)

    def _insert_new(self, triple: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        text_for_embed = f"{triple['subject']} {triple['predicate']} {triple['object']}"
        emb_bytes = embed_text(text_for_embed).astype(np.float32).tobytes()
        cursor.execute(
            """INSERT INTO triples 
               (subject, predicate, object, confidence, evidence, embedding, is_controversial, source, episodic_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (triple['subject'], triple['predicate'], triple['object'],
             triple.get('confidence', 0.5), triple.get('evidence', ''),
             emb_bytes, triple.get('is_controversial', 0), triple.get('source', ''),
             triple.get('episodic_id'))
        )
        conn.commit()
        conn.close()

    def _update_existing(self, new_triple: Dict, existing_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT confidence, evidence FROM triples WHERE id = ?",
            (existing_id,)
        )
        old_conf, old_evi = cursor.fetchone()
        new_conf = max(old_conf, new_triple.get('confidence', 0.5))
        new_evi = old_evi + " | " + new_triple.get('evidence', '')
        cursor.execute(
            "UPDATE triples SET confidence = ?, evidence = ? WHERE id = ?",
            (new_conf, new_evi, existing_id)
        )
        conn.commit()
        conn.close()

    def detect_conflict(self, new_triple: Dict) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object != ?",
            (new_triple['subject'], new_triple['predicate'], new_triple['object'])
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def _resolve_conflict(self, new_triple: Dict, existing_id: int) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT confidence, evidence, source FROM triples WHERE id = ?",
            (existing_id,)
        )
        old_conf, old_evi, old_source = cursor.fetchone()
        conn.close()
        new_conf = new_triple.get('confidence', 0.5)
        if abs(new_conf - old_conf) > 0.3:
            return 'update' if new_conf > old_conf else 'ignore'
        if new_triple.get('source') == 'human' and old_source != 'human':
            return 'update'
        if old_source == 'human' and new_triple.get('source') != 'human':
            return 'ignore'
        return 'keep_both'

    # ---------- 图推理 ----------
    def transitive_reason(self, subject: str, predicate: str, max_depth: int = 2) -> List[Dict]:
        if max_depth < 1:
            return []
        results = []
        visited = set()
        queue = deque([(subject, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                continue
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT subject, predicate, object, confidence, evidence FROM triples WHERE subject = ? AND is_controversial = 0",
                (node,)
            )
            rows = cursor.fetchall()
            conn.close()
            for row in rows:
                s, p, o, conf, evi = row
                key = (s, p, o)
                if key in visited:
                    continue
                visited.add(key)
                results.append({'subject': s, 'predicate': p, 'object': o, 'confidence': conf, 'evidence': evi, 'depth': depth})
                if depth < max_depth:
                    queue.append((o, depth + 1))
        return results

    def reason_with_path(self, subject: str, predicate: str, max_depth: int = 2) -> List[Dict]:
        direct = self.exact_match(subject, predicate)
        transitive = self.transitive_reason(subject, predicate, max_depth)
        seen_objs = set()
        combined = []
        for item in direct + transitive:
            obj = item['object']
            if obj not in seen_objs:
                seen_objs.add(obj)
                combined.append(item)
        return combined

    def exact_match(self, subject: str, predicate: Optional[str] = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if predicate:
            cursor.execute(
                "SELECT subject, predicate, object, confidence, evidence FROM triples WHERE subject = ? AND predicate = ?",
                (subject, predicate)
            )
        else:
            cursor.execute(
                "SELECT subject, predicate, object, confidence, evidence FROM triples WHERE subject = ?",
                (subject,)
            )
        rows = cursor.fetchall()
        conn.close()
        results = []
        for row in rows:
            results.append({
                'subject': row[0],
                'predicate': row[1],
                'object': row[2],
                'confidence': row[3],
                'evidence': row[4],
                'source': 'exact'
            })
        return results

    def multi_hop_reason(self, start_entity: str, target_entity: str = None,
                         max_depth: int = 3, return_paths: bool = True) -> List[Dict]:
        """
        多跳推理：从 start_entity 出发，查找所有可达路径
        支持：
        - 目标已知：查找 start → target 的所有路径
        - 目标未知：查找所有深度 ≤ max_depth 的路径
        """
        from collections import deque
        results = []
        visited = set()
        # 队列元素: (当前实体, 路径列表, 深度)
        queue = deque([(start_entity, [], 0)])

        while queue:
            node, path, depth = queue.popleft()
            if depth > max_depth:
                continue
            if depth > 0 and (target_entity is None or node == target_entity):
                # 记录一条完整路径
                results.append({
                    'path': path + [node],
                    'depth': depth,
                    'entities': [p['subject'] for p in path] + [node],
                    'relations': [p['predicate'] for p in path]
                })
                if target_entity is not None and node == target_entity:
                    continue  # 继续找其他路径

            # 查找从当前节点出发的所有三元组
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT subject, predicate, object, confidence, evidence FROM triples "
                "WHERE subject = ? AND is_controversial = 0",
                (node,)
            )
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                s, p, o, conf, evi = row
                # 避免循环
                edge_key = (s, p, o)
                if edge_key in visited:
                    continue
                visited.add(edge_key)
                new_path = path + [{'subject': s, 'predicate': p, 'object': o,
                                    'confidence': conf, 'evidence': evi}]
                queue.append((o, new_path, depth + 1))

        # 按深度和置信度排序
        results.sort(key=lambda x: (x['depth'], -max([e.get('confidence', 0) for e in x['path']])))
        return results

    def find_paths(self, start: str, end: str, max_depth: int = 4) -> List[Dict]:
        """
        路径查找：查找 start → end 的所有路径（带完整推理链）
        使用 BFS，返回路径列表
        """
        return self.multi_hop_reason(start, target_entity=end, max_depth=max_depth)

    def get_path_context(self, path: List[Dict]) -> str:
        """
        将推理路径格式化为自然语言描述，用于 LLM 上下文
        """
        if not path:
            return ""
        parts = []
        for step in path:
            parts.append(f"{step['subject']} →({step['predicate']})→ {step['object']}")
        return " → ".join(parts)

    # 原方法保留
    def bm25_match(self, query: str) -> List[Dict]:
        return self.exact_match(query)

    def vector_similarity(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query:
            return []
        query_vec = embed_text(query)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT subject, predicate, object, confidence, evidence, embedding FROM triples")
        rows = cursor.fetchall()
        conn.close()
        results = []
        for row in rows:
            if row[5] is None:
                continue
            stored_vec = np.frombuffer(row[5], dtype=np.float32)
            sim = np.dot(query_vec, stored_vec)
            results.append({
                'subject': row[0],
                'predicate': row[1],
                'object': row[2],
                'confidence': row[3] * (0.5 + 0.5 * sim),
                'evidence': row[4],
                'score': sim,
                'source': 'vector'
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def detect_conflict(self, new_triple: Dict) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object != ?",
            (new_triple['subject'], new_triple['predicate'], new_triple['object'])
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def update(self, old_triple: Dict, new_triple: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM triples WHERE subject=? AND predicate=? AND object=?",
            (old_triple['subject'], old_triple['predicate'], old_triple['object'])
        )
        self.store(new_triple)
        conn.close()