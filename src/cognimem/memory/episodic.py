import json
import sqlite3
import time
import numpy as np
from typing import List, Dict, Any, Optional


def _make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    return obj


class EpisodicMemory:
    """
    情景记忆，支持三层分块结构：
    - Turn-level: 单轮对话（最细粒度）
    - Chunk-level: N轮滑动窗口（检索单元）
    - Parent-level: 更大范围聚合（返回单元，提供完整上下文）

    Parent 机制：
    每个 chunk 都有一个 parent_id，指向一个覆盖更广范围的 parent chunk。
    检索时匹配到 chunk，但返回的是它的 parent，从而提供更完整的语境。
    """

    def __init__(self, db_path: str = 'episodic.db',
                 window_size: int = 3,
                 parent_window_size: int = 8):
        self.db_path = db_path
        self.window_size = window_size  # 子块窗口大小
        self.parent_window_size = parent_window_size  # 父块窗口大小
        self._buffer: List[Dict[str, Any]] = []
        self._parent_buffer: List[Dict[str, Any]] = []
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                content TEXT,
                metadata TEXT
            )
        ''')
        # 子块表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_turn_id INTEGER,
                end_turn_id INTEGER,
                content TEXT,
                parent_id INTEGER,
                timestamp REAL
            )
        ''')
        # 父块表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_turn_id INTEGER,
                end_turn_id INTEGER,
                content TEXT,
                timestamp REAL
            )
        ''')
        # 迁移：为已存在的 chunks 表添加 parent_id
        cursor.execute("PRAGMA table_info(chunks)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'parent_id' not in columns:
            cursor.execute("ALTER TABLE chunks ADD COLUMN parent_id INTEGER")

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_range ON chunks (start_turn_id, end_turn_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent_range ON parent_chunks (start_turn_id, end_turn_id)')
        conn.commit()
        conn.close()

    def store(self, item: Dict[str, Any]) -> int:
        """存储一轮对话，并维护滑动窗口生成子块和父块"""
        item_clean = _make_json_serializable(item)
        content = item_clean.get('content', '')
        metadata = item_clean.get('metadata', {})
        timestamp = item_clean.get('timestamp', time.time())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO episodes (timestamp, content, metadata) VALUES (?, ?, ?)',
            (timestamp,
             json.dumps(content, ensure_ascii=False),
             json.dumps(metadata, ensure_ascii=False))
        )
        turn_id = cursor.lastrowid
        conn.commit()
        conn.close()

        content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

        # 维护子块缓冲区
        self._buffer.append({'turn_id': turn_id, 'content': content_str})
        if len(self._buffer) >= self.window_size:
            self._create_chunk()
            self._buffer = self._buffer[-(self.window_size - 1):]

        # 维护父块缓冲区
        self._parent_buffer.append({'turn_id': turn_id, 'content': content_str})
        if len(self._parent_buffer) >= self.parent_window_size:
            self._create_parent_chunk()
            # 父块滑动步长为 1（保留 parent_window_size-1 个）
            self._parent_buffer = self._parent_buffer[-(self.parent_window_size - 1):]

        return turn_id

    def _create_chunk(self):
        """创建子块（window_size 轮），并关联到已有的父块"""
        recent = self._buffer[-self.window_size:]
        chunk_content = "\n".join([b['content'] for b in recent])
        start_turn_id = recent[0]['turn_id']
        end_turn_id = recent[-1]['turn_id']

        # 查找覆盖此范围的父块
        parent_id = self._find_parent_for_range(start_turn_id, end_turn_id)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO chunks (start_turn_id, end_turn_id, content, parent_id, timestamp) '
            'VALUES (?, ?, ?, ?, ?)',
            (start_turn_id, end_turn_id, chunk_content, parent_id, time.time())
        )
        conn.commit()
        conn.close()

    def _create_parent_chunk(self):
        """创建父块（parent_window_size 轮）"""
        recent = self._parent_buffer[-self.parent_window_size:]
        parent_content = "\n".join([b['content'] for b in recent])
        start_turn_id = recent[0]['turn_id']
        end_turn_id = recent[-1]['turn_id']

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO parent_chunks (start_turn_id, end_turn_id, content, timestamp) '
            'VALUES (?, ?, ?, ?)',
            (start_turn_id, end_turn_id, parent_content, time.time())
        )
        parent_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # 回填：将覆盖范围内的子块的 parent_id 更新为此父块
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chunks SET parent_id = ? WHERE start_turn_id >= ? AND end_turn_id <= ? AND parent_id IS NULL",
            (parent_id, start_turn_id, end_turn_id)
        )
        conn.commit()
        conn.close()

    def _find_parent_for_range(self, start_turn_id: int, end_turn_id: int) -> Optional[int]:
        """查找完全覆盖指定轮次范围的父块"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM parent_chunks WHERE start_turn_id <= ? AND end_turn_id >= ? "
            "ORDER BY id DESC LIMIT 1",
            (start_turn_id, end_turn_id)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_chunk_with_parent(self, turn_id: int, use_parent: bool = True) -> Optional[Dict]:
        """
        获取包含指定轮次的子块；如果 use_parent=True 且有父块，返回父块内容。
        这是检索的核心接口。
        """
        if turn_id is None:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, start_turn_id, end_turn_id, content, parent_id FROM chunks "
            "WHERE start_turn_id <= ? AND end_turn_id >= ? ORDER BY id DESC LIMIT 1",
            (turn_id, turn_id)
        )
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        chunk_id, start_id, end_id, content, parent_id = row

        # 如果有父块且要求使用，则返回父块内容
        if use_parent and parent_id:
            cursor.execute(
                "SELECT content FROM parent_chunks WHERE id = ?", (parent_id,)
            )
            parent_row = cursor.fetchone()
            if parent_row and parent_row[0]:
                content = parent_row[0]

        conn.close()
        return {
            'id': chunk_id,
            'start_turn_id': start_id,
            'end_turn_id': end_id,
            'content': content,
            'parent_id': parent_id
        }

    def get_chunk_by_turn(self, turn_id: int) -> Optional[Dict]:
        """兼容旧接口：返回子块（不替换为父块）"""
        return self.get_chunk_with_parent(turn_id, use_parent=False)

    def get_turn_content(self, turn_id: int) -> Optional[str]:
        """获取单轮内容"""
        if turn_id is None:
            return None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM episodes WHERE id = ?", (turn_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            try:
                content = json.loads(row[0])
            except:
                content = row[0]
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return None

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict]:
        """对子块进行关键词搜索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, start_turn_id, end_turn_id, content, parent_id FROM chunks "
            "WHERE content LIKE ? LIMIT ?",
            (f'%{query}%', limit)
        )
        rows = cursor.fetchall()
        conn.close()
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'start_turn_id': row[1],
                'end_turn_id': row[2],
                'content': row[3],
                'parent_id': row[4],
                'source': 'chunk_search',
                'confidence': 0.5
            })
        return results

    def raw_search(self, query: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, content, metadata FROM episodes WHERE content LIKE ? LIMIT ?",
            (f'%{query}%', limit)
        )
        rows = cursor.fetchall()
        conn.close()
        results = []
        for row in rows:
            results.append({
                'timestamp': row[0],
                'content': json.loads(row[1]),
                'metadata': json.loads(row[2]),
                'source': 'episodic',
                'confidence': 0.3
            })
        return results

    def get_recent(self, n: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, content, metadata FROM episodes ORDER BY timestamp DESC LIMIT ?",
            (n,)
        )
        rows = cursor.fetchall()
        conn.close()
        results = []
        for row in rows:
            results.append({
                'timestamp': row[0],
                'content': json.loads(row[1]),
                'metadata': json.loads(row[2])
            })
        return results