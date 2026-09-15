"""
扩散激活（Spreading Activation）模块。
模拟人脑联想记忆机制：从种子实体出发，沿图结构扩散激活能量，
自动发现间接相关的记忆节点。

设计原则：
- 零LLM依赖，纯图算法
- 可配置的衰减系数和扩散深度
- 支持能量预算，防止无限扩散
"""

import sqlite3
from collections import deque, defaultdict
from typing import List, Dict, Any, Set, Optional


class SpreadingActivation:
    """
    扩散激活引擎。

    核心参数：
    - decay_factor: 每跳的能量衰减系数（默认0.5）
    - max_hops: 最大扩散跳数（默认2）
    - activation_threshold: 激活阈值，低于此值不再扩散
    - energy_budget: 总能量预算，防止无限扩散
    """

    def __init__(self, db_path: str, decay_factor: float = 0.5,
                 max_hops: int = 2, activation_threshold: float = 0.1,
                 energy_budget: float = 10.0):
        self.db_path = db_path
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.activation_threshold = activation_threshold
        self.energy_budget = energy_budget
        self._adjacency_cache = None
        self._cache_valid = False

    def _build_adjacency(self) -> Dict[str, List[Dict]]:
        """构建邻接表：从实体到其所有关联边的映射"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT subject, predicate, object, confidence, episodic_id FROM triples "
            "WHERE is_controversial = 0"
        )
        rows = cursor.fetchall()
        conn.close()

        adjacency = defaultdict(list)
        for row in rows:
            subj, pred, obj, conf, ep_id = row
            # 正向边：subject -> object
            adjacency[subj].append({
                'predicate': pred,
                'target': obj,
                'confidence': conf if conf else 0.5,
                'episodic_id': ep_id,
                'direction': 'forward'
            })
            # 反向边：object -> subject
            adjacency[obj].append({
                'predicate': pred,
                'target': subj,
                'confidence': conf if conf else 0.5,
                'episodic_id': ep_id,
                'direction': 'backward'
            })
        return dict(adjacency)

    def _get_adjacency(self) -> Dict[str, List[Dict]]:
        """获取邻接表（带缓存）"""
        if not self._cache_valid or self._adjacency_cache is None:
            self._adjacency_cache = self._build_adjacency()
            self._cache_valid = True
        return self._adjacency_cache

    def invalidate_cache(self):
        """当图结构变化时，清除缓存"""
        self._cache_valid = False

    def activate(self, seed_entities: List[str],
                 seed_scores: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        从种子实体出发，执行扩散激活。

        Args:
            seed_entities: 种子实体列表
            seed_scores: 可选的种子初始分数 {entity: score}

        Returns:
            激活结果列表，每个元素包含：
            - entity: 被激活的实体
            - activation: 激活能量
            - hop: 跳数
            - episodic_ids: 关联的情景记忆ID集合
            - path: 从种子到该实体的路径
        """
        if not seed_entities:
            return []

        adjacency = self._get_adjacency()
        if not adjacency:
            return []

        # 激活能量表：entity -> activation_energy
        activation_map: Dict[str, float] = {}
        # 访问记录：防止重复扩散
        visited: Set[str] = set()
        # 关联的情景记忆ID：entity -> set(episodic_ids)
        entity_episodes: Dict[str, Set[int]] = defaultdict(set)
        # 路径记录
        entity_paths: Dict[str, List[str]] = {}

        # 初始化种子
        queue = deque()
        for ent in seed_entities:
            init_score = seed_scores.get(ent, 1.0) if seed_scores else 1.0
            activation_map[ent] = init_score
            entity_paths[ent] = [ent]
            queue.append((ent, 0, init_score))  # (entity, hop, energy)
            visited.add(ent)

        total_energy_used = 0.0

        while queue:
            current_entity, hop, current_energy = queue.popleft()

            if hop >= self.max_hops:
                continue
            if current_energy < self.activation_threshold:
                continue
            if total_energy_used > self.energy_budget:
                break

            neighbors = adjacency.get(current_entity, [])
            for edge in neighbors:
                target = edge['target']
                edge_conf = edge['confidence']
                ep_id = edge['episodic_id']

                # 能量传播：当前能量 * 衰减系数 * 边置信度
                propagated = current_energy * self.decay_factor * edge_conf

                if propagated < self.activation_threshold:
                    continue

                # 记录关联的情景记忆
                if ep_id is not None:
                    entity_episodes[target].add(ep_id)
                    entity_episodes[current_entity].add(ep_id)

                # 更新激活能量（取最大值）
                if target not in activation_map or propagated > activation_map[target]:
                    activation_map[target] = propagated
                    entity_paths[target] = entity_paths.get(current_entity, []) + [target]

                # 如果未访问过，或能量足够大，继续扩散
                if target not in visited:
                    visited.add(target)
                    queue.append((target, hop + 1, propagated))
                    total_energy_used += propagated

        # 构建结果
        results = []
        for entity, energy in activation_map.items():
            if entity in seed_entities:
                continue  # 跳过种子本身
            results.append({
                'entity': entity,
                'activation': energy,
                'hop': len(entity_paths.get(entity, [])) - 1,
                'episodic_ids': list(entity_episodes.get(entity, set())),
                'path': entity_paths.get(entity, [])
            })

        # 按激活能量降序排序
        results.sort(key=lambda x: x['activation'], reverse=True)
        return results

    def get_activated_turn_ids(self, seed_entities: List[str],
                               seed_scores: Optional[Dict[str, float]] = None,
                               max_results: int = 30) -> List[Dict[str, Any]]:
        """
        便捷方法：直接返回被激活的情景记忆ID及其元信息。
        用于集成到 recall 流水线中。
        """
        activations = self.activate(seed_entities, seed_scores)
        turn_scores: Dict[int, float] = {}

        for act in activations[:max_results]:
            for ep_id in act['episodic_ids']:
                if ep_id not in turn_scores or act['activation'] > turn_scores[ep_id]:
                    turn_scores[ep_id] = act['activation']

        results = []
        for turn_id, score in sorted(turn_scores.items(), key=lambda x: x[1], reverse=True):
            results.append({
                'turn_id': turn_id,
                'activation_score': score,
                'source': 'spreading_activation'
            })
        return results