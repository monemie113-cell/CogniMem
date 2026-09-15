import re
from typing import List, Dict, Optional

# 定义常见虚词和停用词（中文），这些词不应作为实体
STOP_WORDS = {'请', '如何', '什么', '怎么', '为什么', '能否', '可以', '需要', '帮我', '一下', '一个', '这种', '那些',
              '关于', '对于', '根据', '按照', '通过', '从', '到', '在', '是', '的', '了', '吧', '吗', '呢'}

class CuriosityEngine:
    def __init__(self, memory, threshold: float = 0.7):
        self.memory = memory
        self.threshold = threshold

    def generate_questions(self, subject: str, predicate: str) -> List[str]:
        questions = []
        direct = self.memory.semantic.exact_match(subject, predicate)
        if not direct:
            questions.append(f"请问 {subject} 的 {predicate} 是什么？")
        elif direct[0]['confidence'] < self.threshold:
            questions.append(f"你确定 {subject} 的 {predicate} 是 {direct[0]['object']} 吗？")
        transitive = self.memory.semantic.transitive_reason(subject, predicate, max_depth=1)
        if not transitive:
            questions.append(f"你知道 {subject} 与 {predicate} 有间接关系吗？")
        return questions

    def generate_questions_from_text(self, text: str) -> List[str]:
        """
        从文本中提取有意义的实体，为每个实体生成好奇心问题。
        如果提取不到合适实体，返回空列表。
        """
        # 1. 提取连续中文（长度 ≥ 2），或英文专有名词（首字母大写）
        # 中文：至少2个连续汉字
        chinese_entities = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        # 英文专有名词：首字母大写
        english_entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        # 合并
        raw_entities = chinese_entities + english_entities

        # 2. 过滤停用词和虚词
        filtered = []
        for ent in raw_entities:
            # 去除前后空格
            ent = ent.strip()
            if not ent:
                continue
            # 检查是否在停用词列表中
            if ent in STOP_WORDS:
                continue
            # 检查是否以停用词开头（如“请总结”中的“请”）
            # 但如果整句是“请总结”，我们不想用，但“总结”可能有用，但我们只取完整实体
            # 我们可以进一步拆分，但为了简单，只保留长度≥2且不在停用词中的
            # 并且，如果 ent 中包含停用词，我们可以拆分，但这里简化
            # 只过滤整个词
            filtered.append(ent)

        # 3. 去重
        seen = set()
        unique_entities = []
        for ent in filtered:
            if ent not in seen:
                seen.add(ent)
                unique_entities.append(ent)

        # 4. 如果没有有效实体，返回空列表
        if not unique_entities:
            return []

        # 5. 为每个实体生成问题
        questions = []
        for ent in unique_entities:
            # 查询该实体在语义记忆中的关系数量
            rels = self.memory.semantic.exact_match(ent)
            if len(rels) < 2:
                questions.append(f"你能告诉我关于 “{ent}” 的更多信息吗？")
        return questions