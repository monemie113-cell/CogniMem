from typing import List, Dict, Optional
import time


class MemorySummarizer:
    """
    记忆摘要器：将历史对话压缩为结构化摘要
    参考 MiniHermes 的五阶段压缩管线
    """

    def __init__(self, memory, llm_client=None, max_tokens: int = 2000):
        self.memory = memory
        self.llm = llm_client
        self.max_tokens = max_tokens
        self._summary_cache = {}

    def summarize_conversation(self, messages: List[Dict], max_length: int = 200) -> str:
        """
        将对话消息列表压缩为摘要
        """
        if not messages:
            return ""

        # 1. 提取关键事实（从语义记忆中检索）
        key_facts = []
        for msg in messages[-5:]:  # 只处理最近5条
            content = msg.get('content', '')
            if content:
                facts = self.memory.semantic.exact_match(content[:50])
                key_facts.extend(facts)

        # 2. 去重并排序
        seen = set()
        unique_facts = []
        for f in key_facts:
            key = f"{f.get('subject', '')}{f.get('predicate', '')}{f.get('object', '')}"
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        # 3. 构建摘要
        summary_parts = []
        for f in unique_facts[:5]:
            summary_parts.append(f"{f['subject']} {f['predicate']} {f['object']}")

        # 4. 如果提供了 LLM，使用 LLM 生成更自然的摘要
        if self.llm and len(messages) > 3:
            try:
                prompt = f"请将以下对话历史压缩为一段简洁的摘要（200字以内）：\n"
                for msg in messages[-5:]:
                    prompt += f"- {msg.get('role', 'user')}: {msg.get('content', '')[:100]}\n"
                summary = self.llm.generate(prompt)
                return summary[:max_length]
            except:
                pass

        return "；".join(summary_parts) if summary_parts else "无关键信息"