from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMClient(ABC):
    """LLM 客户端抽象基类，定义统一接口"""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        根据提示和上下文生成回复。
        Args:
            prompt: 用户输入或系统提示
            context: 可选的上下文信息（如记忆检索结果、推理状态等）
        Returns:
            生成的回复文本
        """
        pass

    @abstractmethod
    def load_model(self):
        """加载模型（如果支持延迟加载）"""
        pass