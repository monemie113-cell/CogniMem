# CogniMem/src/cognimem/__init__.py
from .core import CogniMemPipeline
from .memory import PersistentMemory

def get_llm_client(provider, **kwargs):
    """延迟加载 LLM 客户端，避免不必要的依赖"""
    if provider == 'huggingface':
        from .llm.huggingface import HuggingFaceClient
        return HuggingFaceClient(**kwargs)
    elif provider == 'ollama':
        from .llm.ollama import OllamaClient
        return OllamaClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

__all__ = ['CogniMemPipeline', 'PersistentMemory', 'get_llm_client']