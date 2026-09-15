# CogniMem/src/cognimem/utils/text_processor.py
import re
import numpy as np
import hashlib
from typing import List, Dict

# 全局缓存
_embedder = None
_nlp = None


def get_embedder():
    """加载 sentence-transformers 模型，带重试和镜像源"""
    global _embedder
    if _embedder is None:
        try:
            # 设置镜像源（国内加速）
            import os
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

            from sentence_transformers import SentenceTransformer
            # 增加超时时间，避免短暂网络问题
            _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Loaded sentence-transformers model.")
        except Exception as e:
            print(f"Failed to load sentence-transformers: {e}. Falling back to spacy.")
            _embedder = None
    return _embedder


def get_nlp():
    """加载 spacy 模型，带离线 fallback"""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Downloading spacy model...")
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            print("Loaded spacy model.")
        except Exception as e:
            print(f"Failed to load spacy: {e}. Using hash-based fallback.")
            _nlp = None
    return _nlp


def embed_text(text: str) -> np.ndarray:
    """
    将文本转为归一化向量。
    策略：sentence-transformers -> spacy -> 哈希向量
    """
    if not text:
        # 空文本返回零向量
        return np.zeros(384)

    # 1. 尝试 sentence-transformers
    embedder = get_embedder()
    if embedder is not None:
        try:
            vec = embedder.encode(text, convert_to_numpy=True)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            print(f"sentence-transformers encoding failed: {e}")

    # 2. 尝试 spacy 词向量
    nlp = get_nlp()
    if nlp is not None:
        try:
            doc = nlp(text)
            # 获取所有有词向量的 token 的平均值
            vectors = [token.vector for token in doc if token.has_vector]
            if vectors:
                vec = np.mean(vectors, axis=0)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec
            else:
                # 如果无向量（例如纯中文），使用字符级回退
                pass
        except Exception as e:
            print(f"spacy encoding failed: {e}")

    # 3. 最终 fallback：基于哈希的确定性伪随机向量
    # 保证相同文本生成相同向量
    hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    np.random.seed(hash_val % 2 ** 32)
    vec = np.random.randn(384)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    np.random.seed()  # 重置随机种子
    return vec


def extract_triples(text: str) -> List[Dict[str, str]]:
    """从文本中提取三元组（依赖 spacy）"""
    nlp = get_nlp()
    if nlp is None:
        return []
    doc = nlp(text[:512])
    triples = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subj = token.text
            pred = token.head.text
            obj = None
            for child in token.head.children:
                if child.dep_ in ("dobj", "attr", "prep"):
                    obj = child.text
                    if obj:
                        triples.append({
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "evidence": text[:100]
                        })
                    break
    if not triples:
        nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        if len(nouns) >= 2:
            triples.append({
                "subject": nouns[0],
                "predicate": "related_to",
                "object": nouns[1],
                "evidence": text[:100]
            })
    return triples