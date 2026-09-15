import time
import re
from typing import Dict, Any, Optional
from ..layers.input_gate import InputGate
from ..layers.engram import EngramLayer
from ..layers.hybrid_encoder import HybridEncoder
from ..layers.liquid_engine import LiquidEngine
from ..memory.persistent_memory import PersistentMemory
from ..memory.summarizer import MemorySummarizer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CogniMemPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.input_gate = InputGate(config.get('input_gate', {}))
        self.engram = EngramLayer(config.get('engram', {}))
        self.encoder = HybridEncoder(config.get('hybrid_encoder', {}))
        self.liquid = LiquidEngine(config.get('liquid_engine', {}))
        self.memory = PersistentMemory(config.get('memory', {}))
        self.stats = {'input_gate': {}, 'engram': {}, 'encoder': {}, 'liquid': {}}

        # 初始化 LLM（如果启用）
        self.llm_client = None
        llm_config = config.get('llm', {})
        print(f"[DEBUG] llm_config = {llm_config}")
        print(f"[DEBUG] enabled = {llm_config.get('enabled')}")

        if llm_config.get('enabled', False):
            provider = llm_config.get('provider', 'huggingface')
            if provider == 'huggingface':
                from ..llm import HuggingFaceClient
                self.llm_client = HuggingFaceClient(
                    model_name=llm_config.get('model_name', 'Qwen/Qwen2.5-0.5B'),
                    device=llm_config.get('device', 'auto'),
                    max_new_tokens=llm_config.get('max_new_tokens', 256),
                    temperature=llm_config.get('temperature', 0.7)
                )
            elif provider == 'ollama':
                from ..llm import OllamaClient
                self.llm_client = OllamaClient(
                    model_name=llm_config.get('model_name', 'qwen2.5:0.5b'),
                    base_url=llm_config.get('base_url', 'http://localhost:11434')
                )
                logger.info("LLM client initialized.")
            else:
                logger.warning(f"Unsupported LLM provider: {provider}")

        # 初始化记忆摘要器（如果启用）
        self.summarizer = None
        if llm_config.get('enabled', False) and config.get('memory', {}).get('summarizer', {}).get('enabled', True):
            self.summarizer = MemorySummarizer(
                memory=self.memory,
                llm_client=self.llm_client,
                max_tokens=config.get('memory', {}).get('summarizer', {}).get('max_summary_tokens', 200)
            )

    def run(self, user_input: str, context: Dict = None) -> Dict:
        # 聚合器优先路径
        try:
            from ..memory.aggregator import Aggregator
            if not hasattr(self, '_aggregator'):
                self._aggregator = Aggregator()
            # 检索一次（后续会复用）
            _early_retrieved = self.memory.recall(user_input, k=5)
            _early_texts = [r.get('content', '') for r in _early_retrieved if r.get('content')]
            _agg = self._aggregator.aggregate(user_input, _early_texts)
            if _agg and _agg.get('confidence', 0) >= 0.6:
                logger.info(f"Aggregator answer: {_agg['answer']} (type={_agg['type']})")
                return {
                    'response': _agg['answer'],
                    'reasoning': f"Aggregator type={_agg['type']}, evidence={_agg['evidence'][:5]}",
                    'raw_output': None,
                    'questions': [],
                    'meta': {'aggregator': _agg}
                }
        except Exception as e:
            logger.warning(f"Aggregator skipped: {e}")
        # 1. 输入门控
        filtered, gate_meta = self.input_gate.process(user_input, context)
        if not gate_meta['passed']:
            return {'output': None, 'reason': 'input_blocked', 'meta': gate_meta}
        self.stats['input_gate'] = gate_meta

        # 2. Engram查找
        engram_embed, engram_meta = self.engram.process(filtered, context)
        self.stats['engram'] = engram_meta

        # 3. 混合上下文编码
        encoded, encoder_meta = self.encoder.process(filtered, engram_embed)
        self.stats['encoder'] = encoder_meta

        # 4. 液态推理引擎
        output, liquid_meta = self.liquid.process(encoded, context)
        self.stats['liquid'] = liquid_meta

        # 5. 存储记忆（当 output 非空时）
        if output is not None:
            memory_item = {
                'input': user_input,
                'output': output,
                'timestamp': time.time(),
                'confidence': liquid_meta.get('confidence', 0.5),
                'type': 'interaction',
                'content': user_input,
                'metadata': {'output': output}
            }
            self.memory.store(memory_item)

        # 6. 生成最终回复（增强版：记忆摘要 + 推理路径 + 思维链 + 相关记忆检索）
        final_response = None
        reasoning = None
        if self.llm_client is not None and output is not None:
            try:
                # ----- 6.0 检索相关记忆（新增） -----
                retrieved_memories = self.memory.recall(user_input, k=5)
                memory_context = ""
                if retrieved_memories:
                    memory_lines = []
                    for item in retrieved_memories:
                        content = item.get('content', '')
                        if content:
                            memory_lines.append(f"- {content}")
                    memory_context = "\n".join(memory_lines)

                # ----- 6.1 获取记忆摘要 -----
                history_summary = ""
                if self.summarizer:
                    recent_messages = self.memory.episodic.get_recent(10)
                    messages = []
                    for item in recent_messages:
                        content = item.get('content', '')
                        messages.append({'role': 'user', 'content': content})
                    if messages:
                        history_summary = self.summarizer.summarize_conversation(messages)

                # ----- 6.2 获取推理路径（多跳推理） -----
                path_context = ""
                entities = re.findall(r'\b[A-Z][a-z]+\b', user_input)
                if not entities:
                    entities = re.findall(r'[\u4e00-\u9fa5]{2,}', user_input)
                if entities:
                    for ent in entities[:2]:
                        try:
                            paths = self.memory.semantic.multi_hop_reason(ent, max_depth=2)
                            if paths and isinstance(paths, list):
                                path_context += f"\n关于 '{ent}' 的推理路径：\n"
                                for p in paths[:2]:
                                    if isinstance(p, dict):
                                        entities_in_path = p.get('entities', [])
                                        if entities_in_path:
                                            path_str = " → ".join(str(e) for e in entities_in_path)
                                            path_context += f"  - {path_str}\n"
                        except Exception as e:
                            logger.warning(f"Path reasoning skipped for '{ent}': {e}")

                # ----- 6.3 构建增强 Prompt（含记忆上下文） -----
                enhanced_prompt = f"""【对话摘要】
{history_summary if history_summary else "（无历史摘要）"}

【相关记忆】
{memory_context if memory_context else "（无相关记忆）"}

【推理上下文】
{path_context if path_context else "（无推理路径）"}

【用户问题】
{user_input}

请按照以下步骤逐步思考，然后给出最终答案：

1. 理解问题核心
2. 分析可用信息（结合摘要、相关记忆和推理上下文）
3. 推理过程
4. 得出最终结论

请先输出你的思考过程（用 <thinking> 标签包围），然后输出最终答案（用 <answer> 标签包围）。"""

                # ----- 6.4 调用 LLM 并解析 CoT 响应 -----
                raw_response = self.llm_client.generate(
                    enhanced_prompt,
                    context={"memory": memory_context}
                )
                logger.info("LLM response generated with CoT.")

                thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL)

                if answer_match:
                    final_response = answer_match.group(1).strip()
                else:
                    final_response = raw_response.strip()

                if thinking_match:
                    reasoning = thinking_match.group(1).strip()

            except Exception as e:
                logger.error(f"Enhanced LLM generation failed: {e}")
                final_response = str(output) if output is not None else "No response generated."
                reasoning = ""

        # 如果 LLM 未启用或失败，回退到原始输出
        if final_response is None:
            final_response = str(output) if output is not None else "No response."
            reasoning = ""

        # 7. 好奇心驱动（可选）
        questions = []
        if self.llm_client is not None and output is not None:
            try:
                curiosity = self.memory.get_curiosity()
                questions = curiosity.generate_questions_from_text(user_input)
            except Exception as e:
                logger.warning(f"Curiosity generation failed: {e}")

        return {
            'response': final_response,
            'reasoning': reasoning,
            'raw_output': output,
            'questions': questions,
            'meta': {
                'gate': gate_meta,
                'engram': engram_meta,
                'encoder': encoder_meta,
                'liquid': liquid_meta
            }
        }

    def get_stats(self):
        return self.stats