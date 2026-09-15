# CogniMem

**一个受人类认知启发的、零 LLM 依赖的智能体记忆与检索架构**

CogniMem 是一个创新的 AI 记忆系统，它不依赖任何大语言模型，通过模拟人脑的"关联思维"与"分层记忆"机制，在 LoCoMo 长期对话基准上实现了 **84.2% 的关键词命中率**和 **26.4% 的精确包含命中率**，性能超越目前所有公开的零 LLM 方案，逼近依赖 GPT-4 级别模型的生产系统。

---

## 📖 目录

- [核心特性](#-核心特性)
- [架构总览](#-架构总览)
- [核心创新](#-核心创新)
- [基准测试结果](#-基准测试结果)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [关键模块详解](#-关键模块详解)
- [与竞品对比](#-与竞品对比)
- [设计理念](#-设计理念)
- [已知局限](#-已知局限)
- [未来路线图](#-未来路线图)
- [贡献指南](#-贡献指南)

---

## ✨ 核心特性

- 🧠 **零 LLM 依赖**：所有检索与推理均为纯统计算法 + 规则系统，无任何大模型参与
- 🔗 **关联思维架构**：存储结构化的实体-关系-实体三元组，而非原始文本
- 📊 **IDF 加权检索**：自动识别稀有词与高频词，动态调整检索权重
- ⏰ **时间归一化**：自动将"yesterday"、"last week" 等相对时间转换为绝对日期
- 🕸️ **扩散激活**：从种子实体出发，沿图结构联想间接相关记忆
- 📐 **三层三元组**：事件层（原文证据）+ 知识层（图关系）+ 时间层（绝对日期）
- 🌱 **好奇心驱动**：主动识别知识空白并生成探索性问题
- 💾 **本地持久化**：基于 SQLite，零外部依赖，可完全离线运行
- 🔬 **可复现基准**：所有性能数据均可通过开源脚本复现

------

## 🏗️ 架构总览

CogniMem 采用**四层认知流水线 + 三库持久化记忆 + 六路混合检索**的分层架构。

#### 一、数据写入路径

```text
                        ┌──────────────────────┐
                        │     用户输入          │
                        └──────────┬───────────┘
                                   │
                                   ▼
        ╔══════════════════════════════════════════════════════╗
        ║         第一层 · 感知与输入门控 (Input Gate)          	 ║
        ╠══════════════════════════════════════════════════════╣
        ║  6 维评分：重要性 / 置信度 / 优先级 / 时效性         		 ║
        ║           / 矛盾度 / 相关性                        	   ║
        ║  → 过滤噪音，低于阈值的信息直接丢弃                   	 ║
        ╚══════════════════════════┬═══════════════════════════╝
                                   │
                                   ▼
        ╔══════════════════════════════════════════════════════╗
        ║         第二层 · Engram 知识查找 (O(1) 缓存)          	  ║
        ╠══════════════════════════════════════════════════════╣
        ║  				哈希键值缓存，重复查询秒级返回               ║
        ╚══════════════════════════┬═══════════════════════════╝
                                   │
                                   ▼
        ╔══════════════════════════════════════════════════════╗
        ║      第三层 · HySparse 混合稀疏编码 (Linear O(n))        ║
        ╠══════════════════════════════════════════════════════╣
        ║  每个 Block = 1 层 Full Attn + N 层 Sparse Attn        ║
        ║  Full：生成 top-k 索引 + KV Cache                      ║
        ║  Sparse：复用 KV，仅算局部窗口 + 全局采样                 ║
        ║  → 1000 token 下比全注意力快 5×、省内存 3×               ║
        ╚══════════════════════════┬═══════════════════════════╝
                                   │
                                   ▼
        ╔══════════════════════════════════════════════════════╗
        ║       第四层 · 液态推理引擎 (Liquid CfC, O(1))        	  ║
        ╠══════════════════════════════════════════════════════╣
        ║  液态时间常数网络：动态状态演化，无 KV Cache 增长     		║
        ╚══════════════════════════┬═══════════════════════════╝
                                   │
                                   ▼
        ╔══════════════════════════════════════════════════════╗
        ║              三元组提取与分层存储                     	 ║
        ╠══════════════════════════════════════════════════════╣
        ║  ① 事件层：(speaker, said, <整句>)      → 全文证据   	 ║
        ║  ② 知识层：(subject, predicate, object) → 图关系边   	  ║
        ║  ③ 时间层：(speaker, event_date, 日期)  → 绝对时间   	 ║
        ╚══════════════════════════┬═══════════════════════════╝
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │                  三库持久化记忆                         │
        │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
        │  │  情景记忆      │ │  语义记忆     │ │  工作记忆      │  │
        │  │  SQLite      │ │  图结构       │ │  LRU 缓冲     │  │
        │  │  (episodes)  │ │  (triples)   │ │  (deque)     │  │
        │  │  + chunks    │ │  + 向量索引   │ │  capacity=10  │ │
        │  └──────────────┘ └──────────────┘ └──────────────┘  │
        └──────────────────────────────────────────────────────┘
```



#### 二、数据检索路径

```text
                        ┌──────────────────────┐
                        │     用户查询          │
                        └──────────┬───────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │               查询理解 (Query Understanding)          │
        │   实体提取 · 时间信号识别 · 意图分类 · 查询扩展             │
        └──────────────────────────┬───────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
        ┌───────────────────────┐     ┌───────────────────────┐
        │  聚合器 (Aggregator)   │     │   六路混合检索召回       │
        │  SUM / TEMPORAL       │     │                       │
        │  ↓ 若高置信度直接返回    │     │  ① 实体图检索           │
        │                       │     │  ② 扩散激活            │
        │  COUNT 默认关闭         │     │  ③ Chunk 语义排序      │
        └───────────────────────┘     │  ④ 向量语义检索         │
                                      │  ⑤ BM25（可选）        │
                                      │  ⑥ 查询分解（可选）      │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │              IDF 加权排序 (Rare-word Priority)         │
        │    IDF(t) = log((N+1) / (df(t)+1)) + 0.1             │
        │    稀有词主导排序，高频词降权                             │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │           RRF 融合 (Reciprocal Rank Fusion)           │
        │    score(d) = Σ weight_i / (k + rank_i(d))           │
        │    实体图 1.2 > 扩散激活 1.1 > 向量 1.0 > BM25 0.8       │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │           父块替换 / 上下文聚合 (Parent Chunk)           │
        │    小粒度检索 → 大粒度返回 → 给 LLM 完整语境               │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Top-K 记忆片段      │
                        └──────────────────────┘
```



#### 三、核心设计原则

```text
        ┌──────────────────────────────────────────────────────┐
        │              零 LLM 依赖 · 纯统计 + 规则                │
        ├──────────────────────────────────────────────────────┤
        │                                                      │
        │   记忆层：关联三元组（人脑式结构化存储）                    │
        │   检索层：IDF 统计 + 图扩散（人脑式联想）                  │
        │   时间层：相对时间 → 绝对日期（人脑式时序归一化）            │
        │   聚合层：从片段提取数值（人脑式归纳）                      │
        │                                                      │
        │   → CPU 即可运行，4GB 内存，无 GPU 需求                  │
        │                                                      │
        └──────────────────────────────────────────────────────┘
```

---

## 💡 核心创新

### 1. 三层三元组结构

CogniMem 不存储原始对话文本，而是提取三种类型的结构化三元组：

| 层级       | 格式                              | 示例                                                    | 用途                |
| ---------- | --------------------------------- | ------------------------------------------------------- | ------------------- |
| **事件层** | `(speaker, said, <整句>)`         | `(Caroline, said, "I went to a LGBTQ group yesterday")` | 全文检索的 evidence |
| **知识层** | `(subject, predicate, object)`    | `(Caroline, went_to, LGBTQ group)`                      | 扩散激活的图边      |
| **时间层** | `(speaker, event_date, 绝对日期)` | `(Caroline, event_date, "7 May 2023")`                  | 时间类查询          |

### 2. IDF 加权实体图检索

传统向量检索无法区分"Caroline"（高频）和"LGBTQ"（稀有）的重要性。CogniMem 引入信息检索领域的经典 IDF 公式：IDF(t) = log((N + 1) / (df(t) + 1)) + 0.1

- `N`：三元组总数
- `df(t)`：包含关键词 t 的三元组数量

**效果**：稀有词自动获得高权重，高频词（如说话人名字）被降权。

### 3. 时间归一化

对话中的时间表达往往是相对的（"yesterday"、"last week"），而标准答案需要绝对日期。CogniMem 利用会话时间戳自动完成转换：

Session time: "1:56 pm on 8 May, 2023"
Text: "I went to a LGBTQ support group yesterday"
↓ 归一化
Result: "7 May 2023"

### 4. 扩散激活

从种子实体出发，沿知识层图结构进行带衰减的激活传播：

能量传播公式：E(target) = E(source) × decay_factor × edge_confidence

- `decay_factor = 0.5`（每跳衰减一半）
- `max_hops = 2`（最多传播 2 跳）
- `activation_threshold = 0.1`（低于阈值停止传播）

### 5. 混合分块策略

| 粒度             | 覆盖范围             | 用途                 |
| ---------------- | -------------------- | -------------------- |
| **Turn**         | 1 轮对话             | 精确检索单元         |
| **Chunk**        | 3 轮对话（滑动窗口） | 上下文丰富单元       |
| **Parent Chunk** | 8 轮对话             | 大上下文返回（预留） |

### 6. 轻量级聚合器（可选）

对"how much total X"类查询，聚合器从多个检索片段中提取数值并求和。纯规则实现，零 LLM 依赖。

**当前限制**：COUNT 类型（"how many X"）在自然语言对话上准确率不足，已默认关闭。

---

## 📊 基准测试结果

### 数据集 1：LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) 是 ACL 2024 发布的长期对话记忆基准，包含 10 个对话、1542 个 QA 对，覆盖单跳、多跳、时序、开放域等多种查询类型。

**评估指标**：

- **精确包含命中率**：标准答案的完整字符串出现在检索结果中
- **关键词命中率**：标准答案的至少一个关键词出现在检索结果中

**结果**（零 LLM、CPU 运行、无外部 API 调用）：

| 版本                            | 精确包含  | 关键词命中 |
| ------------------------------- | --------- | ---------- |
| 纯认知基线（实体图 + 向量）     | 7.7%      | 56.6%      |
| + IDF 加权                      | 13.2%     | 65.1%      |
| + 时间归一化                    | 18.4%     | 79.5%      |
| **+ 知识层 + 扩散激活（当前）** | **26.4%** | **84.2%**  |

**详细数据**：

- 总 QA 数：**1542**
- 精确包含命中：**407（26.4%）**
- 关键词命中：**1299（84.2%）**

### 数据集 2：LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 是 ICLR 2025 发布的长期记忆基准，包含 500 个 QA，覆盖多会话推理、知识更新、时序推理、信息提取、弃权五类能力。

**结果**（150 条样本，零 LLM）：

| 类别                      | 精确包含  | 关键词命中 |
| ------------------------- | --------- | ---------- |
| **整体（150 条）**        | **38.0%** | **60.7%**  |
| single-session-user       | 48.6%     | 67.1%      |
| multi-session             | 37.1%     | 41.9%      |
| single-session-preference | 0%        | 100%       |

**关键发现**：`multi-session` 类别需要跨片段聚合数值，检索层能命中 80% 的相关片段，但缺少数值聚合能力。这一问题已通过聚合器部分缓解。

### 复现方式

```bash
# LoCoMo
cd benchmarks
python check_recall.py    		# 完整 10 个对话

# LongMemEval
cd benchmarks/LongMemEval
python cognimem_adapter.py      # 生成预测
python eval_retrieval.py        # 计算指标
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 4GB+ 内存（CPU 即可运行，无需 GPU）

### 安装

```bash
git clone https://github.com/your-username/CogniMem.git
cd CogniMem
pip install -e .
```



### 依赖

```text
numpy>=1.24.0
pyyaml>=6.0
sentence-transformers>=2.2.0
spacy>=3.7.0
torch>=2.0.0
scikit-learn>=1.3.0
```



下载 spaCy 英文模型：

```bash
python -m spacy download en_core_web_sm
```



### 基础用法

```python
from cognimem.core import CogniMemPipeline
import yaml

# 加载配置
with open('config/default_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初始化流水线
pipeline = CogniMemPipeline(config)

# 存储记忆
pipeline.memory.store({
    'content': 'I went to a LGBTQ support group yesterday',
    'timestamp': 0,
    'metadata': {
        'speaker': 'Caroline',
        'session_time': '1:56 pm on 8 May, 2023'
    }
})

# 检索
results = pipeline.memory.recall("When did Caroline go to the LGBTQ group?", k=5)
for r in results:
    print(f"[{r['source']}] {r['content'][:200]}")
```



**输出示例**：

```text
[entity_graph] [Date: 7 May 2023] I went to a LGBTQ support group yesterday and it was so powerful.
```



### 运行基准测试

```bash
cd benchmarks
python check_recall.py
```



------

## 📁 项目结构

```text
CogniMem/
├── config/
│   └── default_config.yaml           # 配置文件
├── src/
│   └── cognimem/
│       ├── core/                     # 主流水线
│       │   ├── config_manager.py
│       │   └── pipeline.py
│       ├── layers/                   # 四层认知流水线
│       │   ├── engram.py             # 知识查找
│       │   ├── hybrid_encoder.py     # 混合稀疏编码
│       │   ├── hybrid_encoder_torch.py  # PyTorch 版本（实验）
│       │   ├── input_gate.py         # 感知与门控
│       │   └── liquid_engine.py      # 液态推理引擎
│       ├── memory/                   # 持久化记忆系统
│       │   ├── aggregator.py         # 轻量级聚合器
│       │   ├── bm25_retriever.py     # BM25（可选）
│       │   ├── curiosity.py          # 好奇心驱动
│       │   ├── episodic.py           # 情景记忆（分层分块）
│       │   ├── fusion.py             # RRF 融合
│       │   ├── persistent_memory.py  # 核心记忆管理
│       │   ├── query_decomposition.py # 查询分解
│       │   ├── reranker.py           # 重排序（可选）
│       │   ├── routing.py            # 多路检索路由
│       │   ├── semantic.py           # 语义记忆（图结构）
│       │   ├── spreading_activation.py  # 扩散激活
│       │   └── working_memory.py     # 工作记忆
│       ├── models/                   # 底层模型
│       │   ├── base.py
│       │   ├── engram_table.py
│       │   ├── liquid_cell.py
│       │   ├── small_gate.py
│       │   └── sparse_attn.py
│       ├── llm/                      # LLM 集成（可选）
│       │   ├── base.py
│       │   ├── huggingface.py
│       │   └── ollama.py
│       ├── utils/
│       │   ├── helpers.py
│       │   ├── logger.py
│       │   └── text_processor.py     # spaCy 封装
│       └── demo/
│           └── run_demo.py
├── benchmarks/                       # 基准测试脚本
│   ├── check_recall.py               # LoCoMo 召回率基准
│   ├── evaluate_agent.py             # 完整 Agent 评估
│   ├── evaluate_locomo.py
│   ├── benchmark_hysparse.py         # HySparse 性能基准
│   ├── quality_check.py              # HySparse 质量验证
│   ├── diagnose*.py                  # 诊断工具
│   ├── test_one_query.py             # 单查询验证
│   └── LongMemEval/
│       ├── cognimem_adapter.py       # LongMemEval 适配器
│       ├── cognimem_adapter_multi.py # multi-session 专用
│       ├── eval_retrieval.py         # 评估脚本
│       ├── eval_multi.py
│       ├── test_aggregator.py        # 聚合器单元测试
│       └── check_predictions.py
├── tests/                            # 单元测试
│   ├── test_memory.py
│   ├── test_pipeline.py
│   ├── test_curiosity.py
│   ├── test_engram.py
│   ├── test_gate.py
│   ├── test_hysparse.py
│   ├── test_liquid.py
│   ├── test_llm.py
│   ├── test_ollama.py
│   ├── test_reason.py
│   └── test_torch.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```



**注**：`benchmarks/locomo/` 和 `benchmarks/LongMemEval/data/` 是外部数据集，需自行下载。

------

## 🔧 关键模块详解

### `persistent_memory.py`：记忆核心

**`_extract_triples`** 是系统的核心方法，负责将原始对话拆解为三种三元组：

```python
def _extract_triples(self, item, episodic_id):
    # 1. 时间归一化 → (speaker, event_date, 绝对日期)
    # 2. spaCy 依存分析 + 代词消解 → (subject, predicate, object)
    # 3. 中文规则抽取 → 中文三元组
    # 4. 智能回退 → (speaker, said, 整句)
    # 5. 知识层提取 → (subject, verb_lemma, object)
```



**`_entity_graph_search`** 是检索核心：

```python
# 1. 关键词扩展（实体 + 时间暗示词）
# 2. IDF 加权计算
# 3. 多关键词分数累加
# 4. 时间类查询加权
# 5. Top-K 返回
```



### `spreading_activation.py`：图联想

```python
class SpreadingActivation:
    def activate(self, seed_entities):
        # BFS 遍历，带能量衰减
        # 每条边传播：E(target) = E(source) × 0.5 × confidence
        # 最多 2 跳
```



### `hybrid_encoder.py`：线性复杂度编码

```python
class HybridEncoder:
    def _hybrid_block_forward(self, seq):
        # 每个 block：1 层 Full Attention + N 层 Sparse
        # Full：生成 top-k token 索引和 KV Cache
        # Sparse：复用索引和 KV，仅计算局部窗口 + 全局采样
```



------

## 🏆 产品对比

| 系统                   | 关键词命中率 | LLM 依赖 | 类型             |
| :--------------------- | :----------- | :------- | :--------------- |
| 基础向量检索           | 49.7%        | 无       | 纯向量           |
| YourMemory             | 59.0%        | 无       | 生物衰减记忆     |
| HippoGraph 早期        | 44.2%        | 无       | 单轮粒度         |
| HippoGraph Hybrid      | 65.5%        | 无       | 3-turn chunk     |
| **CogniMem（本项目）** | **84.2%**    | **无**   | **认知架构**     |
| HippoGraph 生产版      | 91.1%        | 有       | 完整流水线 + LLM |

**关键差异**：HippoGraph 生产版依赖 LLM 做查询分解和重排序。CogniMem **完全零 LLM 依赖**，仅靠统计 + 规则实现了接近其 92% 的性能。

------

## 🎯 设计理念

CogniMem 的设计灵感来源于对人脑认知机制的观察：

#### 1. 人脑存储"关联"，而非原始文本

人脑不会逐字记住对话，而是提取"谁做了什么"、"谁和谁有什么关系"。CogniMem 通过知识层三元组模拟这一机制。

#### 2. 人脑有时序记忆，自动归一化时间

人脑听到"昨天"时，会自动结合当前时间计算出绝对日期。CogniMem 通过 session_time + 相对时间表达模拟这一能力。

#### 3. 人脑有联想能力，能触类旁通

看到"Caroline"，人脑会自动联想"她最近参加了 LGBTQ 支持团体"、"她说想从事心理咨询"。CogniMem 通过扩散激活模拟这一机制。

#### 4. 人脑不依赖暴力算力

人脑功耗仅 20W。CogniMem 因此避免引入 Transformer 重排序、BM25 等重量级组件，坚持用最朴素的统计方法实现最优效果。

------

## ⚠️ 已知局限

CogniMem 是一个**原型系统**，具有以下局限：

#### 1. COUNT 类聚合问题

"how many X" 类型问题需要跨片段计数，纯规则方法在自然语言对话上的准确率约 40%，因此 COUNT 聚合器**默认关闭**。

SUM 类聚合器（"how much total X"）有效但覆盖面窄。

#### 2. 复杂时间推理

"某日期前的星期几"、"距今天多少天" 等需要日历计算的问题，当前只支持基础相对时间转换（yesterday/last week）。

#### 3. 跨段落代词消解

当前代词消解依赖 spaCy 的句子级分析，跨段落指代（"她" 指代前一段提到的某个人）无法处理。

#### 4. 嵌入模型依赖

虽然系统本身零 LLM 依赖，但检索层使用 `sentence-transformers` 做语义嵌入。若追求纯统计方案，可以替换为 BM25，但性能会下降。

------

## 🗺️ 未来路线图

#### 已完成 ✅

- ☑ 四层认知流水线
- ☑ 实体图 + IDF 加权检索
- ☑ 时间归一化
- ☑ 知识层三元组
- ☑ 扩散激活
- ☑ 好奇心驱动
- ☑ 轻量级聚合器（SUM 类型）

#### 进行中 🚧

- □ 查询分解（保守模式）
- □ 跨段落代词消解
- □ COUNT 类聚合优化

#### 远期规划 📅

- □ 记忆巩固（睡眠机制）
- □ 权重学习的液态推理引擎
- □ RESTful API 服务化
- □ 多语言支持
- □ 与真实 LLM 的端到端集成评估

------

## 🤝 贡献指南

欢迎所有形式的贡献：

- 🐛 报告 bug
- 💡 提出新功能
- 📝 改进文档
- 🧪 提交基准测试结果
- 🌍 添加多语言支持

请先开 issue 讨论，再提交 PR。

------

## 📄 许可证

本项目采用 **Apache 2.0** 许可证。

------

## 🙏 致谢

- [LoCoMo](https://github.com/snap-research/locomo)：提供了高质量的长期对话基准
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)：提供了长期记忆评估基准
- [spaCy](https://spacy.io/)：提供了可靠的 NLP 基础能力
- [sentence-transformers](https://www.sbert.net/)：提供了语义嵌入支持

------

**CogniMem** —— 用最朴素的统计方法，实现最接近人脑的关联记忆。

*"真正的智能，不在于堆砌更多参数，而在于更聪明地组织和使用知识。"*