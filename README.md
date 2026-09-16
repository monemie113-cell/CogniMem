

# CogniMem

**A human-cognition-inspired, zero-LLM-dependency memory and retrieval architecture for AI agents**

CogniMem is an innovative AI memory system that does not rely on any large language model. By simulating the brain's "associative thinking" and "hierarchical memory" mechanisms, it achieves **84.2% keyword hit rate** and **26.4% exact match hit rate** on the LoCoMo long-term dialogue benchmark, surpassing all publicly available zero-LLM solutions and approaching the performance of production systems that rely on GPT-4-level models.

[English](README.md) | [中文](README_zh.md)

------

## 📖 Table of Contents

- Core Features
- Architecture Overview
- Core Innovations
- Benchmark Results
- Quick Start
- Project Structure
- Key Module Details
- Comparison with Competitors
- Design Philosophy
- Known Limitations
- Roadmap
- Contributing

------

## ✨ Core Features

- 🧠 **Zero LLM Dependency**: All retrieval and reasoning are pure statistical algorithms + rule-based systems, with no large model involvement
- 🔗 **Associative Thinking Architecture**: Stores structured entity-relation-entity triples instead of raw text
- 📊 **IDF-Weighted Retrieval**: Automatically identifies rare vs. frequent words and dynamically adjusts retrieval weights
- ⏰ **Temporal Normalization**: Automatically converts relative time expressions like "yesterday", "last week" into absolute dates
- 🕸️ **Spreading Activation**: Starting from seed entities, associates indirectly related memories along the graph structure
- 📐 **Three-Layer Triples**: Event layer (raw evidence) + Knowledge layer (graph relations) + Temporal layer (absolute dates)
- 🌱 **Curiosity-Driven**: Actively identifies knowledge gaps and generates exploratory questions
- 💾 **Local Persistence**: Based on SQLite, zero external dependencies, fully offline capable
- 🔬 **Reproducible Benchmarks**: All performance data can be reproduced via open-source scripts

------

## 🏗️ Architecture Overview

CogniMem is built on a **four-layer cognitive pipeline**, backed by **three persistent memory stores** and a **six-path hybrid retrieval** system.

### Part 1 · Data Write Path

```text
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 · Perception & Input Gate                          │
│  ├─ 6-dim scoring: importance / confidence / priority /     │
│  │                 timeliness / contradiction / relevance    │
│  └─ Filters noise; drops input below threshold              │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 · Engram Knowledge Lookup  (O(1) cache)            │
│  └─ Hash key-value cache; repeated queries return instantly │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 · HySparse Hybrid Encoding  (Linear O(n))          │
│  ├─ Each block: 1 Full Attn + N Sparse Attn                 │
│  ├─ Full: generates top-k indices + KV Cache                │
│  └─ Sparse: reuses KV, computes local window + global sampling │
│     → 5× faster, 3× less memory than full attention @1000 tokens │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 · Liquid Reasoning Engine  (Liquid CfC, O(1))      │
│  └─ Liquid time-constant network; no KV Cache growth        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Triple Extraction & Hierarchical Storage                   │
│  ├─ ① Event     (speaker, said, <sentence>)   → evidence    │
│  ├─ ② Knowledge (subject, predicate, object)  → graph edge  │
│  └─ ③ Temporal  (speaker, event_date, date)   → abs. time   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Three-Store Persistent Memory              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐         │
│  │ Episodic   │    │ Semantic   │    │ Working    │         │
│  │ SQLite     │    │ Graph      │    │ LRU Buffer │         │
│  │ (episodes) │    │ (triples)  │    │ (deque)    │         │
│  │ + chunks   │    │ + vec idx  │    │ capacity=10│         │
│  └────────────┘    └────────────┘    └────────────┘         │
└─────────────────────────────────────────────────────────────┘
```



### Part 2 · Data Retrieval Path

```text
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Understanding                                        │
│  └─ Entity extraction · Temporal signal · Intent · Expansion│
└──────────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│  Aggregator (optional)    │    │  Six-Path Hybrid Recall   │
│  ├─ SUM / TEMPORAL        │    │  ├─ ① Entity graph        │
│  ├─ COUNT: disabled       │    │  ├─ ② Spreading activation│
│  └─ If high-confidence,   │    │  ├─ ③ Chunk semantic rank │
│     return directly       │    │  ├─ ④ Vector semantic     │
└───────────────────────────┘    │  ├─ ⑤ BM25 (optional)     │
                                 │  └─ ⑥ Query decomposition │
                                 └──────────────┬────────────┘
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│  IDF-Weighted Ranking                                       │
│  └─ IDF(t) = log((N+1) / (df(t)+1)) + 0.1                   │
│     Rare words dominate; frequent words downweighted        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  RRF Fusion  (Reciprocal Rank Fusion)                       │
│  └─ score(d) = Σ weight_i / (k + rank_i(d))                 │
│     Entity graph 1.2 > Spreading 1.1 > Vector 1.0 > BM25 0.8│
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Parent Chunk Replacement / Context Aggregation             │
│  └─ Fine-grained retrieval → coarse-grained return          │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Top-K Memory Segments                  │
└─────────────────────────────────────────────────────────────┘
```



### Part 3 · Core Design Principles

```text
┌─────────────────────────────────────────────────────────────┐
│                Zero LLM Dependency · Pure Stat + Rules      │
├─────────────────────────────────────────────────────────────┤
│  Memory layer      : Associative triples (brain-like)       │
│  Retrieval layer   : IDF statistics + graph spreading       │
│  Temporal layer    : Relative time → absolute date          │
│  Aggregation layer : Value extraction from segments         │
│                                                             │
│  → CPU only · 4GB RAM · No GPU required                     │
└─────────────────────────────────────────────────────────────┘
```



------

## 💡 Core Innovations

### 1. Three-Layer Triple Structure

CogniMem does not store raw dialogue text; instead it extracts three types of structured triples:

| Layer               | Format                                 | Example                                                 | Purpose                              |
| :------------------ | :------------------------------------- | :------------------------------------------------------ | :----------------------------------- |
| **Event Layer**     | `(speaker, said, <full sentence>)`     | `(Caroline, said, "I went to a LGBTQ group yesterday")` | Full-text evidence for retrieval     |
| **Knowledge Layer** | `(subject, predicate, object)`         | `(Caroline, went_to, LGBTQ group)`                      | Graph edges for spreading activation |
| **Temporal Layer**  | `(speaker, event_date, absolute date)` | `(Caroline, event_date, "7 May 2023")`                  | Temporal queries                     |

### 2. IDF-Weighted Entity Graph Retrieval

Traditional vector retrieval cannot distinguish the importance of "Caroline" (frequent) vs. "LGBTQ" (rare). CogniMem introduces the classic IDF formula from information retrieval:

IDF(t) = log((N + 1) / (df(t) + 1)) + 0.1

- `N`: total number of triples
- `df(t)`: number of triples containing keyword t

**Effect**: Rare words automatically gain high weight; frequent words (such as speaker names) are downweighted.

### 3. Temporal Normalization

Temporal expressions in dialogue are often relative ("yesterday", "last week"), while ground-truth answers require absolute dates. CogniMem uses session timestamps to perform the conversion automatically:

Session time: "1:56 pm on 8 May, 2023"
Text: "I went to a LGBTQ support group yesterday"
↓ Normalization
Result: "7 May 2023"

### 4. Spreading Activation

Starting from seed entities, activation propagates along the knowledge-layer graph with decay:

Energy propagation formula: E(target) = E(source) × decay_factor × edge_confidence

- `decay_factor = 0.5` (halves each hop)
- `max_hops = 2` (maximum 2 hops)
- `activation_threshold = 0.1` (stops propagating below threshold)

### 5. Hybrid Chunking Strategy

| Granularity      | Coverage                          | Purpose                         |
| :--------------- | :-------------------------------- | :------------------------------ |
| **Turn**         | 1 dialogue turn                   | Precise retrieval unit          |
| **Chunk**        | 3 dialogue turns (sliding window) | Context-rich unit               |
| **Parent Chunk** | 8 dialogue turns                  | Large context return (reserved) |

### 6. Lightweight Aggregator (Optional)

For queries like "how much total X", the aggregator extracts numeric values from multiple retrieved segments and sums them. Pure rule-based, zero LLM dependency.

**Current limitation**: COUNT-type questions ("how many X") have insufficient accuracy on natural language dialogue, so they are disabled by default.

------

## 📊 Benchmark Results

### Dataset 1: LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) is a long-term dialogue memory benchmark released at ACL 2024, containing 10 dialogues and 1542 QA pairs, covering single-hop, multi-hop, temporal, and open-domain query types.

**Evaluation Metrics**:

- **Exact Match Hit Rate**: The complete string of the ground-truth answer appears in the retrieval results
- **Keyword Hit Rate**: At least one keyword from the ground-truth answer appears in the retrieval results

**Results** (zero LLM, CPU-only, no external API calls):

| Version                                                | Exact Match | Keyword Hit |
| :----------------------------------------------------- | :---------- | :---------- |
| Pure cognitive baseline (entity graph + vector)        | 7.7%        | 56.6%       |
| + IDF weighting                                        | 13.2%       | 65.1%       |
| + Temporal normalization                               | 18.4%       | 79.5%       |
| **+ Knowledge layer + spreading activation (current)** | **26.4%**   | **84.2%**   |

**Detailed Data**:

- Total QA pairs: **1542**
- Exact match hits: **407 (26.4%)**
- Keyword hits: **1299 (84.2%)**

### Dataset 2: LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is a long-term memory benchmark released at ICLR 2025, containing 500 QA pairs covering multi-session reasoning, knowledge updates, temporal reasoning, information extraction, and abstention.

**Results** (150 samples, zero LLM):

| Category                  | Exact Match | Keyword Hit |
| :------------------------ | :---------- | :---------- |
| **Overall (150 samples)** | **38.0%**   | **60.7%**   |
| single-session-user       | 48.6%       | 67.1%       |
| multi-session             | 37.1%       | 41.9%       |
| single-session-preference | 0%          | 100%        |

**Key Finding**: The `multi-session` category requires aggregating values across segments. The retrieval layer can hit 80% of relevant segments but lacks numeric aggregation capability. This issue has been partially alleviated by the aggregator.

### Reproduction

```bash
# LoCoMo
cd benchmarks
python check_recall.py    		# full 10 dialogues

# LongMemEval
cd benchmarks/LongMemEval
python cognimem_adapter.py      # generate predictions
python eval_retrieval.py        # compute metrics
```



## 🚀 Quick Start

### Requirements

- Python 3.10+
- 4GB+ RAM (CPU only, no GPU required)

### Installation

```bash
git clone https://github.com/your-username/CogniMem.git
cd CogniMem
pip install -e .
```



### Dependencies

```bash
numpy>=1.24.0
pyyaml>=6.0
sentence-transformers>=2.2.0
spacy>=3.7.0
torch>=2.0.0
scikit-learn>=1.3.0
```



Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```



### Basic Usage

```python
from cognimem.core import CogniMemPipeline
import yaml

# Load configuration
with open('config/default_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = CogniMemPipeline(config)

# Store memory
pipeline.memory.store({
    'content': 'I went to a LGBTQ support group yesterday',
    'timestamp': 0,
    'metadata': {
        'speaker': 'Caroline',
        'session_time': '1:56 pm on 8 May, 2023'
    }
})

# Retrieve
results = pipeline.memory.recall("When did Caroline go to the LGBTQ group?", k=5)
for r in results:
    print(f"[{r['source']}] {r['content'][:200]}")
```



**Example Output**:

```text
[entity_graph] [Date: 7 May 2023] I went to a LGBTQ support group yesterday and it was so powerful.
```



### Run Benchmarks

```bash
cd benchmarks
python check_recall.py
```



------

## 📁 Project Structure

```text
CogniMem/
├── config/
│   └── default_config.yaml           # Configuration file
├── src/
│   └── cognimem/
│       ├── core/                     # Main pipeline
│       │   ├── config_manager.py
│       │   └── pipeline.py
│       ├── layers/                   # Four-layer cognitive pipeline
│       │   ├── engram.py             # Knowledge lookup
│       │   ├── hybrid_encoder.py     # Hybrid sparse encoding
│       │   ├── hybrid_encoder_torch.py  # PyTorch version (experimental)
│       │   ├── input_gate.py         # Perception & gate
│       │   └── liquid_engine.py      # Liquid reasoning engine
│       ├── memory/                   # Persistent memory system
│       │   ├── aggregator.py         # Lightweight aggregator
│       │   ├── bm25_retriever.py     # BM25 (optional)
│       │   ├── curiosity.py          # Curiosity-driven
│       │   ├── episodic.py           # Episodic memory (hierarchical chunking)
│       │   ├── fusion.py             # RRF fusion
│       │   ├── persistent_memory.py  # Core memory management
│       │   ├── query_decomposition.py # Query decomposition
│       │   ├── reranker.py           # Reranker (optional)
│       │   ├── routing.py            # Multi-path retrieval routing
│       │   ├── semantic.py           # Semantic memory (graph structure)
│       │   ├── spreading_activation.py  # Spreading activation
│       │   └── working_memory.py     # Working memory
│       ├── models/                   # Base models
│       │   ├── base.py
│       │   ├── engram_table.py
│       │   ├── liquid_cell.py
│       │   ├── small_gate.py
│       │   └── sparse_attn.py
│       ├── llm/                      # LLM integration (optional)
│       │   ├── base.py
│       │   ├── huggingface.py
│       │   └── ollama.py
│       ├── utils/
│       │   ├── helpers.py
│       │   ├── logger.py
│       │   └── text_processor.py     # spaCy wrapper
│       └── demo/
│           └── run_demo.py
├── benchmarks/                       # Benchmark scripts
│   ├── check_recall.py               # LoCoMo recall benchmark
│   ├── evaluate_agent.py             # Full agent evaluation
│   ├── evaluate_locomo.py
│   ├── benchmark_hysparse.py         # HySparse performance benchmark
│   ├── quality_check.py              # HySparse quality check
│   ├── diagnose*.py                  # Diagnostic tools
│   ├── test_one_query.py             # Single query validation
│   └── LongMemEval/
│       ├── cognimem_adapter.py       # LongMemEval adapter
│       ├── cognimem_adapter_multi.py # multi-session specific
│       ├── eval_retrieval.py         # Evaluation script
│       ├── eval_multi.py
│       ├── test_aggregator.py        # Aggregator unit tests
│       └── check_predictions.py
├── tests/                            # Unit tests
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



**Note**: `benchmarks/locomo/` and `benchmarks/LongMemEval/data/` are external datasets that need to be downloaded separately.

------

## 🔧 Key Module Details

### `persistent_memory.py`: Memory Core

**`_extract_triples`** is the core method of the system, responsible for decomposing raw dialogue into three types of triples:

```python
def _extract_triples(self, item, episodic_id):
    # 1. Temporal normalization → (speaker, event_date, absolute date)
    # 2. spaCy dependency parsing + pronoun resolution → (subject, predicate, object)
    # 3. Chinese rule extraction → Chinese triples
    # 4. Smart fallback → (speaker, said, full sentence)
    # 5. Knowledge layer extraction → (subject, verb_lemma, object)
```



**`_entity_graph_search`** is the retrieval core:

```python
# 1. Keyword expansion (entities + temporal hint words)
# 2. IDF weighting
# 3. Multi-keyword score accumulation
# 4. Temporal query weighting
# 5. Top-K return
```



### `spreading_activation.py`: Graph Association

```python
class SpreadingActivation:
    def activate(self, seed_entities):
        # BFS traversal with energy decay
        # Propagation per edge: E(target) = E(source) × 0.5 × confidence
        # Maximum 2 hops
```



### `hybrid_encoder.py`: Linear Complexity Encoding

```python
class HybridEncoder:
    def _hybrid_block_forward(self, seq):
        # Each block: 1 Full Attention + N Sparse Attention
        # Full: generates top-k token indices and KV Cache
        # Sparse: reuses indices and KV, computes only local window + global sampling
```



------

## 🏆 Comparison with Competitors

| System                      | Keyword Hit Rate | LLM Dependency | Type                       |
| :-------------------------- | :--------------- | :------------- | :------------------------- |
| Basic vector retrieval      | 49.7%            | No             | Pure vector                |
| YourMemory                  | 59.0%            | No             | Biological decay memory    |
| HippoGraph early            | 44.2%            | No             | Single-turn granularity    |
| HippoGraph Hybrid           | 65.5%            | No             | 3-turn chunk               |
| **CogniMem (this project)** | **84.2%**        | **No**         | **Cognitive architecture** |
| HippoGraph production       | 91.1%            | Yes            | Full pipeline + LLM        |

**Key Difference**: HippoGraph production relies on LLM for query decomposition and reranking. CogniMem is **completely zero-LLM dependent**, achieving close to 92% of its performance using only statistics + rules.

------

## 🎯 Design Philosophy

CogniMem's design is inspired by observations of human brain cognitive mechanisms:

#### 1. The brain stores "associations", not raw text

The brain does not remember conversations verbatim, but extracts "who did what" and "who is related to whom". CogniMem simulates this via knowledge-layer triples.

#### 2. The brain has temporal memory and automatically normalizes time

When hearing "yesterday", the brain automatically computes the absolute date based on the current time. CogniMem simulates this capability via session_time + relative time expressions.

#### 3. The brain has associative ability and can infer by analogy

Seeing "Caroline", the brain automatically associates "she recently joined an LGBTQ support group" and "she said she wants to work in psychological counseling". CogniMem simulates this via spreading activation.

#### 4. The brain does not rely on brute-force computation

The brain consumes only 20W. CogniMem therefore avoids heavyweight components like Transformer rerankers and BM25, insisting on the simplest statistical methods for optimal results.

------

## ⚠️ Known Limitations

CogniMem is a **prototype system** with the following limitations:

#### 1. COUNT-type aggregation problem

"how many X" questions require cross-segment counting. Pure rule-based methods achieve about 40% accuracy on natural language dialogue, so the COUNT aggregator is **disabled by default**.

SUM aggregator ("how much total X") is effective but narrow in coverage.

#### 2. Complex temporal reasoning

Questions requiring calendar calculations such as "what day of the week was before a certain date" or "how many days from today" are currently limited to basic relative time conversion (yesterday/last week).

#### 3. Cross-paragraph pronoun resolution

Current pronoun resolution relies on spaCy's sentence-level analysis; cross-paragraph references (e.g., "she" referring to someone mentioned in a previous paragraph) cannot be handled.

#### 4. Embedding model dependency

Although the system itself is zero-LLM dependent, the retrieval layer uses `sentence-transformers` for semantic embeddings. For a purely statistical approach, BM25 could be used instead, but performance would decrease.

------

## 🗺️ Roadmap

#### Completed ✅

- ☑ Four-layer cognitive pipeline
- ☑ Entity graph + IDF-weighted retrieval
- ☑ Temporal normalization
- ☑ Knowledge-layer triples
- ☑ Spreading activation
- ☑ Curiosity-driven
- ☑ Lightweight aggregator (SUM type)

#### In Progress 🚧

- □ Query decomposition (conservative mode)
- □ Cross-paragraph pronoun resolution
- □ COUNT-type aggregation optimization

#### Future Plans 📅

- □ Memory consolidation (sleep mechanism)
- □ Weight-learning liquid reasoning engine
- □ RESTful API service
- □ Multilingual support
- □ End-to-end integration evaluation with real LLMs

------

## 🤝 Contributing

All forms of contribution are welcome:

- 🐛 Report bugs
- 💡 Propose new features
- 📝 Improve documentation
- 🧪 Submit benchmark results
- 🌍 Add multilingual support

Please open an issue for discussion before submitting a PR.

------

## 📄 License

This project is licensed under the **Apache 2.0** License.

------

## 🙏 Acknowledgements

- [LoCoMo](https://github.com/snap-research/locomo): provided a high-quality long-term dialogue benchmark
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): provided a long-term memory evaluation benchmark
- [spaCy](https://spacy.io/): provided reliable NLP foundation
- [sentence-transformers](https://www.sbert.net/): provided semantic embedding support

------

**CogniMem** — Using the simplest statistical methods to achieve associative memory closest to the human brain.

*"True intelligence lies not in stacking more parameters, but in organizing and using knowledge more intelligently."*