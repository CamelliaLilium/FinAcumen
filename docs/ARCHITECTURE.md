# FinAcumen Architecture

FinAcumen is a multi-modal financial QA system with a structured experience memory layer. It consists of two subsystems: **FT** (Financial Tools Runtime — agent loop and tool execution) and **FM** (Financial Memory — experience retrieval and collection).

## 1. System Overview

```
┌────────────────── Benchmark Driver ──────────────────┐
│  finacumen-benchmark --variant X --dataset Y          │
│  for target in dataset: result = await variant.solve() │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────── Variant Strategy Layer ───────────────────┐
│  MemoryAgentVariant (wrapper):                        │
│    Stage A: retrieve_experience(target)              │
│    Stage B: delegate to base variant.solve()         │
│    Stage C: collect_experience(trace) async write     │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────── Agent Execution Layer ────────────────────┐
│  ToolCallAgent ReAct loop: think() → act()           │
│  Tools: PythonExecute, FinancialDataLookup,           │
│         OcrExtract, Terminate, DeepReasoningTool       │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────── Memory Base (persistent) ─────────────────┐
│  MEMORY.md  ← manifest (≤200 lines)                   │
│  pat_*.md   ← YAML frontmatter + experience text      │
│  pat_*.emb  ← pre-computed embedding (float32)         │
│  archive/   ← pruned entries                          │
│  collect_errors.jsonl ← dead-letter queue             │
└──────────────────────────────────────────────────────┘
```

The two subsystems are layered: FT provides agent reasoning with tool use; FM adds a persistent side-channel that accumulates reusable experience across problems and retrofits it into future agent runs.

## 2. FT — Financial Tools Runtime

### 2.1 Agent Hierarchy

```
BaseAgent
  └─ ReActAgent         ← think() → act() loop
       └─ ToolCallAgent  ← LLM function-calling dispatch
            └─ FinanceAgent   ← financial reasoning expert
```

- **BaseAgent**: manages in-session message history, LLM client, and a `run()` loop with configurable `max_steps`.
- **ReActAgent**: defines the `think() → act()` cycle. `think()` calls the LLM with tool specifications; `act()` executes the selected tool calls.
- **ToolCallAgent**: manages a `ToolCollection`, handles function-calling dispatch, anti-loop detection, and the `Terminate` special tool that exits the loop.
- **FinanceAgent**: financial reasoning specialist with a curated tool set and a domain-specific system prompt.

### 2.2 Agent Types

| Agent | Role | When Used |
|-------|------|-----------|
| **FinanceAgent** | Primary solver; ReAct loop with financial tools | Default for all benchmark tasks |
| **PlanningAgent** | High-level planner/coordinator that decomposes complex tasks | Enabled by `use_planning_agent = true` in config |
| **MultimodalAgent** | Handles image-text tasks with vision model integration | Enabled by `use_multimodal_agent = true` in config |

### 2.3 Tools

| Tool | Purpose |
|------|---------|
| `PythonExecute` | Arithmetic, string parsing, table joins on data the agent already has |
| `FinancialDataLookup` | Queries offline knowledge base (stock prices, news, quarterly indicators) |
| `OcrExtract` | Re-reads charts/tables as clean markdown when vision-read values are unreliable |
| `Terminate` | Submits the final answer and exits the agent loop |
| `DeepReasoningTool` | DSER: K parallel reasoning chains with M self-refinement rounds → majority vote |

### 2.4 DSER (Deep Structured Experience Reasoning)

DSER is a parallel-chain reasoning mechanism. For each problem:
1. K independent chains are initialized with base temperature (exploration).
2. Each chain proceeds through M self-refinement rounds at lower temperature (convergence).
3. Final answers from all chains are aggregated by majority voting (per answer type: exact match for MCQ, tolerance-based clustering for numerical).

DSER parameters: `K=5` chains, `M=1` refinement round, `T_init=0.6`, `T_refine=0.3`.

### 2.5 LLM Adapter Layer

The LLM subsystem routes through an adapter layer that handles model-specific differences:

| Adapter | Models |
|---------|--------|
| `Qwen3VLAdapter` | Qwen3-VL-8B-Instruct |
| `GLMFlashAdapter` | GLM-4.1V-9B-Flash |
| `GLMThinkingAdapter` | GLM-4.1V-9B-Thinking |
| `DefaultAdapter` | Claude, DeepSeek, GPT, and other OpenAI-compatible models |

Each adapter determines: function calling support, image support, temperature clamping, API key resolution, and tool prompt injection/parsing.

## 3. FM — Financial Memory

### 3.1 Memory Bank Structure

Each experience entry is stored as two files:

```
bank/
├── MEMORY.md                    # Manifest index (≤200 lines)
├── pat_<name>.md                # YAML frontmatter + key_insight body
├── pat_<name>.emb               # Pre-computed embedding (float32, raw bytes)
├── archive/                     # Pruned entries (moved, not deleted)
└── collect_errors.jsonl         # Dead-letter queue for failed collections
```

### 3.2 Entry Schema

Each entry in the bank contains:

| Field | Purpose |
|-------|---------|
| `name` | Unique snake_case identifier |
| `question_class` | `mcq` / `boolean` / `numerical` / `free_text` |
| `question_type` | `conceptual` / `computational` / `chart` |
| `tool_tags` | Subset of `[OCR, code, none]` — tools useful for this problem type |
| `key_insight` | 50–70 word distilled experience (guiding-path or guard-rule) |
| `polarity` | `positive` (success recipe) or `negative` (failure guard rule) |
| `description` | One-line summary for the manifest index |
| `source_dataset` | Origin dataset (bizbench / finmme / finmmr / fintmm) |
| `source_target_id` | Origin problem ID |
| `question` / `gold_answer` | Original problem text and answer (for strategy A/B injection) |
| `use_count` / `hit_count` | Retrieval statistics for pruning |

### 3.3 Polarity: Guiding-Path vs Guard-Rule

- **Positive** (`polarity = "positive"`): Written when the agent solves correctly. Contains a *guiding-path* recipe in `first ... then ... next ... based on ... derive` form — reusable steps that produced the correct answer.
- **Negative** (`polarity = "negative"`): Written when the agent solves incorrectly. Contains a *guard rule* in `When <condition>, <action>` or `Before finalizing, verify <check>` form — a concrete corrective action triggered by a specific condition.

### 3.4 Memory Pipeline

The full memory lifecycle per problem:

```
retrieve → inject → run → collect → cross-verify
```

#### 3.4.1 Retrieve (three-stage retrieval)

**Stage 0 — Tagger**: LLM classifies the incoming problem into `(question_class, question_type, tool_tags)`. This provides structured filtering keys.

**Stage 1 — Hard Gate**: Filter bank entries by exact tag match. For small banks (< 50 entries), both `question_class` and `question_type` must match. For larger banks, only `question_class` must match.

**Stage 2 — Embedding Ranking**: Compute cosine similarity between the query embedding and each candidate entry's pre-computed `.emb`. A structural match bonus (`+0.1`) is added for tag overlap. A source-dataset bonus (`+0.15`) is added when the entry originates from the same benchmark dataset. Candidates below `cosine < 0.55` with no structural match are dropped.

**Stage 3 — LLM Rerank**: Top candidates (up to 20) are presented to an LLM reranker that performs strict applicability judgment. The reranker distinguishes true relevance from superficial tag similarity. Top-6 applicable entries are returned.

**Fallback**: If any stage fails or returns zero candidates, the system returns `no-memory` and the agent runs without experience injection. This is by design — the system never forces retrieval when no applicable experience exists.

#### 3.4.2 Inject

Retrieved experiences are prepended to the problem context. The agent's system prompt switches from the *no-memory* template to the *with-memory* template, which instructs the agent to read and follow Past approaches and Guard rules before reasoning. The two prompt templates are byte-identical in their base reasoning instructions; the only difference is the addition of memory-aware instructions.

#### 3.4.3 Run

The agent solves the problem using the standard ReAct loop. If experiences were injected, the `use_count` of each retrieved entry is incremented. If the answer is correct, `hit_count` is also incremented.

#### 3.4.4 Collect

After solving, the system asynchronously reflects on the agent's reasoning trace:

1. **Trace extraction**: The agent's final chain-of-thought and tool outputs are captured.
2. **Reflect (LLM)**: Depending on correctness, either `PROMPT_REFLECT_POSITIVE` or `PROMPT_REFLECT_NEGATIVE` is used to distill the trace into a structured `(question_class, question_type, tool_tags, key_insight, name_hint, description)` tuple.
3. **Anti-give-up filter**: Entries that codify "give up" behavior (e.g., concluding "data not available" when a concrete answer exists) are blocked at write time.
4. **Deduplication**: For banks ≥ 20 entries, cosine similarity against same-tag entries prevents storing near-duplicate experiences.
5. **Write**: Entry is written as `pat_<name>.md` + `pat_<name>.emb`; the manifest is updated.

Collection runs asynchronously (`asyncio.create_task`) to avoid blocking the benchmark loop. A semaphore limits concurrent collection tasks.

#### 3.4.5 Cross-Verify

A separate `memory_judge` LLM evaluates the quality of collected and retrieved entries. This feeds into pruning decisions and provides audit signals for memory health.

### 3.5 Memory Strategies A–E

The `retrieve` module supports five strategies controlling what content is injected into the agent context:

| Strategy | Injected Content | Purpose |
|----------|-----------------|---------|
| **A** | Question only (raw past problem text) | Ablation: tests if mere problem similarity helps |
| **B** | Question + Answer | Ablation: tests format reference value |
| **C** | Full experience: Question + Answer + distilled Findings/Cautions | Default strategy; full experience injection |
| **D** | Question + Answer + Annotation (directive instructions) | Tests explicit MUST/DO NOT directives |
| **E** | Question + Answer + Experience + Annotation (all fields) | Maximum information; all available signals |

Strategy C is the default and primary contribution: structured, polarity-tagged experiences distilled from agent reasoning traces.

### 3.6 Pruning

The bank maintains quality through two pruning mechanisms:

**Store-level prune** (statistical):
| Rule | Condition | Action |
|------|-----------|--------|
| Early Noisy | `use ≥ 3` AND `hit_ratio < 0.3` | Archive |
| Standard Noisy | `use ≥ 10` AND `hit_ratio < 0.5` | Archive |
| Dead | `use = 0` AND `total_queries ≥ 500` | Archive |

**Framework-level prune** (rule-based anti-give-up patterns): Entries matching anti-give-up patterns are blocked at write time and logged to `collect_errors.jsonl`.

Archived entries are moved to `archive/` rather than deleted, preserving provenance.

## 4. Benchmarks

The system supports six financial QA benchmarks:

| Benchmark | Type | Source |
|-----------|------|--------|
| BizBench | Financial QA | Text + tables |
| FinMME | Financial multi-modal | Text + images |
| FinMMR Easy/Med/Hard | Financial multi-modal reasoning | Text + images |
| FinTMMBench | Financial text + multi-modal | Text + images + charts |

The benchmark driver (`finacumen-benchmark`) loads datasets, iterates through problems, runs the chosen variant's `solve()`, and writes results as JSONL.

## 5. Key Design Invariants

1. **No-memory baseline integrity**: When retrieval returns empty, the solver prompt is byte-identical to running without the memory layer. No "memory-aware" tokens leak into the no-memory path.

2. **Collect is off-critical-path**: Experience collection runs asynchronously and never blocks the next problem's reasoning.

3. **Failure resilience**: Any stage of retrieval that fails gracefully falls back to `no-memory`. A single retrieval failure never crashes the benchmark run.

4. **Variant abstraction**: All reasoning strategies implement the same `DSERVariant.solve(target) -> result` interface, enabling transparent swapping between memory-aware and baseline configurations.
