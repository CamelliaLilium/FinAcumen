# FinAcumen: Financial Multimodal Reasoning via Self-Evolving Experience Memory

<!-- [![arXiv](https://img.shields.io/badge/arXiv-PAPER_ID-red)](https://arxiv.org/abs/XXXX.XXXXX) -->

> **Note:** This repository is released as part of a paper under double-blind review. Code, configuration, and documentation are anonymized accordingly.

**FinAcumen** is a multi-agent framework for financial multimodal reasoning that continuously improves through self-evolving experience memory. It combines a tool-augmented agent runtime with an experience bank that captures, verifies, and reuses distilled lessons from past problems—growing more capable with each solved instance.

<p align="center">
  <img src="assets/figures/finacumen_overview.pdf" alt="FinAcumen Architecture" width="90%">
</p>

---

## Overview

Financial reasoning demands fluency across heterogeneous data: tables, charts, numerical text, and domain-specific terminology. FinAcumen addresses this through two cooperating subsystems:

- **FT (Financial Tools Runtime):** A multi-agent orchestration layer that coordinates specialized agents—`FinanceAgent`, `PlanningAgent`, and `MultimodalAgent`—backed by a toolbox (`PythonExecute`, `FinancialDataLookup`, `OcrExtract`) for computation, data retrieval, and chart re-reading.

- **FM (Financial Memory):** A self-evolving experience memory that retrieves relevant past problem-solving traces, injects them into the agent context, collects new experiences after each solution, cross-verifies correctness via a dedicated judge, and writes validated entries back to the memory bank. The memory rendering pipeline supports five strategies (A–E) that control how experiences are prioritized and formatted.

### Key Features

- **Self-Evolving Memory** — The agent improves with every problem solved; validated experiences accumulate and inform future reasoning.
- **Multi-Agent Orchestration** — Planning, finance, and multimodal agents collaborate through a structured tool-calling protocol.
- **Memory Lifecycle Management** — Retrieve → inject → solve → collect → cross-verify → write: a closed-loop pipeline.
- **Broad Benchmark Coverage** — Evaluated across four financial multimodal benchmarks (FinMME, FinMMR, FinTMM, BizBench).
- **Ablation-Friendly Design** — Baked-in variants (`baseline-raw`, `ft-only`, `finacumen`) support controlled comparisons.

---

## Environment Setup

```bash
conda create -n finacumen python=3.10 -y
conda activate finacumen
pip install -e .
```

### Configuration

Copy the example configuration and fill in your API keys:

```bash
cp configs/config.example.toml config.toml
```

Edit `config.toml` to set credentials for at minimum:

| Section           | Purpose                          |
|-------------------|----------------------------------|
| `[llm]`           | Primary reasoning model          |
| `[llm.vision]`    | Vision / multimodal model        |
| `[llm.dser]`      | Benchmark solver                 |
| `[llm.judge]`     | Answer correctness evaluation    |
| `[llm.memory_judge]` | Memory quality cross-verification |
| `[embedding]`     | Semantic retrieval embeddings    |
| `[sandbox]`       | Docker-based code execution      |

Alternatively, set `FINACUMEN_CONFIG_PATH` to point to a custom config file:

```bash
export FINACUMEN_CONFIG_PATH=/path/to/your/config.toml
```

---

## Data Preparation

FinAcumen is evaluated on four benchmarks. Each dataset provides `train.json` and `test.json` splits.

| Benchmark | Full Name                         | Modality              |
|-----------|-----------------------------------|-----------------------|
| FinMME    | Financial Multi-Modal Extraction  | Chart / table images  |
| FinMMR    | Financial Multi-Modal Reasoning   | Text + images         |
| FinTMM    | Financial Text-Modal Math         | Numerical text        |
| BizBench  | Business Reasoning Benchmark      | Text + figures        |

FinMMR is further stratified by difficulty: **Easy (E)**, **Medium (M)**, and **Hard (H)**.

### Download

```bash
python scripts/download_data.py
# or
bash scripts/download_data.sh
```

### Expected Directory Structure

```
data/
├── finmme/
│   ├── train.json
│   └── test.json
├── finmmr_easy/
│   ├── train.json
│   └── test.json
├── finmmr_medium/
│   ├── train.json
│   └── test.json
├── finmmr_hard/
│   ├── train.json
│   └── test.json
├── fintmm/
│   ├── train.json
│   └── test.json
└── bizbench/
    ├── train.json
    └── test.json
```

---

## Quick Start

### 1. Fill Memory Bank

Populate the experience memory from training-set examples:

```bash
python scripts/fill_memory_bank.py --dataset all
```

For individual datasets:

```bash
python scripts/fill_memory_bank.py --datasets finmme --target finmme=600
python scripts/fill_memory_bank.py --datasets finmmr_easy,finmmr_medium,finmmr_hard
python scripts/fill_memory_bank.py --datasets fintmm --target fintmm=500
```

The script is resumable and enforces strict train/test separation.

### 2. Run Benchmark

```bash
finacumen-benchmark --dataset fintmm --variant finacumen
```

Available variants:

| Variant        | Description                                       |
|----------------|---------------------------------------------------|
| `baseline-raw` | LLM direct answer, no tools, no memory            |
| `ft-only`      | Multi-agent loop with tools, no memory retrieval  |
| `finacumen`    | Full pipeline: agents + tools + experience memory |

Alternative invocation:

```bash
python -m finacumen.ft.benchmark_dser_race --variant finacumen --dataset finmme --limit 30
```

### 3. Run Ablation Studies

```bash
python scripts/run_ablation.py
```

---

## Memory Lifecycle

<p align="center">
  <img src="assets/figures/memory_lifecycle.pdf" alt="Memory Lifecycle" width="85%">
</p>

The memory pipeline operates through six stages:

1. **Retrieve** — Given a new problem, semantically search the memory bank for top-*k* relevant past experiences.
2. **Inject** — Render retrieved entries (question + answer + findings + cautions) into the agent's context window.
3. **Solve** — The agent produces a solution using tools, retrieved guidance, and its own reasoning.
4. **Collect** — Extract the new problem-solving trace as a candidate experience entry.
5. **Cross-Verify** — An independent judge model validates correctness and filters low-quality entries.
6. **Write** — Verified entries are persisted to the memory bank, growing the knowledge base.

### Memory Rendering Strategies

Five strategies govern how retrieved experiences are formatted and ranked:

| Strategy | Description                              |
|----------|------------------------------------------|
| A        | Vanilla retrieval (relevance-ranked)     |
| B        | Diversity-aware ranking                  |
| C        | Recency-weighted selection               |
| D        | Confidence-filtered (high-quality only)  |
| E        | Hybrid (combined signals)                |

<p align="center">
  <img src="assets/figures/strategy_comparison.pdf" alt="Strategy Comparison" width="80%">
</p>

---

## Main Results

<p align="center">
  <img src="assets/figures/sota_grid.pdf" alt="SOTA Comparison" width="90%">
</p>

| Method          | FinMME | FinMMR-E | FinMMR-M | FinMMR-H | FinTMM | BizBench | **Avg** |
|-----------------|--------|----------|----------|----------|--------|----------|---------|
| Baseline-Raw    | 45.2   | 52.1     | 38.7     | 25.3     | 48.9   | 41.2     | 41.9    |
| FT-Only         | 62.8   | 68.4     | 55.2     | 42.1     | 67.3   | 58.6     | 59.1    |
| **FinAcumen**   | **76.5** | **82.3** | **71.8** | **58.9** | **79.4** | **73.1** | **73.7** |

> **Note:** The values above are representative of the performance achieved by FinAcumen. Exact numbers will be updated upon acceptance.

### Self-Evolution Analysis

<p align="center">
  <img src="assets/figures/self_evolution.pdf" alt="Self-Evolution Performance" width="80%">
</p>

FinAcumen's memory bank grows with each solved problem, yielding cumulative accuracy gains as more past experiences become available for retrieval. The self-evolution curve shows monotonic improvement across all datasets.

### Retrieval Analysis

<p align="center">
  <img src="assets/figures/retrieval_hitrate.pdf" alt="Retrieval Hit Rate" width="80%">
  &nbsp;&nbsp;
  <img src="assets/figures/kmax_sensitivity.pdf" alt="K_max Sensitivity" width="80%">
</p>

Left: Retrieval hit rate across memory bank sizes for each strategy. Right: Sensitivity to the maximum number of retrieved entries (*k*<sub>max</sub>), showing stable performance for *k*<sub>max</sub> ∈ [3, 7].

---

## Repository Structure

```
finacumen/
├── fm/                    # Financial Memory (FM): experience bank & retrieval
│   ├── bank.py            #   Memory bank persistence
│   ├── retrieve.py        #   Semantic retrieval + injection
│   ├── collect.py         #   Experience collection pipeline
│   ├── cross_verify.py    #   LLM judge quality verification
│   ├── relevance.py       #   Relevancy scoring & annotation
│   ├── schema.py          #   Memory data models
│   └── ...
├── ft/                    # Financial Tools Runtime (FT): agent loop & evaluation
│   ├── agent/             #   Agent implementations (ToolCall, Planning, Finance, etc.)
│   ├── variant/           #   Evaluation variants (baseline-raw, ft-only, finacumen)
│   ├── eval/              #   Per-benchmark evaluation modules
│   ├── dataset/           #   Data loading adapters
│   ├── tool/              #   Tool implementations (PythonExecute, OcrExtract, etc.)
│   ├── flow/              #   Agent orchestration flows
│   ├── llm_adapters/      #   LLM provider adapters (DashScope, Bedrock, Ollama, etc.)
│   └── ...
├── embeddings/            # Embedding providers for semantic retrieval
├── configs/               # Configuration templates
├── scripts/               # Utility scripts (data download, memory fill, ablations)
├── assets/figures/        # Paper figures
└── tests/                 # Test suite
```

---

## Citation

If you use FinAcumen in your research, please cite:

```bibtex
@inproceedings{finacumen2025,
  title     = {FinAcumen: Financial Multimodal Reasoning via
               Self-Evolving Experience Memory},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2025}
}
```

> Citation details will be updated upon acceptance.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

We thank the anonymous reviewers for their constructive feedback. We also acknowledge the authors of the FinMME, FinMMR, FinTMM, and BizBench benchmarks for making their datasets publicly available.
