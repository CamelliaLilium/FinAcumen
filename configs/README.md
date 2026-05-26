# FinAcumen Configuration

## Quick Start

1. Copy the example config:
   ```
   cp configs/config.example.toml configs/config.toml
   ```
   Or set `FINACUMEN_CONFIG_PATH` to any path.

2. Fill in your API keys in the copied file.

## Sections

| Section | Purpose |
|---------|---------|
| `[llm]` | Primary reasoning model (e.g., Qwen3-VL-8B-Instruct) |
| `[llm.vision]` | Multimodal model for image-text tasks |
| `[llm.dser]` | DSER solver profile (deep structured reasoning) |
| `[llm.judge]` | LLM-as-judge for answer correctness evaluation |
| `[llm.memory_judge]` | Memory quality evaluation judge |
| `[ocr]` | OCR model for chart/table text extraction |
| `[embedding]` | Semantic embedding provider for experience retrieval |
| `[sandbox]` | Docker-based code execution sandbox |
| `[daytona]` | Daytona sandbox (placeholder if unused) |
| `[mcp]` | Model Context Protocol server reference |
| `[runflow]` | Agent orchestration (PlanningAgent, MultimodalAgent toggles) |
| `[browser]` | (Optional) Browser automation settings |
| `[search]` | (Optional) Web search engine settings |

Models used in paper experiments: Qwen3-VL-8B-Instruct (solver), gpt-4o-mini (judge), text-embedding-v3 (embeddings).
