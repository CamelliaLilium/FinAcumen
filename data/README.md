# Datasets

FinAcumen is evaluated on four financial QA benchmarks. Download links will be provided after the review process is complete.

## Benchmarks

### FinMME (Financial Multi-Modal Evaluation)
- **Size**: ~500 test questions
- **Format**: JSON with `id`, `question`, `context`, `gold_answer`, `answer_type`, `difficulty`
- **Domains**: Financial reports, charts, tables, news articles

### FinMMR (Financial Multi-Modal Reasoning)
- **Splits**: easy (~300), medium (~300), hard (~300)
- **Size**: ~900 test questions across three difficulty levels
- **Format**: JSON with `id`, `question`, `context`, `gold_answer`, `answer_type`, `difficulty`
- **Domains**: Numerical reasoning, financial statement analysis, market data interpretation

### FinTMM (Financial Text + Multi-Modal)
- **Size**: ~500 test questions
- **Format**: JSON with `id`, `question`, `context`, `gold_answer`, `answer_type`, `difficulty`
- **Domains**: Mixed text and table-based financial reasoning

### BizBench (Business Benchmark)
- **Size**: ~645 questions (SEC-NUM subset used for training)
- **Format**: JSON with `id`, `question`, `context`, `gold_answer`, `answer_type`, `task`
- **Domains**: SEC filings, business metrics, financial ratios

## Directory Structure

After downloading, the `data/` directory should look like:

```
data/
  finmme/
    train.json
    test.json
  finmmr_easy/
    train.json
    test.json
  finmmr_hard/
    train.json
    test.json
  finmmr_medium/
    train.json
    test.json
  fintmm/
    train.json
    test.json
  bizbench/
    train.json
    test.json
  sample/
    README.md
```

## Data Format

Each `train.json` and `test.json` is a JSON array of question objects:

```json
[
  {
    "id": "unique_question_id",
    "question": "What was the company's revenue in Q3 2023?",
    "context": "In its Q3 2023 earnings report, the company reported...",
    "gold_answer": "14.2 billion",
    "answer_type": "numerical",
    "difficulty": "easy"
  }
]
```

Additional fields may be present depending on the dataset (e.g., `task` for BizBench).

## Download

Use the download helper scripts:

```bash
# Shell
bash scripts/download_data.sh

# Python
python scripts/download_data.py
```

See `scripts/download_data.sh` and `scripts/download_data.py` for placeholder download URLs and instructions.
