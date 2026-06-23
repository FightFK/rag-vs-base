# HotpotQA Benchmark: Base LLM vs RAG

Compare a standalone LLM (Ollama + Gemma 4 E2B-it) against a Retrieval-Augmented
Generation (RAG) system (Dify + same Gemma model) on the HotpotQA dataset.

The **only difference** between the two systems is the retrieval component, so
the comparison isolates the contribution of retrieval.

---

## 1. Requirements

- Python 3.11+
- Ollama running locally (serving the Gemma 4 E2B-it model)
- Dify instance (only needed for the RAG / `both` modes)
- Internet access for downloading the HotpotQA dataset from Hugging Face

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Pull the model into Ollama

```bash
ollama pull gemma4:e2b
ollama list                        # confirm the tag appears
```

### Configure Dify (only for RAG mode)

```bash
export DIFY_API_KEY="your-dify-api-key"
export DIFY_BASE_URL="http://localhost/v1"   # optional, has a default
```

---

## 2. Project structure

```
ragvsbase/
├── data/                 # dataset cache (HF datasets)
├── outputs/
│   ├── results.jsonl     # one JSON row per sample (append-only)
│   ├── checkpoints/       # checkpoint_<count>.csv every 1000 samples
│   └── final/             # summary.csv + publication tables (per experiment tag)
├── metrics/               # em_f1_<tag>.csv
├── logs/                  # experiment.log
├── src/
│   ├── config.py
│   ├── dataset/loader.py
│   ├── clients/
│   │   ├── ollama_client.py
│   │   └── dify_client.py
│   ├── evaluation/
│   │   ├── metrics.py     # normalize_answer / EM / F1
│   │   └── analysis.py
│   ├── experiment/runner.py
│   └── utils/{prompts,logging_utils}.py
└── run_experiment.py      # CLI entrypoint
```

---

## 3. Running the experiment

The CLI has three subcommands: `run`, `analyze`, and `all`.

### Common flags

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--split` | `validation` | `validation` / `train` / `test` | HotpotQA split |
| `--dataset-config` | `distractor` | `distractor` / `fullwiki` | HotpotQA builder config |
| `--mode` | `both` | `base` / `rag` / `both` | Which system(s) to run |
| `--limit` | (none) | int | Max samples to process this run |
| `--model` | `gemma-4-e2b-it` | str | Publication model name written to JSONL |
| `--ollama-model` | `gemma4:e2b` | str | Actual Ollama tag sent to the API |

### Step 1 — small smoke test (base model only)

```bash
python run_experiment.py run --limit 10 --mode base
```

Check `outputs/results.jsonl` — each row should have a non-empty `base_answer`.

### Step 2 — run 100 samples, base only

```bash
python run_experiment.py run --limit 100 --mode base
```

### Step 3 — run 50,000 samples from the train split with fullwiki config

```bash
python run_experiment.py run \
  --split train \
  --dataset-config fullwiki \
  --limit 50000 \
  --mode base \
  --ollama-model gemma4:e2b
```

### Step 4 — continue the remaining samples of the split

Simply drop `--limit` and run again. The resume logic reads `results.jsonl`,
builds a set of processed ids, and skips them — so the run continues from
where it stopped:

```bash
python run_experiment.py run \
  --split train \
  --dataset-config fullwiki \
  --mode base
```

### Step 5 — run the RAG system

Once Dify is configured:

```bash
export DIFY_API_KEY="..."
python run_experiment.py run \
  --split train \
  --dataset-config fullwiki \
  --mode rag
```

To run both systems together:

```bash
python run_experiment.py run --split train --dataset-config fullwiki --mode both
```

---

## 4. Analyzing results

```bash
python run_experiment.py analyze --split train --dataset-config fullwiki
```

This reads `outputs/results.jsonl`, filters rows matching the selected
`split` + `dataset_config`, computes EM / F1 / latency / error counts, and
writes (tagged with `<dataset_config>_<split>`, e.g. `fullwiki_train`):

- `metrics/em_f1_fullwiki_train.csv`
- `outputs/final/summary_fullwiki_train.csv`
- `outputs/final/table1_em_f1_fullwiki_train.csv`
- `outputs/final/table2_latency_fullwiki_train.csv`
- `outputs/final/table3_errors_fullwiki_train.csv`

Run `all` to run the experiment and immediately produce metrics:

```bash
python run_experiment.py all --split train --dataset-config fullwiki --mode base --limit 100
```

---

## 5. Resume & checkpointing

- Each completed sample is **appended immediately** to `outputs/results.jsonl`.
- A checkpoint CSV is written every 1000 samples in
  `outputs/checkpoints/checkpoint_<count>.csv`.
- On startup the runner loads existing ids from `results.jsonl` and skips them.
- Resume is **mode-aware**: running `--mode base` first and `--mode both`
  later will only re-process samples whose `base_answer` (or `rag_answer`)
  is still missing.

### Restart from scratch

```bash
rm outputs/results.jsonl
rm -rf outputs/checkpoints/
```

---

## 6. Output format (`results.jsonl`)

Each line is one JSON record:

```json
{
  "id": "train-42",
  "question": "What government position was held by ...?",
  "ground_truth": "Prime Minister",
  "split": "train",
  "dataset_config": "fullwiki",
  "base_answer": "...",
  "rag_answer": "...",
  "base_latency": 1.23,
  "rag_latency": 2.10,
  "timestamp": "2026-06-24T01:09:51.892585+00:00",
  "model": "gemma-4-e2b-it",
  "error": false,
  "error_type": null
}
```

When a system fails after 3 retries with exponential backoff, the record is
still written with `error: true` and `error_type` set (e.g.
`"ollama:HTTPError"`), and the run continues.

---

## 7. Configuration

Defaults live in `src/config.py` and can be overridden via environment
variables or CLI flags:

| Setting | Env var | CLI flag | Default |
|---------|---------|----------|---------|
| Ollama base URL | `OLLAMA_BASE_URL` | — | `http://localhost:11434` |
| Ollama model tag | — | `--ollama-model` | `gemma4:e2b` |
| Publication model name | — | `--model` | `gemma-4-e2b-it` |
| Dify base URL | `DIFY_BASE_URL` | — | `http://localhost/v1` |
| Dify API key | `DIFY_API_KEY` | — | (none) |
| Dataset split | — | `--split` | `validation` |
| Dataset config | — | `--dataset-config` | `distractor` |

---

## 8. Experimental workflow (from CLAUDE.md)

| Step | Samples | Purpose |
|------|---------|---------|
| 1 | 100 | Validate outputs manually |
| 2 | 1,000 | Validate metrics |
| 3 | 5,000 | Evaluate stability |
| 4 | full split | Final benchmark |

---

## 9. Metrics

Primary:

- **Exact Match (EM)** — normalized prediction equals normalized gold answer.
- **F1** — token-overlap F1 between normalized prediction and gold answer.

Secondary:

- Average latency (seconds per sample)
- Error count (number of failed requests after retries)
- Throughput (samples / second)

`normalize_answer` lowercases, strips punctuation/articles/extra whitespace,
following the standard SQuAD / HotpotQA evaluation protocol.