# CLAUDE.md

## Project

HotpotQA Benchmark: Base LLM vs RAG

Objective:

Compare a standalone LLM against a Retrieval-Augmented Generation (RAG) system using the same underlying language model.

Target publication:

Academic conference paper.

---

## Experimental Design

### Base System

Question
→ Ollama
→ Gemma 4 E2B-it
→ Answer

### RAG System

Question
→ Dify API
→ Retrieval
→ Gemma 4 E2B-it
→ Answer

The language model must be identical for both systems.

The only difference between systems is the retrieval component.

---

## Model Configuration

LLM:

* Gemma 4 E2B-it

Serving:

* Ollama

Base URL:

http://localhost:11434

Embedding:

* BGE-M3

RAG Platform:

* Dify

---

## Dataset

Dataset:

HotpotQA

Source:

https://huggingface.co/datasets/hotpotqa/hotpot_qa

Default Split:

validation

Support:

* validation
* train
* test

---

## Evaluation Metrics

Primary Metrics:

* Exact Match (EM)
* F1

Secondary Metrics:

* Average Latency
* Error Rate
* Throughput

Optional Metrics:

* Recall@k
* Context Precision
* Context Recall

---

## Required Output Format

Each completed sample must be immediately appended to disk.

Use JSONL.

Example:

{
"id": 0,
"question": "...",
"ground_truth": "...",

"base_answer": "...",
"rag_answer": "...",

"base_latency": 1.23,
"rag_latency": 2.10,

"timestamp": "...",

"model": "gemma-4-e2b-it"
}

Never keep all results only in memory.

---

## Checkpointing

Requirements:

* append every completed sample
* checkpoint every 1000 samples
* support resume
* skip processed ids

Example:

outputs/results.jsonl

outputs/checkpoints/checkpoint_1000.csv

outputs/checkpoints/checkpoint_2000.csv

---

## Resume Logic

On startup:

1. Read existing results.jsonl
2. Build processed_ids set
3. Skip processed rows
4. Continue experiment automatically

The experiment must never restart from zero unless explicitly requested.

---

## Error Handling

If Ollama fails:

* retry 3 times
* exponential backoff

If Dify fails:

* retry 3 times
* exponential backoff

If still failing:

store

{
"error": true,
"error_type": "..."
}

and continue.

Never stop the entire experiment.

---

## Project Structure

project/

├── data/

├── outputs/
│   ├── results.jsonl
│   ├── checkpoints/
│   └── final/

├── metrics/

├── logs/

├── src/
│   ├── dataset/
│   ├── clients/
│   │   ├── ollama_client.py
│   │   └── dify_client.py
│   ├── evaluation/
│   ├── experiment/
│   └── utils/

├── run_experiment.py

└── CLAUDE.md

---

## Ollama Client

Use Ollama REST API.

Endpoint:

POST /api/generate

Temperature:

0

Response must be parsed into plain text answer.

Measure latency.

---

## Dify Client

Use Dify Chat API.

Endpoint:

/v1/chat-messages

Response Mode:

blocking

Measure latency.

Store raw response for future error analysis.

---

## Logging

Log:

* start time
* end time
* processed samples
* failures
* retries

Location:

logs/

---

## Metrics Calculation

Implement:

normalize_answer()

exact_match()

f1_score()

Generate:

metrics/em_f1.csv

Example:

Method,EM,F1,Latency
Base,0.52,0.64,1.23
RAG,0.67,0.78,2.01

---

## Analysis Outputs

Generate:

summary.csv

containing:

* EM
* F1
* Avg Latency
* Error Count

---

## Publication Tables

Automatically generate:

Table 1:

Method | EM | F1

Table 2:

Method | Avg Latency

Table 3:

Method | Error Count

Export CSV files suitable for conference papers.

---

## Experimental Workflow

Step 1:

Run 100 samples

Validate outputs manually.

Step 2:

Run 1000 samples

Validate metrics.

Step 3:

Run 5000 samples

Evaluate stability.

Step 4:

Run full validation set.

---

## Coding Requirements

Python 3.11+

Required Libraries:

* requests
* pandas
* tqdm
* datasets
* numpy

Use:

* type hints
* dataclasses
* pathlib

Avoid notebooks.

Everything must run from CLI.

---

## Success Criteria

The benchmark must be able to:

* run for multiple days
* recover from crashes
* resume automatically
* produce publication-ready metrics
* compare Base vs RAG fairly
