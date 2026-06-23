"""Compute aggregate metrics, summary CSV and publication tables from results.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import Config
from src.evaluation.metrics import exact_match, f1_score


def load_results(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_metrics(rows: List[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        gt = r.get("ground_truth", "")
        base = r.get("base_answer", "")
        rag = r.get("rag_answer", "")
        records.append(
            {
                "base_em": exact_match(base, gt),
                "rag_em": exact_match(rag, gt),
                "base_f1": f1_score(base, gt),
                "rag_f1": f1_score(rag, gt),
                "base_latency": float(r.get("base_latency", 0.0)),
                "rag_latency": float(r.get("rag_latency", 0.0)),
                "base_error": bool(r.get("error") and "ollama" in (r.get("error_type") or "")),
                "rag_error": bool(r.get("error") and "dify" in (r.get("error_type") or "")),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = pd.DataFrame(
        {
            "Method": ["Base", "RAG"],
            "EM": [df["base_em"].mean(), df["rag_em"].mean()],
            "F1": [df["base_f1"].mean(), df["rag_f1"].mean()],
            "Avg Latency": [df["base_latency"].mean(), df["rag_latency"].mean()],
            "Error Count": [int(df["base_error"].sum()), int(df["rag_error"].sum())],
        }
    )
    return summary


def _wrapper(methods, values) -> pd.DataFrame:
    return pd.DataFrame({"Method": methods, "Value": values})


def analyze(config: Config) -> Dict[str, pd.DataFrame]:
    config.ensure_dirs()
    rows = load_results(config.results_file)
    df = compute_metrics(rows)
    summary = build_summary(df)
    dtl = {}

    if summary.empty:
        return dtl

    # metrics/em_f1.csv
    em_f1 = pd.DataFrame(
        {
            "Method": ["Base", "RAG"],
            "EM": [df["base_em"].mean(), df["rag_em"].mean()],
            "F1": [df["base_f1"].mean(), df["rag_f1"].mean()],
            "Latency": [df["base_latency"].mean(), df["rag_latency"].mean()],
        }
    )
    em_f1.to_csv(config.metrics_dir / "em_f1.csv", index=False)

    # summary.csv in outputs/final
    summary.to_csv(config.final_dir / "summary.csv", index=False)

    # Publication tables
    table1 = summary[["Method", "EM", "F1"]].copy()
    table2 = summary[["Method", "Avg Latency"]].copy()
    table3 = summary[["Method", "Error Count"]].copy()

    table1.to_csv(config.final_dir / "table1_em_f1.csv", index=False)
    table2.to_csv(config.final_dir / "table2_latency.csv", index=False)
    table3.to_csv(config.final_dir / "table3_errors.csv", index=False)

    return {"summary": summary, "em_f1": em_f1, "table1": table1, "table2": table2, "table3": table3}