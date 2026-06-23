"""CLI entrypoint for the HotpotQA Base-vs-RAG benchmark.

Usage examples:
    python run_experiment.py run --limit 100
    python run_experiment.py run --split validation
    python run_experiment.py analyze
    python run_experiment.py run --limit 100 && python run_experiment.py analyze
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from src.config import Config
from src.evaluation.analysis import analyze
from src.experiment.runner import ExperimentRunner


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.model = args.model or cfg.model
    cfg.split = args.split
    cfg.dify_api_key = os.getenv("DIFY_API_KEY", cfg.dify_api_key)
    cfg.dify_base_url = os.getenv("DIFY_BASE_URL", cfg.dify_base_url)
    cfg.ollama_base_url = os.getenv("OLLAMA_BASE_URL", cfg.ollama_base_url)
    return cfg


def cmd_run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    runner = ExperimentRunner(cfg)
    runner.run(limit=args.limit)


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    results = analyze(cfg)
    if not results:
        print("No results found to analyze.")
        sys.exit(1)
    print("\n=== Summary ===")
    print(results["summary"].to_string(index=False))
    print("\nMetrics written to:", cfg.metrics_dir)
    print("Final tables written to:", cfg.final_dir)


def cmd_all(args: argparse.Namespace) -> None:
    cmd_run(args)
    cmd_analyze(args)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="HotpotQA Base LLM vs RAG benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the experiment.")
    p_run.add_argument("--limit", type=int, default=None, help="Max samples to process this run.")
    p_run.add_argument("--split", type=str, default="validation", choices=["validation", "train", "test"])
    p_run.add_argument("--model", type=str, default=None, help="Override the Ollama model name.")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="Compute metrics and tables from existing results.")
    p_an.add_argument("--split", type=str, default="validation", choices=["validation", "train", "test"])
    p_an.add_argument("--model", type=str, default=None)
    p_an.set_defaults(func=cmd_analyze)

    p_all = sub.add_parser("all", help="Run experiment then analyze.")
    p_all.add_argument("--limit", type=int, default=None)
    p_all.add_argument("--split", type=str, default="validation", choices=["validation", "train", "test"])
    p_all.add_argument("--model", type=str, default=None)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()