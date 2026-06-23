"""Central configuration for the HotpotQA Base-vs-RAG benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Model / serving
    model: str = "gemma-4-e2b-it"
    ollama_base_url: str = "http://localhost:11434"
    ollama_endpoint: str = "/api/generate"
    temperature: float = 0.0

    # Dify
    dify_base_url: str = "http://localhost/v1"
    dify_chat_endpoint: str = "/chat-messages"
    dify_api_key: Optional[str] = None
    dify_user: str = "hotpotqa-benchmark"

    # Dataset
    dataset_name: str = "hotpotqa/hotpot_qa"
    split: str = "validation"

    # Retry behaviour
    max_retries: int = 3
    backoff_base: float = 2.0  # exponential backoff: 2^n seconds

    # Checkpointing
    checkpoint_every: int = 1000

    # Paths (resolved relative to project root)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    results_file: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "results.jsonl")
    checkpoints_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "checkpoints")
    final_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "final")
    metrics_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "metrics")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.outputs_dir,
            self.checkpoints_dir,
            self.final_dir,
            self.metrics_dir,
            self.logs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()