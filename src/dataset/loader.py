"""HotpotQA dataset loader using the Hugging Face `datasets` library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

from src.config import Config


@dataclass
class Sample:
    id: str
    question: str
    ground_truth: str


class HotpotQALoader:
    def __init__(self, config: Config) -> None:
        self.config = config

    def __iter__(self) -> Iterator[Sample]:
        ds = load_dataset(self.config.dataset_name, split=self.config.split)
        for i, row in enumerate(ds):
            answer = row.get("answer", "")
            if isinstance(answer, dict):
                answer = answer.get("value", "") or answer.get("text", "") or ""
            yield Sample(
                id=str(i),
                question=row["question"],
                ground_truth=str(answer).strip(),
            )

    def __len__(self) -> int:
        # Avoid loading twice in common usage; callers usually iterate.
        ds = load_dataset(self.config.dataset_name, split=self.config.split)
        return len(ds)