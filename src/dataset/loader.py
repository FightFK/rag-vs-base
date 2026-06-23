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

    def load(self):
        # HotpotQA requires a builder config: 'distractor' (standard) or 'fullwiki'.
        return load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.split,
        )

    def iter_samples(self) -> Iterator[Sample]:
        ds = self.load()
        split = self.config.split
        for i, row in enumerate(ds):
            answer = row.get("answer", "")
            if isinstance(answer, dict):
                answer = answer.get("value", "") or answer.get("text", "") or ""
            yield Sample(
                id=f"{split}-{i}",
                question=row["question"],
                ground_truth=str(answer).strip(),
            )

    def __iter__(self) -> Iterator[Sample]:
        return self.iter_samples()

    def __len__(self) -> int:
        return len(self.load())