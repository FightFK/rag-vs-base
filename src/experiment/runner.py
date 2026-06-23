"""Experiment runner: orchestrates Base+RAG inference, checkpointing and resume."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.clients.dify_client import DifyClient
from src.clients.ollama_client import OllamaClient
from src.config import Config
from src.dataset.loader import HotpotQALoader
from src.utils.logging_utils import setup_logger
from src.utils.prompts import BASE_PROMPT_TEMPLATE


@dataclass
class SampleResult:
    id: str
    question: str
    ground_truth: str
    base_answer: str
    rag_answer: str
    base_latency: float
    rag_latency: float
    timestamp: str
    model: str
    error: bool = False
    error_type: Optional[str] = None


class ExperimentRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        self.logger = setup_logger("experiment", config.logs_dir / "experiment.log")
        self.ollama = OllamaClient(config)
        self.dify = DifyClient(config)
        self.processed_ids: set[str] = self._load_processed_ids()

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def _load_processed_ids(self) -> set[str]:
        ids: set[str] = set()
        rf = self.config.results_file
        if not rf.exists():
            return ids
        with rf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "id" in row:
                        ids.add(str(row["id"]))
                except json.JSONDecodeError:
                    continue
        self.logger.info("Resumed: %d samples already processed.", len(ids))
        return ids

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def _append_result(self, result: SampleResult) -> None:
        with self.config.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def _write_checkpoint(self, count: int) -> None:
        cp = self.config.checkpoints_dir / f"checkpoint_{count}.csv"
        with cp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "question", "ground_truth", "base_answer", "rag_answer"])
            with self.config.results_file.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    writer.writerow(
                        [
                            row.get("id", ""),
                            row.get("question", ""),
                            row.get("ground_truth", ""),
                            row.get("base_answer", ""),
                            row.get("rag_answer", ""),
                        ]
                    )
        self.logger.info("Wrote checkpoint %s", cp)

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------
    def _process_sample(self, sample) -> SampleResult:  # noqa: ANN001
        ts = datetime.now(timezone.utc).isoformat()
        base_answer = ""
        rag_answer = ""
        base_latency = 0.0
        rag_latency = 0.0
        error = False
        error_type: Optional[str] = None

        # Base system
        try:
            prompt = BASE_PROMPT_TEMPLATE.format(question=sample.question)
            res = self.ollama.generate(prompt)
            base_answer = res.answer
            base_latency = res.latency
        except Exception as exc:  # noqa: BLE001
            error = True
            error_type = f"ollama:{type(exc).__name__}"
            self.logger.error("Ollama failed for id=%s: %s", sample.id, exc)

        # RAG system
        try:
            res = self.dify.chat(sample.question)
            rag_answer = res.answer
            rag_latency = res.latency
        except Exception as exc:  # noqa: BLE001
            error = True
            et = f"dify:{type(exc).__name__}"
            error_type = f"{error_type};{et}" if error_type else et
            self.logger.error("Dify failed for id=%s: %s", sample.id, exc)

        return SampleResult(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            base_answer=base_answer,
            rag_answer=rag_answer,
            base_latency=base_latency,
            rag_latency=rag_latency,
            timestamp=ts,
            model=self.config.model,
            error=error,
            error_type=error_type,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, limit: Optional[int] = None) -> None:
        start_ts = datetime.now(timezone.utc).isoformat()
        self.logger.info("Experiment started at %s", start_ts)
        self.logger.info("Model: %s", self.config.model)

        loader = HotpotQALoader(self.config)
        processed_this_run = 0
        failures = 0
        retries_observed = 0  # tracked via error_type heuristic
        t0 = time.perf_counter()

        try:
            pbar = tqdm(loader, desc="HotpotQA")
            for sample in pbar:
                if limit is not None and processed_this_run >= limit:
                    break
                if sample.id in self.processed_ids:
                    continue
                result = self._process_sample(sample)
                self._append_result(result)
                self.processed_ids.add(sample.id)
                processed_this_run += 1
                if result.error:
                    failures += 1
                pbar.set_postfix(done=len(self.processed_ids), fails=failures)
                if processed_this_run > 0 and processed_this_run % self.config.checkpoint_every == 0:
                    self._write_checkpoint(len(self.processed_ids))
        finally:
            end_ts = datetime.now(timezone.utc).isoformat()
            elapsed = time.perf_counter() - t0
            self.logger.info("Experiment ended at %s", end_ts)
            self.logger.info("Processed this run: %d", processed_this_run)
            self.logger.info("Total processed:    %d", len(self.processed_ids))
            self.logger.info("Failures this run:  %d", failures)
            self.logger.info("Elapsed seconds:    %.2f", elapsed)
            if processed_this_run > 0:
                final_cp = len(self.processed_ids)
                self._write_checkpoint(final_cp)