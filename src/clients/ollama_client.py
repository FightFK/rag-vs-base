"""Ollama REST API client for the Base LLM system."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

from src.config import Config


@dataclass
class GenerationResult:
    answer: str
    latency: float
    raw: dict


class OllamaClient:
    """Thin wrapper around Ollama's /api/generate endpoint."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.url = f"{config.ollama_base_url.rstrip('/')}{config.ollama_endpoint}"

    def generate(self, prompt: str, retries: Optional[int] = None) -> GenerationResult:
        retries = self.config.max_retries if retries is None else retries
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.config.temperature},
                }
                start = time.perf_counter()
                resp = requests.post(self.url, json=payload, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                latency = time.perf_counter() - start
                answer = (data.get("response") or "").strip()
                return GenerationResult(answer=answer, latency=latency, raw=data)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries - 1:
                    backoff = self.config.backoff_base ** attempt
                    time.sleep(backoff)
        raise last_err  # type: ignore[misc]