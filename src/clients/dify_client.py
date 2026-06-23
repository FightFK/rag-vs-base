"""Dify Chat API client for the RAG system."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import requests

from src.config import Config


@dataclass
class ChatResult:
    answer: str
    latency: float
    raw: dict


class DifyClient:
    """Wrapper around Dify's /v1/chat-messages endpoint (blocking mode)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.url = f"{config.dify_base_url.rstrip('/')}{config.dify_chat_endpoint}"
        self.headers = {"Authorization": f"Bearer {config.dify_api_key}"}

    def chat(self, query: str, retries: Optional[int] = None) -> ChatResult:
        if not self.config.dify_api_key:
            raise RuntimeError("DIFY_API_KEY is not configured.")
        retries = self.config.max_retries if retries is None else retries
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                payload = {
                    "inputs": {},
                    "query": query,
                    "response_mode": "blocking",
                    "user": self.config.dify_user,
                    "conversation_id": "",
                }
                start = time.perf_counter()
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                latency = time.perf_counter() - start
                answer = (data.get("answer") or "").strip()
                return ChatResult(answer=answer, latency=latency, raw=data)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries - 1:
                    backoff = self.config.backoff_base ** attempt
                    time.sleep(backoff)
        raise last_err  # type: ignore[misc]