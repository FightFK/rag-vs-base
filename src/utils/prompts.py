"""Prompt templates for the Base and RAG systems."""
from __future__ import annotations

BASE_PROMPT_TEMPLATE = """You are a question answering assistant. Answer the question as concisely as possible with a short factual answer only. Do not add explanations.

Question: {question}

Answer:"""