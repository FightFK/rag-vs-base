"""Evaluation metrics: normalize_answer, Exact Match (EM) and F1.

Implementation follows the standard SQuAD / HotpotQA evaluation protocol.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


@dataclass
class SampleMetrics:
    base_em: float
    rag_em: float
    base_f1: float
    rag_f1: float


def evaluate_sample(base_answer: str, rag_answer: str, ground_truth: str) -> SampleMetrics:
    return SampleMetrics(
        base_em=exact_match(base_answer, ground_truth),
        rag_em=exact_match(rag_answer, ground_truth),
        base_f1=f1_score(base_answer, ground_truth),
        rag_f1=f1_score(rag_answer, ground_truth),
    )