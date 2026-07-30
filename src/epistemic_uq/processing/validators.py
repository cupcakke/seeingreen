from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

from epistemic_uq.processing.normalization import canonicalize_answer, normalize_text, parse_number
from epistemic_uq.schemas import CanonicalEvaluationUnit, EvaluationLabel, ExtractedAnswer, TaskType


def exact_match(prediction: str, references: list[str]) -> bool:
    normalized = normalize_text(prediction, remove_articles=True)
    return any(normalized == normalize_text(reference, remove_articles=True) for reference in references)


def regex_match(prediction: str, patterns: list[str], flags: int = re.IGNORECASE | re.DOTALL) -> bool:
    return any(re.fullmatch(pattern, prediction.strip(), flags=flags) is not None for pattern in patterns)


def numeric_match(prediction: str, references: list[Any], absolute_tolerance: float, relative_tolerance: float) -> bool:
    predicted = parse_number(prediction)
    if predicted is None:
        return False
    predicted_float = float(predicted)
    for reference in references:
        reference_number = parse_number(str(reference))
        if reference_number is None:
            continue
        if math.isclose(predicted_float, float(reference_number), abs_tol=absolute_tolerance, rel_tol=relative_tolerance):
            return True
    return False


def structured_match(prediction: ExtractedAnswer, references: list[Any], required_keys: list[str]) -> bool:
    if not prediction.valid or not isinstance(prediction.value, (dict, list)):
        return False
    if isinstance(prediction.value, dict) and any(key not in prediction.value for key in required_keys):
        return False
    canonical_prediction = json.dumps(prediction.value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for reference in references:
        try:
            value = json.loads(reference) if isinstance(reference, str) else reference
        except json.JSONDecodeError:
            continue
        canonical_reference = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if canonical_prediction == canonical_reference:
            return True
    return False


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_text(prediction, remove_articles=True).split()
    reference_tokens = normalize_text(reference, remove_articles=True).split()
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    predicted_counts: dict[str, int] = {}
    reference_counts: dict[str, int] = {}
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = sum(min(count, reference_counts.get(token, 0)) for token, count in predicted_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def evaluate_answer(example: CanonicalEvaluationUnit, answer: ExtractedAnswer) -> EvaluationLabel | None:
    references = list(example.valid_answers)
    if example.reference_label is not None:
        references.insert(0, example.reference_label)
    if not references:
        return None
    config = example.validator_config
    method = str(config.get("method", "auto"))
    if method == "regex":
        correct = regex_match(answer.raw, [str(value) for value in references])
        return EvaluationLabel(correct=correct, score=float(correct), validator="regex")
    if method == "numeric" or example.expected_format in {"number", "numeric", "float", "integer", "percentage"}:
        correct = numeric_match(
            answer.raw,
            references,
            float(config.get("absolute_tolerance", 1e-6)),
            float(config.get("relative_tolerance", 1e-6)),
        )
        return EvaluationLabel(correct=correct, score=float(correct), validator="numeric")
    if method == "structured" or example.task_type == TaskType.STRUCTURED:
        correct = structured_match(answer, references, list(config.get("required_keys", [])))
        return EvaluationLabel(correct=correct, score=float(correct), validator="structured")
    if method == "token_f1":
        score = max(token_f1(answer.raw, str(reference)) for reference in references)
        threshold = float(config.get("threshold", 0.8))
        return EvaluationLabel(correct=score >= threshold, score=score, validator="token_f1")
    correct = exact_match(answer.raw, [str(value) for value in references])
    return EvaluationLabel(correct=correct, score=float(correct), validator="exact_match")
