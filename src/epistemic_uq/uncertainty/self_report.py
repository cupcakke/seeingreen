from __future__ import annotations

import json
import re
from typing import Any

from epistemic_uq.errors import ParsingError
from epistemic_uq.schemas import SelfReportedConfidence


JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
PERCENT_PATTERN = re.compile(r"(?<!\d)(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%")
DECIMAL_PATTERN = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")
SCORE_PATTERN = re.compile(r"(?i)(?:confidence|probability|score)\s*[:=]\s*(\d+(?:\.\d+)?)")


def _normalize(value: float) -> float:
    if value > 1.0 and value <= 100.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ParsingError(f"Confidence {value} is outside [0, 1]")
    return float(value)


def _from_json(text: str) -> tuple[float, str] | None:
    candidates = [text.strip()]
    candidates.extend(match.group(0) for match in JSON_OBJECT_PATTERN.finditer(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        for key in ("confidence", "probability", "correct_probability", "score"):
            if key in value and isinstance(value[key], (int, float, str)):
                try:
                    return _normalize(float(value[key])), f"json:{key}"
                except (ValueError, ParsingError):
                    continue
    return None


def parse_self_report(text: str) -> SelfReportedConfidence:
    parsed = _from_json(text)
    if parsed:
        value, parser = parsed
        return SelfReportedConfidence(value=value, raw=text, parser=parser)
    percent = PERCENT_PATTERN.search(text)
    if percent:
        return SelfReportedConfidence(value=_normalize(float(percent.group(1))), raw=text, parser="percent")
    score = SCORE_PATTERN.search(text)
    if score:
        return SelfReportedConfidence(value=_normalize(float(score.group(1))), raw=text, parser="labeled_number")
    decimal = DECIMAL_PATTERN.search(text)
    if decimal:
        return SelfReportedConfidence(value=_normalize(float(decimal.group(1))), raw=text, parser="decimal")
    raise ParsingError("No machine-readable confidence value found")


def parse_truth_probability(text: str) -> float:
    parsed = _from_json(text)
    if parsed:
        return parsed[0]
    lowered = text.strip().casefold()
    if lowered in {"true", "correct", "yes"}:
        return 1.0
    if lowered in {"false", "incorrect", "no"}:
        return 0.0
    raise ParsingError("No truth probability found")
