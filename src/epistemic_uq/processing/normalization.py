from __future__ import annotations

import csv
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from epistemic_uq.schemas import CanonicalEvaluationUnit, ExtractedAnswer, TaskType


ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s.+\-%/]", re.UNICODE)
JSON_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def normalize_text(value: str, remove_articles: bool = False) -> str:
    normalized = normalize_unicode(value).strip().casefold()
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)
    if remove_articles:
        normalized = ARTICLE_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def parse_number(value: str) -> Decimal | None:
    cleaned = normalize_unicode(value).strip().replace(",", "")
    cleaned = cleaned.removesuffix("%").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value.strip().endswith("%"):
        number /= Decimal(100)
    return number


def extract_json(value: str) -> Any:
    candidate = value.strip()
    fence = JSON_FENCE_PATTERN.match(candidate)
    if fence:
        candidate = fence.group(1)
    return json.loads(candidate)


def canonicalize_answer(raw: str, task_type: TaskType, expected_format: str | None = None) -> ExtractedAnswer:
    errors: list[str] = []
    if task_type == TaskType.CLASSIFICATION:
        canonical = normalize_text(raw)
        return ExtractedAnswer(raw=raw, canonical=canonical, value=canonical, parser="classification")
    if task_type == TaskType.EXTRACTION:
        canonical = normalize_text(raw, remove_articles=False)
        return ExtractedAnswer(raw=raw, canonical=canonical, value=canonical, parser="extraction")
    if task_type == TaskType.STRUCTURED:
        try:
            value = extract_json(raw)
            canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return ExtractedAnswer(raw=raw, canonical=canonical, value=value, parser="json")
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(str(exc))
            canonical = normalize_text(raw)
            return ExtractedAnswer(
                raw=raw,
                canonical=canonical,
                value=None,
                parser="json",
                valid=False,
                validation_errors=tuple(errors),
            )
    number = parse_number(raw)
    if expected_format in {"number", "numeric", "float", "integer", "percentage"} and number is not None:
        canonical = format(number.normalize(), "f")
        return ExtractedAnswer(raw=raw, canonical=canonical, value=float(number), parser="numeric")
    canonical = normalize_text(raw, remove_articles=True)
    return ExtractedAnswer(raw=raw, canonical=canonical, value=canonical, parser="qa")


class DatasetLoader:
    adapter = TypeAdapter(CanonicalEvaluationUnit)

    def load(self, path: Path, dataset_id: str | None = None) -> tuple[CanonicalEvaluationUnit, ...]:
        suffix = path.suffix.casefold()
        if suffix in {".jsonl", ".ndjson"}:
            records = self._load_jsonl(path)
        elif suffix == ".json":
            records = self._load_json(path)
        elif suffix == ".csv":
            records = self._load_csv(path)
        else:
            raise ValueError(f"Unsupported dataset format {path.suffix}")
        normalized: list[CanonicalEvaluationUnit] = []
        for index, record in enumerate(records):
            if dataset_id and "dataset_id" not in record:
                record["dataset_id"] = dataset_id
            if "example_id" not in record:
                record["example_id"] = f"{record.get('dataset_id', dataset_id or path.stem)}:{index}"
            normalized.append(self.adapter.validate_python(record))
        seen = set()
        for example in normalized:
            if example.example_id in seen:
                raise ValueError(f"Duplicate example_id {example.example_id}")
            seen.add(example.example_id)
        return tuple(normalized)

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Line {line_number} is not an object")
                records.append(value)
        return records

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and isinstance(value.get("examples"), list):
            value = value["examples"]
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON dataset must be an array of objects or an object containing examples")
        return list(value)

    def _load_csv(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        parsed: list[dict[str, Any]] = []
        for record in records:
            value: dict[str, Any] = dict(record)
            for key in ("valid_answers", "subgroup_metadata", "perturbation_rules", "validator_config", "metadata"):
                if value.get(key):
                    value[key] = json.loads(value[key])
            parsed.append(value)
        return parsed
