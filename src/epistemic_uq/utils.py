import hashlib
import json
import math
import random
import time
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import orjson


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_timestamp() -> str:
    return utc_now().isoformat()


def stable_json(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value)).hexdigest()


def trace_id() -> str:
    return uuid.uuid4().hex


def clamp_probability(value: float, epsilon: float = 1e-12) -> float:
    if math.isnan(value):
        raise ValueError("Probability cannot be NaN")
    return min(1.0 - epsilon, max(epsilon, float(value)))


def deterministic_random(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def chunked(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def jsonable(value: Any) -> Any:
    return json.loads(stable_json(value))
