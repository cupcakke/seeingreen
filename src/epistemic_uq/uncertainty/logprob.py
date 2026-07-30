from __future__ import annotations

import math
from enum import StrEnum

from epistemic_uq.schemas import Generation, TokenProbability
from epistemic_uq.utils import clamp_probability


class AggregationStrategy(StrEnum):
    GEOMETRIC_MEAN = "geometric_mean"
    ARITHMETIC_MEAN = "arithmetic_mean"
    MINIMUM = "minimum"
    PRODUCT = "product"
    LENGTH_NORMALIZED_PRODUCT = "length_normalized_product"


def select_answer_tokens(
    generation: Generation,
    answer_start: int | None = None,
    answer_end: int | None = None,
) -> tuple[TokenProbability, ...]:
    tokens = generation.token_probabilities
    if answer_start is None and answer_end is None:
        return tokens
    selected = []
    for token in tokens:
        start = token.start_char if token.start_char is not None else 0
        end = token.end_char if token.end_char is not None else start + len(token.token)
        if answer_start is not None and end <= answer_start:
            continue
        if answer_end is not None and start >= answer_end:
            continue
        selected.append(token)
    return tuple(selected)


def aggregate_token_probabilities(
    tokens: tuple[TokenProbability, ...],
    strategy: AggregationStrategy = AggregationStrategy.GEOMETRIC_MEAN,
) -> float | None:
    if not tokens:
        return None
    logprobs = [float(token.logprob) for token in tokens]
    probabilities = [clamp_probability(float(token.probability or math.exp(token.logprob))) for token in tokens]
    if strategy == AggregationStrategy.GEOMETRIC_MEAN:
        return clamp_probability(math.exp(sum(logprobs) / len(logprobs)))
    if strategy == AggregationStrategy.ARITHMETIC_MEAN:
        return clamp_probability(sum(probabilities) / len(probabilities))
    if strategy == AggregationStrategy.MINIMUM:
        return clamp_probability(min(probabilities))
    if strategy == AggregationStrategy.PRODUCT:
        return clamp_probability(math.exp(sum(logprobs)))
    if strategy == AggregationStrategy.LENGTH_NORMALIZED_PRODUCT:
        length = max(1, sum(max(1, len(token.token.strip())) for token in tokens))
        return clamp_probability(math.exp(sum(logprobs) / math.sqrt(length)))
    raise ValueError(f"Unknown aggregation strategy {strategy}")


def answer_logprob_confidence(
    generation: Generation,
    strategy: AggregationStrategy = AggregationStrategy.GEOMETRIC_MEAN,
    answer_start: int | None = None,
    answer_end: int | None = None,
) -> float | None:
    return aggregate_token_probabilities(select_answer_tokens(generation, answer_start, answer_end), strategy)
