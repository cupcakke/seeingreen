import math

import pytest

from epistemic_uq.errors import ParsingError
from epistemic_uq.schemas import Generation, TokenProbability, Usage
from epistemic_uq.uncertainty.logprob import AggregationStrategy, aggregate_token_probabilities
from epistemic_uq.uncertainty.self_report import parse_self_report, parse_truth_probability
from epistemic_uq.utils import utc_now


def test_parse_json_confidence() -> None:
    parsed = parse_self_report('{"confidence":0.83}')
    assert parsed.value == 0.83


def test_parse_percent_confidence() -> None:
    parsed = parse_self_report("Confidence: 75%")
    assert parsed.value == 0.75


def test_parse_truth_probability() -> None:
    assert parse_truth_probability('{"correct_probability":0.91,"incorrect_probability":0.09}') == 0.91


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ParsingError):
        parse_self_report("unknown")


def test_geometric_logprob_aggregation() -> None:
    tokens = (
        TokenProbability(token="a", logprob=math.log(0.8), probability=0.8, position=0),
        TokenProbability(token="b", logprob=math.log(0.2), probability=0.2, position=1),
    )
    value = aggregate_token_probabilities(tokens, AggregationStrategy.GEOMETRIC_MEAN)
    assert value is not None
    assert math.isclose(value, 0.4)


def test_empty_logprobs_return_none() -> None:
    assert aggregate_token_probabilities(()) is None
