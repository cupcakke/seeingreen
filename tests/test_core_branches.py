from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from epistemic_uq.calibration.metrics import (
    calibration_bins,
    calibration_report,
    optimize_threshold,
    selective_curve,
    threshold_for_target_coverage,
)
from epistemic_uq.policy import DecisionPolicy
from epistemic_uq.processing.normalization import DatasetLoader, canonicalize_answer, extract_json
from epistemic_uq.processing.perturbation import PerturbationEngine
from epistemic_uq.processing.validators import evaluate_answer, regex_match, structured_match
from epistemic_uq.schemas import (
    CanonicalEvaluationUnit,
    Criticality,
    DecisionAction,
    ExtractedAnswer,
    Generation,
    PolicyConfig,
    Prompt,
    TaskType,
    TokenProbability,
    UncertaintyFeatures,
    Usage,
)
from epistemic_uq.uncertainty.agreement import agreement_statistics, cross_group_agreement, normalized_entropy
from epistemic_uq.uncertainty.composite import EpistemicRiskEstimator
from epistemic_uq.uncertainty.logprob import (
    AggregationStrategy,
    aggregate_token_probabilities,
    answer_logprob_confidence,
    select_answer_tokens,
)
from epistemic_uq.utils import chunked, deep_merge, deterministic_random, elapsed_ms, jsonable, stable_hash, utc_timestamp


def minimal_features(contradiction: bool = False) -> UncertaintyFeatures:
    return UncertaintyFeatures(
        contradiction=contradiction,
        model_knowledge_uncertainty=0.5,
        prompt_sensitivity_uncertainty=0.5,
        decoding_instability_uncertainty=0.5,
        epistemic_risk=0.5,
    )


def test_metrics_validation_and_objectives() -> None:
    with pytest.raises(ValueError):
        calibration_report([], [])
    with pytest.raises(ValueError):
        calibration_report([0.2], [0, 1])
    with pytest.raises(ValueError):
        calibration_report([1.2], [1])
    with pytest.raises(ValueError):
        calibration_report([0.2], [2])
    with pytest.raises(ValueError):
        calibration_bins([0.2], [0], n_bins=0)
    with pytest.raises(ValueError):
        calibration_bins([0.2], [0], strategy="invalid")
    report = calibration_report([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1], n_bins=8, strategy="uniform")
    assert len(report.bins) == 8
    assert report.auroc == 1.0
    single_class = calibration_report([0.1, 0.2], [0, 0])
    assert single_class.auroc is None
    assert threshold_for_target_coverage([0.1, 0.4, 0.9], 0.0) == 1.0
    assert 0.0 <= threshold_for_target_coverage([0.1, 0.4, 0.9], 2 / 3) <= 1.0
    with pytest.raises(ValueError):
        threshold_for_target_coverage([], 0.5)
    with pytest.raises(ValueError):
        threshold_for_target_coverage([0.5], 1.5)
    target = optimize_threshold([0.9, 0.7, 0.4, 0.1], [1, 0, 1, 0], objective="target_coverage", target_coverage=0.5)
    assert abs(target.coverage - 0.5) <= 0.25
    utility = optimize_threshold(
        [0.9, 0.7, 0.4, 0.1],
        [1, 0, 1, 0],
        objective="utility",
        correct_utility=2.0,
        incorrect_utility=-3.0,
        abstain_utility=0.1,
    )
    assert 0.0 <= utility.coverage <= 1.0
    infeasible = optimize_threshold([0.9, 0.8], [0, 0], objective="minimum_risk", max_risk=0.0)
    assert infeasible.risk == 1.0


def test_dataset_loader_all_formats(tmp_path: Path) -> None:
    record = {
        "example_id": "x",
        "dataset_id": "d",
        "task_type": "classification",
        "user_input": "Classify",
        "reference_label": "yes",
        "valid_answers": [],
        "subgroup_metadata": {},
        "perturbation_rules": {},
        "validator_config": {},
        "criticality": "low",
        "metadata": {},
    }
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps({"examples": [record]}), encoding="utf-8")
    assert DatasetLoader().load(json_path)[0].example_id == "x"
    array_path = tmp_path / "array.json"
    array_path.write_text(json.dumps([record]), encoding="utf-8")
    assert len(DatasetLoader().load(array_path)) == 1
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "example_id,dataset_id,task_type,user_input,reference_label,valid_answers,subgroup_metadata,perturbation_rules,validator_config,criticality,metadata\n"
        'x,d,classification,Classify,yes,[],{},{},{},low,{}\n',
        encoding="utf-8",
    )
    assert len(DatasetLoader().load(csv_path)) == 1
    unsupported = tmp_path / "data.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        DatasetLoader().load(unsupported)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"not_examples": True}), encoding="utf-8")
    with pytest.raises(ValueError):
        DatasetLoader().load(bad_json)
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        DatasetLoader().load(duplicate)


def test_canonicalization_invalid_structured_and_json() -> None:
    answer = canonicalize_answer("not json", TaskType.STRUCTURED)
    assert not answer.valid
    with pytest.raises(json.JSONDecodeError):
        extract_json("not json")
    classification = canonicalize_answer("YES!", TaskType.CLASSIFICATION)
    extraction = canonicalize_answer("Value: Alpha", TaskType.EXTRACTION)
    assert classification.canonical == "yes"
    assert extraction.canonical == "value alpha"


def test_perturbation_all_transforms_and_unknown() -> None:
    example = CanonicalEvaluationUnit(
        example_id="p",
        dataset_id="d",
        task_type=TaskType.CLASSIFICATION,
        user_input="Provide the following answer.\n\nChoose a label.",
        expected_format="label",
    )
    prompt = Prompt(user=example.user_input, template_id="base", template_version="1")
    family = PerturbationEngine(seed=3).build_family(
        example,
        prompt,
        ["instruction_order", "formatting", "task_framing", "lexical"],
        10,
    )
    assert {item.transform for item in family.variants} == {
        "instruction_order",
        "formatting",
        "task_framing",
        "lexical",
    }
    short_prompt = Prompt(user="Choose the answer", template_id="base", template_version="1")
    short_family = PerturbationEngine().build_family(example, short_prompt, ["instruction_order"], 1)
    assert len(short_family.variants) == 1
    with pytest.raises(ValueError):
        PerturbationEngine().build_family(example, prompt, ["unknown"], 1)


def test_validator_branches() -> None:
    assert regex_match("ABC-123", [r"[A-Z]+-\d+"])
    structured_answer = canonicalize_answer('{"a":1,"b":2}', TaskType.STRUCTURED)
    assert structured_match(structured_answer, [{"a": 1, "b": 2}], ["a"])
    assert not structured_match(structured_answer, [{"a": 1}], ["c"])
    invalid = ExtractedAnswer(raw="x", canonical="x", parser="json", valid=False)
    assert not structured_match(invalid, [{}], [])
    regex_example = CanonicalEvaluationUnit(
        example_id="r",
        dataset_id="d",
        task_type=TaskType.EXTRACTION,
        user_input="extract",
        valid_answers=(r"[A-Z]{3}",),
        validator_config={"method": "regex"},
    )
    assert evaluate_answer(regex_example, canonicalize_answer("ABC", TaskType.EXTRACTION)).correct
    token_example = CanonicalEvaluationUnit(
        example_id="f",
        dataset_id="d",
        task_type=TaskType.QUESTION_ANSWERING,
        user_input="answer",
        valid_answers=("blue car",),
        validator_config={"method": "token_f1", "threshold": 0.5},
    )
    result = evaluate_answer(token_example, canonicalize_answer("blue vehicle", TaskType.QUESTION_ANSWERING))
    assert result is not None and result.correct
    unlabeled = CanonicalEvaluationUnit(
        example_id="u",
        dataset_id="d",
        task_type=TaskType.QUESTION_ANSWERING,
        user_input="answer",
    )
    assert evaluate_answer(unlabeled, canonicalize_answer("x", TaskType.QUESTION_ANSWERING)) is None


def test_policy_every_action() -> None:
    config = PolicyConfig(
        answer_threshold=0.8,
        warning_threshold=0.6,
        clarification_threshold=0.45,
        external_verification_threshold=0.35,
        criticality_adjustments={"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0},
    )
    policy = DecisionPolicy(config)
    assert policy.decide(0.9, minimal_features(), Criticality.LOW).action == DecisionAction.ANSWER
    assert policy.decide(0.7, minimal_features(), Criticality.LOW).action == DecisionAction.WARN
    assert policy.decide(0.5, minimal_features(), Criticality.LOW).action == DecisionAction.CLARIFY
    assert policy.decide(0.4, minimal_features(), Criticality.LOW).action == DecisionAction.VERIFY
    assert policy.decide(0.2, minimal_features(), Criticality.LOW).action == DecisionAction.ABSTAIN
    assert policy.decide(0.2, minimal_features(), Criticality.HIGH).action == DecisionAction.ESCALATE


def test_logprob_strategies_and_selection() -> None:
    tokens = (
        TokenProbability(token="aa", logprob=math.log(0.8), probability=0.8, position=0, start_char=0, end_char=2),
        TokenProbability(token="bb", logprob=math.log(0.5), probability=0.5, position=1, start_char=2, end_char=4),
    )
    assert aggregate_token_probabilities(tokens, AggregationStrategy.ARITHMETIC_MEAN) == pytest.approx(0.65)
    assert aggregate_token_probabilities(tokens, AggregationStrategy.MINIMUM) == pytest.approx(0.5)
    assert aggregate_token_probabilities(tokens, AggregationStrategy.PRODUCT) == pytest.approx(0.4)
    length_normalized = aggregate_token_probabilities(tokens, AggregationStrategy.LENGTH_NORMALIZED_PRODUCT)
    assert length_normalized is not None and 0.0 < length_normalized < 1.0
    generation = Generation(
        generation_id="g",
        request_id="r",
        backend_id="b",
        model="m",
        text="aabb",
        token_probabilities=tokens,
        usage=Usage(),
        latency_ms=1.0,
        created_at=np.datetime64("2026-01-01").astype("datetime64[ms]").astype(object),
    )
    assert select_answer_tokens(generation, answer_start=2, answer_end=4) == (tokens[1],)
    assert answer_logprob_confidence(generation, answer_start=2, answer_end=4) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        aggregate_token_probabilities(tokens, "invalid")


def test_agreement_empty_cross_group_and_composite() -> None:
    from epistemic_uq.uncertainty.semantic import SemanticAdjudicator

    adjudicator = SemanticAdjudicator()
    stats, clusters = agreement_statistics((), (), adjudicator)
    assert stats.normalized_entropy == 1.0
    assert clusters == ()
    yes = ExtractedAnswer(raw="yes", canonical="yes", parser="x")
    no = ExtractedAnswer(raw="no", canonical="no", parser="x")
    assert cross_group_agreement(((yes,), (yes,), (no,)), adjudicator) == pytest.approx(1 / 3)
    assert cross_group_agreement(((yes,),), adjudicator) is None
    assert normalized_entropy([1.0]) == 0.0
    features = EpistemicRiskEstimator().estimate(
        self_report_confidence=0.9,
        logprob_confidence=0.8,
        truth_confidence=0.7,
        self_consistency_confidence=0.6,
        perturbation_stability=0.5,
        cross_model_agreement=0.4,
        semantic_agreement=0.6,
        semantic_entropy=0.7,
        contradiction=True,
        calibration_residual=0.4,
        local_overconfidence=0.3,
    )
    assert features.epistemic_risk > 0.5


def test_utils_branches() -> None:
    assert stable_hash({"a": 1}) == stable_hash({"a": 1})
    assert deterministic_random(1, "x").random() == deterministic_random(1, "x").random()
    assert deep_merge({"a": {"b": 1}, "c": 1}, {"a": {"d": 2}, "c": 2}) == {"a": {"b": 1, "d": 2}, "c": 2}
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        list(chunked([1], 0))
    start = __import__("time").perf_counter()
    assert elapsed_ms(start) >= 0.0
    assert isinstance(utc_timestamp(), str)
    assert jsonable({"x": np.array([1, 2])}) == {"x": [1, 2]}
