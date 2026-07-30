from epistemic_uq.processing.normalization import canonicalize_answer, normalize_text, parse_number
from epistemic_uq.processing.validators import evaluate_answer, token_f1
from epistemic_uq.schemas import CanonicalEvaluationUnit, TaskType


def test_normalize_text_handles_articles_and_punctuation() -> None:
    assert normalize_text("  The, Quick! fox  ", remove_articles=True) == "quick fox"


def test_parse_percentage() -> None:
    assert float(parse_number("25%")) == 0.25


def test_structured_answer_canonicalization() -> None:
    answer = canonicalize_answer('```json\n{"b":2,"a":1}\n```', TaskType.STRUCTURED)
    assert answer.valid
    assert answer.canonical == '{"a":1,"b":2}'


def test_numeric_evaluation_with_tolerance() -> None:
    example = CanonicalEvaluationUnit(
        example_id="x",
        dataset_id="d",
        task_type=TaskType.QUESTION_ANSWERING,
        user_input="value",
        expected_format="number",
        reference_label=1.0,
        validator_config={"method": "numeric", "absolute_tolerance": 0.01},
    )
    answer = canonicalize_answer("1.005", TaskType.QUESTION_ANSWERING, "number")
    result = evaluate_answer(example, answer)
    assert result is not None and result.correct


def test_token_f1() -> None:
    assert token_f1("the blue car", "blue car") == 1.0
