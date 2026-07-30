from pathlib import Path

from epistemic_uq.reports import ReportGenerator
from epistemic_uq.schemas import CanonicalEvaluationUnit, TaskType


def test_single_example_end_to_end(runner) -> None:
    example = CanonicalEvaluationUnit(
        example_id="math",
        dataset_id="d",
        task_type=TaskType.QUESTION_ANSWERING,
        user_input="What is 8 + 9?",
        expected_format="number",
        reference_label=17,
        subgroup_metadata={"domain": "arithmetic"},
        validator_config={"method": "numeric"},
    )
    results = runner.evaluate_example(example, ("reference-local",))
    assert len(results) == 1
    result = results[0]
    assert result.answer.canonical == "17"
    assert result.evaluation is not None and result.evaluation.correct
    assert result.features.self_consistency_confidence == 1.0
    assert result.features.perturbation_stability == 1.0
    assert Path(result.audit_reference).exists()


def test_dataset_run_resume_and_report(tmp_path: Path, runner, repository) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"example_id":"m1","dataset_id":"d","task_type":"question_answering","user_input":"What is 4 + 6?","expected_format":"number","reference_label":10,"valid_answers":[],"subgroup_metadata":{"domain":"arithmetic"},"perturbation_rules":{},"validator_config":{"method":"numeric"},"criticality":"low","metadata":{}}\n',
        encoding="utf-8",
    )
    run = runner.run_dataset(dataset, "d", ("reference-local",))
    assert run.status.value == "completed"
    resumed = runner.run_dataset(dataset, "d", ("reference-local",), resume_experiment_id=run.experiment_id)
    assert resumed.experiment_id == run.experiment_id
    paths = ReportGenerator(repository).generate(run.experiment_id, tmp_path / "report")
    assert Path(paths["json"]).exists()
    assert Path(paths["html"]).exists()
