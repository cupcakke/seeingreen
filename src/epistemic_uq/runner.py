from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from epistemic_uq.backends.base import ModelBackend
from epistemic_uq.backends.registry import BackendRegistry
from epistemic_uq.calibration.calibrators import calibrator_from_artifact
from epistemic_uq.calibration.fusion import RuleBasedFusion, TransparentFusionModel
from epistemic_uq.logging import bind_trace, reset_trace
from epistemic_uq.monitoring import MODEL_CALLS, MODEL_LATENCY, MODEL_TOKENS, PARSING_FAILURES, POLICY_ACTIONS, DriftMonitor
from epistemic_uq.policy import DecisionPolicy
from epistemic_uq.processing.normalization import DatasetLoader, canonicalize_answer
from epistemic_uq.processing.perturbation import PerturbationEngine
from epistemic_uq.processing.validators import evaluate_answer
from epistemic_uq.schemas import (
    BackendConfig,
    CanonicalEvaluationUnit,
    DatasetManifest,
    ExperimentRun,
    ExperimentStatus,
    ExtractedAnswer,
    ModelRequest,
    PolicyConfig,
    Prompt,
    SelfReportedConfidence,
    UncertaintyResult,
)
from epistemic_uq.storage import Repository
from epistemic_uq.uncertainty.agreement import agreement_statistics
from epistemic_uq.uncertainty.composite import EpistemicRiskEstimator
from epistemic_uq.uncertainty.logprob import AggregationStrategy, answer_logprob_confidence
from epistemic_uq.uncertainty.self_report import parse_self_report, parse_truth_probability
from epistemic_uq.uncertainty.semantic import SemanticAdjudicator, SemanticConfig
from epistemic_uq.utils import deep_merge, stable_hash, trace_id, utc_now


logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        repository: Repository,
        registry: BackendRegistry,
        config: dict[str, Any],
        drift_monitor: DriftMonitor | None = None,
    ) -> None:
        self.repository = repository
        self.registry = registry
        self.config = config
        semantic_config = config.get("semantic", {})
        self.adjudicator = SemanticAdjudicator(
            SemanticConfig(
                embedding_model=semantic_config.get("embedding_model"),
                cosine_threshold=float(semantic_config.get("cosine_threshold", 0.82)),
                numeric_absolute_tolerance=float(semantic_config.get("numeric_absolute_tolerance", 1e-6)),
                numeric_relative_tolerance=float(semantic_config.get("numeric_relative_tolerance", 1e-6)),
            )
        )
        self.perturbation_engine = PerturbationEngine(seed=int(config.get("execution", {}).get("seed", 1729)))
        self.risk_estimator = EpistemicRiskEstimator()
        self.rule_fusion = RuleBasedFusion(config.get("fusion", {}).get("weights"))
        policy_config = PolicyConfig.model_validate(config.get("policy", {}))
        self.policy = DecisionPolicy(policy_config)
        observability = config.get("observability", {})
        self.drift_monitor = drift_monitor or DriftMonitor(
            baseline_window=int(observability.get("baseline_window", 1000)),
            current_window=int(observability.get("drift_window", 200)),
            psi_threshold=float(observability.get("psi_threshold", 0.2)),
            calibration_threshold=float(observability.get("calibration_alarm_threshold", 0.08)),
        )
        self.calibrator = self._load_calibrator(config)
        self.fusion_model = self._load_fusion(config)

    def _load_calibrator(self, config: dict[str, Any]):
        calibrator_id = config.get("calibration", {}).get("calibrator_id")
        if not calibrator_id:
            return None
        artifact = self.repository.get_calibration(str(calibrator_id))
        return calibrator_from_artifact(artifact)

    def _load_fusion(self, config: dict[str, Any]) -> TransparentFusionModel | None:
        fusion_id = config.get("fusion", {}).get("model_id")
        if not fusion_id:
            return None
        artifact = self.repository.get_calibration(str(fusion_id))
        parameters = artifact.get("parameters", {})
        model = TransparentFusionModel(
            feature_names=tuple(parameters["feature_names"]),
            monotonic=bool(parameters.get("monotonic", True)),
        )
        import numpy as np

        model.coefficients = np.asarray(parameters["coefficients"], dtype=float)
        model.intercept = float(parameters["intercept"])
        model.means = np.asarray(parameters["means"], dtype=float)
        model.scales = np.asarray(parameters["scales"], dtype=float)
        model.missing_values = np.asarray(parameters["missing_values"], dtype=float)
        model.fitted = True
        return model

    def register_dataset(self, path: Path, dataset_id: str, version: str) -> DatasetManifest:
        examples = DatasetLoader().load(path, dataset_id=dataset_id)
        payload = [example.model_dump(mode="json") for example in examples]
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            content_hash=stable_hash(payload),
            example_count=len(examples),
            schema_version=1,
            created_at=utc_now(),
            source_path=str(path.resolve()),
            metadata={"format": path.suffix.casefold()},
        )
        self.repository.register_dataset(manifest)
        self.repository.artifacts.write(f"datasets/{dataset_id}", payload)
        return manifest

    def register_backend(self, config: BackendConfig) -> None:
        self.repository.register_backend(config)
        self.registry.register(config)

    def load_registered_backends(self, backend_ids: tuple[str, ...]) -> None:
        registered = {backend.config.backend_id for backend in self.registry.all()}
        for backend_id in backend_ids:
            if backend_id not in registered:
                self.registry.register(self.repository.get_backend(backend_id))

    def _baseline_prompt(self, example: CanonicalEvaluationUnit) -> Prompt:
        format_instruction = ""
        if example.expected_format:
            format_instruction = f"\n\nReturn the answer in this format: {example.expected_format}."
        return Prompt(
            system="Follow the user task precisely. Return only the requested answer without confidence commentary.",
            user=f"{example.user_input}{format_instruction}",
            template_id="baseline",
            template_version="1.0.0",
            variables={
                "example_id": example.example_id,
                "task_type": example.task_type.value,
                "expected_format": example.expected_format,
            },
        )

    def _model_request(
        self,
        example: CanonicalEvaluationUnit,
        backend: ModelBackend,
        prompt: Prompt,
        purpose: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed_offset: int = 0,
        logprobs: bool = False,
    ) -> ModelRequest:
        base_seed = int(self.config.get("execution", {}).get("seed", 1729))
        seed = base_seed + seed_offset if backend.capabilities.seed else None
        return ModelRequest(
            request_id=f"{example.example_id}:{backend.config.backend_id}:{purpose}:{trace_id()}",
            prompt=prompt,
            model=backend.config.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs and backend.capabilities.logprobs,
            metadata={
                "example_id": example.example_id,
                "dataset_id": example.dataset_id,
                "purpose": purpose,
                "task_type": example.task_type.value,
                "subgroup_metadata": example.subgroup_metadata,
                "criticality": example.criticality.value,
            },
        )

    def _record_generation(self, backend_id: str, generation) -> None:
        MODEL_CALLS.labels(backend=backend_id, status="success").inc()
        MODEL_LATENCY.labels(backend=backend_id).observe(generation.latency_ms / 1000.0)
        MODEL_TOKENS.labels(backend=backend_id, kind="prompt").inc(generation.usage.prompt_tokens)
        MODEL_TOKENS.labels(backend=backend_id, kind="completion").inc(generation.usage.completion_tokens)

    def _call(self, backend: ModelBackend, request: ModelRequest):
        try:
            generation = backend.generate(request)
        except Exception:
            MODEL_CALLS.labels(backend=backend.config.backend_id, status="error").inc()
            raise
        self._record_generation(backend.config.backend_id, generation)
        return generation

    def _parse_confidence(self, text: str) -> SelfReportedConfidence | None:
        try:
            return parse_self_report(text)
        except Exception:
            PARSING_FAILURES.labels(parser="self_report").inc()
            logger.warning("self_report_parse_failed", extra={"text": text[:500]})
            return None

    def _parse_truth(self, text: str) -> float | None:
        try:
            return parse_truth_probability(text)
        except Exception:
            PARSING_FAILURES.labels(parser="truth_probability").inc()
            logger.warning("truth_probability_parse_failed", extra={"text": text[:500]})
            return None

    def _generate_baselines(
        self,
        example: CanonicalEvaluationUnit,
        backend_ids: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        generation_config = self.config.get("sampling", {})
        max_tokens = int(generation_config.get("max_tokens", 512))
        baseline_prompt = self._baseline_prompt(example)
        baselines: dict[str, dict[str, Any]] = {}
        for backend_id in backend_ids:
            backend = self.registry.get(backend_id)
            request = self._model_request(
                example=example,
                backend=backend,
                prompt=baseline_prompt,
                purpose="baseline",
                temperature=float(generation_config.get("baseline_temperature", 0.0)),
                top_p=1.0,
                max_tokens=max_tokens,
                logprobs=True,
            )
            generation = self._call(backend, request)
            answer = canonicalize_answer(generation.text, example.task_type, example.expected_format)
            baselines[backend_id] = {
                "backend": backend,
                "request": request,
                "generation": generation,
                "answer": answer,
                "prompt": baseline_prompt,
            }
        return baselines

    def _cross_model_agreement(
        self,
        backend_id: str,
        baselines: dict[str, dict[str, Any]],
    ) -> float | None:
        others = [value["answer"] for key, value in baselines.items() if key != backend_id]
        if not others:
            return None
        source = baselines[backend_id]["answer"]
        return sum(self.adjudicator.equivalent(source, other) for other in others) / len(others)

    def _evaluate_backend(
        self,
        example: CanonicalEvaluationUnit,
        backend_id: str,
        baselines: dict[str, dict[str, Any]],
    ) -> UncertaintyResult:
        baseline = baselines[backend_id]
        backend: ModelBackend = baseline["backend"]
        request: ModelRequest = baseline["request"]
        generation = baseline["generation"]
        answer: ExtractedAnswer = baseline["answer"]
        uncertainty_config = self.config.get("uncertainty", {})
        sampling_config = self.config.get("sampling", {})
        raw_artifacts: dict[str, Any] = {
            "baseline": generation.model_dump(mode="json"),
            "baseline_request": request.model_dump(mode="json"),
        }
        self_report = None
        if bool(uncertainty_config.get("self_report", True)) and backend.capabilities.forced_confidence:
            try:
                confidence_generation = backend.forced_confidence(request, generation.text)
            except Exception:
                MODEL_CALLS.labels(backend=backend_id, status="error").inc()
                raise
            self._record_generation(backend_id, confidence_generation)
            self_report = self._parse_confidence(confidence_generation.text)
            raw_artifacts["self_report_generation"] = confidence_generation.model_dump(mode="json")
            raw_artifacts["self_report"] = self_report.model_dump(mode="json") if self_report else None
        truth_confidence = None
        if bool(uncertainty_config.get("truth_verification", True)):
            try:
                truth_generation = backend.truth_verification(request, generation.text)
            except Exception:
                MODEL_CALLS.labels(backend=backend_id, status="error").inc()
                raise
            self._record_generation(backend_id, truth_generation)
            truth_confidence = self._parse_truth(truth_generation.text)
            raw_artifacts["truth_generation"] = truth_generation.model_dump(mode="json")
            raw_artifacts["truth_confidence"] = truth_confidence
        logprob_confidence = answer_logprob_confidence(
            generation,
            strategy=AggregationStrategy(str(uncertainty_config.get("logprob_strategy", "geometric_mean"))),
        )
        sample_count = int(sampling_config.get("sample_count", 8))
        sampled_generations = ()
        sampled_answers: tuple[ExtractedAnswer, ...] = ()
        sample_confidences: tuple[float | None, ...] = ()
        sample_statistics = None
        clusters = ()
        if bool(uncertainty_config.get("sampling_agreement", True)) and sample_count > 0:
            sample_request = self._model_request(
                example=example,
                backend=backend,
                prompt=baseline["prompt"],
                purpose="self_consistency",
                temperature=float(sampling_config.get("temperature", 0.7)),
                top_p=float(sampling_config.get("top_p", 0.95)),
                max_tokens=int(sampling_config.get("max_tokens", 512)),
                seed_offset=10000,
                logprobs=True,
            )
            sampled_generations = backend.sample(sample_request, sample_count)
            for sampled_generation in sampled_generations:
                self._record_generation(backend_id, sampled_generation)
            sampled_answers = tuple(
                canonicalize_answer(item.text, example.task_type, example.expected_format)
                for item in sampled_generations
            )
            sample_confidences = tuple(answer_logprob_confidence(item) for item in sampled_generations)
            sample_statistics, clusters = agreement_statistics(
                sampled_answers,
                tuple(item.generation_id for item in sampled_generations),
                self.adjudicator,
                high_confidence=sample_confidences,
                contradiction_threshold=float(uncertainty_config.get("contradiction_threshold", 0.75)),
            )
            raw_artifacts["sample_request"] = sample_request.model_dump(mode="json")
            raw_artifacts["sampled_generations"] = [item.model_dump(mode="json") for item in sampled_generations]
            raw_artifacts["sampled_answers"] = [item.model_dump(mode="json") for item in sampled_answers]
            raw_artifacts["semantic_clusters"] = [item.model_dump(mode="json") for item in clusters]
            raw_artifacts["sample_statistics"] = sample_statistics.model_dump(mode="json")
        perturbation_stability = None
        perturbation_outputs: list[dict[str, Any]] = []
        perturbation_config = self.config.get("perturbation", {})
        if bool(uncertainty_config.get("prompt_perturbation", True)) and bool(perturbation_config.get("enabled", True)):
            family = self.perturbation_engine.build_family(
                example,
                baseline["prompt"],
                tuple(perturbation_config.get("transforms", [])),
                int(perturbation_config.get("max_variants", 6)),
            )
            equivalent = 0
            for index, perturbation in enumerate(family.variants):
                perturbation_request = self._model_request(
                    example=example,
                    backend=backend,
                    prompt=perturbation.prompt,
                    purpose=f"perturbation:{perturbation.transform}",
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(sampling_config.get("max_tokens", 512)),
                    seed_offset=20000 + index,
                    logprobs=False,
                )
                perturbation_generation = self._call(backend, perturbation_request)
                perturbation_answer = canonicalize_answer(
                    perturbation_generation.text,
                    example.task_type,
                    example.expected_format,
                )
                is_equivalent = self.adjudicator.equivalent(answer, perturbation_answer)
                equivalent += int(is_equivalent)
                perturbation_outputs.append(
                    {
                        "perturbation": perturbation.model_dump(mode="json"),
                        "request": perturbation_request.model_dump(mode="json"),
                        "generation": perturbation_generation.model_dump(mode="json"),
                        "answer": perturbation_answer.model_dump(mode="json"),
                        "equivalent_to_baseline": is_equivalent,
                    }
                )
            perturbation_stability = equivalent / len(family.variants) if family.variants else None
            raw_artifacts["perturbation_family"] = family.model_dump(mode="json")
            raw_artifacts["perturbation_outputs"] = perturbation_outputs
        cross_model_agreement = (
            self._cross_model_agreement(backend_id, baselines)
            if bool(uncertainty_config.get("cross_model", True))
            else None
        )
        self_consistency_confidence = sample_statistics.dominant_mass if sample_statistics else None
        semantic_agreement = sample_statistics.semantic_agreement if sample_statistics else None
        semantic_entropy = sample_statistics.normalized_entropy if sample_statistics else None
        contradiction = sample_statistics.contradiction if sample_statistics else False
        features = self.risk_estimator.estimate(
            self_report_confidence=self_report.value if self_report else None,
            logprob_confidence=logprob_confidence,
            truth_confidence=truth_confidence,
            self_consistency_confidence=self_consistency_confidence,
            perturbation_stability=perturbation_stability,
            cross_model_agreement=cross_model_agreement,
            semantic_agreement=semantic_agreement,
            semantic_entropy=semantic_entropy,
            contradiction=contradiction,
            raw={
                "sample_count": len(sampled_generations),
                "cluster_count": len(clusters),
                "perturbation_count": len(perturbation_outputs),
            },
        )
        feature_row = features.model_dump(mode="python")
        feature_row["semantic_entropy_inverse"] = 1.0 - features.semantic_entropy if features.semantic_entropy is not None else None
        feature_row["contradiction_inverse"] = 0.0 if features.contradiction else 1.0
        if self.fusion_model:
            fusion_prediction = self.fusion_model.explain(feature_row)
        else:
            fusion_prediction = self.rule_fusion.predict_one(feature_row)
        calibrated_confidence = fusion_prediction.probability
        if self.calibrator:
            calibrated_confidence = float(self.calibrator.predict([calibrated_confidence])[0])
        evaluation = evaluate_answer(example, answer)
        decision = self.policy.decide(
            calibrated_confidence=calibrated_confidence,
            features=features,
            criticality=example.criticality,
            subgroup_metadata=example.subgroup_metadata,
        )
        POLICY_ACTIONS.labels(action=decision.action.value, task_type=example.task_type.value).inc()
        raw_artifacts["cross_model_baselines"] = {
            key: {
                "answer": value["answer"].model_dump(mode="json"),
                "generation_id": value["generation"].generation_id,
            }
            for key, value in baselines.items()
        }
        raw_artifacts["features"] = features.model_dump(mode="json")
        raw_artifacts["fusion"] = {
            "probability": fusion_prediction.probability,
            "contributions": fusion_prediction.contributions,
            "calibrated_probability": calibrated_confidence,
        }
        raw_artifacts["evaluation"] = evaluation.model_dump(mode="json") if evaluation else None
        raw_artifacts["decision"] = decision.model_dump(mode="json")
        audit_reference = self.repository.artifacts.write(
            f"audit/{example.dataset_id}/{example.example_id}/{backend_id}",
            raw_artifacts,
        )
        snapshot = self.drift_monitor.observe(
            signal="calibrated_confidence",
            value=calibrated_confidence,
            task_type=example.task_type,
            subgroup=example.subgroup_metadata,
            label=int(evaluation.correct) if evaluation else None,
        )
        if snapshot:
            self.repository.save_drift_snapshot(
                signal=snapshot.signal,
                task_type=snapshot.task_type.value if snapshot.task_type else None,
                subgroup=snapshot.subgroup,
                snapshot=snapshot.model_dump(mode="json"),
            )
        return UncertaintyResult(
            example_id=example.example_id,
            backend_id=backend_id,
            answer=answer,
            baseline_generation=generation,
            features=features,
            calibrated_confidence=calibrated_confidence,
            decision=decision,
            evaluation=evaluation,
            audit_reference=audit_reference,
        )

    def evaluate_example(
        self,
        example: CanonicalEvaluationUnit,
        backend_ids: tuple[str, ...],
        existing_baselines: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[UncertaintyResult, ...]:
        trace = trace_id()
        token = bind_trace(trace)
        try:
            baselines = self._generate_baselines(example, backend_ids)
            baselines.update(existing_baselines or {})
            return tuple(self._evaluate_backend(example, backend_id, baselines) for backend_id in backend_ids)
        finally:
            reset_trace(token)

    def run_dataset(
        self,
        dataset_path: Path,
        dataset_id: str,
        backend_ids: tuple[str, ...],
        version: str = "1",
        overrides: dict[str, Any] | None = None,
        resume_experiment_id: str | None = None,
    ) -> ExperimentRun:
        effective_config = deep_merge(self.config, overrides or {})
        if effective_config != self.config:
            derived = ExperimentRunner(self.repository, self.registry, effective_config, self.drift_monitor)
            return derived.run_dataset(
                dataset_path=dataset_path,
                dataset_id=dataset_id,
                backend_ids=backend_ids,
                version=version,
                resume_experiment_id=resume_experiment_id,
            )
        manifest = self.register_dataset(dataset_path, dataset_id, version)
        examples = DatasetLoader().load(dataset_path, dataset_id=dataset_id)
        self.load_registered_backends(backend_ids)
        if resume_experiment_id:
            run = self.repository.get_experiment(resume_experiment_id)
            if run.dataset_hash != manifest.content_hash:
                raise ValueError("Resume dataset hash does not match the original experiment")
            if run.config_hash != stable_hash(self.config):
                raise ValueError("Resume configuration hash does not match the original experiment")
        else:
            experiment_id = trace_id()
            manifest_payload = {
                "experiment_id": experiment_id,
                "dataset": manifest.model_dump(mode="json"),
                "backends": [self.registry.get(backend_id).config.model_dump(mode="json") for backend_id in backend_ids],
                "config": self.config,
                "created_at": utc_now().isoformat(),
            }
            manifest_path = self.repository.artifacts.write(f"experiments/{experiment_id}/manifest", manifest_payload)
            run = ExperimentRun(
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                backend_ids=backend_ids,
                config_hash=stable_hash(self.config),
                dataset_hash=manifest.content_hash,
                status=ExperimentStatus.CREATED,
                created_at=utc_now(),
                manifest_path=manifest_path,
                metadata={"version": version},
            )
            self.repository.create_experiment(run)
        self.repository.update_experiment_status(run.experiment_id, ExperimentStatus.RUNNING)
        max_workers = int(self.config.get("execution", {}).get("max_workers", 4))
        failures: list[dict[str, Any]] = []

        def execute(example: CanonicalEvaluationUnit) -> tuple[UncertaintyResult, ...]:
            pending = [
                backend_id
                for backend_id in backend_ids
                if self.repository.begin_result(run.experiment_id, example.example_id, backend_id)
            ]
            if not pending:
                return ()
            existing_baselines: dict[str, dict[str, Any]] = {}
            for backend_id in backend_ids:
                if backend_id in pending:
                    continue
                record = self.repository.get_result(run.experiment_id, example.example_id, backend_id)
                if record and record.get("result"):
                    completed = UncertaintyResult.model_validate(record["result"])
                    existing_baselines[backend_id] = {
                        "generation": completed.baseline_generation,
                        "answer": completed.answer,
                    }
            return self.evaluate_example(example, tuple(pending), existing_baselines)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="euq-runner") as executor:
            futures: dict[Future[tuple[UncertaintyResult, ...]], CanonicalEvaluationUnit] = {
                executor.submit(execute, example): example for example in examples
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    results = future.result()
                    for result in results:
                        self.repository.complete_result(
                            run.experiment_id,
                            result.example_id,
                            result.backend_id,
                            result,
                        )
                except Exception as exc:
                    logger.exception("example_evaluation_failed", extra={"example_id": example.example_id})
                    failure = {"type": type(exc).__name__, "message": str(exc)}
                    failures.append({"example_id": example.example_id, **failure})
                    for backend_id in backend_ids:
                        self.repository.fail_result(run.experiment_id, example.example_id, backend_id, failure)
        final_status = ExperimentStatus.FAILED if failures else ExperimentStatus.COMPLETED
        self.repository.update_experiment_status(run.experiment_id, final_status)
        completed = self.repository.get_experiment(run.experiment_id)
        if failures:
            self.repository.artifacts.write(f"experiments/{run.experiment_id}/failures", failures)
        return completed
