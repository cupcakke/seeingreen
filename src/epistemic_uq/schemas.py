from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskType(StrEnum):
    QUESTION_ANSWERING = "question_answering"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    STRUCTURED = "structured"


class BackendType(StrEnum):
    HTTP = "http"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    SUBPROCESS = "subprocess"


class DecisionAction(StrEnum):
    ANSWER = "answer"
    WARN = "answer_with_warning"
    ABSTAIN = "abstain"
    CLARIFY = "request_clarification"
    VERIFY = "request_external_verification"
    ESCALATE = "escalate_human"


class Criticality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TokenProbability(FrozenModel):
    token: str
    logprob: float
    probability: float | None = None
    position: int = Field(ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def derive_probability(self) -> TokenProbability:
        if self.probability is None:
            import math

            object.__setattr__(self, "probability", float(math.exp(self.logprob)))
        return self


class Prompt(FrozenModel):
    system: str | None = None
    user: str
    template_id: str
    template_version: str
    variables: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(FrozenModel):
    request_id: str
    prompt: Prompt
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    seed: int | None = None
    stop: tuple[str, ...] = ()
    logprobs: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(FrozenModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    currency: str = "USD"


class Generation(FrozenModel):
    generation_id: str
    request_id: str
    backend_id: str
    model: str
    text: str
    finish_reason: str | None = None
    token_probabilities: tuple[TokenProbability, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(ge=0.0)
    created_at: datetime
    raw_response: dict[str, Any] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)


class ExtractedAnswer(FrozenModel):
    raw: str
    canonical: str
    value: Any = None
    parser: str
    valid: bool = True
    validation_errors: tuple[str, ...] = ()


class SelfReportedConfidence(FrozenModel):
    value: float = Field(ge=0.0, le=1.0)
    raw: str
    parser: str


class SampledResponse(FrozenModel):
    generation: Generation
    answer: ExtractedAnswer
    self_report: SelfReportedConfidence | None = None


class SampledResponseSet(FrozenModel):
    example_id: str
    backend_id: str
    prompt_variant_id: str
    samples: tuple[SampledResponse, ...]


class SemanticCluster(FrozenModel):
    cluster_id: str
    member_generation_ids: tuple[str, ...]
    canonical_representative: str
    mass: float = Field(ge=0.0, le=1.0)
    lexical_consistency: float = Field(ge=0.0, le=1.0)


class Perturbation(FrozenModel):
    perturbation_id: str
    transform: str
    prompt: Prompt
    parameters: dict[str, Any] = Field(default_factory=dict)


class PerturbationFamily(FrozenModel):
    example_id: str
    baseline: Prompt
    variants: tuple[Perturbation, ...]


class CalibrationBin(FrozenModel):
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    empirical_accuracy: float = Field(ge=0.0, le=1.0)
    absolute_gap: float = Field(ge=0.0, le=1.0)


class EvaluationLabel(FrozenModel):
    correct: bool
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    validator: str
    details: dict[str, Any] = Field(default_factory=dict)


class SubgroupAudit(FrozenModel):
    subgroup_key: str
    subgroup_value: str
    count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0)
    brier: float = Field(ge=0.0)
    overconfidence: float
    grouping_loss_proxy: float = Field(ge=0.0)
    risk_multiplier: float = Field(ge=0.0)


class AbstentionDecision(FrozenModel):
    action: DecisionAction
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    adjusted_confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...]
    policy_version: str


class CanonicalEvaluationUnit(FrozenModel):
    example_id: str
    dataset_id: str
    task_type: TaskType
    user_input: str
    expected_format: str | None = None
    reference_label: Any = None
    valid_answers: tuple[Any, ...] = ()
    subgroup_metadata: dict[str, str] = Field(default_factory=dict)
    perturbation_rules: dict[str, Any] = Field(default_factory=dict)
    validator_config: dict[str, Any] = Field(default_factory=dict)
    criticality: Criticality = Criticality.MEDIUM
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_reference(self) -> CanonicalEvaluationUnit:
        if self.reference_label is None and not self.valid_answers:
            return self
        return self


class BackendCapabilities(FrozenModel):
    logprobs: bool = False
    seed: bool = False
    multi_sample: bool = True
    forced_confidence: bool = True
    streaming: bool = False
    local: bool = False


class BackendConfig(FrozenModel):
    backend_id: str
    backend_type: BackendType
    model: str
    endpoint: str | None = None
    api_key_env: str | None = None
    command: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    retries: int = Field(default=3, ge=0)
    concurrency: int = Field(default=4, ge=1)
    pricing: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backend_fields(self) -> BackendConfig:
        if self.backend_type in {BackendType.HTTP, BackendType.OLLAMA} and not self.endpoint:
            raise ValueError("HTTP and Ollama backends require endpoint")
        if self.backend_type == BackendType.SUBPROCESS and not self.command:
            raise ValueError("Subprocess backend requires command")
        return self


class AgreementStatistics(FrozenModel):
    lexical_agreement: float = Field(ge=0.0, le=1.0)
    semantic_agreement: float = Field(ge=0.0, le=1.0)
    dominant_mass: float = Field(ge=0.0, le=1.0)
    normalized_entropy: float = Field(ge=0.0, le=1.0)
    contradiction: bool
    contradiction_pairs: tuple[tuple[str, str], ...] = ()


class UncertaintyFeatures(FrozenModel):
    self_report_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    logprob_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    truth_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    self_consistency_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    perturbation_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    cross_model_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction: bool = False
    model_knowledge_uncertainty: float = Field(ge=0.0, le=1.0)
    prompt_sensitivity_uncertainty: float = Field(ge=0.0, le=1.0)
    decoding_instability_uncertainty: float = Field(ge=0.0, le=1.0)
    epistemic_risk: float = Field(ge=0.0, le=1.0)
    raw: dict[str, Any] = Field(default_factory=dict)


class UncertaintyResult(FrozenModel):
    example_id: str
    backend_id: str
    answer: ExtractedAnswer
    baseline_generation: Generation
    features: UncertaintyFeatures
    calibrated_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: AbstentionDecision | None = None
    evaluation: EvaluationLabel | None = None
    audit_reference: str


class MetricSummary(FrozenModel):
    accuracy: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0)
    mce: float = Field(ge=0.0)
    brier: float = Field(ge=0.0)
    nll: float = Field(ge=0.0)
    auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    aurc: float = Field(ge=0.0)
    bins: tuple[CalibrationBin, ...]


class ExperimentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentRun(FrozenModel):
    experiment_id: str
    dataset_id: str
    backend_ids: tuple[str, ...]
    config_hash: str
    dataset_hash: str
    status: ExperimentStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    manifest_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(FrozenModel):
    dataset_id: str
    version: str
    content_hash: str
    example_count: int = Field(ge=0)
    schema_version: int = Field(ge=1)
    created_at: datetime
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationModelArtifact(FrozenModel):
    calibrator_id: str
    method: str
    feature_names: tuple[str, ...]
    training_dataset_hash: str
    fitted_at: datetime
    parameters: dict[str, Any]
    metrics: dict[str, float]
    task_type: TaskType | None = None
    subgroup: dict[str, str] = Field(default_factory=dict)


class DriftSnapshot(FrozenModel):
    signal: str
    task_type: TaskType | None = None
    subgroup: dict[str, str] = Field(default_factory=dict)
    reference_count: int = Field(ge=0)
    current_count: int = Field(ge=0)
    population_stability_index: float = Field(ge=0.0)
    mean_shift: float
    calibration_shift: float | None = None
    alarm: bool
    observed_at: datetime


class QueryRequest(FrozenModel):
    backend_ids: tuple[str, ...]
    task: CanonicalEvaluationUnit
    generation: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(FrozenModel):
    dataset_path: str
    backend_ids: tuple[str, ...]
    config_path: str | None = None


class ThresholdSimulationRequest(FrozenModel):
    confidences: tuple[float, ...]
    labels: tuple[int, ...]
    utilities: dict[str, Any] = Field(default_factory=dict)
    target_coverage: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def aligned(self) -> ThresholdSimulationRequest:
        if len(self.confidences) != len(self.labels):
            raise ValueError("Confidences and labels must have the same length")
        return self


class PolicyConfig(FrozenModel):
    answer_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    warning_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    clarification_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    external_verification_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    criticality_adjustments: dict[str, float] = Field(default_factory=dict)
    contradiction_action: DecisionAction = DecisionAction.ESCALATE
    policy_version: str = "1.0.0"

    @field_validator("criticality_adjustments")
    @classmethod
    def valid_adjustments(cls, value: dict[str, float]) -> dict[str, float]:
        for key, adjustment in value.items():
            if key not in {item.value for item in Criticality}:
                raise ValueError(f"Unknown criticality {key}")
            if not 0.0 <= adjustment <= 1.0:
                raise ValueError("Criticality adjustments must be probabilities")
        return value
