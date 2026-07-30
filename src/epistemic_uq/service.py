from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

try:
    import redis
except ImportError:
    redis = None
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from epistemic_uq.backends.registry import BackendRegistry
from epistemic_uq.calibration.metrics import optimize_threshold
from epistemic_uq.logging import bind_trace, configure_logging, reset_trace
from epistemic_uq.policy import DecisionPolicy
from epistemic_uq.runner import ExperimentRunner
from epistemic_uq.schemas import (
    AbstentionDecision,
    BatchRequest,
    Criticality,
    PolicyConfig,
    QueryRequest,
    SubgroupAudit,
    ThresholdSimulationRequest,
    UncertaintyFeatures,
    UncertaintyResult,
)
from epistemic_uq.settings import settings
from epistemic_uq.storage import ArtifactStore, Repository
from epistemic_uq.utils import trace_id


configure_logging()
logger = logging.getLogger(__name__)
artifact_store = ArtifactStore(settings.artifact_root)
repository = Repository(settings.database_url, artifact_store)
repository.create_schema()
registry = BackendRegistry()
pipeline_config = settings.load_pipeline_config()
runner = ExperimentRunner(repository, registry, pipeline_config)


class LocalRateLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        boundary = now - 60.0
        with self._lock:
            events = self._events[key]
            while events and events[0] < boundary:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


class RateLimiter:
    def __init__(self, url: str, limit: int) -> None:
        self.limit = limit
        self.local = LocalRateLimiter(limit)
        self.client = None if redis is None else redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1.0, socket_timeout=1.0)

    def allow(self, key: str) -> bool:
        window = int(time.time() // 60)
        redis_key = f"euq:rate:{key}:{window}"
        if self.client is None:
            return self.local.allow(key)
        try:
            pipeline = self.client.pipeline()
            pipeline.incr(redis_key)
            pipeline.expire(redis_key, 120)
            count, _ = pipeline.execute()
            return int(count) <= self.limit
        except Exception:
            logger.warning("redis_rate_limit_unavailable")
            return self.local.allow(key)

    def ready(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


rate_limiter = RateLimiter(settings.redis_url, settings.rate_limit_per_minute)


class PolicyInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    features: UncertaintyFeatures
    criticality: Criticality = Criticality.MEDIUM
    subgroup_metadata: dict[str, str] = Field(default_factory=dict)
    subgroup_audits: tuple[SubgroupAudit, ...] = ()
    policy: PolicyConfig | None = None


class ThresholdSimulationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float
    coverage: float
    accuracy: float
    risk: float
    answered: int
    abstained: int


def authenticate(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    if x_api_key is None or x_api_key not in settings.api_key_set:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not rate_limiter.allow(x_api_key):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    return x_api_key


app = FastAPI(
    title="Epistemic UQ",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    value = request.headers.get("X-Trace-ID") or trace_id()
    token = bind_trace(value)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", extra={"method": request.method, "path": request.url.path})
        raise
    finally:
        reset_trace(token)
    response.headers["X-Trace-ID"] = value
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000.0:.3f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception", extra={"path": request.url.path})
    return ORJSONResponse(status_code=500, content={"error": "internal_server_error"})


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready() -> dict[str, Any]:
    database_ready = False
    try:
        with repository.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ready = True
    except Exception:
        logger.exception("database_readiness_failed")
    redis_ready = rate_limiter.ready()
    status_value = "ok" if database_ready else "unavailable"
    return {"status": status_value, "database": database_ready, "redis": redis_ready}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/uncertainty", response_model=list[UncertaintyResult], dependencies=[Depends(authenticate)])
def estimate_uncertainty(payload: QueryRequest) -> list[UncertaintyResult]:
    runner.load_registered_backends(payload.backend_ids)
    effective_runner = runner
    if payload.config_overrides:
        from epistemic_uq.utils import deep_merge

        effective_runner = ExperimentRunner(repository, registry, deep_merge(pipeline_config, payload.config_overrides))
    return list(effective_runner.evaluate_example(payload.task, payload.backend_ids))


@app.post("/v1/batch/evaluate", dependencies=[Depends(authenticate)])
def batch_evaluate(payload: BatchRequest) -> dict[str, Any]:
    root = settings.allowed_data_root.resolve()
    source = Path(payload.dataset_path).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Dataset path must be within {root}") from exc
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Dataset file does not exist")
    config = pipeline_config
    if payload.config_path is not None:
        config_source = Path(payload.config_path).resolve()
        try:
            config_source.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Configuration path must be within {root}") from exc
        config = settings.load_pipeline_config(config_source)
    effective_runner = ExperimentRunner(repository, registry, config)
    run = effective_runner.run_dataset(
        dataset_path=source,
        dataset_id=source.stem,
        backend_ids=payload.backend_ids,
    )
    return run.model_dump(mode="json")


@app.get("/v1/calibrations/{calibrator_id}", dependencies=[Depends(authenticate)])
def calibration_lookup(calibrator_id: str) -> dict[str, Any]:
    try:
        return repository.get_calibration(calibrator_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/policy/decide", response_model=AbstentionDecision, dependencies=[Depends(authenticate)])
def policy_inference(payload: PolicyInferenceRequest) -> AbstentionDecision:
    policy = DecisionPolicy(payload.policy or PolicyConfig.model_validate(pipeline_config.get("policy", {})))
    return policy.decide(
        calibrated_confidence=payload.calibrated_confidence,
        features=payload.features,
        criticality=payload.criticality,
        subgroup_audits=payload.subgroup_audits,
        subgroup_metadata=payload.subgroup_metadata,
    )


@app.post("/v1/thresholds/simulate", response_model=ThresholdSimulationResponse, dependencies=[Depends(authenticate)])
def threshold_simulation(payload: ThresholdSimulationRequest) -> ThresholdSimulationResponse:
    utilities = payload.utilities
    objective = "target_coverage" if payload.target_coverage is not None else str(utilities.get("objective", "utility"))
    point = optimize_threshold(
        payload.confidences,
        payload.labels,
        objective=objective,
        correct_utility=float(utilities.get("correct", 1.0)),
        incorrect_utility=float(utilities.get("incorrect", -1.0)),
        abstain_utility=float(utilities.get("abstain", 0.0)),
        max_risk=float(utilities.get("max_risk", 0.1)),
        target_coverage=float(payload.target_coverage if payload.target_coverage is not None else utilities.get("target_coverage", 0.8)),
    )
    return ThresholdSimulationResponse(
        threshold=point.threshold,
        coverage=point.coverage,
        accuracy=point.accuracy,
        risk=point.risk,
        answered=point.answered,
        abstained=point.abstained,
    )
