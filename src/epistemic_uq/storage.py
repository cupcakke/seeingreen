from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from epistemic_uq.errors import StorageError
from epistemic_uq.schemas import BackendConfig, DatasetManifest, ExperimentRun, ExperimentStatus
from epistemic_uq.utils import stable_hash, stable_json, utc_now


class Base(DeclarativeBase):
    pass


class DatasetRecord(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    example_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BackendRecord(Base):
    __tablename__ = "backends"

    backend_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=False)
    backend_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    results: Mapped[list[ResultRecord]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class ResultRecord(Base):
    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("experiment_id", "example_id", "backend_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.experiment_id"), nullable=False)
    example_id: Mapped[str] = mapped_column(String(255), nullable=False)
    backend_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    error_json: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    experiment: Mapped[ExperimentRecord] = relationship(back_populates="results")


class CalibrationRecord(Base):
    __tablename__ = "calibrations"

    calibrator_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    method: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DriftRecord(Base):
    __tablename__ = "drift_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64))
    subgroup_json: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ArtifactStore:
    def __init__(self, root: Path, compression: str = "gzip") -> None:
        self.root = root
        self.compression = compression
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, digest: str) -> Path:
        suffix = ".json.gz" if self.compression == "gzip" else ".json"
        return self.root / namespace / digest[:2] / f"{digest}{suffix}"

    def write(self, namespace: str, value: Any) -> str:
        payload = stable_json(value)
        digest = stable_hash(value)
        destination = self._path(namespace, digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return str(destination)
        descriptor, temporary_name = tempfile.mkstemp(prefix="artifact-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as raw:
                if self.compression == "gzip":
                    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                        handle.write(payload)
                else:
                    raw.write(payload)
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temporary_name, destination)
        except Exception as exc:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise StorageError(str(exc)) from exc
        return str(destination)

    def read(self, path: str | Path) -> Any:
        source = Path(path)
        if not source.exists():
            raise StorageError(f"Artifact does not exist: {source}")
        if source.suffix == ".gz":
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)


class Repository:
    def __init__(self, database_url: str, artifact_store: ArtifactStore) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self.artifacts = artifact_store

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def register_dataset(self, manifest: DatasetManifest) -> None:
        with self.session_factory.begin() as session:
            existing = session.get(DatasetRecord, manifest.dataset_id)
            data = manifest.model_dump_json()
            if existing is None:
                session.add(
                    DatasetRecord(
                        dataset_id=manifest.dataset_id,
                        version=manifest.version,
                        content_hash=manifest.content_hash,
                        example_count=manifest.example_count,
                        source_path=manifest.source_path,
                        manifest_json=data,
                        created_at=manifest.created_at,
                    )
                )
            else:
                existing.version = manifest.version
                existing.content_hash = manifest.content_hash
                existing.example_count = manifest.example_count
                existing.source_path = manifest.source_path
                existing.manifest_json = data

    def get_dataset(self, dataset_id: str) -> DatasetManifest:
        with self.session_factory() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise StorageError(f"Unknown dataset {dataset_id}")
            return DatasetManifest.model_validate_json(record.manifest_json)

    def register_backend(self, config: BackendConfig) -> None:
        payload = config.model_dump(mode="json")
        digest = stable_hash(payload)
        now = utc_now()
        with self.session_factory.begin() as session:
            existing = session.get(BackendRecord, config.backend_id)
            if existing is None:
                session.add(
                    BackendRecord(
                        backend_id=config.backend_id,
                        config_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        config_hash=digest,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                existing.config_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                existing.config_hash = digest
                existing.updated_at = now

    def get_backend(self, backend_id: str) -> BackendConfig:
        with self.session_factory() as session:
            record = session.get(BackendRecord, backend_id)
            if record is None:
                raise StorageError(f"Unknown backend {backend_id}")
            return BackendConfig.model_validate_json(record.config_json)

    def list_backends(self) -> tuple[BackendConfig, ...]:
        with self.session_factory() as session:
            records = session.scalars(select(BackendRecord).order_by(BackendRecord.backend_id)).all()
            return tuple(BackendConfig.model_validate_json(record.config_json) for record in records)

    def create_experiment(self, run: ExperimentRun) -> None:
        with self.session_factory.begin() as session:
            session.add(
                ExperimentRecord(
                    experiment_id=run.experiment_id,
                    dataset_id=run.dataset_id,
                    backend_ids_json=json.dumps(list(run.backend_ids)),
                    config_hash=run.config_hash,
                    dataset_hash=run.dataset_hash,
                    status=run.status.value,
                    manifest_path=run.manifest_path,
                    metadata_json=json.dumps(run.metadata, ensure_ascii=False, sort_keys=True),
                    created_at=run.created_at,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                )
            )

    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise StorageError(f"Unknown experiment {experiment_id}")
            record.status = status.value
            if status == ExperimentStatus.RUNNING and record.started_at is None:
                record.started_at = now
            if status in {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}:
                record.completed_at = now

    def get_experiment(self, experiment_id: str) -> ExperimentRun:
        with self.session_factory() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise StorageError(f"Unknown experiment {experiment_id}")
            return ExperimentRun(
                experiment_id=record.experiment_id,
                dataset_id=record.dataset_id,
                backend_ids=tuple(json.loads(record.backend_ids_json)),
                config_hash=record.config_hash,
                dataset_hash=record.dataset_hash,
                status=ExperimentStatus(record.status),
                created_at=record.created_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                manifest_path=record.manifest_path,
                metadata=json.loads(record.metadata_json),
            )

    def list_experiments(self) -> tuple[ExperimentRun, ...]:
        with self.session_factory() as session:
            records = session.scalars(select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())).all()
            return tuple(
                ExperimentRun(
                    experiment_id=record.experiment_id,
                    dataset_id=record.dataset_id,
                    backend_ids=tuple(json.loads(record.backend_ids_json)),
                    config_hash=record.config_hash,
                    dataset_hash=record.dataset_hash,
                    status=ExperimentStatus(record.status),
                    created_at=record.created_at,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    manifest_path=record.manifest_path,
                    metadata=json.loads(record.metadata_json),
                )
                for record in records
            )

    def begin_result(self, experiment_id: str, example_id: str, backend_id: str) -> bool:
        with self.session_factory.begin() as session:
            statement = select(ResultRecord).where(
                ResultRecord.experiment_id == experiment_id,
                ResultRecord.example_id == example_id,
                ResultRecord.backend_id == backend_id,
            )
            record = session.scalar(statement)
            if record is not None and record.status == "completed":
                return False
            if record is None:
                record = ResultRecord(
                    experiment_id=experiment_id,
                    example_id=example_id,
                    backend_id=backend_id,
                    status="running",
                    started_at=utc_now(),
                )
                session.add(record)
            else:
                record.status = "running"
                record.started_at = utc_now()
                record.error_json = None
            return True

    def complete_result(self, experiment_id: str, example_id: str, backend_id: str, result: Any) -> str:
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        artifact_path = self.artifacts.write("results", payload)
        with self.session_factory.begin() as session:
            statement = select(ResultRecord).where(
                ResultRecord.experiment_id == experiment_id,
                ResultRecord.example_id == example_id,
                ResultRecord.backend_id == backend_id,
            )
            record = session.scalar(statement)
            if record is None:
                raise StorageError("Result record was not initialized")
            record.status = "completed"
            record.artifact_path = artifact_path
            record.result_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            record.completed_at = utc_now()
        return artifact_path

    def fail_result(self, experiment_id: str, example_id: str, backend_id: str, error: dict[str, Any]) -> None:
        with self.session_factory.begin() as session:
            statement = select(ResultRecord).where(
                ResultRecord.experiment_id == experiment_id,
                ResultRecord.example_id == example_id,
                ResultRecord.backend_id == backend_id,
            )
            record = session.scalar(statement)
            if record is None:
                record = ResultRecord(
                    experiment_id=experiment_id,
                    example_id=example_id,
                    backend_id=backend_id,
                    status="failed",
                    started_at=utc_now(),
                )
                session.add(record)
            elif record.status == "completed":
                return
            record.status = "failed"
            record.error_json = json.dumps(error, ensure_ascii=False, sort_keys=True)
            record.completed_at = utc_now()

    def get_result(self, experiment_id: str, example_id: str, backend_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            statement = select(ResultRecord).where(
                ResultRecord.experiment_id == experiment_id,
                ResultRecord.example_id == example_id,
                ResultRecord.backend_id == backend_id,
            )
            record = session.scalar(statement)
            if record is None:
                return None
            return {
                "example_id": record.example_id,
                "backend_id": record.backend_id,
                "status": record.status,
                "artifact_path": record.artifact_path,
                "result": json.loads(record.result_json) if record.result_json else None,
                "error": json.loads(record.error_json) if record.error_json else None,
            }

    def list_results(self, experiment_id: str) -> tuple[dict[str, Any], ...]:
        with self.session_factory() as session:
            records = session.scalars(
                select(ResultRecord).where(ResultRecord.experiment_id == experiment_id).order_by(ResultRecord.id)
            ).all()
            values: list[dict[str, Any]] = []
            for record in records:
                values.append(
                    {
                        "example_id": record.example_id,
                        "backend_id": record.backend_id,
                        "status": record.status,
                        "artifact_path": record.artifact_path,
                        "result": json.loads(record.result_json) if record.result_json else None,
                        "error": json.loads(record.error_json) if record.error_json else None,
                    }
                )
            return tuple(values)

    def save_calibration(self, calibrator_id: str, method: str, artifact: dict[str, Any]) -> None:
        with self.session_factory.begin() as session:
            existing = session.get(CalibrationRecord, calibrator_id)
            payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
            if existing is None:
                session.add(
                    CalibrationRecord(
                        calibrator_id=calibrator_id,
                        method=method,
                        artifact_json=payload,
                        created_at=utc_now(),
                    )
                )
            else:
                existing.method = method
                existing.artifact_json = payload

    def get_calibration(self, calibrator_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            record = session.get(CalibrationRecord, calibrator_id)
            if record is None:
                raise StorageError(f"Unknown calibration {calibrator_id}")
            return json.loads(record.artifact_json)

    def save_drift_snapshot(self, signal: str, task_type: str | None, subgroup: dict[str, str], snapshot: dict[str, Any]) -> None:
        with self.session_factory.begin() as session:
            session.add(
                DriftRecord(
                    signal=signal,
                    task_type=task_type,
                    subgroup_json=json.dumps(subgroup, ensure_ascii=False, sort_keys=True),
                    snapshot_json=json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                    observed_at=utc_now(),
                )
            )
