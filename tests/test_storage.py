from pathlib import Path

from epistemic_uq.schemas import BackendConfig, DatasetManifest, ExperimentRun, ExperimentStatus
from epistemic_uq.storage import ArtifactStore, Repository
from epistemic_uq.utils import utc_now


def test_artifact_store_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    path = store.write("test", {"b": 2, "a": 1})
    assert store.read(path) == {"a": 1, "b": 2}
    assert store.write("test", {"a": 1, "b": 2}) == path


def test_repository_roundtrip(repository: Repository, backend_config: BackendConfig) -> None:
    manifest = DatasetManifest(
        dataset_id="d",
        version="1",
        content_hash="a" * 64,
        example_count=1,
        schema_version=1,
        created_at=utc_now(),
        source_path="dataset.jsonl",
    )
    repository.register_dataset(manifest)
    repository.register_backend(backend_config)
    run = ExperimentRun(
        experiment_id="e",
        dataset_id="d",
        backend_ids=(backend_config.backend_id,),
        config_hash="b" * 64,
        dataset_hash=manifest.content_hash,
        status=ExperimentStatus.CREATED,
        created_at=utc_now(),
        manifest_path="manifest.json",
    )
    repository.create_experiment(run)
    repository.update_experiment_status("e", ExperimentStatus.RUNNING)
    loaded = repository.get_experiment("e")
    assert loaded.status == ExperimentStatus.RUNNING
    assert repository.get_backend(backend_config.backend_id) == backend_config
