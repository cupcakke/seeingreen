from pathlib import Path

from epistemic_uq.backends.registry import BackendRegistry
from epistemic_uq.reports import ReportGenerator
from epistemic_uq.runner import ExperimentRunner
from epistemic_uq.schemas import BackendConfig
from epistemic_uq.settings import settings
from epistemic_uq.storage import ArtifactStore, Repository


repository = Repository(settings.database_url, ArtifactStore(settings.artifact_root))
repository.create_schema()
registry = BackendRegistry()
runner = ExperimentRunner(repository, registry, settings.load_pipeline_config(Path("config/e2e.yaml")))
backend = BackendConfig.model_validate_json(Path("examples/backend-subprocess.json").read_text(encoding="utf-8"))
runner.register_backend(backend)
run = runner.run_dataset(
    dataset_path=Path("examples/dataset.jsonl"),
    dataset_id="reference-benchmark",
    backend_ids=(backend.backend_id,),
    version="1",
)
ReportGenerator(repository).generate(run.experiment_id, Path("var/reports") / run.experiment_id)
print(run.model_dump_json(indent=2))
