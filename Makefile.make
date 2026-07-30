PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-build-isolation -e . --no-deps

install-test:
	$(PYTHON) -m pip install -r requirements-test.txt
	$(PYTHON) -m pip install --no-build-isolation -e . --no-deps

migrate:
	alembic upgrade head

bootstrap:
	$(PYTHON) scripts/bootstrap.py

test:
	pytest

coverage:
	pytest --cov=epistemic_uq --cov-report=term-missing --cov-report=html

e2e:
	$(PYTHON) scripts/e2e.py

serve:
	uvicorn epistemic_uq.service:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build
