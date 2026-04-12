SHELL := /bin/bash

up:
	docker compose up --build

down:
	docker compose down -v

train:
	docker compose run --rm trainer python -m src.training.train --train-dir data/processed/train --val-dir data/processed/val --test-dir data/processed/test --params params.yaml --export-dir models/latest

ingest:
	docker compose run --rm trainer python -m src.data.ingest --source-dir data/external/galaxy_dataset --output-dir data/processed

test:
	python -m pytest tests -q

lint:
	python -m compileall src api/app model_service/app frontend tests

dvc-repro:
	dvc repro
