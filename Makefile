SHELL := /bin/bash

up:
	docker compose up --build

down:
	docker compose down -v

train:
	docker compose exec airflow-api-server /opt/venvs/training/bin/python -m src.training.train

fetch-raw:
	docker compose exec airflow-api-server /opt/venvs/training/bin/python -m dvc repro fetch_raw

preprocess:
	docker compose exec airflow-api-server /opt/venvs/training/bin/python -m dvc repro preprocess_final

test:
	python -m pytest tests -q

lint:
	python -m compileall src api/app model_service/app frontend tests airflow/dags

dvc-repro:
	docker compose exec airflow-api-server /opt/venvs/training/bin/python -m dvc repro report
