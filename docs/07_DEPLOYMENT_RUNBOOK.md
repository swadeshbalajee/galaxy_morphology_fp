
# Deployment Runbook

## Start stack

```bash
docker compose up --build
```

## Run DVC pipeline manually

```bash
docker compose exec trainer dvc repro report
```

## Trigger Airflow controller

Use the Airflow UI and trigger `galaxy_morphology_control_plane`.

## Regenerate report only

```bash
docker compose exec trainer python -m src.reporting.generate_report
```
