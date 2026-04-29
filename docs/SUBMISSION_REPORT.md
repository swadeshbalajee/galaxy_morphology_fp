# Galaxy Morphology Classification with MLOps

Submission report generated on 28 April 2026.

## Executive Summary

This project is an end-to-end AI application for classifying galaxy morphology into elliptical, spiral, lenticular, irregular, and merger classes. It combines a Streamlit user interface, FastAPI services, a PyTorch model, DVC reproducible pipelines, Airflow orchestration, MLflow experiment tracking and registry, Postgres-backed application state, Prometheus/Grafana monitoring, Loki log aggregation, Alertmanager email notifications, and Airflow report/failure emails.

The design follows the assignment guideline by separating the UI, API gateway, model inference engine, ML lifecycle, control plane, and observability stack into independently deployable services connected through REST APIs and Docker Compose networking.

## Rubric Alignment

| Evaluation area | Project evidence |
|---|---|
| Frontend UI/UX and user manual | Streamlit app with single prediction, ZIP batch prediction, correction CSV upload, recent prediction history, and `docs/06_USER_MANUAL.md`. |
| ML pipeline visualization | DVC DAG, Airflow DAG, Streamlit pipeline console, Prometheus targets, Grafana dashboard, and MLflow run views. |
| Software design | Architecture, HLD, LLD, endpoint definitions, database model, and module map in `docs/01_ARCHITECTURE.md`, `docs/02_HLD.md`, and `docs/03_LLD.md`. |
| MLOps implementation | DVC, Airflow, MLflow, Docker Compose, Prometheus, Grafana, Loki, Alertmanager, and Postgres. |
| Testing | Unit tests, integration health contracts, functional test plan, observability test cases, and final verification checklist. |
| Deployment and packaging | Dockerized frontend, API, model service, trainer, Airflow, MLflow, monitoring services, and a deployment runbook. |

## Problem Statement and Goals

The application supports local galaxy morphology classification with a complete MLOps lifecycle. A non-technical user can upload one image or a ZIP batch, receive predictions, submit corrections, and inspect recent predictions. The engineering workflow supports data ingestion, preprocessing, training, evaluation, model registration, feedback-aware retraining, reporting, monitoring, and email alerts.

Success is measured through ML metrics such as accuracy and macro F1, application metrics such as readiness and latency, and operational evidence such as reproducible DVC runs, MLflow registry decisions, dashboard visibility, and alert delivery.

## Architecture Narrative

The architecture has six main layers:

1. User layer: Streamlit frontend and pipeline console.
2. Serving layer: FastAPI API gateway and FastAPI model service.
3. ML lifecycle: DVC stages, trainer container, PyTorch training code, generated reports, and MLflow tracking.
4. Control plane: Airflow DAG with branching, registry, model reload, provenance, email reporting, and hook-backed task failure email.
5. State and artifacts: Postgres for predictions, feedback, service logs, control state, and artifact snapshots; filesystem/DVC for datasets and model exports.
6. Observability: Prometheus, Grafana, Loki, Promtail, and Alertmanager.

```text
User -> Streamlit -> API Gateway -> Model Service -> MLflow Registry
                  -> Postgres
Airflow -> DVC Pipeline -> Trainer -> MLflow + Artifacts + Reports
API/Model/Exporter -> Prometheus -> Grafana + Alertmanager -> Email
Logs -> Promtail -> Loki -> Grafana
```

## Data and ML Pipeline

The DVC pipeline owns deterministic artifact lineage. It starts with Galaxy Zoo data acquisition, performs two-stage preprocessing, trains a ResNet18-based classifier, evaluates it, records metrics, and generates Markdown/HTML reports.

| Stage | Purpose | Main outputs |
|---|---|---|
| `fetch_raw` | Download and summarize Galaxy Zoo source data | Raw dataset and raw summary |
| `preprocess_v1` | Resize and normalize source image structure | Processed v1 dataset and summary |
| `preprocess_final` | Create final train/validation/test split and drift baseline | Final dataset and drift baseline |
| `train` | Train classifier and log experiment metadata | Model export, train metrics, validation metrics |
| `evaluate` | Evaluate offline and live feedback metrics | Test metrics and live metrics |
| `report` | Generate pipeline report artifacts | `latest_report.md` and `latest_report.html` |

## Experiment Tracking and Registry

MLflow tracks model runs, metrics, parameters, and artifacts. The registry stores the galaxy morphology classifier and uses the `champion` alias for serving. Airflow registers a candidate model after a successful DVC run, validates accuracy and macro F1 thresholds, compares the candidate with the current champion, and reloads serving only if promotion succeeds.

Runtime provenance is preserved by saving and logging `dvc.lock` and `provenance.json`. The provenance record includes the Airflow run id, MLflow run id, DVC lock hash, feedback snapshot information, and optional deployment metadata such as Git commit SHA, app version, container image, and CI run id.

## Serving and User Experience

The frontend is intentionally decoupled from model inference. Streamlit calls the API gateway, and the API gateway calls the model service through configurable REST URLs. The model service owns model loading and prediction; the API owns persistence, validation, and response shaping.

| Workflow | User action | System response |
|---|---|---|
| Single prediction | Upload one supported image | Returns label, top-k scores, latency, and model version |
| Batch prediction | Upload a ZIP file | Stores a prediction batch and returns per-image results |
| Single correction | Submit true label for one prediction | Stores feedback correction in Postgres |
| CSV correction | Export recent predictions, fill corrections, upload CSV | Validates rows and stores accepted corrections |
| Pipeline console | Open service links and health checks | Gives non-technical visibility into the MLOps stack |

## Feedback and Continuous Improvement

Predictions and corrections are stored in Postgres. Airflow inspects runtime state and decides whether retraining is needed based on missing artifacts, degraded metrics, feedback thresholds, or configuration changes. Accepted feedback can be materialized into training data and used in future DVC runs.

This creates a closed loop: prediction -> correction -> feedback snapshot -> retraining decision -> candidate model -> validation -> champion promotion or rejection -> runtime report.

## Observability, Alerts, and Logging

Prometheus scrapes the API, model service, and pipeline exporter. Grafana visualizes service health, prediction behavior, latency, drift, and pipeline metrics. Loki receives log streams through Promtail, while service logs are also persisted in Postgres for queryable evidence.

Alertmanager sends email through Mailtrap. Airflow sends runtime report emails and task failure notifications through the configured `smtp_default` connection. The live alert route was verified with TLS enabled, and `GalaxyLiveAccuracyLow` moved from pending to firing after its five-minute `for` window.

| Signal | Tooling | Evidence |
|---|---|---|
| Service readiness | `/health`, `/ready`, Docker health checks | API and model readiness endpoints |
| Metrics | Prometheus client and exporter | `/metrics` endpoints and Prometheus targets |
| Dashboards | Grafana | Provisioned dashboard JSON |
| Logs | Loki, Promtail, Postgres service logs | Grafana Loki view and SQL queries |
| Alerts | Prometheus rules, Alertmanager, and Airflow failure callback | Mailtrap alert/report/failure email |

## Low-Level Design and API Contracts

The codebase is organized around separate modules for configuration, data, training, registry, reporting, monitoring, API gateway, and model service.

| Service | Endpoint | Purpose |
|---|---|---|
| API | `GET /health` | Liveness |
| API | `GET /ready` | Checks model service and database |
| API | `POST /predict` | Single image prediction |
| API | `POST /predict-batch` | ZIP batch prediction |
| API | `POST /feedback` | Single correction |
| API | `POST /feedback/upload-csv` | Validated correction upload |
| API | `GET /recent-predictions` | Prediction history |
| API | `GET /recent-predictions/export` | CSV correction template |
| API | `GET /metrics` | Prometheus metrics |
| Model service | `GET /ready` | Model readiness |
| Model service | `POST /predict` | Model inference |
| Model service | `POST /reload` | Reload champion/local model |
| Model service | `GET /metrics` | Model metrics |

## Deployment and Operations

The local deployment uses Docker Compose. Services include Postgres, Redis, Airflow components, trainer, pipeline exporter, model service, API gateway, frontend, MLflow, Prometheus, Alertmanager, Loki, Promtail, Grafana, and Adminer.

Key URLs:

| Tool | URL |
|---|---|
| Frontend | `http://localhost:8501` |
| API docs | `http://localhost:8000/docs` |
| Model service docs | `http://localhost:8001/docs` |
| Airflow | `http://localhost:8080` |
| MLflow | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Alertmanager | `http://localhost:9093` |
| Grafana | `http://localhost:3000` |
| Adminer | `http://localhost:8081` |

## Testing and Acceptance Criteria

The test strategy combines unit tests, integration health checks, functional tests, Airflow test cases, observability checks, and manual proof capture.

| Test area | Coverage |
|---|---|
| Unit tests | Schemas, preprocessing, metrics, live feedback metrics, config contracts |
| Integration tests | API health contract |
| Functional tests | DVC DAG, DVC report pipeline, frontend upload, batch upload, feedback CSV, model reload, report generation |
| Airflow tests | Branching, DVC run, candidate registration, validation, promotion/rejection, runtime report email, failure email callback |
| Observability tests | Metrics endpoints, Prometheus targets, Grafana dashboard, Loki readiness, Alertmanager email |

Acceptance criteria are met when the stack starts successfully, the user can perform predictions and feedback, the pipeline can regenerate artifacts, MLflow records candidate/champion decisions, monitoring surfaces runtime signals, alerts can deliver email, and proof screenshots are captured.

## Current Run Snapshot

The latest project snapshot records a 500-image raw dataset with 100 images per class, a 70/15/15 train-validation-test split per class, validation accuracy of 0.52, validation macro F1 of 0.4976, candidate version 10, champion version 7, and a registry decision that kept the champion because the candidate did not beat the champion macro F1 of 0.5496.

## User Manual Summary

For a non-technical evaluator, the application starts at `http://localhost:8501`. The main workflows are:

1. Open the single image tab, upload an image, and run prediction.
2. Review the predicted class, confidence chart, model version, and latency.
3. If the prediction is wrong, submit the correct label as feedback.
4. Use ZIP batch upload for multiple images.
5. Use recent predictions to export a correction CSV template.
6. Upload the completed correction CSV and fix any row-level validation errors shown by the UI.
7. Open the pipeline console for health checks and links to Airflow, MLflow, Prometheus, Grafana, Loki, and API docs.

## Proof Image Placeholders

### Image Placeholder: Docker Compose Services

Evidence expected: Show all required containers running with healthy core services.

![Placeholder - Docker Compose Services](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Streamlit Single Prediction

Evidence expected: Show the frontend prediction result with label, confidence, model version, and latency.

![Placeholder - Streamlit Single Prediction](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: ZIP Batch Prediction

Evidence expected: Show batch upload results table and stored batch id.

![Placeholder - ZIP Batch Prediction](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Feedback Submission or CSV Upload

Evidence expected: Show accepted correction feedback or CSV validation success.

![Placeholder - Feedback Submission or CSV Upload](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Airflow DAG Run

Evidence expected: Show `galaxy_morphology_control_plane` DAG success or branch behavior.

![Placeholder - Airflow DAG Run](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: DVC Pipeline DAG

Evidence expected: Show DVC pipeline visualization or successful `dvc repro report`.

![Placeholder - DVC Pipeline DAG](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: MLflow Experiment and Registry

Evidence expected: Show tracked run, candidate/champion model version, and registry alias.

![Placeholder - MLflow Experiment and Registry](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Prometheus Targets and Firing Alert

Evidence expected: Show scrape targets and `GalaxyLiveAccuracyLow` firing or another test alert.

![Placeholder - Prometheus Targets and Firing Alert](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Grafana Dashboard

Evidence expected: Show service and model monitoring dashboard panels.

![Placeholder - Grafana Dashboard](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Loki Logs

Evidence expected: Show logs visible through Grafana/Loki or Promtail pipeline.

![Placeholder - Loki Logs](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Mailtrap Email

Evidence expected: Show Alertmanager email, Airflow report email, or Airflow failure email received in Mailtrap.

![Placeholder - Mailtrap Email](image/proof/<replace-with-screenshot>.png)

### Image Placeholder: Generated Runtime Report

Evidence expected: Show latest generated report artifact or email report attachment.

![Placeholder - Generated Runtime Report](image/proof/<replace-with-screenshot>.png)


## Known Gaps and Mitigations

| Gap | Mitigation |
|---|---|
| Screenshots are captured manually | Reserved proof placeholders are included in this report and screenshots should be saved under `image/proof/`. |
| Large image/model artifacts are kept outside Postgres | DVC and mounted volumes preserve reproducibility without bloating the database. |
| Full production CI/CD is represented locally | DVC, MLflow provenance, Docker Compose, and optional deployment metadata provide reproducible local CI-style evidence. |

## Source Documentation

This report consolidates the following Markdown documents into one submission narrative:

- `docs/00_REQUIREMENT_COVERAGE.md`
- `docs/01_ARCHITECTURE.md`
- `docs/02_HLD.md`
- `docs/03_LLD.md`
- `docs/04_TEST_PLAN_AND_CASES.md`
- `docs/05_TEST_REPORT.md`
- `docs/06_USER_MANUAL.md`
- `docs/07_DEPLOYMENT_RUNBOOK.md`
