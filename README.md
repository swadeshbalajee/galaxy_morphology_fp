# Galaxy Morphology MLOps Project Report

## Cover

- Project: `galaxy-mlops-v2`
- Repository: `galaxy_morphology_fp`
- Stack: DVC + Airflow + FastAPI + Streamlit + MLflow + Postgres + Prometheus + Grafana + Loki
- Report owner: `<Add name>`
- Submission date: `<Add date>`

## Abstract

This project implements an end-to-end galaxy morphology classification platform with a full MLOps stack. The system supports dataset preparation, model training, evaluation, registry promotion, batch and single-image inference, feedback capture, retraining decisions, observability, alerting, and operational reporting. The final architecture separates the ML artifact pipeline from the operational control plane: DVC manages reproducible ML stages, while Airflow orchestrates runtime decisions such as retraining, report generation, service reloads, and email delivery.

## Problem Statement

The goal is to classify galaxy images into:

- `elliptical`
- `spiral`
- `lenticular`
- `irregular`
- `merger`

The system is expected to:

- train and evaluate a reproducible image classifier
- expose prediction APIs and a usable frontend
- capture user corrections for continuous improvement
- store operational and application state in Postgres
- monitor model and service behavior with observability tooling
- support demonstrable deployment with proof artifacts

## Objectives Delivered

- Reproducible data and model pipeline with DVC
- Airflow-based control plane for retraining and reporting
- Postgres-backed application state and pipeline artifact summaries
- MLflow experiment tracking and model registry promotion
- Streamlit frontend for prediction, batch upload, and feedback
- FastAPI API and model-service deployment
- Prometheus, Grafana, Loki, and Alertmanager integration
- SMTP-based email delivery for alerts and Airflow report email

## Final System Architecture

### High-Level Flow

`Frontend -> API -> Model Service -> Postgres`

`DVC -> Airflow Control Plane -> Train / Evaluate / Report -> MLflow + Monitoring Stack`

### Major Components

- `frontend/`: Streamlit user interface
- `api/`: FastAPI gateway for inference, history, and feedback
- `model_service/`: model-serving service
- `src/data/`: ingestion and preprocessing stages
- `src/training/`: training and evaluation logic
- `src/reporting/`: report generation
- `src/registry/`: MLflow model registration and champion promotion
- `airflow/dags/`: control-plane DAGs
- `monitoring/`: Prometheus, Grafana, Loki, Promtail, Alertmanager config
- `postgres/`: database initialization scripts

## Database Design Summary

### Operational Databases

- `airflow`
- `galaxy_app`
- `mlflow`

### Application Tables

- `prediction_batches`
- `predictions`
- `feedback_uploads`
- `feedback_corrections`
- `control_plane_state`
- `pipeline_artifact_snapshots`

### Artifact Storage Design

Most pipeline summaries and metric snapshots now live in Postgres instead of local JSON files. The main table is:

- `pipeline_artifact_snapshots`

Design characteristics:

- stores summary and metric payloads as `JSONB`
- partitioned by `recorded_date`
- queryable directly with SQL
- latest snapshot accessible through the view:
  - `latest_pipeline_artifact_snapshots`

Artifacts moved into Postgres include:

- raw ingestion summary
- preprocess v1 summary
- preprocess final summary
- feedback training summary
- train metrics
- validation metrics
- test metrics
- live metrics
- classification report
- pipeline runtime summary
- registry status

Artifacts intentionally kept as files:

- processed datasets
- trained model export
- report files
- confusion matrix CSV
- feedback manifest CSV
- drift baseline file

## Pipeline Design

### DVC Stages

1. `fetch_raw`
2. `preprocess_v1`
3. `preprocess_final`
4. `train`
5. `evaluate`
6. `report`

### Airflow Control Plane

The Airflow DAG:

- checks runtime state
- inspects feedback growth
- decides whether retraining is required
- runs the DVC report pipeline when necessary
- registers the best model in MLflow
- reloads the serving model
- sends the latest report through the configured SMTP connection

## Deployment Topology

The project is deployed with Docker Compose and includes:

- `postgres`
- `redis`
- `airflow-init`
- `airflow-api-server`
- `airflow-scheduler`
- `airflow-dag-processor`
- `airflow-worker`
- `airflow-triggerer`
- `trainer`
- `pipeline-exporter`
- `model-service`
- `api`
- `frontend`
- `mlflow`
- `prometheus`
- `alertmanager`
- `loki`
- `promtail`
- `grafana`
- `adminer`

## Runtime Endpoints

- Frontend: `http://localhost:8501`
- API: `http://localhost:8000/docs`
- Model Service: `http://localhost:8001/docs`
- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`
- Grafana: `http://localhost:3000`
- Adminer: `http://localhost:8081`

## SQL Evidence Queries

### Latest Artifact Snapshots

```sql
SELECT artifact_key, stage_name, recorded_at
FROM latest_pipeline_artifact_snapshots
ORDER BY artifact_key;
```

### Historical Train Metrics by Date

```sql
SELECT recorded_date, recorded_at, payload
FROM pipeline_artifact_snapshots
WHERE artifact_key = 'train_metrics'
ORDER BY recorded_at DESC;
```

### Recent Feedback Corrections

```sql
SELECT prediction_id, predicted_label, corrected_label, created_at
FROM feedback_corrections
ORDER BY created_at DESC
LIMIT 20;
```

## Deployment Procedure

### Environment Preparation

1. Configure `.env`
2. Confirm Mailtrap SMTP values for Alertmanager
3. Confirm Airflow SMTP connection for report email

### Deploy

```bash
docker compose up -d --build
```

### Validate

```bash
docker compose ps
```

## Reported Features

### Inference and Feedback

- single image prediction
- ZIP batch prediction
- recent prediction filtering
- CSV template export
- validated correction CSV upload
- feedback storage in Postgres

### Continuous Improvement

- feedback-aware retraining trigger
- feedback materialization into training dataset
- live metrics from accepted corrections
- control-plane state stored in Postgres

### Monitoring and Alerting

- Prometheus service and pipeline metrics
- Grafana dashboards
- Loki log aggregation
- Alertmanager email routing
- Mailtrap SMTP integration

## Proof of Deployment

Store screenshots inside [`image/proof`](<c:\Users\Swadesh B\Downloads\galaxy_morphology_fp\image\proof>).

### Proof 1: Docker Compose Running

Expected file: `image/proof/01-docker-compose-ps.png`

![Proof Placeholder - Docker Compose](image/proof/01-docker-compose-ps.png)

### Proof 2: Frontend Home / Prediction UI

Expected file: `image/proof/02-frontend-ui.png`

![Proof Placeholder - Frontend UI](image/proof/02-frontend-ui.png)

### Proof 3: Airflow DAG View

Expected file: `image/proof/03-airflow-dag.png`

![Proof Placeholder - Airflow DAG](image/proof/03-airflow-dag.png)

### Proof 4: MLflow Experiment / Registry

Expected file: `image/proof/04-mlflow.png`

![Proof Placeholder - MLflow](image/proof/04-mlflow.png)

### Proof 5: Prometheus Targets / Alerts

Expected file: `image/proof/05-prometheus.png`

![Proof Placeholder - Prometheus](image/proof/05-prometheus.png)

### Proof 6: Grafana Dashboard

Expected file: `image/proof/06-grafana-dashboard.png`

![Proof Placeholder - Grafana Dashboard](image/proof/06-grafana-dashboard.png)

### Proof 7: Adminer / SQL Evidence

Expected file: `image/proof/07-adminer-sql.png`

![Proof Placeholder - Adminer SQL](image/proof/07-adminer-sql.png)

### Proof 8: Alert Email / Mailtrap

Expected file: `image/proof/08-mailtrap-alert.png`

![Proof Placeholder - Mailtrap Alert](image/proof/08-mailtrap-alert.png)

### Proof 9: Latest Generated Report

Expected file: `image/proof/09-latest-report.png`

![Proof Placeholder - Latest Report](image/proof/09-latest-report.png)

## Result Summary

The project now demonstrates a full MLOps lifecycle:

- reproducible training with DVC
- operational orchestration with Airflow
- durable application and pipeline metadata in Postgres
- experiment tracking and registry promotion with MLflow
- user-facing prediction and feedback workflows
- monitoring, logs, dashboards, and alerts
- deployable multi-service Docker stack

## Limitations

- reports still remain file-based by design
- large binary datasets and trained model files remain on disk rather than in Postgres
- proof images must still be captured manually after deployment

## Final Submission Checklist

- [ ] `.env` configured
- [ ] `docker compose up -d --build` completed
- [ ] frontend prediction tested
- [ ] feedback upload tested
- [ ] Airflow DAG triggered
- [ ] MLflow registry checked
- [ ] Prometheus verified
- [ ] Grafana verified
- [ ] Adminer SQL proof captured
- [ ] Mailtrap alert proof captured
- [ ] screenshots saved in `image/proof/`

