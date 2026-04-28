# Architecture

The project uses a two-layer MLOps architecture: DVC owns deterministic artifact lineage, while Airflow owns runtime decisions and operational actions.

## Component Architecture

```mermaid
flowchart TB
  subgraph UI["User Layer"]
    User[Evaluator or user]
    Frontend[Streamlit frontend]
  end

  subgraph Serving["Serving Layer"]
    API[FastAPI API gateway]
    Model[FastAPI model service]
  end

  subgraph ML["ML Lifecycle"]
    DVC[DVC pipeline]
    Trainer[Trainer container]
    MLflow[MLflow tracking and registry]
    Reports[Markdown and HTML reports]
  end

  subgraph Control["Control Plane"]
    Airflow[Airflow DAG]
    Redis[Redis broker]
  end

  subgraph State["State and Artifacts"]
    PG[(Postgres)]
    Files[(Datasets and model export)]
  end

  subgraph Obs["Observability"]
    Prom[Prometheus]
    Grafana[Grafana]
    Loki[Loki]
    Alert[Alertmanager]
  end

  User --> Frontend --> API --> Model
  API --> PG
  Model --> MLflow
  Model --> Files
  Model --> PG
  Airflow --> Redis
  Airflow --> DVC
  DVC --> Trainer
  Trainer --> Files
  Trainer --> PG
  Trainer --> MLflow
  DVC --> Reports
  Airflow --> Model
  API --> Prom
  Model --> Prom
  PG --> Prom
  Prom --> Grafana
  Loki --> Grafana
  Prom --> Alert
```

## DVC Artifact Pipeline

```mermaid
flowchart LR
  Raw[Galaxy Zoo sources] --> Fetch[fetch_raw]
  Fetch --> RawSet[data/raw/galaxy_dataset]
  RawSet --> V1[preprocess_v1]
  V1 --> ProcV1[data/processed/v1]
  ProcV1 --> Final[preprocess_final]
  Final --> Split[data/processed/final train/val/test]
  Final --> Baseline[drift_baseline.json]
  Split --> Train[train]
  Feedback[data/feedback/training_feedback] --> Train
  Train --> Model[models/latest]
  Train --> ValMetrics[validation metrics]
  Model --> Eval[evaluate]
  Split --> Eval
  Eval --> TestLive[test and live metrics]
  TestLive --> Report[report]
  Report --> Latest[latest_report.md and latest_report.html]
```

## Serving Architecture

```mermaid
sequenceDiagram
  actor U as User
  participant S as Streamlit
  participant A as API Gateway
  participant M as Model Service
  participant R as MLflow Registry
  participant P as Postgres

  U->>S: Upload image
  S->>A: POST /predict
  A->>M: POST /predict
  M->>R: Resolve champion alias
  M-->>A: Prediction, top-k, latency, z-score
  A->>P: Insert prediction batch and prediction
  A-->>S: Prediction response
```

## Feedback and Continuous Improvement

```mermaid
flowchart TD
  Predict[Predictions stored in Postgres] --> Correct[User correction]
  Correct --> Feedback[feedback_corrections]
  Feedback --> AirflowCheck[Airflow inspection]
  AirflowCheck -->|new feedback >= threshold| Materialize[Materialize feedback images]
  Materialize --> Train[Retrain with base + feedback samples]
  Train --> Register[Register candidate model]
  Register --> Validate{Meets accuracy and macro F1 thresholds?}
  Validate -->|No| Reject[Persist rejected validation status]
  Validate -->|Yes| Compare{Better than champion?}
  Compare -->|Yes| Promote[Move champion alias]
  Compare -->|No| Keep[Keep current champion]
  Promote --> Reload[Reload model service]
  Reject --> RuntimeReport[Generate runtime report]
  Keep --> RuntimeReport[Generate runtime report]
```

Airflow saves the generated `dvc.lock` and `provenance.json` for each DVC run under `artifacts/runtime/runs/<airflow_run_id>/` and logs both to the corresponding MLflow run. When CI/CD supplies deployment metadata through environment variables, the provenance records the Git commit SHA, app version, container image, and CI run id, and MLflow receives matching `deployment.*` tags.

## Database Architecture

```mermaid
erDiagram
  prediction_batches ||--o{ predictions : contains
  predictions ||--o| feedback_corrections : corrected_by
  feedback_uploads ||--o{ feedback_corrections : uploads
  control_plane_state ||--|| control_plane_state : singleton
  pipeline_artifact_snapshots ||--|| latest_pipeline_artifact_snapshots : view

  prediction_batches {
    uuid batch_id PK
    text source_filename
    text source_type
    int total_files
    timestamptz created_at
  }

  predictions {
    uuid prediction_id PK
    uuid batch_id FK
    text original_filename
    bytea image_bytes
    text predicted_label
    jsonb top_k
    text model_version
    float latency_ms
    float brightness_zscore
    timestamptz created_at
  }

  feedback_uploads {
    uuid upload_id PK
    text source_filename
    bytea raw_csv
    int row_count
    timestamptz created_at
  }

  feedback_corrections {
    uuid correction_id PK
    uuid upload_id FK
    uuid prediction_id FK
    text predicted_label
    text corrected_label
    text model_version
    timestamptz created_at
  }

  pipeline_artifact_snapshots {
    bigint artifact_id
    text artifact_key
    text stage_name
    text run_id
    jsonb payload
    date recorded_date
    timestamptz recorded_at
  }
```

## Observability Architecture

```mermaid
flowchart LR
  API[API /metrics] --> Prom[Prometheus]
  Model[Model service /metrics] --> Prom
  Exporter[Pipeline exporter /metrics] --> Prom
  Prom --> Grafana[Grafana dashboards]
  ServiceLogs[Service logs] --> PG[(Postgres service_logs)]
  AirflowLogs[Airflow log files] --> Promtail[Promtail]
  Promtail --> Loki[Loki]
  Loki --> Grafana
  Prom --> Alertmanager[Alertmanager]
  Alertmanager --> Email[SMTP email]
```

## Artifact Storage Policy

| Artifact class | Storage |
|---|---|
| Dataset directories | Filesystem and DVC outputs |
| Model export | `models/latest`, MLflow artifacts, registry alias |
| Drift baseline | Local JSON plus Postgres snapshot |
| Metrics and summaries | Postgres JSONB snapshots |
| DVC pipeline reports | `artifacts/reports/latest_report.md` and `.html` |
| Airflow runtime reports | `artifacts/runtime/latest_runtime_report.md` and `.html` |
| Images uploaded for prediction | Postgres `BYTEA` in `predictions` |
| Feedback training images | Filesystem snapshot from accepted corrections |
