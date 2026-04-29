# High-Level Design

## Goal

Build a local, demonstrable MLOps platform for galaxy morphology classification that supports training, serving, feedback, retraining decisions, reporting, and monitoring.

## Logical View

```mermaid
flowchart TB
  Data[Data subsystem] --> ML[ML subsystem]
  ML --> Registry[Registry subsystem]
  Registry --> Serving[Serving subsystem]
  Serving --> Feedback[Feedback subsystem]
  Feedback --> Control[Control subsystem]
  Control --> ML
  Serving --> Observability[Observability subsystem]
  ML --> Observability
  Control --> Reporting[Reporting subsystem]
```

## Design Decisions

| Decision | Rationale |
|---|---|
| DVC for ML stages | Gives deterministic dependencies, outputs, and rerun behavior |
| Airflow for control plane | Handles scheduled checks, branching, registry, reload, report email, and failure email callbacks |
| Postgres for state | Keeps predictions, feedback, service logs, control state, and artifact summaries durable |
| MLflow registry alias | Lets serving use `models:/galaxy_morphology_classifier@champion` |
| API gateway in front of model service | Keeps UI decoupled from model loading and storage |
| Prometheus plus Loki | Separates numeric metrics from logs while Grafana visualizes both |

## Data Lifecycle

```mermaid
stateDiagram-v2
  [*] --> Downloaded
  Downloaded --> RawSubset: threshold label assignment
  RawSubset --> ProcessedV1: RGB resize to v1 size
  ProcessedV1 --> TrainingReady: train/val/test split
  TrainingReady --> TrainedModel: train
  TrainedModel --> EvaluatedModel: evaluate
  EvaluatedModel --> Reported: report
  Reported --> [*]
```

## Inference Lifecycle

```mermaid
stateDiagram-v2
  [*] --> Uploaded
  Uploaded --> Validated
  Validated --> Predicted
  Predicted --> Persisted
  Persisted --> Displayed
  Displayed --> Corrected: user says prediction is wrong
  Displayed --> [*]: no correction
  Corrected --> FeedbackStored
  FeedbackStored --> [*]
```

## Control-Plane Decision Logic

The DAG schedule is read from `continuous_improvement.monitor_schedule` and defaults to `30 12 * * *`; `catchup` is disabled.

```mermaid
flowchart TD
  Start[Scheduled or manual DAG run] --> Inspect[Inspect runtime state]
  Inspect --> Raw{Raw data exists?}
  Raw -->|No| Run[Run DVC report pipeline]
  Raw -->|Yes| Model{Model export exists?}
  Model -->|No| Run
  Model -->|Yes| Metrics{Metrics below thresholds?}
  Metrics -->|Yes| Run
  Metrics -->|No| Feedback{New feedback threshold reached?}
  Feedback -->|Yes| Run
  Feedback -->|No| Config{Pipeline config changed?}
  Config -->|Yes| Run
  Config -->|No| Skip[Skip retraining]
  Run --> Push[Push DVC artifacts]
  Push --> Provenance[Save and log DVC provenance]
  Provenance --> Register[Register candidate in MLflow]
  Register --> Validate{Candidate passes thresholds?}
  Validate -->|No| Reject[Persist rejection status]
  Validate -->|Yes| Promote[Promote if candidate beats champion]
  Promote --> Reload[Reload model service]
  Reject --> RuntimeReport[Generate runtime report]
  Reload --> RuntimeReport[Generate runtime report]
  Skip --> RuntimeReport
  RuntimeReport --> Email[Send runtime report]
  Start -. task failure .-> FailureEmail[Send failure email via smtp_default]
```

## Deployment View

```mermaid
flowchart LR
  subgraph Compose["Docker Compose"]
    PG[postgres]
    Redis[redis]
    Airflow[airflow-api-server/scheduler/worker/triggerer]
    Trainer[trainer]
    Exporter[pipeline-exporter]
    Model[model-service]
    API[api]
    UI[frontend]
    MLflow[mlflow]
    Prom[prometheus]
    Alert[alertmanager]
    Loki[loki]
    Promtail[promtail]
    Grafana[grafana]
    Adminer[adminer]
  end

  UI --> API --> Model
  Model --> MLflow
  API --> PG
  Airflow --> Trainer
  Airflow --> Redis
  Trainer --> PG
  Trainer --> MLflow
  Exporter --> PG
  Prom --> Alert
  Promtail --> Loki
  Grafana --> Prom
  Grafana --> Loki
  Adminer --> PG
```

## Key Quality Attributes

| Attribute | Design support |
|---|---|
| Reproducibility | DVC stages and config-driven paths |
| Traceability | MLflow runs, registry status, artifact snapshots |
| Operability | Airflow branching, health endpoints, reload endpoint |
| Observability | Metrics, logs, dashboards, alert routing |
| Extensibility | Service boundaries and config-driven class/model settings |
| Demonstrability | Streamlit UI, Adminer, generated report, proof folder |
