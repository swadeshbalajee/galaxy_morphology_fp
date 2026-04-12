
# Architecture

## Core design decision

This project uses a **two-layer orchestration model**.

### Layer 1: DVC artifact pipeline

DVC owns the deterministic artifact flow:

`fetch_raw -> preprocess_v1 -> preprocess_final -> train -> evaluate -> report`

This gives:

- reproducibility
- dependency tracking
- incremental rebuilds
- clear DAG lineage
- CI-friendly pipeline structure

### Layer 2: Airflow control plane

Airflow owns runtime decisions:

- inspect current state
- detect model degradation
- detect whether enough new feedback exists
- decide retrain vs skip
- trigger `dvc repro report`
- reload model service
- email the latest report

## Serving architecture

`Frontend -> API Gateway -> Model Service`

The frontend never talks directly to the model artifact. This preserves loose coupling and clean REST boundaries.

## Monitoring architecture

- Prometheus scrapes API, model service, and pipeline exporter
- Grafana visualizes metrics
- Loki + Promtail collect logs from app files and Airflow logs

## Data versions

The project explicitly has two processing versions before training:

- `data/processed/v1` = normalized intermediate dataset
- `data/processed/final` = final train/val/test training-ready dataset
