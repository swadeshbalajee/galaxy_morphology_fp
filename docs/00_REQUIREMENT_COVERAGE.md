
# Requirement Coverage Matrix

| Requirement | Implemented in |
|---|---|
| Separate frontend + backend + model service | `frontend/`, `api/`, `model_service/` |
| DVC DAG for reproducibility | `dvc.yaml` |
| Airflow orchestration / control plane | `airflow/dags/galaxy_pipeline.py` |
| MLflow experiment tracking | `src/training/train.py`, `mlflow/` |
| Monitoring metrics | `api/app/metrics.py`, `model_service/app/main.py`, `src/monitoring/pipeline_exporter.py` |
| Grafana dashboards | `monitoring/grafana/dashboards/galaxy_mlops_dashboard.json` |
| Logs in Grafana | `monitoring/loki/*`, `monitoring/promtail/*` |
| Continuous improvement loop | `airflow/dags/galaxy_pipeline.py`, `src/training/evaluate.py` |
| Email report delivery | `src/common/email_utils.py`, Airflow DAG |
| Two-stage preprocessing | `src/data/preprocess_v1.py`, `src/data/preprocess_final.py` |
| Non-technical user manual | `docs/06_USER_MANUAL.md` |
| HLD / LLD / architecture docs | `docs/01_ARCHITECTURE.md`, `docs/02_HLD.md`, `docs/03_LLD.md` |
