
# Low-Level Design

## Configuration source

All core project behavior is driven by `config.yaml`.

Key sections:

- `paths`
- `data`
- `training`
- `continuous_improvement`
- `email`
- `services`

## DVC stages

### `fetch_raw`
- module: `src.data.download_galaxy_zoo`
- input: source URLs from config
- output: `data/raw/galaxy_dataset`, `artifacts/raw_data_summary.json`

### `preprocess_v1`
- module: `src.data.preprocess_v1`
- input: raw dataset
- output: `data/processed/v1`, `artifacts/processed_v1_summary.json`

### `preprocess_final`
- module: `src.data.preprocess_final`
- input: processed v1
- output: `data/processed/final`, manifest, drift baseline

### `train`
- module: `src.training.train`
- input: final train/val/test split
- output: model artifact, train metrics, test metrics, confusion matrix

### `evaluate`
- module: `src.training.evaluate`
- input: final test set + model + feedback DB
- output: offline + live metrics JSON

### `report`
- module: `src.reporting.generate_report`
- input: summaries + metrics
- output: latest Markdown and HTML report

## Airflow tasks

### `inspect_and_branch`
Reads current model state, raw-data existence, offline metrics, live feedback metrics, and feedback count.

### `run_dvc_pipeline`
Runs `dvc repro report`.

### `reload_model_service`
Calls `POST /reload` on the model service.

### `send_report_email`
Reads the generated report and sends it through SMTP when enabled.

## REST endpoints

### API Gateway
- `GET /health`
- `GET /ready`
- `POST /predict`
- `POST /feedback`
- `GET /recent-predictions`

### Model Service
- `GET /health`
- `GET /ready`
- `POST /reload`
- `POST /predict`

## Observability

### Prometheus metrics
- API request counts and latencies
- model latencies and readiness
- drift z-score
- pipeline offline accuracy
- live accuracy
- raw images by class
- training duration
- tracked log sizes

### Loki logs
- `logs/*.log`
- `airflow/logs/**/*.log`
