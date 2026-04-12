
# Galaxy Morphology MLOps Project

This repository is now structured as a **full-stack local MLOps application** where:

- **DVC** is the single source of truth for the artifact pipeline.
- **Airflow** is the control plane that checks runtime state, decides whether retraining is needed, triggers DVC, reloads the service, and emails the latest report.
- **MLflow** tracks experiments and artifacts.
- **Prometheus + Grafana + Loki** visualize metrics and logs.
- **Frontend + API + Model Service** form a loosely coupled serving stack.

## Final flow

### DVC pipeline

`fetch_raw -> preprocess_v1 -> preprocess_final -> train -> evaluate -> report`

### Airflow control plane

`inspect runtime -> decide retrain/skip -> dvc repro report -> reload model service -> email latest report`

## Why this split?

- DVC is best for deterministic, reproducible artifact builds.
- Airflow is best for scheduling, branching, monitoring, alerts, and operational control.
- This directly matches the assignment expectation of **DVC-based CI/reproducibility** plus **Airflow-based data/ML orchestration**.

## Data policy in this version

This version is intentionally capped for a manageable local demo:

- The dataset still comes from **Galaxy Zoo** via code.
- `config.yaml` sets `data.max_images_per_class: 1000` and `data.target_images_per_class: 1000`.
- Only the required subset is materialized into the raw dataset folder.
- The project uses **5 classes**: `elliptical`, `spiral`, `lenticular`, `irregular`, `merger`.
- Processing is split into **two versions** before training:
  - `data/processed/v1`
  - `data/processed/final`

---

# 1. Fresh setup from zero

## 1.1 Unzip and enter the project

```bash
unzip zip.zip -d galaxy-mlops
cd galaxy-mlops
```

## 1.2 Initialize Git

```bash
git init
git branch -M main
git add .
git commit -m "Initial full-stack MLOps project"
```

## 1.3 Initialize Git LFS

```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.pkl" "*.onnx" "*.zip" "models/**"
git add .gitattributes
git commit -m "Enable Git LFS for large model artifacts"
```

## 1.4 Initialize DVC

```bash
pip install dvc

dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## 1.5 Create a local DVC remote

```bash
mkdir -p .dvc/storage

dvc remote add -d localstorage .dvc/storage
git add .dvc/config
git commit -m "Configure local DVC remote"
```

## 1.6 Prepare environment file

```bash
cp .env.example .env
```

Edit `.env` only if you want SMTP delivery or different ports.

---

# 2. Main configuration

All essential project behavior is controlled from `config.yaml`.

Important keys:

- dataset source URLs
- class names
- `MAX_IMAGES_PER_CLASS`
- train/val/test split ratios
- model hyperparameters
- continuous-improvement thresholds
- report paths
- email recipients
- service URLs used inside the app

This project expects you to change behavior by editing `config.yaml`, not by hardcoding constants in code.

---

# 3. Start the full stack

## 3.1 Build and launch

```bash
docker compose up --build
```

## 3.2 Open the services

- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Model service docs: `http://localhost:8001/docs`
- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Loki ready check: `http://localhost:3100/ready`

---

# 4. Run the artifact pipeline with DVC

## 4.1 See the DAG

```bash
dvc dag
```

## 4.2 Run the complete pipeline

```bash
docker compose exec trainer dvc repro report
```

This will run, in order:

1. download a capped raw subset from Galaxy Zoo
2. create `processed/v1`
3. create `processed/final`
4. train the model
5. evaluate offline and live-feedback metrics
6. generate Markdown + HTML reports

## 4.3 Push cached artifacts into the local remote

```bash
docker compose exec trainer dvc push
```

---

# 5. Trigger the Airflow control plane

The Airflow DAG name is:

`galaxy_morphology_control_plane`

## 5.1 Trigger from UI

Open Airflow, find the DAG, and click **Trigger DAG**.

## 5.2 What Airflow does

The Airflow DAG does **not** rebuild the ML pipeline itself stage by stage.
It does this instead:

1. reads current offline metrics and live feedback metrics
2. checks whether the raw data is missing
3. checks whether the model is missing
4. checks whether performance dropped below threshold
5. checks whether enough new feedback has arrived
6. decides whether retraining should happen
7. if yes, runs `dvc repro report`
8. reloads the model service
9. emails the latest report

That is the intended control-plane behavior for continuous improvement.

---

# 6. Continuous improvement behavior

Configured in `config.yaml` under `continuous_improvement`.

Current logic uses:

- `accuracy_threshold`
- `macro_f1_threshold`
- `min_new_feedback_samples`
- `min_total_feedback_samples`

Retraining is triggered if any of these hold:

- raw data is missing
- model artifact is missing
- offline metrics are below threshold
- live feedback metrics are below threshold once enough feedback exists
- enough new feedback has arrived to justify another cycle

---

# 7. Logs and observability

## 7.1 File logs

Application logs go to:

- `logs/*.log`
- `airflow/logs/**`

## 7.2 Metrics in Grafana

Grafana now includes:

- API throughput
- model latency
- API readiness
- model readiness
- offline test accuracy
- live feedback accuracy
- drift z-score
- raw images by class
- training duration
- tracked log file sizes

## 7.3 Logs in Grafana

Logs are available in Grafana through **Loki + Promtail**.
They include:

- data download logs
- preprocessing logs
- training logs
- evaluation logs
- API logs
- model service logs
- frontend logs
- Airflow task logs

---

# 8. Demo sequence for evaluation

Use this sequence for your screen recording.

## 8.1 Show the repo and config

Open:

- `config.yaml`
- `dvc.yaml`
- `airflow/dags/galaxy_pipeline.py`
- `docs/01_ARCHITECTURE.md`
- `docs/02_HLD.md`
- `docs/03_LLD.md`

Explain:

- DVC handles artifact lineage
- Airflow handles runtime orchestration and retraining decisions
- data processing has two stages: `processed/v1` and `processed/final`

## 8.2 Show DVC DAG

```bash
dvc dag
```

## 8.3 Run DVC pipeline

```bash
docker compose exec trainer dvc repro report
```

## 8.4 Show MLflow

Show:

- params
- metrics
- artifacts
- confusion matrix / report files

## 8.5 Show Airflow control plane

Trigger `galaxy_morphology_control_plane` and show:

- inspect step
- decision step
- DVC trigger step
- reload step
- email step

## 8.6 Show serving stack

- upload an image in Streamlit
- get a prediction
- submit feedback
- refresh recent predictions table

## 8.7 Show monitoring

- Prometheus targets
- Grafana metrics panels
- Grafana logs panel from Loki

## 8.8 Show final report

Open:

- `artifacts/reports/latest_report.md`
- `artifacts/reports/latest_report.html`

If SMTP is configured, also show the email received with the attached report.

---

# 9. Useful commands

## Re-run only from preprocessing onward

```bash
docker compose exec trainer dvc repro preprocess_final
```

## Re-run only training onward

```bash
docker compose exec trainer dvc repro train
```

## Regenerate only the report

```bash
docker compose exec trainer dvc repro report
```

## Run tests

```bash
pytest tests -q
```

## Validate syntax

```bash
python -m compileall src api/app model_service/app frontend airflow/dags tests
```

---

# 10. Submission checklist

Before submission, make sure you have:

- run the DVC pipeline successfully
- triggered the Airflow control plane successfully
- shown MLflow experiment tracking
- shown Grafana metrics
- shown Loki logs
- shown frontend prediction + feedback
- shown docs
- shown the final report artifacts
- recorded the demo
