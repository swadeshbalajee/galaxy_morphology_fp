# Galaxy Morphology MLOps Project

This version upgrades the stack in five important ways:

1. **Postgres replaces SQLite** for application state and MLflow backend metadata.
2. **ZIP batch inference** is supported from the UI and API.
3. **Correction CSV upload with validation** is supported with row-level error reporting.
4. **Grafana dashboards + Loki logs + Prometheus alerting** are wired more completely.
5. **Adminer** is added as a lightweight SQL UI for querying Postgres from the browser.

## What changed

### Serving and feedback workflow

- Single-image prediction still works.
- Batch prediction now accepts a `.zip` containing multiple images.
- Every prediction row is stored in **Postgres**.
- Image bytes are also stored in **Postgres** so prediction state stays durable.
- Recent predictions can be filtered by date range in the UI.
- Filtered predictions can be downloaded as a CSV template for corrections.
- Correction CSV upload validates the file before inserting rows into Postgres.

### Database changes

The application no longer uses `artifacts/predictions.db`.

State now lives in Postgres tables:

- `prediction_batches`
- `predictions`
- `feedback_uploads`
- `feedback_corrections`

The operational behavior is:

- indexes are created on `created_at`
- the corrections and upload tables are **clustered by date index** after insert-heavy operations
- Airflow and evaluation now read feedback counts and live metrics from Postgres-backed correction data
- MLflow backend metadata is stored in Postgres instead of SQLite

### Monitoring changes

- Grafana datasources are provisioned with fixed UIDs.
- The dashboard JSON is fixed and expanded.
- The Loki logs panel is fixed.
- Alertmanager is added and connected to Prometheus.
- Mailtrap SMTP fields are exposed in `.env`.

---

## Architecture

### Main flow

`Frontend -> API Gateway -> Model Service -> Postgres`

### MLOps flow

`DVC -> Airflow control plane -> training/evaluation/report -> MLflow + Prometheus + Grafana + Loki`

### Database usage

- **Airflow DB**: `airflow`
- **Application DB**: `galaxy_app`
- **MLflow DB**: `mlflow`

All three run in the same Postgres container, but as separate logical databases.

---

## Services

After startup these should be available:

- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Model service docs: `http://localhost:8001/docs`
- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`
- Grafana: `http://localhost:3000`
- Loki readiness: `http://localhost:3100/ready`
- Adminer SQL UI: `http://localhost:8081`
- Postgres: `localhost:5432`

---

## First-time setup

## 1. Prepare environment file

```bash
cp .env.example .env
```

Edit `.env` and set at least these values:

```env
DATABASE_URL=postgresql://galaxy:galaxy@postgres:5432/galaxy_app
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg://mlflow:mlflow@postgres:5432/mlflow
MAILTRAP_SMTP_HOST=sandbox.smtp.mailtrap.io
MAILTRAP_SMTP_PORT=2525
MAILTRAP_SMTP_USERNAME=YOUR_MAILTRAP_USERNAME
MAILTRAP_SMTP_PASSWORD=YOUR_MAILTRAP_PASSWORD
MAILTRAP_FROM_EMAIL=alerts@example.com
ALERT_EMAIL_TO=you@example.com
```

If you also want the report-email step from Airflow to send mail, update the `email:` section in `config.yaml` and set the SMTP credential env vars used there.

## 2. Important note about existing Postgres volumes

The project now uses a Postgres init script to create:

- user/db: `airflow` / `airflow`
- user/db: `galaxy` / `galaxy_app`
- user/db: `mlflow` / `mlflow`

These init scripts run only on a **fresh Postgres volume**.

If you already started an older version of the stack, reset the Postgres volume once:

```bash
docker compose down -v
```

Then start again.

This is important because otherwise the new databases and roles may not be created.

## 3. Start the full stack

```bash
docker compose up --build
```

---

## Postgres usage and SQL UI

## Browser-based SQL UI

Open Adminer:

`http://localhost:8081`

Use:

- **System**: `PostgreSQL`
- **Server**: `postgres`
- **Username**: `galaxy`
- **Password**: `galaxy`
- **Database**: `galaxy_app`

You can also log into:

- `airflow`
- `mlflow`

by changing the database and username.

## CLI access without the root-user issue

If you want shell access instead of Adminer, use:

```bash
docker compose exec postgres psql -U postgres -d postgres
```

or for the application DB:

```bash
docker compose exec postgres psql -U postgres -d galaxy_app
```

Do not try to run `psql` from a root shell inside a random container. Use the Postgres service directly, or use Adminer.

## Useful SQL checks

Count predictions:

```sql
SELECT COUNT(*) FROM predictions;
```

See latest prediction rows:

```sql
SELECT prediction_id, original_filename, predicted_label, created_at
FROM predictions
ORDER BY created_at DESC
LIMIT 20;
```

See uploaded correction rows:

```sql
SELECT prediction_id, predicted_label, corrected_label, created_at
FROM feedback_corrections
ORDER BY created_at DESC
LIMIT 20;
```

---

## UI workflow

## 1. Single image prediction

In the **Single image** tab:

- upload one image
- click **Predict Morphology**
- optionally submit a manual correction

That creates:

- one batch row in `prediction_batches`
- one prediction row in `predictions`
- one correction row in `feedback_corrections` if feedback is submitted

## 2. ZIP batch prediction

In the **ZIP batch upload** tab:

- upload a `.zip` containing images
- click **Run batch prediction**

The API will:

- validate the ZIP
- scan supported image files inside it
- call the model service once per image
- store every image and prediction in Postgres
- return the batch id and prediction table to the UI

## 3. Filter recent predictions and export CSV

In the **Recent predictions** tab:

- choose `start_date`
- choose `end_date`
- choose row limit
- inspect the filtered prediction table
- click **Download filtered prediction CSV template**

This downloaded CSV contains these columns:

- `prediction_id`
- `batch_id`
- `original_filename`
- `predicted_label`
- `model_version`
- `latency_ms`
- `created_at`
- `corrected_label`

Only the rows that need correction should be kept in the file you upload back.

## 4. Upload correction CSV

In the **Correction CSV upload** tab:

- upload the edited CSV
- click **Validate and upload CSV**

The backend validates every row before storing it.

### Validation rules

For each uploaded row the API checks:

1. required columns are present
2. required columns are filled
3. `predicted_label != corrected_label`
4. `prediction_id` exists in Postgres
5. the non-correction columns match the stored prediction row

If validation fails, the UI shows a popup with:

- row number
- exact mistake

No rows are inserted when validation fails.

If validation succeeds:

- the raw CSV is stored in `feedback_uploads`
- the correction rows are upserted into `feedback_corrections`

---

## DVC and Airflow

## Run the DVC pipeline manually

```bash
docker compose exec airflow-api-server /opt/venvs/training/bin/python -m dvc repro report
```

## Trigger Airflow control plane

Open Airflow and trigger:

`galaxy_morphology_control_plane`

The Airflow DAG now reads live feedback count from Postgres instead of SQLite.

---

## MLflow

MLflow is still available at:

`http://localhost:5000`

The difference is that its backend metadata now lives in Postgres instead of `mlflow.db`.

Artifacts still go to the mounted `mlruns` volume.

---

## Prometheus, Grafana, Loki, and Alertmanager

## What is wired

### Prometheus scrapes

- API metrics
- model-service metrics
- pipeline-exporter metrics
- Prometheus self-metrics

### Loki/Promtail scrapes

- `logs/*.log`
- `airflow/logs/**`

### Grafana dashboard now includes

- API requests/sec
- API latency p95
- offline test accuracy
- live feedback accuracy
- API readiness
- DB readiness
- model readiness
- feedback count
- predictions stored
- corrections stored
- CSV validation failures
- brightness drift z-score
- application logs from Loki

## How to verify Grafana

1. Open `http://localhost:3000`
2. Go to **Dashboards**
3. Open **Galaxy Morphology MLOps Dashboard**
4. Set time range to **Last 6 hours** or **Last 24 hours**
5. Generate traffic from the frontend by:
   - running a single prediction
   - running a ZIP batch upload
   - uploading one invalid correction CSV
6. Refresh the dashboard

You should now see live changes in:

- predictions stored
- correction counts
- validation failures
- log stream

## How to verify logs in Grafana

1. Open Grafana
2. Open the dashboard logs panel, or go to **Explore**
3. Select datasource **Loki**
4. Run this query:

```text
{job=~"project-logs|airflow-logs"}
```

If logs are still empty, check:

```bash
docker compose logs promtail
docker compose logs loki
docker compose logs grafana
```

and also confirm log files are being written under:

- `logs/`
- `airflow/logs/`

---

## Email alerting with Mailtrap

This stack uses:

- **Prometheus** for alert rules
- **Alertmanager** for notification routing
- **Mailtrap SMTP** for sending emails

## Step-by-step Mailtrap setup

### 1. In Mailtrap

Create or open an inbox and copy:

- SMTP host
- SMTP port
- username
- password

### 2. Put those values in `.env`

```env
MAILTRAP_SMTP_HOST=sandbox.smtp.mailtrap.io
MAILTRAP_SMTP_PORT=2525
MAILTRAP_SMTP_USERNAME=YOUR_VALUE
MAILTRAP_SMTP_PASSWORD=YOUR_VALUE
MAILTRAP_FROM_EMAIL=alerts@example.com
ALERT_EMAIL_TO=you@example.com
```

### 3. Restart alerting services

```bash
docker compose up -d --build prometheus alertmanager grafana
```

### 4. Confirm Alertmanager is healthy

Open:

`http://localhost:9093`

### 5. Confirm Prometheus sees Alertmanager

Open:

`http://localhost:9090/config`

and verify the `alerting` section contains `alertmanager:9093`.

### 6. Trigger a test alert

An easy test is to stop the API temporarily:

```bash
docker compose stop api
```

Wait about a minute. This should fire `GalaxyAPIUnavailable`.

Then bring it back:

```bash
docker compose start api
```

Mailtrap should receive the alert email and then a resolved notification.

## Current alert rules

- `GalaxyAPIUnavailable`
- `GalaxyDBUnavailable`
- `GalaxyModelUnavailable`
- `GalaxyDriftHigh`
- `GalaxyLiveAccuracyLow`
- `GalaxyAPILatencyHigh`
- `GalaxyCSVValidationFailuresSpike`

---

## Troubleshooting

## 1. No dashboard appears in Grafana

Check:

```bash
docker compose logs grafana
```

Then confirm these paths exist in the repo:

- `monitoring/grafana/provisioning/datasources/datasource.yml`
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- `monitoring/grafana/dashboards/galaxy_mlops_dashboard.json`

## 2. Logs do not show in Grafana

Check:

```bash
docker compose logs promtail
docker compose logs loki
```

Then verify your application is actually writing files into `logs/*.log` and `airflow/logs/**`.

## 3. Adminer cannot connect to Postgres

Make sure the stack is up and use:

- server: `postgres`
- username: `galaxy`
- password: `galaxy`
- database: `galaxy_app`

## 4. MLflow fails on startup after switching from SQLite

The most common cause is an old Postgres volume or missing DB/user initialization.

Reset once:

```bash
docker compose down -v
docker compose up --build
```

## 5. Correction CSV upload fails immediately

Make sure the CSV was downloaded from the **Recent predictions** export button and only edited in the `corrected_label` column for rows that need correction.

---

## Validation and sanity commands

```bash
python -m compileall src api/app model_service/app frontend airflow/dags tests
pytest tests -q
```

---

## Demo sequence

For evaluation/demo recording, show this order:

1. `docker compose up --build`
2. Frontend single-image prediction
3. Frontend ZIP batch upload
4. Recent predictions date filter
5. CSV export
6. Upload an intentionally bad correction CSV and show popup errors
7. Upload a corrected CSV successfully
8. Open Adminer and query `predictions` / `feedback_corrections`
9. Open MLflow
10. Open Prometheus targets
11. Open Grafana dashboard and logs
12. Trigger an alert test and show Mailtrap receiving the email
