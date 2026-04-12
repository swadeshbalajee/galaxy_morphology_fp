
# Test Plan and Test Cases

## Objectives

Verify that the project is correct, reproducible, observable, and demo-ready.

## Functional tests

1. DVC DAG renders successfully.
2. `dvc repro report` completes successfully.
3. Airflow DAG triggers and branches correctly.
4. Frontend can upload an image and receive a prediction.
5. Feedback submission stores rows in SQLite.
6. Model service reload endpoint works after retraining.

## Observability tests

1. Prometheus can scrape API, model service, and pipeline exporter.
2. Grafana dashboard loads.
3. Loki receives app logs and Airflow logs.

## Continuous-improvement tests

1. With metrics above threshold, Airflow skips retraining.
2. With metrics below threshold, Airflow runs DVC.
3. With sufficient new feedback, Airflow retriggers the DVC pipeline.
4. Latest report is generated and emailed when SMTP is enabled.
