
from __future__ import annotations

import os

import requests
import streamlit as st

from src.common.config import get_config_value, load_config
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('frontend_console')
CONFIG = load_config()
API_URL = os.getenv('API_URL') or get_config_value(CONFIG, 'services.api_url', 'http://localhost:8000')
AIRFLOW_URL = os.getenv('AIRFLOW_URL') or get_config_value(CONFIG, 'services.airflow_url', 'http://localhost:8080')
MLFLOW_URL = os.getenv('MLFLOW_URL') or get_config_value(CONFIG, 'services.mlflow_url', 'http://localhost:5000')
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL') or get_config_value(CONFIG, 'services.prometheus_url', 'http://localhost:9090')
GRAFANA_URL = os.getenv('GRAFANA_URL') or get_config_value(CONFIG, 'services.grafana_url', 'http://localhost:3000')
LOKI_URL = os.getenv('LOKI_URL') or get_config_value(CONFIG, 'services.loki_url', 'http://localhost:3100')
PIPELINE_EXPORTER_URL = os.getenv('PIPELINE_EXPORTER_URL', 'http://pipeline-exporter:8010')

st.set_page_config(page_title='Pipeline Console', page_icon='🛠️', layout='wide')
st.title('🛠️ MLOps Pipeline Console')

st.markdown(
    """
    This page is the evaluator-facing operational console. It highlights the split between the **DVC artifact pipeline** and the **Airflow control plane**.
    """
)

services = {
    'API Gateway': f'{API_URL}/health',
    'Airflow': f'{AIRFLOW_URL}/api/v2/monitor/health',
    'MLflow': MLFLOW_URL,
    'Prometheus': f'{PROMETHEUS_URL}/-/healthy',
    'Grafana': f'{GRAFANA_URL}/api/health',
    'Loki': f'{LOKI_URL}/ready',
    'Pipeline Exporter': f'{PIPELINE_EXPORTER_URL}/health',
}

columns = st.columns(len(services))
for column, (name, url) in zip(columns, services.items()):
    with column:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                st.success(name)
            else:
                st.error(name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning('Service check failed name=%s error=%s', name, exc)
            st.error(name)

st.divider()
st.subheader('Tool Links')
st.markdown(f'- [Airflow DAGs]({AIRFLOW_URL})')
st.markdown(f'- [MLflow Experiments]({MLFLOW_URL})')
st.markdown(f'- [Prometheus Targets]({PROMETHEUS_URL}/targets)')
st.markdown(f'- [Grafana Dashboard]({GRAFANA_URL})')
st.markdown(f'- [Loki Ready Check]({LOKI_URL}/ready)')
st.markdown(f'- [FastAPI Docs]({API_URL}/docs)')

st.divider()
st.subheader('Pipeline Narrative')
st.markdown(
    """
    1. **DVC fetch_raw** downloads the Galaxy Zoo source metadata and materializes a capped raw subset.
    2. **DVC preprocess_v1** standardizes images into a first processed dataset version.
    3. **DVC preprocess_final** creates the final train / val / test split used for training.
    4. **DVC train** fine-tunes the classifier and logs rich metrics and artifacts to MLflow.
    5. **DVC evaluate** writes offline and live-feedback metrics.
    6. **DVC report** creates the latest Markdown + HTML project report.
    7. **Airflow control plane** checks for missing data, degraded metrics, and new feedback before deciding whether to retrigger DVC.
    8. **Airflow email step** sends the latest report after a successful control-plane run.
    9. **Grafana + Loki** show metrics and logs from API, model service, training, preprocessing, and Airflow.
    """
)
