from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from src.common.config import get_config_value, load_config
from src.common.logging_utils import configure_logging

LOGGER = configure_logging('frontend')
CONFIG = load_config()
API_URL = os.getenv('API_URL') or get_config_value(CONFIG, 'services.api_url', 'http://api:8000')
REPORT_PATH = get_config_value(CONFIG, 'paths.latest_report_md_path', 'artifacts/reports/latest_report.md')

st.set_page_config(page_title='Galaxy Morphology Classifier', page_icon='🌌', layout='wide')


@st.dialog('CSV validation errors')
def show_csv_errors(errors: list[dict]):
    st.error('The uploaded correction CSV has validation errors. Fix these rows and upload again.')
    st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)


st.title('🌌 Galaxy Morphology Classification Portal')
st.caption('Run single-image or ZIP batch inference, download filtered prediction history, and upload correction CSV files backed by Postgres state.')

with st.sidebar:
    st.header('System Overview')
    st.write('Frontend → API Gateway → Model Service')
    st.write('Postgres stores predictions, uploaded images, and correction CSVs')
    st.write('DVC pipeline → Airflow control plane → MLflow + Prometheus + Grafana + Loki')
    st.write(f'API URL: `{API_URL}`')
    if st.button('Check API Health'):
        try:
            response = requests.get(f'{API_URL}/health', timeout=5)
            st.success(response.json())
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception('API health check failed: %s', exc)
            st.error(str(exc))

single_tab, batch_tab, feedback_tab, recent_tab = st.tabs([
    'Single image',
    'ZIP batch upload',
    'Correction CSV upload',
    'Recent predictions',
])

with single_tab:
    uploaded = st.file_uploader('Upload a galaxy image', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])
    left, right = st.columns([1, 1])
    with left:
        if uploaded is not None:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, caption='Uploaded image')

    with right:
        if uploaded is not None and st.button('Predict Morphology', type='primary'):
            files = {'file': (uploaded.name, uploaded.getvalue(), uploaded.type or 'application/octet-stream')}
            with st.spinner('Running inference...'):
                response = requests.post(f'{API_URL}/predict', files=files, timeout=120)
            if response.ok:
                result = response.json()
                st.success(f"Predicted class: **{result['predicted_label']}**")
                st.write(f"Model version: `{result['model_version']}`")
                st.write(f"Latency: `{result['latency_ms']:.2f} ms`")
                df = pd.DataFrame(result['top_k'])
                st.bar_chart(df.set_index('label'))
                st.session_state['last_prediction_id'] = result['prediction_id']
                st.session_state['last_prediction'] = result
                LOGGER.info('Prediction displayed prediction_id=%s', result['prediction_id'])
            else:
                st.error(response.text)

    if 'last_prediction_id' in st.session_state:
        st.divider()
        st.subheader('Single-record feedback')
        label = st.text_input('Correct / actual label', placeholder='Example: spiral')
        notes = st.text_area('Optional notes', placeholder='Was the image noisy? Was it ambiguous?')
        if st.button('Submit Feedback'):
            payload = {
                'prediction_id': st.session_state['last_prediction_id'],
                'ground_truth_label': label,
                'notes': notes,
            }
            response = requests.post(f'{API_URL}/feedback', json=payload, timeout=30)
            if response.ok:
                LOGGER.info('Feedback submitted for prediction_id=%s', payload['prediction_id'])
                st.success('Feedback recorded successfully.')
            else:
                st.error(response.text)

with batch_tab:
    st.subheader('Upload multiple images as a ZIP')
    zip_file = st.file_uploader('Upload ZIP archive of images', type=['zip'], key='zip_uploader')
    if zip_file is not None and st.button('Run batch prediction', type='primary'):
        with st.spinner('Running batch inference...'):
            response = requests.post(
                f'{API_URL}/predict-batch',
                files={'zip_file': (zip_file.name, zip_file.getvalue(), 'application/zip')},
                timeout=600,
            )
        if response.ok:
            payload = response.json()
            st.success(f"Batch stored in Postgres. Batch id: `{payload['batch_id']}` | Images: `{payload['total_images']}`")
            items = pd.DataFrame(payload['items'])
            st.dataframe(items[['prediction_id', 'batch_id', 'original_filename', 'predicted_label', 'model_version', 'latency_ms']], use_container_width=True)
        else:
            st.error(response.text)

with feedback_tab:
    st.subheader('Upload correction CSV')
    st.caption('Use the exported recent-predictions CSV template, fill only the rows that need correction, and set `corrected_label`.')
    feedback_csv = st.file_uploader('Upload correction CSV', type=['csv'], key='feedback_csv_uploader')
    if feedback_csv is not None and st.button('Validate and upload CSV', type='primary'):
        response = requests.post(
            f'{API_URL}/feedback/upload-csv',
            files={'file': (feedback_csv.name, feedback_csv.getvalue(), 'text/csv')},
            timeout=120,
        )
        payload = response.json()
        if response.ok:
            st.success(f"Correction CSV uploaded successfully. Upload id: `{payload['upload_id']}` | Rows stored: `{payload['row_count']}`")
        else:
            st.session_state['csv_errors'] = payload.get('errors', [])
            show_csv_errors(st.session_state['csv_errors'])

with recent_tab:
    st.subheader('Recent predictions')
    default_end = date.today()
    default_start = default_end - timedelta(days=7)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_date = st.date_input('Start date', value=default_start)
    with col2:
        end_date = st.date_input('End date', value=default_end)
    with col3:
        limit = st.number_input('Rows', min_value=1, max_value=5000, value=200, step=50)

    params = {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'limit': int(limit),
    }
    try:
        response = requests.get(f'{API_URL}/recent-predictions', params=params, timeout=30)
        payload = response.json()
        recent = payload.get('items', [])
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True)
        else:
            st.info('No predictions recorded for the selected date range.')
        summary = payload.get('summary', {})
        st.caption(
            f"Prediction count: {summary.get('prediction_count', 0)} | Correction count: {summary.get('feedback_count', 0)}"
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f'Unable to fetch recent predictions: {exc}')

    export_response = requests.get(f'{API_URL}/recent-predictions/export', params=params, timeout=60)
    if export_response.ok:
        st.download_button(
            label='Download filtered prediction CSV template',
            data=export_response.content,
            file_name=f'predictions_{start_date.isoformat()}_{end_date.isoformat()}.csv',
            mime='text/csv',
        )

st.divider()
st.subheader('Latest pipeline report')
try:
    report_path = os.path.join('/app', REPORT_PATH) if not os.path.isabs(REPORT_PATH) else REPORT_PATH
    if os.path.exists(report_path):
        st.code(open(report_path, 'r', encoding='utf-8').read(), language='markdown')
    else:
        st.info('No report generated yet. Run the DVC pipeline or Airflow controller first.')
except Exception as exc:  # noqa: BLE001
    st.warning(f'Unable to load report: {exc}')
