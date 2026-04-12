
from __future__ import annotations

import os

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

st.title('🌌 Galaxy Morphology Classification Portal')
st.caption('Upload a galaxy image, inspect the latest pipeline report, and submit feedback for continuous improvement.')

with st.sidebar:
    st.header('System Overview')
    st.write('Frontend → API Gateway → Model Service')
    st.write('DVC pipeline → Airflow control plane → Report email + monitoring')
    st.write(f'API URL: `{API_URL}`')
    if st.button('Check API Health'):
        try:
            response = requests.get(f'{API_URL}/health', timeout=5)
            st.success(response.json())
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception('API health check failed: %s', exc)
            st.error(str(exc))

uploaded = st.file_uploader('Upload a galaxy image', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])
left, right = st.columns([1, 1])
with left:
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption='Uploaded image', use_container_width=True)

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
    st.subheader('Feedback Loop')
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

st.divider()
st.subheader('Recent Predictions')
try:
    response = requests.get(f'{API_URL}/recent-predictions', timeout=10)
    payload = response.json()
    recent = payload.get('items', [])
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
    else:
        st.info('No predictions recorded yet.')
    st.caption(f"Feedback count: {payload.get('summary', {}).get('feedback_count', 0)}")
except Exception as exc:  # noqa: BLE001
    st.warning(f'Unable to fetch recent predictions: {exc}')

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
