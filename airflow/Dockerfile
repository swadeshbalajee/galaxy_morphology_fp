FROM apache/airflow:3.1.8-python3.12

USER root
COPY requirements/training.txt /tmp/training-requirements.txt
RUN python -m venv /opt/venvs/training     && /opt/venvs/training/bin/pip install --no-cache-dir --upgrade pip setuptools wheel     && /opt/venvs/training/bin/pip install --no-cache-dir -r /tmp/training-requirements.txt     && chown -R airflow:0 /opt/venvs

USER airflow
COPY requirements/airflow.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
RUN pip install --no-cache-dir "psycopg[binary]==3.2.13"
ENV TRAINING_PYTHON=/opt/venvs/training/bin/python
WORKDIR /opt/airflow
