FROM apache/airflow:3.1.8-python3.12

USER airflow
COPY requirements/airflow.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt

WORKDIR /opt/airflow
