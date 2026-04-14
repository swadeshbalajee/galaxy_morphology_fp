FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow==2.22.0 "psycopg[binary]==3.2.13"

CMD ["mlflow", "--version"]