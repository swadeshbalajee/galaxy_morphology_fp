FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow==3.11.1 psycopg2-binary==2.9.10

CMD ["mlflow", "--version"]
