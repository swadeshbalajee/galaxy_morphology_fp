#!/usr/bin/env bash
(
set -euo pipefail

: "${AIRFLOW_DB_USER:?AIRFLOW_DB_USER is required}"
: "${AIRFLOW_DB_PASSWORD:?AIRFLOW_DB_PASSWORD is required}"
: "${AIRFLOW_DB_NAME:?AIRFLOW_DB_NAME is required}"
: "${GALAXY_DB_USER:?GALAXY_DB_USER is required}"
: "${GALAXY_DB_PASSWORD:?GALAXY_DB_PASSWORD is required}"
: "${GALAXY_DB_NAME:?GALAXY_DB_NAME is required}"
: "${MLFLOW_DB_USER:?MLFLOW_DB_USER is required}"
: "${MLFLOW_DB_PASSWORD:?MLFLOW_DB_PASSWORD is required}"
: "${MLFLOW_DB_NAME:?MLFLOW_DB_NAME is required}"

psql -v ON_ERROR_STOP=1 \
  -v airflow_user="$AIRFLOW_DB_USER" \
  -v airflow_password="$AIRFLOW_DB_PASSWORD" \
  -v airflow_db="$AIRFLOW_DB_NAME" \
  -v galaxy_user="$GALAXY_DB_USER" \
  -v galaxy_password="$GALAXY_DB_PASSWORD" \
  -v galaxy_db="$GALAXY_DB_NAME" \
  -v mlflow_user="$MLFLOW_DB_USER" \
  -v mlflow_password="$MLFLOW_DB_PASSWORD" \
  -v mlflow_db="$MLFLOW_DB_NAME" <<'EOSQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'airflow_user', :'airflow_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'airflow_user') \gexec

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'galaxy_user', :'galaxy_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'galaxy_user') \gexec

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'mlflow_user', :'mlflow_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'mlflow_user') \gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'airflow_db', :'airflow_user')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'airflow_db') \gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'galaxy_db', :'galaxy_user')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'galaxy_db') \gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'mlflow_db', :'mlflow_user')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'mlflow_db') \gexec
EOSQL
)
