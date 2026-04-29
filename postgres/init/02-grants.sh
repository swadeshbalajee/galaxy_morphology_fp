#!/usr/bin/env bash
(
set -euo pipefail

: "${AIRFLOW_DB_USER:?AIRFLOW_DB_USER is required}"
: "${AIRFLOW_DB_NAME:?AIRFLOW_DB_NAME is required}"
: "${GALAXY_DB_USER:?GALAXY_DB_USER is required}"
: "${GALAXY_DB_NAME:?GALAXY_DB_NAME is required}"
: "${MLFLOW_DB_USER:?MLFLOW_DB_USER is required}"
: "${MLFLOW_DB_NAME:?MLFLOW_DB_NAME is required}"

grant_database_privileges() {
  local db_name="$1"
  local db_user="$2"

  psql -v ON_ERROR_STOP=1 \
    -v db_name="$db_name" \
    -v db_user="$db_user" \
    --dbname "$db_name" <<'EOSQL'
ALTER DATABASE :"db_name" OWNER TO :"db_user";
ALTER SCHEMA public OWNER TO :"db_user";
GRANT ALL ON SCHEMA public TO :"db_user";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO :"db_user";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO :"db_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO :"db_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO :"db_user";
EOSQL
}

grant_database_privileges "$AIRFLOW_DB_NAME" "$AIRFLOW_DB_USER"
grant_database_privileges "$GALAXY_DB_NAME" "$GALAXY_DB_USER"
grant_database_privileges "$MLFLOW_DB_NAME" "$MLFLOW_DB_USER"
)
