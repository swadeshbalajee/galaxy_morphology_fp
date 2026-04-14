DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
      CREATE ROLE airflow LOGIN PASSWORD 'airflow';
   END IF;

   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'galaxy') THEN
      CREATE ROLE galaxy LOGIN PASSWORD 'galaxy';
   END IF;

   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow') THEN
      CREATE ROLE mlflow LOGIN PASSWORD 'mlflow';
   END IF;
END
$$;

SELECT 'CREATE DATABASE airflow OWNER airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow') \gexec

SELECT 'CREATE DATABASE galaxy_app OWNER galaxy'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'galaxy_app') \gexec

SELECT 'CREATE DATABASE mlflow OWNER mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow') \gexec
