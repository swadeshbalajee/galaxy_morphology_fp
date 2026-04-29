from prometheus_client import Counter, Gauge, Histogram

API_REQUESTS = Counter(
    "galaxy_api_requests_total",
    "Total number of API requests.",
    ["endpoint", "method", "status"],
)
API_LATENCY = Histogram(
    "galaxy_api_request_latency_seconds", "API request latency.", ["endpoint"]
)
API_READY = Gauge("galaxy_api_ready", "API readiness state.")
DB_READY = Gauge("galaxy_db_ready", "Database readiness state.")
FEEDBACK_COUNT = Counter("galaxy_feedback_total", "Ground truth feedback submissions.")
RECENT_PREDICTIONS_SIZE = Gauge(
    "galaxy_recent_predictions_count", "Recent prediction rows returned by the API."
)
PREDICTION_ROWS_STORED = Counter(
    "galaxy_predictions_rows_stored_total", "Prediction rows stored in Postgres."
)
BATCH_UPLOADS_TOTAL = Counter(
    "galaxy_batch_uploads_total", "ZIP batch uploads processed by the API."
)
CSV_UPLOADS_TOTAL = Counter(
    "galaxy_feedback_csv_uploads_total", "Feedback CSV uploads accepted by the API."
)
CSV_VALIDATION_FAILURES_TOTAL = Counter(
    "galaxy_feedback_csv_validation_failures_total", "Feedback CSV validation failures."
)
CORRECTION_ROWS_STORED = Counter(
    "galaxy_feedback_correction_rows_total", "Correction rows stored in Postgres."
)
