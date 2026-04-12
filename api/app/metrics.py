
from prometheus_client import Counter, Gauge, Histogram

API_REQUESTS = Counter('galaxy_api_requests_total', 'Total number of API requests.', ['endpoint', 'method', 'status'])
API_LATENCY = Histogram('galaxy_api_request_latency_seconds', 'API request latency.', ['endpoint'])
API_READY = Gauge('galaxy_api_ready', 'API readiness state.')
FEEDBACK_COUNT = Counter('galaxy_feedback_total', 'Ground truth feedback submissions.')
RECENT_PREDICTIONS_SIZE = Gauge('galaxy_recent_predictions_count', 'Recent prediction rows returned by the API.')
