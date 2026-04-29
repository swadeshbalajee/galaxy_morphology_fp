from api.app.main import app as api_app
from fastapi.testclient import TestClient


def test_api_health_contract():
    client = TestClient(api_app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
