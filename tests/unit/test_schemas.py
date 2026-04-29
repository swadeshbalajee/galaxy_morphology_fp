from api.app.schemas import FeedbackRequest, PredictionResponse


def test_feedback_request_model():
    payload = FeedbackRequest(
        prediction_id="abc", ground_truth_label="spiral", notes="clear arms"
    )
    assert payload.ground_truth_label == "spiral"


def test_prediction_response_model():
    response = PredictionResponse(
        prediction_id="1",
        predicted_label="elliptical",
        top_k=[{"label": "elliptical", "probability": 0.8}],
        model_version="latest",
        latency_ms=12.5,
    )
    assert response.predicted_label == "elliptical"
