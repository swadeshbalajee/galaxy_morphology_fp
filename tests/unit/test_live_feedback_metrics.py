from src.training.evaluate import compute_live_feedback_metrics


def test_live_feedback_metrics_assume_missing_feedback_is_correct():
    rows = [
        {'model_version': '7', 'predicted_label': 'spiral', 'corrected_label': None},
        {'model_version': '7', 'predicted_label': 'elliptical', 'corrected_label': None},
        {'model_version': '7', 'predicted_label': 'spiral', 'corrected_label': 'merger'},
    ]

    metrics = compute_live_feedback_metrics(rows)

    assert metrics['latest_model_version'] == '7'
    assert metrics['prediction_count'] == 3
    assert metrics['feedback_count'] == 1
    assert metrics['assumed_correct_count'] == 2
    assert metrics['accuracy'] == 0.666667


def test_live_feedback_metrics_empty_rows_have_no_accuracy():
    metrics = compute_live_feedback_metrics([])

    assert metrics == {
        'feedback_count': 0,
        'prediction_count': 0,
        'assumed_correct_count': 0,
        'latest_model_version': None,
        'accuracy': None,
        'macro_f1': None,
    }


def test_live_feedback_metrics_support_tuple_rows_from_database_cursor():
    rows = [
        ('8', 'spiral', None),
        ('8', 'elliptical', 'spiral'),
    ]

    metrics = compute_live_feedback_metrics(rows)

    assert metrics['latest_model_version'] == '8'
    assert metrics['prediction_count'] == 2
    assert metrics['feedback_count'] == 1
    assert metrics['assumed_correct_count'] == 1
    assert metrics['accuracy'] == 0.5
