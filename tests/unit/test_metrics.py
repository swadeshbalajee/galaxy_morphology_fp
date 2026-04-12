from math import isclose

from src.features.baseline import compute_image_baseline


def test_baseline_empty_folder(tmp_path):
    result = compute_image_baseline(tmp_path)
    assert result['count'] == 0
    assert isclose(result['brightness_mean'], 0.0)
