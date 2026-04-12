
from src.data.ingest import split_counts
from src.data.validate import DatasetValidationResult


def test_split_counts_sum_matches_total():
    train, val, test = split_counts(101, 0.70, 0.15)
    assert train + val + test == 101


def test_dataset_validation_result_dataclass():
    result = DatasetValidationResult(True, 5, 100, [], {'spiral': 20})
    assert result.is_valid is True
    assert result.class_count == 5
    assert result.total_images == 100
    assert result.per_class_counts['spiral'] == 20
