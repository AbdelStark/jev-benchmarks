from jev_benchmarks.models import Prediction
from jev_benchmarks.runner import _validate_prediction


def test_rounding_tolerance_preserves_raw_sum_and_normalizes() -> None:
    prediction = Prediction(
        experiment_id="test",
        backend="test",
        model_requested="test",
        model_resolved="test",
        dataset="fixture",
        example_id="1",
        target_index=0,
        predicted_index=0,
        labels=("a", "b"),
        probabilities=(0.50, 0.49),
        latency_seconds=0.1,
    )
    normalized = _validate_prediction(prediction)
    assert normalized.probability_sum_raw == 0.99
    assert sum(normalized.probabilities) == 1.0
