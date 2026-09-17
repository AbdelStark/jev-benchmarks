from jev_benchmarks.data import _balanced_indices, _class_count


def test_class_count_from_repeated_text() -> None:
    assert _class_count(["first", "first", "first", "second", "second", "second"]) == 3


def test_balanced_indices_are_deterministic_and_cover_classes() -> None:
    targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    first = _balanced_indices(targets, 6, 42)
    assert first == _balanced_indices(targets, 6, 42)
    assert {targets[index] for index in first} == {0, 1, 2}
