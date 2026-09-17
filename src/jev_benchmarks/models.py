from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Example:
    dataset: str
    task: str
    example_id: str
    text: str
    text_sha256: str
    labels: tuple[str, ...]
    target_index: int

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["labels"] = list(self.labels)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Example:
        return cls(**{**row, "labels": tuple(row["labels"])})


@dataclass(frozen=True)
class Prediction:
    experiment_id: str
    backend: str
    model_requested: str
    model_resolved: str
    dataset: str
    example_id: str
    target_index: int
    predicted_index: int
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    latency_seconds: float
    input_tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["labels"] = list(self.labels)
        row["probabilities"] = list(self.probabilities)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Prediction:
        return cls(
            **{
                **row,
                "labels": tuple(row["labels"]),
                "probabilities": tuple(row["probabilities"]),
            }
        )
