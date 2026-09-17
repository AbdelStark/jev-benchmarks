from __future__ import annotations

from typing import Protocol

from ..models import Example, Prediction


class Backend(Protocol):
    name: str

    def warmup(self, example: Example) -> None: ...

    def predict(self, experiment_id: str, example: Example) -> Prediction: ...

    def close(self) -> None: ...
