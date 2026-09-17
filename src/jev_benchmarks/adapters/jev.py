from __future__ import annotations

import time

from typesafe_sdk import Choice, TypeSafeClient

from ..models import Example, Prediction


class JevBackend:
    name = "jev"

    def __init__(self, model_id: str, question: str) -> None:
        self.model_id = model_id
        self.question = question
        self.client = TypeSafeClient(model=model_id)

    @staticmethod
    def _criteria(labels: tuple[str, ...]) -> dict[str, str]:
        return {f"label_{index:03d}": label for index, label in enumerate(labels)}

    def warmup(self, example: Example) -> None:
        # Hosted latency includes connection establishment in the first measured call. Avoiding an
        # unscored API call keeps billed inference fully accounted for.
        return None

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        criteria = self._criteria(example.labels)
        start = time.perf_counter()
        response = self.client.system_one(
            state={"text": example.text},
            questions={"label": Choice(instructions=self.question, criteria=criteria)},
            model=self.model_id,
        )
        latency = time.perf_counter() - start
        answer = response.choices["label"]
        probabilities = tuple(float(answer.probabilities[key]) for key in criteria)
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Prediction(
            experiment_id=experiment_id,
            backend=self.name,
            model_requested=self.model_id,
            model_resolved=response.model,
            dataset=example.dataset,
            example_id=example.example_id,
            target_index=example.target_index,
            predicted_index=predicted,
            labels=example.labels,
            probabilities=probabilities,
            latency_seconds=latency,
            input_tokens=response.usage.input_tokens,
        )

    def close(self) -> None:
        self.client.close()
