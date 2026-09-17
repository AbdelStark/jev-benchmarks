from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from .config import BenchmarkConfig
from .io import write_jsonl
from .models import Example


def _class_count(texts: list[str]) -> int:
    first = texts[0]
    for index in range(1, len(texts)):
        if texts[index] != first:
            return index
    raise ValueError("could not infer class count from repeated BTZSC texts")


def _balanced_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    chosen: list[int] = []
    classes = sorted(by_class)
    while len(chosen) < min(limit, len(targets)):
        made_progress = False
        for class_id in classes:
            if by_class[class_id] and len(chosen) < limit:
                chosen.append(by_class[class_id].pop())
                made_progress = True
        if not made_progress:
            break
    return sorted(chosen)


def load_examples(config: BenchmarkConfig) -> list[Example]:
    spec = config.raw["dataset"]
    output: list[Example] = []
    for dataset_offset, dataset_spec in enumerate(spec["datasets"]):
        name = dataset_spec["name"]
        rows = load_dataset(
            spec["repository"],
            name=name,
            split="test",
            revision=spec["revision"],
            cache_dir=str(config.output_dir.parent / "cache"),
        )
        binary = [int(value) for value in rows["labels"]]
        texts = [str(value) for value in rows["text"]]
        n_classes = _class_count(texts)
        total = len(rows) // n_classes
        labels = tuple(str(rows[index]["hypothesis"]) for index in range(n_classes))
        valid_sample_indices: list[int] = []
        targets: list[int] = []
        for sample_index in range(total):
            offset = sample_index * n_classes
            values = binary[offset : offset + n_classes]
            if sum(values) == 1:
                valid_sample_indices.append(sample_index)
                targets.append(values.index(1))
        selected_positions = _balanced_indices(
            targets,
            int(spec["samples_per_dataset"]),
            config.seed + dataset_offset,
        )
        for position in selected_positions:
            sample_index = valid_sample_indices[position]
            offset = sample_index * n_classes
            text = texts[offset]
            output.append(
                Example(
                    dataset=name,
                    task=str(dataset_spec["task"]),
                    example_id=f"{name}:{sample_index}",
                    text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    labels=labels,
                    target_index=targets[position],
                )
            )
    return output


def prepare_manifest(config: BenchmarkConfig) -> Path:
    examples = load_examples(config)
    path = config.output_dir / "manifest.jsonl"
    write_jsonl(path, [example.to_dict() for example in examples])
    return path
