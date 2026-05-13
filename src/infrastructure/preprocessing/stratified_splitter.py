import random
from collections import defaultdict
from typing import Sequence

from src.domain.entities import PreprocessedSample
from src.domain.interfaces import DatasetSplitter


class StratifiedDatasetSplitter(DatasetSplitter):
    def __init__(
        self,
        validation_size: float,
        random_seed: int,
    ):
        if validation_size <= 0 or validation_size >= 1:
            raise ValueError("validation_size must be between 0 and 1.")

        self.validation_size = validation_size
        self.random_seed = random_seed

    def split(
        self,
        samples: Sequence[PreprocessedSample],
    ) -> tuple[list[PreprocessedSample], list[PreprocessedSample]]:
        grouped_samples = self._group_by_label(samples)

        train_samples = []
        validation_samples = []

        for label, label_samples in grouped_samples.items():
            shuffled = list(label_samples)
            random.Random(self._seed_for_label(label)).shuffle(shuffled)

            validation_count = self._validation_count(len(shuffled))

            validation_samples.extend(shuffled[:validation_count])
            train_samples.extend(shuffled[validation_count:])

        self._sort_by_id(train_samples)
        self._sort_by_id(validation_samples)

        return train_samples, validation_samples

    def _group_by_label(
        self,
        samples: Sequence[PreprocessedSample],
    ) -> dict[str, list[PreprocessedSample]]:
        grouped_samples = defaultdict(list)

        for sample in samples:
            grouped_samples[sample.label or "__missing__"].append(sample)

        return dict(grouped_samples)

    def _validation_count(self, sample_count: int) -> int:
        if sample_count <= 1:
            return 0

        count = round(sample_count * self.validation_size)

        return max(1, min(sample_count - 1, count))

    def _seed_for_label(self, label: str) -> int:
        label_offset = sum(ord(character) for character in label)
        return self.random_seed + label_offset

    def _sort_by_id(self, samples: list[PreprocessedSample]) -> None:
        samples.sort(key=lambda sample: sample.id)
