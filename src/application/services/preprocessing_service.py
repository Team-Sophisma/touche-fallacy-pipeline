from collections import Counter
from pathlib import Path
from typing import Sequence

from src.domain.entities import PreprocessedSample
from src.domain.interfaces import (
    DatasetSplitter,
    InputBuilder,
    LabelMapper,
    PreprocessedDatasetWriter,
)


class PreprocessingService:
    def __init__(
        self,
        input_builder: InputBuilder,
        label_mapper: LabelMapper,
        splitter: DatasetSplitter,
        writer: PreprocessedDatasetWriter,
        tag: str,
        output_dir: str,
    ):
        self.input_builder = input_builder
        self.label_mapper = label_mapper
        self.splitter = splitter
        self.writer = writer
        self.tag = tag
        self.output_dir = Path(output_dir)

    def run(
        self,
        train_items: Sequence[dict],
        test_items: Sequence[dict],
        task_names: Sequence[str],
    ) -> None:
        for task_name in task_names:
            self._run_task(train_items, test_items, task_name)

    def _run_task(
        self,
        train_items: Sequence[dict],
        test_items: Sequence[dict],
        task_name: str,
    ) -> None:
        train_samples = self._build_train_samples(train_items, task_name)
        test_samples = self._build_test_samples(test_items, task_name)

        train_split, validation_split = self.splitter.split(train_samples)

        task_output_dir = self.output_dir / task_name
        self.writer.write_jsonl(
            train_split,
            str(task_output_dir / "train.jsonl"),
        )
        self.writer.write_jsonl(
            validation_split,
            str(task_output_dir / "validation.jsonl"),
        )
        self.writer.write_jsonl(
            test_samples,
            str(task_output_dir / "test.jsonl"),
        )
        self.writer.write_summary(
            self._build_summary(
                task_name=task_name,
                train_samples=train_split,
                validation_samples=validation_split,
                test_samples=test_samples,
            ),
            str(task_output_dir / "summary.json"),
        )

    def _build_train_samples(
        self,
        items: Sequence[dict],
        task_name: str,
    ) -> list[PreprocessedSample]:
        samples = []

        for item in items:
            if not self.label_mapper.should_use_for_training(item, task_name):
                continue

            label = self.label_mapper.map_label(item, task_name)
            if label is None:
                continue

            samples.append(self._build_sample(item, task_name, label))

        return samples

    def _build_test_samples(
        self,
        items: Sequence[dict],
        task_name: str,
    ) -> list[PreprocessedSample]:
        return [
            self._build_sample(item, task_name, label=None)
            for item in items
        ]

    def _build_sample(
        self,
        item: dict,
        task_name: str,
        label: str | None,
    ) -> PreprocessedSample:
        return PreprocessedSample(
            id=item["id"],
            task=task_name,
            text=self.input_builder.build(item),
            label=label,
            tag=self.tag,
            meta={
                "source_id": item["id"],
            },
        )

    def _build_summary(
        self,
        task_name: str,
        train_samples: Sequence[PreprocessedSample],
        validation_samples: Sequence[PreprocessedSample],
        test_samples: Sequence[PreprocessedSample],
    ) -> dict:
        all_train_samples = list(train_samples) + list(validation_samples)

        return {
            "task": task_name,
            "tag": self.tag,
            "train_count": len(train_samples),
            "validation_count": len(validation_samples),
            "test_count": len(test_samples),
            "label_distribution": dict(
                Counter(sample.label for sample in all_train_samples)
            ),
        }
