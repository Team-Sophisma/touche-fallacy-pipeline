
from collections import Counter
from pathlib import Path
import json

from src.domain.interfaces import DatasetAdapter
from src.infrastructure.persistence.readable_dataset_writer import ReadableDatasetWriter


class DatasetInspectionService:
    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        writer: ReadableDatasetWriter,
    ):
        self.dataset_adapter = dataset_adapter
        self.writer = writer

    def export_readable_dataset(
        self,
        split: str,
        csv_output_path: str,
        markdown_output_path: str,
    ) -> None:
        if split == "train":
            samples = self.dataset_adapter.load_train()
        elif split == "test":
            samples = self.dataset_adapter.load_test()
        else:
            raise ValueError(f"Unknown split: {split}")

        self.writer.write_csv(samples, csv_output_path)
        self.writer.write_markdown_preview(samples, markdown_output_path)

    def write_summary(
        self,
        output_path: str,
    ) -> None:
        samples = self.dataset_adapter.load_train()

        summary = {
            "total_train_samples": len(samples),
            "task_name": samples[0].task_name if samples else None,
            "label_distribution": dict(
                Counter(sample.label for sample in samples)
            ),
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )