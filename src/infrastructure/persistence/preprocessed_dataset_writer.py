import json
from pathlib import Path
from typing import Sequence

from src.domain.entities import PreprocessedSample
from src.domain.interfaces import PreprocessedDatasetWriter


class JsonlPreprocessedDatasetWriter(PreprocessedDatasetWriter):
    def write_jsonl(
        self,
        samples: Sequence[PreprocessedSample],
        output_path: str,
    ) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as file:
            for sample in samples:
                file.write(json.dumps(self._to_record(sample), ensure_ascii=False))
                file.write("\n")

    def write_summary(
        self,
        summary: dict,
        output_path: str,
    ) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _to_record(self, sample: PreprocessedSample) -> dict:
        return {
            "id": sample.id,
            "task": sample.task,
            "text": sample.text,
            "label": sample.label,
            "tag": sample.tag,
            "meta": sample.meta,
        }
