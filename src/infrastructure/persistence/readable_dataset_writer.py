
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.domain.entities import TextSample


class ReadableDatasetWriter:
    def write_csv(
        self,
        samples: Sequence[TextSample],
        output_path: str
    ) -> None:
        rows = [self._sample_to_row(sample) for sample in samples]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)
        try:
            df.to_csv(output, index=False, encoding="utf-8-sig")
        except PermissionError as error:
            raise PermissionError(
                f"Cannot write CSV output to {output}. Close the file if it is "
                "open in Excel or another program, then run the script again."
            ) from error

    def write_markdown_preview(
        self,
        samples: Sequence[TextSample],
        output_path: str,
        limit: int = 30
    ) -> None:
        rows = [self._sample_to_row(sample) for sample in samples[:limit]]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            output.write_text(self._rows_to_markdown(rows), encoding="utf-8")
        except PermissionError as error:
            raise PermissionError(
                f"Cannot write markdown output to {output}. Close the file if it "
                "is open in another program, then run the script again."
            ) from error

    def _rows_to_markdown(self, rows: list[dict]) -> str:
        if not rows:
            return ""

        headers = list(rows[0].keys())
        table = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]

        for row in rows:
            table.append(
                "| "
                + " | ".join(self._markdown_cell(row.get(header)) for header in headers)
                + " |"
            )

        return "\n".join(table)

    def _markdown_cell(self, value: object) -> str:
        text = "" if value is None else str(value)
        return (
            text.replace("\\", "\\\\")
            .replace("\r\n", "<br>")
            .replace("\n", "<br>")
            .replace("|", "\\|")
        )

    def _sample_to_row(self, sample: TextSample) -> dict:
        meta = sample.meta

        return {
            "id": sample.id,
            "task_name": sample.task_name,
            "label": sample.label,

            "source_text": sample.source_text,
            "argument_text": sample.argument_text,

            "title": meta.get("title"),
            "parent_text": meta.get("parent_text"),

            "text_raw": meta.get("text_raw"),
            "text_base": meta.get("text_base"),
            "text_enhanced": meta.get("text_enhanced"),

            "base_claim": meta.get("base_claim"),
            "base_supports": " | ".join(meta.get("base_supports", [])),

            "enhanced_claim": meta.get("enhanced_claim"),
            "enhanced_supports": " | ".join(meta.get("enhanced_supports", [])),

            "fallacy_exists": meta.get("fallacy_exists"),
            "fallacy_type": meta.get("fallacy_type"),
            "resembles_fallacy": meta.get("resembles_fallacy"),

            "argument_goal": meta.get("argument_goal"),
            "argument_basis": meta.get("argument_basis"),
        }   
