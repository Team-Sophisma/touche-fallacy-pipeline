import json
from pathlib import Path

import pandas as pd


class SentimentReportWriter:
    def write_csv(self, rows: list[dict], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable_rows = [
            self._serialize_nested_columns(row)
            for row in rows
        ]

        pd.DataFrame(serializable_rows).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

    def write_json(self, data: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _serialize_nested_columns(self, row: dict) -> dict:
        serialized = dict(row)
        serialized["sentiment_scores"] = json.dumps(
            serialized.get("sentiment_scores", {}),
            ensure_ascii=False,
        )
        return serialized
