import json
from pathlib import Path

import pandas as pd


class MaskedDatasetWriter:
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

    def _serialize_nested_columns(self, row: dict) -> dict:
        serialized = dict(row)

        for column in ("entities", "entity_roles"):
            serialized[column] = json.dumps(
                serialized.get(column, []),
                ensure_ascii=False,
            )

        return serialized
