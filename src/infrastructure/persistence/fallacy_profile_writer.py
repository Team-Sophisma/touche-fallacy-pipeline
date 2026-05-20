import json
from pathlib import Path

import pandas as pd


class FallacyProfileWriter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_dataframe_csv(self, df: pd.DataFrame, filename: str) -> None:
        df.to_csv(self.output_dir / filename, index=False, encoding="utf-8-sig")

    def write_json(self, data: dict, filename: str) -> None:
        (self.output_dir / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
