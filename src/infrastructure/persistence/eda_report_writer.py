import json
from pathlib import Path

import pandas as pd


class EDAReportWriter:
    def __init__(self, report_dir: str):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write_summary_json(self, summary: dict, filename: str) -> None:
        output_path = self.report_dir / filename
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def write_dataframe_csv(self, df: pd.DataFrame, filename: str) -> None:
        output_path = self.report_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def write_cross_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        cross_table_dir = self.report_dir / "cross_tables"
        cross_table_dir.mkdir(parents=True, exist_ok=True)

        for name, table in tables.items():
            table.to_csv(cross_table_dir / f"{name}.csv", encoding="utf-8-sig")