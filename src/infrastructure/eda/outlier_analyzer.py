from collections import defaultdict
from typing import Iterable

import pandas as pd

class OutlierAnalyzer:
    """
    Detects EDA outlier candidates without mutating or removing samples.
    """

    DEFAULT_METRICS = [
        "source_word_count",
        "argument_word_count",
        "text_raw_word_count",
        "text_base_word_count",
        "text_enhanced_word_count",
        "base_claim_word_count",
        "base_supports_word_count",
        "base_support_count",
        "enhanced_claim_word_count",
        "enhanced_supports_word_count",
        "enhanced_support_count",
    ]

    def __init__(
        self,
        metrics: Iterable[str] | None = None,
        group_columns: Iterable[str] | None = None,
        iqr_multiplier: float = 3.0,
        max_examples_per_metric: int = 50,
    ):
        self.metrics = list(metrics) if metrics is not None else list(self.DEFAULT_METRICS)
        self.group_columns = list(group_columns) if group_columns is not None else []
        self.iqr_multiplier = iqr_multiplier
        self.max_examples_per_metric = max_examples_per_metric

    def analyze(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return self._empty_report()

        metric_reports = []
        flags_by_id: dict[str, list[str]] = defaultdict(list)

        for metric in self._available_metrics(df):
            report = self._analyze_metric(df, metric)
            metric_reports.append(report)
            self._collect_flags(flags_by_id, report, "global")

        grouped_reports = {}
        for group_column in self.group_columns:
            if group_column not in df.columns:
                continue

            grouped_reports[group_column] = self._analyze_groups(df, group_column)
            for group_value, group_metrics in grouped_reports[group_column].items():
                for report in group_metrics:
                    self._collect_flags(
                        flags_by_id,
                        report,
                        f"{group_column}={group_value}",
                    )

        outlier_ids = self._outlier_ids(df, flags_by_id)

        return {
            "method": "iqr",
            "iqr_multiplier": float(self.iqr_multiplier),
            "metrics_analyzed": self._available_metrics(df),
            "group_columns": [
                column for column in self.group_columns if column in df.columns
            ],
            "total_samples": int(len(df)),
            "unique_outlier_count": int(len(outlier_ids)),
            "unique_outlier_ids": outlier_ids,
            "global": metric_reports,
            "grouped": grouped_reports,
        }

    def flag_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            flagged = df.copy()
            flagged["is_outlier"] = False
            flagged["outlier_reasons"] = ""
            return flagged

        report = self.analyze(df)
        reasons_by_id: dict[str, list[str]] = defaultdict(list)

        for metric_report in report["global"]:
            self._collect_reasons_by_id(reasons_by_id, metric_report, "global")

        for group_column, grouped_metrics in report["grouped"].items():
            for group_value, metric_reports in grouped_metrics.items():
                context = f"{group_column}={group_value}"
                for metric_report in metric_reports:
                    self._collect_reasons_by_id(reasons_by_id, metric_report, context)

        flagged = df.copy()
        ids = flagged["id"].astype(str) if "id" in flagged.columns else flagged.index.astype(str)
        flagged["is_outlier"] = ids.map(lambda value: value in reasons_by_id)
        flagged["outlier_reasons"] = ids.map(
            lambda value: "; ".join(sorted(set(reasons_by_id.get(value, []))))
        )

        return flagged[flagged["is_outlier"]].reset_index(drop=True)

    def _analyze_groups(self, df: pd.DataFrame, group_column: str) -> dict[str, list[dict]]:
        grouped_reports = {}

        for group_value, group_df in df.groupby(group_column, dropna=False):
            if len(group_df) < 4:
                continue

            reports = []
            for metric in self._available_metrics(group_df):
                reports.append(self._analyze_metric(group_df, metric))

            grouped_reports[str(group_value)] = reports

        return grouped_reports

    def _analyze_metric(self, df: pd.DataFrame, metric: str) -> dict:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()

        if series.empty:
            return self._empty_metric_report(metric)

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_rows = df.loc[outlier_mask.index[outlier_mask]]

        return {
            "metric": metric,
            "sample_count": int(series.count()),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(len(outlier_rows)),
            "outlier_ids": self._sample_ids(outlier_rows),
            "examples": self._examples(outlier_rows, metric),
        }

    def _available_metrics(self, df: pd.DataFrame) -> list[str]:
        return [metric for metric in self.metrics if metric in df.columns]

    def _collect_flags(
        self,
        flags_by_id: dict[str, list[str]],
        metric_report: dict,
        context: str,
    ) -> None:
        reason = f"{context}:{metric_report['metric']}"
        for sample_id in metric_report.get("outlier_ids", []):
            flags_by_id[str(sample_id)].append(reason)

    def _collect_reasons_by_id(
        self,
        reasons_by_id: dict[str, list[str]],
        metric_report: dict,
        context: str,
    ) -> None:
        metric = metric_report["metric"]
        for sample_id in metric_report.get("outlier_ids", []):
            reasons_by_id[str(sample_id)].append(f"{context}:{metric}")

    def _outlier_ids(
        self,
        df: pd.DataFrame,
        flags_by_id: dict[str, list[str]],
    ) -> list[str]:
        if "id" not in df.columns:
            return sorted(flags_by_id)

        return [
            str(sample_id)
            for sample_id in df["id"].astype(str)
            if sample_id in flags_by_id
        ]

    def _sample_ids(self, rows: pd.DataFrame) -> list[str]:
        if "id" not in rows.columns:
            return [str(index) for index in rows.index]

        return [str(value) for value in rows["id"]]

    def _examples(self, rows: pd.DataFrame, metric: str) -> list[dict]:
        if rows.empty:
            return []

        sorted_rows = rows.copy()
        sorted_rows["_distance"] = (
            pd.to_numeric(sorted_rows[metric], errors="coerce")
            - pd.to_numeric(sorted_rows[metric], errors="coerce").median()
        ).abs()
        sorted_rows = sorted_rows.sort_values("_distance", ascending=False)

        examples = []
        for _, row in sorted_rows.head(self.max_examples_per_metric).iterrows():
            examples.append({
                "id": str(row.get("id", row.name)),
                "label": self._nullable_str(row.get("label")),
                "fallacy_type": self._nullable_str(row.get("fallacy_type")),
                "value": float(row[metric]),
            })

        return examples

    def _empty_report(self) -> dict:
        return {
            "method": "iqr",
            "iqr_multiplier": float(self.iqr_multiplier),
            "metrics_analyzed": [],
            "group_columns": [],
            "total_samples": 0,
            "unique_outlier_count": 0,
            "unique_outlier_ids": [],
            "global": [],
            "grouped": {},
        }

    def _empty_metric_report(self, metric: str) -> dict:
        return {
            "metric": metric,
            "sample_count": 0,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_ids": [],
            "examples": [],
        }

    def _nullable_str(self, value) -> str | None:
        if pd.isna(value):
            return None

        return str(value)
