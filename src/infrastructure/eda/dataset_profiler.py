from typing import Sequence

import pandas as pd

from src.domain.entities import TextSample


class DatasetProfiler:
    """
    Converts TextSample objects into a pandas DataFrame suitable for EDA.
    """

    def samples_to_dataframe(self, samples: Sequence[TextSample]) -> pd.DataFrame:
        rows = []

        for sample in samples:
            meta = sample.meta or {}

            argument_goal = meta.get("argument_goal")
            argument_basis = meta.get("argument_basis")

            argument_scheme = None
            if argument_goal and argument_basis:
                argument_scheme = f"{argument_goal}-{argument_basis}"

            base_supports = self._safe_list(meta.get("base_supports"))
            enhanced_supports = self._safe_list(meta.get("enhanced_supports"))

            base_supports_text = " ".join(base_supports)
            enhanced_supports_text = " ".join(enhanced_supports)

            rows.append({
                "id": sample.id,
                "task_name": sample.task_name,
                "label": sample.label,

                "source_text": sample.source_text,
                "argument_text": sample.argument_text,

                "source_word_count": self._word_count(sample.source_text),
                "argument_word_count": self._word_count(sample.argument_text),

                "title": meta.get("title"),
                "parent_text": meta.get("parent_text"),

                "text_raw": meta.get("text_raw"),
                "text_base": meta.get("text_base"),
                "text_enhanced": meta.get("text_enhanced"),

                "text_raw_word_count": self._word_count(meta.get("text_raw")),
                "text_base_word_count": self._word_count(meta.get("text_base")),
                "text_enhanced_word_count": self._word_count(meta.get("text_enhanced")),

                "base_claim": meta.get("base_claim"),
                "base_supports": base_supports,
                "base_supports_text": base_supports_text,
                "base_claim_word_count": self._word_count(meta.get("base_claim")),
                "base_supports_word_count": self._word_count(base_supports_text),
                "base_support_count": len(base_supports),

                "enhanced_claim": meta.get("enhanced_claim"),
                "enhanced_supports": enhanced_supports,
                "enhanced_supports_text": enhanced_supports_text,
                "enhanced_claim_word_count": self._word_count(meta.get("enhanced_claim")),
                "enhanced_supports_word_count": self._word_count(enhanced_supports_text),
                "enhanced_support_count": len(enhanced_supports),

                "fallacy_exists": meta.get("fallacy_exists"),
                "fallacy_type": meta.get("fallacy_type"),
                "resembles_fallacy": meta.get("resembles_fallacy"),

                "argument_goal": argument_goal,
                "argument_basis": argument_basis,
                "argument_scheme": argument_scheme,
            })

        return pd.DataFrame(rows)

    def summarize_dataframe(self, df: pd.DataFrame) -> dict:
        return {
            "total_samples": int(len(df)),
            "duplicate_id_count": int(df["id"].duplicated().sum()) if "id" in df else 0,
            "empty_source_text_count": self._empty_text_count(df, "source_text"),
            "empty_argument_text_count": self._empty_text_count(df, "argument_text"),

            "label_distribution": self._value_counts(df, "label"),
            "fallacy_exists_distribution": self._value_counts(df, "fallacy_exists"),
            "fallacy_type_distribution": self._value_counts(df, "fallacy_type"),
            "resembles_fallacy_distribution": self._value_counts(df, "resembles_fallacy"),
            "argument_goal_distribution": self._value_counts(df, "argument_goal"),
            "argument_basis_distribution": self._value_counts(df, "argument_basis"),
            "argument_scheme_distribution": self._value_counts(df, "argument_scheme"),

            "source_word_count_stats": self._numeric_stats(df, "source_word_count"),
            "argument_word_count_stats": self._numeric_stats(df, "argument_word_count"),
            "base_claim_word_count_stats": self._numeric_stats(df, "base_claim_word_count"),
            "base_supports_word_count_stats": self._numeric_stats(df, "base_supports_word_count"),
            "base_support_count_stats": self._numeric_stats(df, "base_support_count"),
        }

    def create_cross_tables(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "fallacy_exists_by_resembles": self._cross_table(
                df,
                "fallacy_exists",
                "resembles_fallacy",
            ),
            "fallacy_type_by_argument_scheme": self._cross_table(
                df,
                "fallacy_type",
                "argument_scheme",
            ),
            "argument_goal_by_basis": self._cross_table(
                df,
                "argument_goal",
                "argument_basis",
            ),
        }

    def _safe_list(self, value) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item) for item in value if item is not None]

        return [str(value)]

    def _word_count(self, text) -> int:
        if text is None:
            return 0

        text = str(text).strip()

        if not text:
            return 0

        return len(text.split())

    def _empty_text_count(self, df: pd.DataFrame, column: str) -> int:
        if column not in df.columns:
            return 0

        return int((df[column].fillna("").astype(str).str.strip() == "").sum())

    def _value_counts(self, df: pd.DataFrame, column: str) -> dict:
        if column not in df.columns:
            return {}

        return {
            str(key): int(value)
            for key, value in df[column].fillna("None").value_counts().items()
        }

    def _numeric_stats(self, df: pd.DataFrame, column: str) -> dict:
        if column not in df.columns or df.empty:
            return {}

        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            return {}

        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
        }

    def _cross_table(
        self,
        df: pd.DataFrame,
        row_column: str,
        column_column: str,
    ) -> pd.DataFrame:
        if row_column not in df.columns or column_column not in df.columns:
            return pd.DataFrame()

        return pd.crosstab(
            df[row_column].fillna("None"),
            df[column_column].fillna("None"),
        )
