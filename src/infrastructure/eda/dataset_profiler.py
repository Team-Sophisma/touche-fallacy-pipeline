from collections import Counter
from typing import Sequence

import pandas as pd

from src.domain.entities import TextSample


class DatasetProfiler:
    def samples_to_dataframe(self, samples: Sequence[TextSample]) -> pd.DataFrame:
        rows = []

        for sample in samples:
            meta = sample.meta

            argument_goal = meta.get("argument_goal")
            argument_basis = meta.get("argument_basis")

            argument_scheme = None
            if argument_goal and argument_basis:
                argument_scheme = f"{argument_goal}-{argument_basis}"

            rows.append({
                "id": sample.id,
                "task_name": sample.task_name,
                "label": sample.label,
                "source_text": sample.source_text,
                "argument_text": sample.argument_text,

                "source_word_count": self._word_count(sample.source_text),
                "argument_word_count": self._word_count(sample.argument_text),

                "fallacy_exists": meta.get("fallacy_exists"),
                "fallacy_type": meta.get("fallacy_type"),
                "resembles_fallacy": meta.get("resembles_fallacy"),

                "argument_goal": argument_goal,
                "argument_basis": argument_basis,
                "argument_scheme": argument_scheme,

                "text_raw_word_count": self._word_count(meta.get("text_raw", "")),
                "text_base_word_count": self._word_count(meta.get("text_base", "")),
                "text_enhanced_word_count": self._word_count(meta.get("text_enhanced", "")),

                "base_claim_word_count": self._word_count(meta.get("base_claim", "")),
                "enhanced_claim_word_count": self._word_count(meta.get("enhanced_claim", "")),

                "base_support_count": len(meta.get("base_supports", []) or []),
                "enhanced_support_count": len(meta.get("enhanced_supports", []) or []),
            })

        return pd.DataFrame(rows)

    def summarize_dataframe(self, df: pd.DataFrame) -> dict:
        return {
            "total_samples": len(df),
            "duplicate_id_count": int(df["id"].duplicated().sum()),
            "empty_source_text_count": int((df["source_text"].fillna("").str.strip() == "").sum()),
            "empty_argument_text_count": int((df["argument_text"].fillna("").str.strip() == "").sum()),

            "label_distribution": self._value_counts(df, "label"),
            "fallacy_exists_distribution": self._value_counts(df, "fallacy_exists"),
            "fallacy_type_distribution": self._value_counts(df, "fallacy_type"),
            "resembles_fallacy_distribution": self._value_counts(df, "resembles_fallacy"),
            "argument_goal_distribution": self._value_counts(df, "argument_goal"),
            "argument_basis_distribution": self._value_counts(df, "argument_basis"),
            "argument_scheme_distribution": self._value_counts(df, "argument_scheme"),

            "source_word_count_stats": self._numeric_stats(df, "source_word_count"),
            "argument_word_count_stats": self._numeric_stats(df, "argument_word_count"),
            "base_support_count_stats": self._numeric_stats(df, "base_support_count"),
        }

    def create_cross_tables(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        tables = {}

        tables["fallacy_exists_by_resembles"] = pd.crosstab(
            df["resembles_fallacy"],
            df["fallacy_exists"],
            dropna=False
        )

        tables["fallacy_type_by_argument_scheme"] = pd.crosstab(
            df["fallacy_type"],
            df["argument_scheme"],
            dropna=False
        )

        tables["argument_goal_by_basis"] = pd.crosstab(
            df["argument_goal"],
            df["argument_basis"],
            dropna=False
        )

        return tables

    def _word_count(self, text: str | None) -> int:
        if not text:
            return 0
        return len(str(text).split())

    def _value_counts(self, df: pd.DataFrame, column: str) -> dict:
        if column not in df.columns:
            return {}
        return df[column].fillna("None").value_counts().to_dict()

    def _numeric_stats(self, df: pd.DataFrame, column: str) -> dict:
        if column not in df.columns:
            return {}

        return {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "std": float(df[column].std()),
        }