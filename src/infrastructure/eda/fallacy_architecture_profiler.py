from collections import Counter
import re

import pandas as pd

from src.domain.fallacy_profile import FallacyProfile


class FallacyArchitectureProfiler:
    def __init__(self, knowledge_base, hard_negative_analyzer, example_sampler):
        self.knowledge_base = knowledge_base
        self.hard_negative_analyzer = hard_negative_analyzer
        self.example_sampler = example_sampler

    def profile(self, df: pd.DataFrame, examples_per_fallacy: int = 5) -> list[FallacyProfile]:
        if df.empty or "fallacy_type" not in df.columns:
            return []

        profiles = []
        fallacy_types = sorted(
            value for value in df["fallacy_type"].dropna().unique()
            if str(value).strip()
        )

        for fallacy_type in fallacy_types:
            fallacy_type = str(fallacy_type)
            architecture = self.knowledge_base.describe(fallacy_type)
            normalized_fallacy_type = architecture.get("fallacy_type", fallacy_type)
            fallacious_rows = df[df["fallacy_type"].astype(str) == fallacy_type]
            valid_lookalike_rows = self._valid_lookalike_rows(df, fallacy_type)
            hard_negatives = self.hard_negative_analyzer.analyze(df, fallacy_type)

            profile = FallacyProfile(
                fallacy_type=normalized_fallacy_type,
                display_name=architecture.get(
                    "display_name",
                    self._display_name(normalized_fallacy_type),
                ),
                total_fallacious=int(len(fallacious_rows)),
                total_valid_lookalikes=int(len(valid_lookalike_rows)),
                definition=architecture.get("definition", ""),
                claim_pattern=architecture.get("claim_pattern", ""),
                support_pattern=architecture.get("support_pattern", ""),
                hidden_reasoning_bridge=architecture.get("hidden_reasoning_bridge", ""),
                failure_point=architecture.get("failure_point", ""),
                valid_lookalike=architecture.get("valid_lookalike", ""),
                common_cues=list(architecture.get("common_cues", [])),
                confusion_risks=list(architecture.get("confusion_risks", [])),
                top_keywords_fallacious=self._top_keywords(
                    fallacious_rows.get("argument_text", []),
                ),
                top_keywords_valid_lookalikes=self._top_keywords(
                    valid_lookalike_rows.get("argument_text", []),
                ),
                fallacious_scheme_distribution=self._value_counts(
                    fallacious_rows,
                    "argument_scheme",
                ),
                valid_lookalike_scheme_distribution=self._value_counts(
                    valid_lookalike_rows,
                    "argument_scheme",
                ),
                fallacious_argument_goal_distribution=self._value_counts(
                    fallacious_rows,
                    "argument_goal",
                ),
                valid_lookalike_argument_goal_distribution=self._value_counts(
                    valid_lookalike_rows,
                    "argument_goal",
                ),
                fallacious_argument_basis_distribution=self._value_counts(
                    fallacious_rows,
                    "argument_basis",
                ),
                valid_lookalike_argument_basis_distribution=self._value_counts(
                    valid_lookalike_rows,
                    "argument_basis",
                ),
                avg_fallacious_source_words=self._mean(
                    fallacious_rows,
                    "source_word_count",
                ),
                avg_valid_lookalike_source_words=self._mean(
                    valid_lookalike_rows,
                    "source_word_count",
                ),
                avg_fallacious_support_count=self._mean(
                    fallacious_rows,
                    "enhanced_support_count",
                ),
                avg_valid_lookalike_support_count=self._mean(
                    valid_lookalike_rows,
                    "enhanced_support_count",
                ),
                representative_fallacious_examples=self._sample_examples(
                    fallacious_rows,
                    examples_per_fallacy,
                ),
                representative_valid_lookalikes=self._sample_examples(
                    valid_lookalike_rows,
                    examples_per_fallacy,
                ),
                interpretation_notes=self._interpretation_notes(
                    architecture,
                    hard_negatives,
                ),
            )
            profiles.append(profile)

        return profiles

    def architecture_matrix(self, profiles: list[FallacyProfile]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fallacy_type": profile.fallacy_type,
                    "display_name": profile.display_name,
                    "total_fallacious": profile.total_fallacious,
                    "total_valid_lookalikes": profile.total_valid_lookalikes,
                    "definition": profile.definition,
                    "claim_pattern": profile.claim_pattern,
                    "support_pattern": profile.support_pattern,
                    "failure_point": profile.failure_point,
                    "valid_lookalike": profile.valid_lookalike,
                    "confusion_risks": "; ".join(profile.confusion_risks),
                }
                for profile in profiles
            ]
        )

    def _valid_lookalike_rows(
        self,
        df: pd.DataFrame,
        fallacy_type: str,
    ) -> pd.DataFrame:
        if "resembles_fallacy" not in df.columns:
            return df.iloc[0:0]

        resembles = df["resembles_fallacy"].fillna("").astype(str) == fallacy_type

        if "fallacy_exists" not in df.columns:
            return df[resembles]

        return df[resembles & ~self._truthy_series(df["fallacy_exists"])]

    def _truthy_series(self, series: pd.Series) -> pd.Series:
        normalized = series.fillna("").astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "fallacy", "yes"})

    def _top_keywords(self, texts, top_n: int = 20) -> list[tuple[str, int]]:
        stopwords = getattr(
            self.hard_negative_analyzer.keyword_extractor,
            "_stopwords",
            set(),
        )
        counter: Counter[str] = Counter()

        for text in texts:
            words = re.findall(r"[A-Za-z][A-Za-z']+", str(text or "").lower())
            counter.update(word for word in words if word not in stopwords)

        return [(word, int(count)) for word, count in counter.most_common(top_n)]

    def _value_counts(self, df: pd.DataFrame, column: str) -> dict[str, int]:
        if df.empty or column not in df.columns:
            return {}

        return {
            str(key): int(value)
            for key, value in df[column].fillna("None").value_counts().items()
        }

    def _mean(self, df: pd.DataFrame, column: str) -> float:
        if df.empty or column not in df.columns:
            return 0.0

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            return 0.0

        return round(float(values.mean()), 6)

    def _sample_examples(self, df: pd.DataFrame, limit: int) -> list[dict]:
        if df.empty:
            return []

        count = min(limit, len(df))
        random_seed = getattr(self.example_sampler, "random_seed", 42)
        sampled = df.sample(n=count, random_state=random_seed)
        return [self._format_example(row) for _, row in sampled.iterrows()]

    def _format_example(self, row: pd.Series) -> dict:
        return {
            "id": self._clean(row.get("id")),
            "fallacy_exists": self._clean(row.get("fallacy_exists")),
            "fallacy_type": self._clean(row.get("fallacy_type")),
            "resembles_fallacy": self._clean(row.get("resembles_fallacy")),
            "argument_scheme": self._clean(row.get("argument_scheme")),
            "source_text": self._clean(row.get("source_text")),
            "claim": self._clean(row.get("enhanced_claim") or row.get("base_claim")),
            "supports": self._clean(
                row.get("enhanced_supports_text")
                or row.get("base_supports_text")
                or row.get("argument_text")
            ),
        }

    def _interpretation_notes(
        self,
        architecture: dict,
        hard_negatives: list[dict],
    ) -> list[str]:
        notes = list(architecture.get("notes", []))
        if hard_negatives:
            notes.append(
                f"Found {len(hard_negatives)} keyword-similar candidates in other classes."
            )
        return notes

    def _display_name(self, fallacy_type: str) -> str:
        return fallacy_type.replace("_", " ").replace("-", " ").title()

    def _clean(self, value):
        if value is None:
            return ""

        if isinstance(value, float) and pd.isna(value):
            return ""

        return value
