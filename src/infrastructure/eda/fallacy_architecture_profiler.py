import pandas as pd

from src.domain.fallacy_profile import FallacyProfile


class FallacyArchitectureProfiler:
    def __init__(self, knowledge_base, hard_negative_analyzer, example_sampler):
        self.knowledge_base = knowledge_base
        self.hard_negative_analyzer = hard_negative_analyzer
        self.example_sampler = example_sampler

    def profile(self, df: pd.DataFrame, examples_per_fallacy: int = 5) -> dict:
        if df.empty or "fallacy_type" not in df.columns:
            return {"fallacies": []}

        profiles = []
        fallacy_types = sorted(
            value for value in df["fallacy_type"].dropna().unique()
            if str(value).strip()
        )

        for fallacy_type in fallacy_types:
            rows = df[df["fallacy_type"] == fallacy_type]
            examples = self.example_sampler.sample(df, fallacy_type, examples_per_fallacy)
            hard_negatives = self.hard_negative_analyzer.analyze(df, fallacy_type)
            keywords = self.hard_negative_analyzer.keyword_extractor.extract(
                rows.get("argument_text", []),
                top_n=20,
            )

            profile = FallacyProfile(
                fallacy_type=str(fallacy_type),
                sample_count=int(len(rows)),
                common_keywords=keywords,
                representative_examples=examples,
                hard_negatives=hard_negatives,
                notes=self.knowledge_base.describe(str(fallacy_type)),
            )
            profiles.append(profile.__dict__)

        return {"fallacies": profiles}
