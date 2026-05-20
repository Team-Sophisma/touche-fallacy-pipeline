import pandas as pd


class HardNegativeAnalyzer:
    def __init__(self, keyword_extractor):
        self.keyword_extractor = keyword_extractor

    def analyze(self, df: pd.DataFrame, fallacy_type: str, limit: int = 10) -> list[dict]:
        if df.empty or "fallacy_type" not in df.columns:
            return []

        fallacy_rows = df[df["fallacy_type"] == fallacy_type]
        keywords = self.keyword_extractor.extract(fallacy_rows.get("argument_text", []))

        if not keywords or "argument_text" not in df.columns:
            return []

        negatives = df[df["fallacy_type"] != fallacy_type].copy()
        negatives["_keyword_hits"] = negatives["argument_text"].fillna("").astype(str).str.lower().apply(
            lambda text: sum(1 for keyword in keywords if keyword in text)
        )

        rows = negatives[negatives["_keyword_hits"] > 0].sort_values(
            "_keyword_hits",
            ascending=False,
        ).head(limit)

        return rows.drop(columns=["_keyword_hits"], errors="ignore").to_dict("records")
