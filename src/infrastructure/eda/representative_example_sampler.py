import pandas as pd


class RepresentativeExampleSampler:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def sample(self, df: pd.DataFrame, fallacy_type: str, limit: int = 5) -> list[dict]:
        if df.empty or "fallacy_type" not in df.columns:
            return []

        rows = df[df["fallacy_type"] == fallacy_type]
        if rows.empty:
            return []

        count = min(limit, len(rows))
        return rows.sample(n=count, random_state=self.random_seed).to_dict("records")
