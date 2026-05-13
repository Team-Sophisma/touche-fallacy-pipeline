from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class EDAPlotter:
    def __init__(self, plots_dir: str):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_value_counts(self, df: pd.DataFrame, column: str, filename: str) -> None:
        if column not in df.columns:
            return

        counts = df[column].fillna("None").value_counts()

        plt.figure(figsize=(10, 5))
        counts.plot(kind="bar")
        plt.title(f"{column} distribution")
        plt.xlabel(column)
        plt.ylabel("count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename)
        plt.close()

    def plot_histogram(self, df: pd.DataFrame, column: str, filename: str) -> None:
        if column not in df.columns:
            return

        plt.figure(figsize=(10, 5))
        df[column].dropna().plot(kind="hist", bins=30)
        plt.title(f"{column} distribution")
        plt.xlabel(column)
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename)
        plt.close()