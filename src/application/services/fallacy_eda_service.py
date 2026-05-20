from src.domain.interfaces import DatasetAdapter
from src.infrastructure.eda.dataset_profiler import DatasetProfiler


class FallacyEDAService:
    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        dataset_profiler: DatasetProfiler,
        architecture_profiler,
        writer,
        outlier_analyzer=None,
        extra_splits: dict | None = None,
    ):
        self.dataset_adapter = dataset_adapter
        self.dataset_profiler = dataset_profiler
        self.architecture_profiler = architecture_profiler
        self.writer = writer
        self.outlier_analyzer = outlier_analyzer
        self.extra_splits = extra_splits or {}

    def run(self, examples_per_fallacy: int = 5) -> None:
        self._run_split("train", self.dataset_adapter.load_train(), examples_per_fallacy)
        self._run_split("test", self.dataset_adapter.load_test(), examples_per_fallacy)
        for split_name, samples in self.extra_splits.items():
            self._run_split(split_name, samples, examples_per_fallacy)

    def _run_split(self, split_name: str, samples, examples_per_fallacy: int) -> None:
        df = self.dataset_profiler.samples_to_dataframe(samples)
        summary = self.dataset_profiler.summarize_dataframe(df)
        architecture_profile = self.architecture_profiler.profile(
            df,
            examples_per_fallacy=examples_per_fallacy,
        )

        self.writer.write_dataframe_csv(df, f"{split_name}_fallacy_profile.csv")
        self.writer.write_json(summary, f"{split_name}_summary.json")
        self.writer.write_json(
            architecture_profile,
            f"{split_name}_fallacy_architecture_profile.json",
        )

        if self.outlier_analyzer is not None:
            self.writer.write_json(
                self.outlier_analyzer.analyze(df),
                f"{split_name}_outlier_analysis.json",
            )
            self.writer.write_dataframe_csv(
                self.outlier_analyzer.flag_dataframe(df),
                f"{split_name}_outlier_candidates.csv",
            )
