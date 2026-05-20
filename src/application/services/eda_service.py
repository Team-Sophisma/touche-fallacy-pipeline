from typing import Optional

from src.domain.interfaces import DatasetAdapter
from src.infrastructure.eda.dataset_profiler import DatasetProfiler
from src.infrastructure.eda.outlier_analyzer import OutlierAnalyzer
from src.infrastructure.persistence.eda_report_writer import EDAReportWriter
from src.infrastructure.visualization.eda_plots import EDAPlotter


class EDAService:
    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        profiler: DatasetProfiler,
        writer: EDAReportWriter,
        plotter: EDAPlotter | None = None,
        outlier_analyzer: Optional[OutlierAnalyzer] = None,
    ):
        self.dataset_adapter = dataset_adapter
        self.profiler = profiler
        self.writer = writer
        self.plotter = plotter
        self.outlier_analyzer = outlier_analyzer

    def run(self) -> None:
        train_samples = self.dataset_adapter.load_train()
        test_samples = self.dataset_adapter.load_test()

        train_df = self.profiler.samples_to_dataframe(train_samples)
        test_df = self.profiler.samples_to_dataframe(test_samples)

        train_summary = self.profiler.summarize_dataframe(train_df)
        test_summary = self.profiler.summarize_dataframe(test_df)

        self.writer.write_dataframe_csv(train_df, "train_eda_table.csv")
        self.writer.write_dataframe_csv(test_df, "test_eda_table.csv")

        self.writer.write_summary_json(train_summary, "train_summary.json")
        self.writer.write_summary_json(test_summary, "test_summary.json")

        cross_tables = self.profiler.create_cross_tables(train_df)
        self.writer.write_cross_tables(cross_tables)

        if self.outlier_analyzer is not None:
            self._write_outlier_report("train", train_df)
            self._write_outlier_report("test", test_df)

        if self.plotter:
            self._generate_plots(train_df)

    def _write_outlier_report(self, split_name: str, df):
        outlier_report = self.outlier_analyzer.analyze(df)
        self.writer.write_summary_json(
            outlier_report,
            f"{split_name}_outlier_analysis.json",
        )
        if split_name == "train":
            self.writer.write_summary_json(outlier_report, "outlier_analysis.json")

        flagged_df = self.outlier_analyzer.flag_dataframe(df)
        self.writer.write_dataframe_csv(
            flagged_df,
            f"{split_name}_outlier_candidates.csv",
        )

    def _generate_plots(self, train_df):
        self.plotter.plot_value_counts(
            train_df,
            "fallacy_exists",
            "fallacy_exists_distribution.png"
        )

        self.plotter.plot_value_counts(
            train_df,
            "fallacy_type",
            "fallacy_type_distribution.png"
        )

        self.plotter.plot_value_counts(
            train_df,
            "argument_scheme",
            "argument_scheme_distribution.png"
        )

        self.plotter.plot_histogram(
            train_df,
            "source_word_count",
            "source_word_count_distribution.png"
        )

        self.plotter.plot_histogram(
            train_df,
            "base_support_count",
            "base_support_count_distribution.png"
        )
