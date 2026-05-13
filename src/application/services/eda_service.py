from src.domain.interfaces import DatasetAdapter
from src.infrastructure.eda.dataset_profiler import DatasetProfiler
from src.infrastructure.persistence.eda_report_writer import EDAReportWriter
from src.infrastructure.visualization.eda_plots import EDAPlotter


class EDAService:
    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        profiler: DatasetProfiler,
        writer: EDAReportWriter,
        plotter: EDAPlotter | None = None,
    ):
        self.dataset_adapter = dataset_adapter
        self.profiler = profiler
        self.writer = writer
        self.plotter = plotter

    def run(self) -> None:
        self._run_split("train", self.dataset_adapter.load_train())
        self._run_split("test", self.dataset_adapter.load_test())

    def _run_split(self, split_name: str, samples) -> None:
        df = self.profiler.samples_to_dataframe(samples)
        summary = self.profiler.summarize_dataframe(df)
        cross_tables = self.profiler.create_cross_tables(df)

        self.writer.write_dataframe_csv(df, f"{split_name}_profile.csv")
        self.writer.write_summary_json(summary, f"{split_name}_summary.json")
        self.writer.write_cross_tables({
            f"{split_name}_{name}": table
            for name, table in cross_tables.items()
        })

        if self.plotter is not None:
            self._write_plots(split_name, df)

    def _write_plots(self, split_name: str, df) -> None:
        self.plotter.plot_value_counts(
            df,
            "fallacy_exists",
            f"{split_name}_fallacy_exists.png",
        )
        self.plotter.plot_value_counts(
            df,
            "fallacy_type",
            f"{split_name}_fallacy_type.png",
        )
        self.plotter.plot_value_counts(
            df,
            "argument_scheme",
            f"{split_name}_argument_scheme.png",
        )
        self.plotter.plot_histogram(
            df,
            "source_word_count",
            f"{split_name}_source_word_count.png",
        )
        self.plotter.plot_histogram(
            df,
            "argument_word_count",
            f"{split_name}_argument_word_count.png",
        )
