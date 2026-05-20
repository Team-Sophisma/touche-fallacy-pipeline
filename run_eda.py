import yaml

from src.infrastructure.datasets.touche_fallacy_adapter import ToucheFallacyAdapter
from src.infrastructure.eda.dataset_profiler import DatasetProfiler
from src.infrastructure.eda.outlier_analyzer import OutlierAnalyzer
from src.infrastructure.persistence.eda_report_writer import EDAReportWriter
from src.infrastructure.visualization.eda_plots import EDAPlotter
from src.application.services.eda_service import EDAService


def main():
    config_path = "configs/experiment/eda_touche_fallacy.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        experiment_config = yaml.safe_load(file)

    with open(
        experiment_config["dataset"]["config_path"],
        "r",
        encoding="utf-8"
    ) as file:
        dataset_config = yaml.safe_load(file)

    dataset_adapter = ToucheFallacyAdapter(
        train_path=dataset_config["paths"]["train"],
        test_path=dataset_config["paths"]["test"],
        task_name=experiment_config["dataset"]["task_name"],
        text_variant=experiment_config["dataset"]["text_variant"],
        argument_variant=experiment_config["dataset"]["argument_variant"],
    )

    report_dir = experiment_config["outputs"]["report_dir"]
    plots_dir = experiment_config["outputs"]["plots_dir"]

    profiler = DatasetProfiler()
    writer = EDAReportWriter(report_dir=report_dir)

    plotter = None
    if experiment_config["eda"].get("generate_plots", True):
        plotter = EDAPlotter(plots_dir=plots_dir)

    outlier_config = experiment_config.get("outliers", {})
    outlier_analyzer = None
    if outlier_config.get("enabled", True):
        outlier_analyzer = OutlierAnalyzer(
            metrics=outlier_config.get("metrics"),
            group_columns=outlier_config.get("group_columns", []),
            iqr_multiplier=outlier_config.get("iqr_multiplier", 3.0),
            max_examples_per_metric=outlier_config.get(
                "max_examples_per_metric",
                50,
            ),
        )

    service = EDAService(
        dataset_adapter=dataset_adapter,
        profiler=profiler,
        writer=writer,
        plotter=plotter,
        outlier_analyzer=outlier_analyzer,
    )

    service.run()

    print("EDA completed successfully.")


if __name__ == "__main__":
    main()
