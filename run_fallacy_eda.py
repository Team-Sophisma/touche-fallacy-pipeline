import yaml

from src.application.services.fallacy_eda_service import FallacyEDAService
from src.infrastructure.datasets.touche_fallacy_adapter import ToucheFallacyAdapter
from src.infrastructure.datasets.preprocessed_jsonl_adapter import (
    PreprocessedJsonlAdapter,
)
from src.infrastructure.eda.dataset_profiler import DatasetProfiler
from src.infrastructure.eda.fallacy_architecture_knowledge_base import (
    FallacyArchitectureKnowledgeBase,
)
from src.infrastructure.eda.fallacy_architecture_profiler import (
    FallacyArchitectureProfiler,
)
from src.infrastructure.eda.hard_negative_analyzer import HardNegativeAnalyzer
from src.infrastructure.eda.outlier_analyzer import OutlierAnalyzer
from src.infrastructure.eda.representative_example_sampler import (
    RepresentativeExampleSampler,
)
from src.infrastructure.eda.text_keyword_extractor import TextKeywordExtractor
from src.infrastructure.persistence.fallacy_profile_writer import (
    FallacyProfileWriter,
)


def main():
    config_path = "configs/experiment/fallacy_architecture_eda.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        experiment_config = yaml.safe_load(file)

    dataset_config_path = experiment_config["dataset"]["config_path"]

    with open(dataset_config_path, "r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    dataset_adapter = ToucheFallacyAdapter(
        train_path=dataset_config["paths"]["train"],
        test_path=dataset_config["paths"]["test"],
        task_name=experiment_config["dataset"]["task_name"],
        text_variant=experiment_config["dataset"]["text_variant"],
        argument_variant=experiment_config["dataset"]["argument_variant"],
    )

    keyword_extractor = TextKeywordExtractor()

    hard_negative_analyzer = HardNegativeAnalyzer(
        keyword_extractor=keyword_extractor,
    )

    example_sampler = RepresentativeExampleSampler(
        random_seed=experiment_config.get("sampling", {}).get("random_seed", 42),
    )

    knowledge_base = FallacyArchitectureKnowledgeBase()

    architecture_profiler = FallacyArchitectureProfiler(
        knowledge_base=knowledge_base,
        hard_negative_analyzer=hard_negative_analyzer,
        example_sampler=example_sampler,
    )

    writer = FallacyProfileWriter(
        output_dir=experiment_config["outputs"]["report_dir"],
    )

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

    extra_splits = {}
    for split_name, split_config in experiment_config.get("extra_splits", {}).items():
        if split_config.get("format") != "preprocessed_jsonl":
            raise ValueError(f"Unsupported extra split format: {split_config.get('format')}")

        extra_splits[split_name] = PreprocessedJsonlAdapter(
            split_config["path"],
        ).load()

    service = FallacyEDAService(
        dataset_adapter=dataset_adapter,
        dataset_profiler=DatasetProfiler(),
        architecture_profiler=architecture_profiler,
        writer=writer,
        outlier_analyzer=outlier_analyzer,
        extra_splits=extra_splits,
    )

    service.run(
        examples_per_fallacy=experiment_config.get("sampling", {}).get(
            "examples_per_fallacy",
            5,
        )
    )

    print("Deep fallacy architecture EDA completed successfully.")


if __name__ == "__main__":
    main()
