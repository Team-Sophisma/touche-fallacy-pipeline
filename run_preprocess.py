import yaml

from src.application.services.preprocessing_service import PreprocessingService
from src.infrastructure.datasets.jsonl_adapter import JsonlAdapter
from src.infrastructure.persistence.preprocessed_dataset_writer import (
    JsonlPreprocessedDatasetWriter,
)
from src.infrastructure.preprocessing.argument_input_builder import (
    ArgumentInputBuilder,
)
from src.infrastructure.preprocessing.stratified_splitter import (
    StratifiedDatasetSplitter,
)
from src.infrastructure.preprocessing.text_normalizer import TextNormalizer
from src.infrastructure.preprocessing.touche_label_mapper import ToucheLabelMapper


def main():
    experiment_config_path = "configs/experiment/preprocess_touche_enhanced.yaml"

    with open(experiment_config_path, "r", encoding="utf-8") as file:
        experiment_config = yaml.safe_load(file)

    with open(
        experiment_config["dataset"]["config_path"],
        "r",
        encoding="utf-8",
    ) as file:
        dataset_config = yaml.safe_load(file)

    train_items = list(JsonlAdapter(dataset_config["paths"]["train"]).read_items())
    test_items = list(JsonlAdapter(dataset_config["paths"]["test"]).read_items())

    input_config = experiment_config["input"]
    split_config = experiment_config["split"]

    input_builder = ArgumentInputBuilder(
        text_field=input_config["text_field"],
        argument_field=input_config["argument_field"],
        normalizer=TextNormalizer(),
        include_title=input_config.get("include_title", False),
        include_parent=input_config.get("include_parent", False),
    )

    service = PreprocessingService(
        input_builder=input_builder,
        label_mapper=ToucheLabelMapper(),
        splitter=StratifiedDatasetSplitter(
            validation_size=split_config["validation_size"],
            random_seed=split_config["random_seed"],
        ),
        writer=JsonlPreprocessedDatasetWriter(),
        tag=experiment_config["dataset"]["tag"],
        output_dir=experiment_config["outputs"]["output_dir"],
    )

    enabled_tasks = [
        task["name"]
        for task in experiment_config["tasks"]
        if task.get("enabled", True)
    ]

    service.run(
        train_items=train_items,
        test_items=test_items,
        task_names=enabled_tasks,
    )

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
