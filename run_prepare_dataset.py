import yaml

from src.infrastructure.datasets.touche_fallacy_adapter import ToucheFallacyAdapter
from src.infrastructure.persistence.readable_dataset_writer import ReadableDatasetWriter
from src.application.services.dataset_inspection_service import DatasetInspectionService


def main():
    experiment_config_path = "configs/experiment/fallacy_binary.yaml"

    with open(experiment_config_path, "r", encoding="utf-8") as file:
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

    writer = ReadableDatasetWriter()

    service = DatasetInspectionService(
        dataset_adapter=dataset_adapter,
        writer=writer,
    )

    outputs = experiment_config["outputs"]

    service.export_readable_dataset(
        split="train",
        csv_output_path=outputs["train_csv"],
        markdown_output_path=outputs["train_preview"],
    )

    service.export_readable_dataset(
        split="test",
        csv_output_path=outputs["test_csv"],
        markdown_output_path=outputs["test_preview"],
    )

    service.write_summary(outputs["summary"])

    print("Readable dataset export completed.")


if __name__ == "__main__":
    main()