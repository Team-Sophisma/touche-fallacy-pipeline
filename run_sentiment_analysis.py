import yaml

from src.application.services.sentiment_analysis_service import (
    SentimentAnalysisService,
)
from src.infrastructure.datasets.touche_fallacy_adapter import ToucheFallacyAdapter
from src.infrastructure.persistence.sentiment_report_writer import (
    SentimentReportWriter,
)
from src.infrastructure.preprocessing.huggingface_sentiment_analyzer import (
    HuggingFaceSentimentAnalyzer,
)


def main() -> None:
    dataset_config = _load_yaml("configs/dataset/touche_fallacy.yaml")
    sentiment_config = _load_yaml("configs/preprocessing/sentiment_analysis.yaml")

    dataset_adapter = ToucheFallacyAdapter(
        train_path=dataset_config["paths"]["train"],
        test_path=dataset_config["paths"]["test"],
        task_name=sentiment_config["dataset"]["task_name"],
        text_variant=sentiment_config["dataset"]["text_variant"],
        argument_variant=sentiment_config["dataset"]["argument_variant"],
    )

    processing_config = sentiment_config.get("processing", {})

    service = SentimentAnalysisService(
        dataset_adapter=dataset_adapter,
        sentiment_analyzer=HuggingFaceSentimentAnalyzer(
            model_name=sentiment_config["sentiment"]["model_name"],
            batch_size=sentiment_config["sentiment"].get("batch_size", 16),
            device=sentiment_config["sentiment"].get("device", "auto"),
            truncation=sentiment_config["sentiment"].get("truncation", True),
            max_length=sentiment_config["sentiment"].get("max_length", 512),
        ),
        text_source=sentiment_config["dataset"].get("text_source", "source_text"),
        max_samples=processing_config.get("max_samples"),
        max_samples_per_fallacy_type=processing_config.get(
            "max_samples_per_fallacy_type",
        ),
    )

    rows = service.create_train_rows()
    summary = service.summarize_rows(rows)

    writer = SentimentReportWriter()
    writer.write_csv(rows, sentiment_config["outputs"]["train_csv"])
    writer.write_json(summary, sentiment_config["outputs"]["summary_json"])

    print(f"Sentiment analysis completed successfully. Rows written: {len(rows)}")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
