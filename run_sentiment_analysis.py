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

    writer = SentimentReportWriter()
    outputs = sentiment_config["outputs"]

    # ── Train split ──────────────────────────────────────────────
    train_rows = service.create_train_rows()
    train_summary = service.summarize_rows(train_rows)
    train_enrichment = service.build_enrichment_map(train_rows)

    writer.write_csv(train_rows, outputs["train_csv"])
    writer.write_json(train_summary, outputs["summary_json"])
    writer.write_enriched_jsonl(
        raw_jsonl_path=dataset_config["paths"]["train"],
        enrichment_map=train_enrichment,
        output_path=outputs["enriched_train_jsonl"],
    )
    print(f"Train sentiment completed. Rows: {len(train_rows)}")

    # ── Test split ───────────────────────────────────────────────
    test_rows = service.create_test_rows()
    test_summary = service.summarize_rows(test_rows)
    test_enrichment = service.build_enrichment_map(test_rows)

    writer.write_csv(test_rows, outputs["test_csv"])
    writer.write_json(test_summary, outputs["test_summary_json"])
    writer.write_enriched_jsonl(
        raw_jsonl_path=dataset_config["paths"]["test"],
        enrichment_map=test_enrichment,
        output_path=outputs["enriched_test_jsonl"],
    )
    print(f"Test sentiment completed.  Rows: {len(test_rows)}")

    print("Sentiment analysis & enrichment finished for both splits.")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()

