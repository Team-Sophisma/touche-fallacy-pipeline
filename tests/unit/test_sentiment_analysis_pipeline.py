import json

import pandas as pd

from src.application.services.sentiment_analysis_service import (
    SentimentAnalysisService,
)
from src.domain.entities import SentimentResult, TextSample
from src.infrastructure.persistence.sentiment_report_writer import (
    SentimentReportWriter,
)
from src.infrastructure.preprocessing.huggingface_sentiment_analyzer import (
    HuggingFaceSentimentAnalyzer,
)


class FakeDatasetAdapter:
    def load_train(self) -> list[TextSample]:
        return [
            TextSample(
                id="sample-1",
                source_text="This is terrible.",
                argument_text="Claim: This is terrible.",
                label="authority",
                task_name="fallacy_type",
                meta={
                    "fallacy_exists": 1,
                    "fallacy_type": "authority",
                    "resembles_fallacy": "authority",
                    "argument_goal": "evaluate",
                    "argument_basis": "source",
                },
            ),
            TextSample(
                id="sample-2",
                source_text="This is fine.",
                argument_text="Claim: This is fine.",
                label="popular",
                task_name="fallacy_type",
                meta={
                    "fallacy_exists": 1,
                    "fallacy_type": "popular",
                    "resembles_fallacy": "popular",
                },
            ),
        ]

    def load_test(self) -> list[TextSample]:
        return []


class FakeSentimentAnalyzer:
    def analyze(self, text: str) -> SentimentResult:
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts) -> list[SentimentResult]:
        return [
            SentimentResult(
                label="negative",
                score=0.91,
                scores={"negative": 0.91, "positive": 0.09},
            )
            for _ in texts
        ]


def test_sentiment_analysis_service_creates_rows_and_summary():
    service = SentimentAnalysisService(
        dataset_adapter=FakeDatasetAdapter(),
        sentiment_analyzer=FakeSentimentAnalyzer(),
        text_source="source_text",
        max_samples=1,
    )

    rows = service.create_train_rows()
    summary = service.summarize_rows(rows)

    assert len(rows) == 1
    assert rows[0]["id"] == "sample-1"
    assert rows[0]["sentiment_label"] == "negative"
    assert rows[0]["sentiment_scores"]["negative"] == 0.91
    assert summary["sentiment_labels"] == ["negative", "positive"]
    assert summary["sentiment_distribution"] == {
        "negative": 1,
        "positive": 0,
    }
    assert summary["fallacy_type_distribution"] == {"authority": 1}
    assert summary["by_fallacy_type"]["authority"]["total_samples"] == 1
    assert summary["by_fallacy_type"]["authority"]["sentiment_distribution"] == {
        "negative": 1,
        "positive": 0,
    }


def test_sentiment_analysis_service_can_sample_per_fallacy_type():
    service = SentimentAnalysisService(
        dataset_adapter=FakeDatasetAdapter(),
        sentiment_analyzer=FakeSentimentAnalyzer(),
        text_source="source_text",
        max_samples_per_fallacy_type=1,
    )

    rows = service.create_train_rows()
    summary = service.summarize_rows(rows)

    assert [row["id"] for row in rows] == ["sample-1", "sample-2"]
    assert summary["fallacy_type_distribution"] == {
        "authority": 1,
        "popular": 1,
    }
    assert sorted(summary["by_fallacy_type"]) == ["authority", "popular"]


def test_sentiment_report_writer_serializes_scores(tmp_path):
    output_path = tmp_path / "sentiment.csv"
    rows = [{
        "id": "sample-1",
        "sentiment_scores": {"negative": 0.91},
    }]

    SentimentReportWriter().write_csv(rows, str(output_path))

    df = pd.read_csv(output_path)
    assert json.loads(df.loc[0, "sentiment_scores"]) == {"negative": 0.91}


def test_huggingface_sentiment_analyzer_normalizes_pipeline_scores():
    analyzer = HuggingFaceSentimentAnalyzer.__new__(HuggingFaceSentimentAnalyzer)

    result = analyzer._to_sentiment_result([
        {"label": "Negative", "score": 0.7},
        {"label": "Positive", "score": 0.3},
    ])

    assert result.label == "negative"
    assert result.score == 0.7
    assert result.scores == {
        "negative": 0.7,
        "positive": 0.3,
    }


def test_huggingface_sentiment_analyzer_resolves_configured_device():
    assert HuggingFaceSentimentAnalyzer._resolve_device("cpu") == -1
    assert HuggingFaceSentimentAnalyzer._resolve_device("0") == 0
