from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Sequence

from src.domain.entities import TextSample
from src.domain.interfaces import DatasetAdapter, SentimentAnalyzer


class SentimentAnalysisService:
    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        sentiment_analyzer: SentimentAnalyzer,
        text_source: str = "source_text",
        max_samples: int | None = None,
        max_samples_per_fallacy_type: int | None = None,
    ):
        self.dataset_adapter = dataset_adapter
        self.sentiment_analyzer = sentiment_analyzer
        self.text_source = text_source
        self.max_samples = max_samples
        self.max_samples_per_fallacy_type = max_samples_per_fallacy_type

    def create_train_rows(self) -> list[dict]:
        samples = self._select_samples(self.dataset_adapter.load_train())
        texts = [self._text_for_sample(sample) for sample in samples]
        sentiments = self.sentiment_analyzer.analyze_batch(texts)

        return [
            self._create_row(sample, text, sentiment)
            for sample, text, sentiment in zip(samples, texts, sentiments)
        ]

    def summarize_rows(self, rows: list[dict]) -> dict:
        sentiment_labels = self._sentiment_labels(rows)
        label_counts = Counter(row["sentiment_label"] for row in rows)
        fallacy_sentiment_counts = Counter(
            (self._fallacy_type_key(row), row["sentiment_label"])
            for row in rows
        )

        return {
            "total_samples": len(rows),
            "text_source": self.text_source,
            "sentiment_labels": sentiment_labels,
            "sentiment_distribution": self._ordered_counts(
                label_counts,
                sentiment_labels,
            ),
            "fallacy_type_distribution": dict(
                Counter(self._fallacy_type_key(row) for row in rows)
            ),
            "fallacy_type_sentiment_distribution": {
                f"{fallacy_type}|{sentiment_label}": count
                for fallacy_type in sorted(
                    {self._fallacy_type_key(row) for row in rows}
                )
                for sentiment_label, count in [
                    (
                        sentiment_label,
                        fallacy_sentiment_counts.get(
                            (fallacy_type, sentiment_label),
                            0,
                        ),
                    )
                    for sentiment_label in sentiment_labels
                ]
            },
            "by_fallacy_type": self._summarize_by_fallacy_type(
                rows,
                sentiment_labels,
            ),
        }

    def _ordered_counts(self, counts: Counter, labels: list[str]) -> dict:
        return {
            label: int(counts.get(label, 0))
            for label in labels
        }

    def _sentiment_labels(self, rows: list[dict]) -> list[str]:
        preferred_order = ["negative", "neutral", "positive"]
        labels = set()

        for row in rows:
            labels.add(row["sentiment_label"])
            labels.update(row.get("sentiment_scores", {}).keys())

        ordered_labels = [
            label for label in preferred_order
            if label in labels
        ]
        ordered_labels.extend(sorted(labels - set(preferred_order)))

        return ordered_labels

    def _create_row(self, sample: TextSample, text: str, sentiment) -> dict:
        meta = sample.meta or {}

        return {
            "id": sample.id,
            "task_name": sample.task_name,
            "label": sample.label,
            "fallacy_exists": meta.get("fallacy_exists"),
            "fallacy_type": meta.get("fallacy_type"),
            "resembles_fallacy": meta.get("resembles_fallacy"),
            "argument_goal": meta.get("argument_goal"),
            "argument_basis": meta.get("argument_basis"),
            "text_source": self.text_source,
            "text": text,
            "sentiment_label": sentiment.label,
            "sentiment_score": sentiment.score,
            "sentiment_scores": asdict(sentiment)["scores"],
        }

    def _text_for_sample(self, sample: TextSample) -> str:
        if self.text_source == "source_text":
            return sample.source_text

        if self.text_source == "argument_text":
            return sample.argument_text

        raise ValueError(f"Unsupported text_source: {self.text_source}")

    def _select_samples(self, samples: Sequence[TextSample]) -> list[TextSample]:
        selected = self._limit_samples_per_fallacy_type(samples)
        return self._limit_samples(selected)

    def _limit_samples_per_fallacy_type(
        self,
        samples: Sequence[TextSample],
    ) -> list[TextSample]:
        if self.max_samples_per_fallacy_type is None:
            return list(samples)

        if self.max_samples_per_fallacy_type <= 0:
            return []

        counts = Counter()
        selected = []

        for sample in samples:
            fallacy_type = self._fallacy_type_key(sample)
            if counts[fallacy_type] >= self.max_samples_per_fallacy_type:
                continue

            selected.append(sample)
            counts[fallacy_type] += 1

        return selected

    def _limit_samples(self, samples: Sequence[TextSample]) -> list[TextSample]:
        if self.max_samples is None:
            return list(samples)

        if self.max_samples <= 0:
            return []

        return list(samples[:self.max_samples])

    def _summarize_by_fallacy_type(
        self,
        rows: list[dict],
        sentiment_labels: list[str],
    ) -> dict:
        rows_by_fallacy_type = defaultdict(list)
        for row in rows:
            rows_by_fallacy_type[self._fallacy_type_key(row)].append(row)

        grouped_summary = {}
        for fallacy_type in sorted(rows_by_fallacy_type):
            group_rows = rows_by_fallacy_type[fallacy_type]
            grouped_summary[fallacy_type] = {
                "total_samples": len(group_rows),
                "sentiment_distribution": self._ordered_counts(
                    Counter(row["sentiment_label"] for row in group_rows),
                    sentiment_labels,
                ),
                "mean_sentiment_score": self._mean(
                    row["sentiment_score"] for row in group_rows
                ),
                "mean_sentiment_scores": self._mean_score_map(group_rows),
            }

        return grouped_summary

    def _mean_score_map(self, rows: list[dict]) -> dict:
        score_sums = defaultdict(float)
        score_counts = Counter()

        for row in rows:
            for label, score in row.get("sentiment_scores", {}).items():
                score_sums[label] += float(score)
                score_counts[label] += 1

        return {
            label: round(score_sums[label] / score_counts[label], 6)
            for label in sorted(score_sums)
            if score_counts[label]
        }

    def _mean(self, values) -> float | None:
        values = [float(value) for value in values]
        if not values:
            return None

        return round(sum(values) / len(values), 6)

    def _fallacy_type_key(self, value) -> str:
        if isinstance(value, TextSample):
            raw_value = (value.meta or {}).get("fallacy_type")
        else:
            raw_value = value.get("fallacy_type")

        if raw_value is None or str(raw_value).strip() == "":
            return "no_fallacy"

        return str(raw_value)
