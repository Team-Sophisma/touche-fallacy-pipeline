from typing import Sequence

import torch
from transformers import pipeline

from src.domain.entities import SentimentResult
from src.domain.interfaces import SentimentAnalyzer


class HuggingFaceSentimentAnalyzer(SentimentAnalyzer):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str | int = "auto",
        truncation: bool = True,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.truncation = truncation
        self.max_length = max_length
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=self.device,
        )

    def analyze(self, text: str) -> SentimentResult:
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: Sequence[str]) -> list[SentimentResult]:
        if not texts:
            return []

        raw_results = self.classifier(
            list(texts),
            batch_size=self.batch_size,
            truncation=self.truncation,
            max_length=self.max_length,
            top_k=None,
        )

        return [
            self._to_sentiment_result(raw_result)
            for raw_result in raw_results
        ]

    def _to_sentiment_result(self, raw_result) -> SentimentResult:
        scores = self._normalize_raw_scores(raw_result)
        best = max(scores, key=lambda item: item["score"])

        score_by_label = {
            self._normalize_label(item["label"]): float(item["score"])
            for item in scores
        }

        return SentimentResult(
            label=self._normalize_label(best["label"]),
            score=float(best["score"]),
            scores=score_by_label,
        )

    def _normalize_raw_scores(self, raw_result) -> list[dict]:
        if isinstance(raw_result, dict):
            return [raw_result]

        if raw_result and isinstance(raw_result[0], list):
            return raw_result[0]

        return list(raw_result)

    def _normalize_label(self, label: str) -> str:
        return str(label).strip().lower().replace(" ", "_")

    @staticmethod
    def _resolve_device(device: str | int) -> int:
        if isinstance(device, int):
            return device

        normalized_device = str(device).strip().lower()

        if normalized_device == "auto":
            return 0 if torch.cuda.is_available() else -1

        if normalized_device == "cpu":
            return -1

        if normalized_device in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            return 0

        if normalized_device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")

            device_index = int(normalized_device.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index is not available: {device_index}")

            return device_index

        return int(normalized_device)
