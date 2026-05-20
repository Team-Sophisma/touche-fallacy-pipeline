from abc import ABC, abstractmethod
from typing import Sequence

from src.domain.entities import (
    EntityMention,
    PreprocessedSample,
    RoleMention,
    SentimentResult,
    TextSample,
)


class DatasetAdapter(ABC):
    @abstractmethod
    def load_train(self) -> list[TextSample]:
        pass

    @abstractmethod
    def load_test(self) -> list[TextSample]:
        pass


class PredictionWriter(ABC):
    @abstractmethod
    def write(self, samples: Sequence[TextSample], output_path: str) -> None:
        pass


class InputBuilder(ABC):
    @abstractmethod
    def build(self, item: dict) -> str:
        pass


class LabelMapper(ABC):
    @abstractmethod
    def map_label(self, item: dict, task_name: str) -> str | None:
        pass

    @abstractmethod
    def should_use_for_training(self, item: dict, task_name: str) -> bool:
        pass


class DatasetSplitter(ABC):
    @abstractmethod
    def split(
        self,
        samples: Sequence[PreprocessedSample],
    ) -> tuple[list[PreprocessedSample], list[PreprocessedSample]]:
        pass


class PreprocessedDatasetWriter(ABC):
    @abstractmethod
    def write_jsonl(
        self,
        samples: Sequence[PreprocessedSample],
        output_path: str,
    ) -> None:
        pass

    @abstractmethod
    def write_summary(
        self,
        summary: dict,
        output_path: str,
    ) -> None:
        pass


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> list[EntityMention]:
        pass


class EntityRoleClassifier(ABC):
    @abstractmethod
    def classify_roles(
        self,
        text: str,
        entities: list[EntityMention],
    ) -> list[RoleMention]:
        pass


class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        pass

    @abstractmethod
    def analyze_batch(self, texts: Sequence[str]) -> list[SentimentResult]:
        pass
