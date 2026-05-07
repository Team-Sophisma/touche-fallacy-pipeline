from abc import ABC, abstractmethod
from typing import Sequence

from src.domain.entities import TextSample


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