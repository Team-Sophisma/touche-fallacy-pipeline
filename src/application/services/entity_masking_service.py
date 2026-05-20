from dataclasses import asdict
from typing import Sequence

from src.domain.entities import TextSample
from src.domain.interfaces import (
    DatasetAdapter,
    EntityExtractor,
    EntityRoleClassifier,
)
from src.infrastructure.preprocessing.role_aware_masker import RoleAwareMasker
from src.infrastructure.preprocessing.simple_entity_masker import SimpleEntityMasker


class EntityMaskingService:
    def __init__(
        self,
        dataset_adapter: DatasetAdapter,
        entity_extractor: EntityExtractor,
        role_classifier: EntityRoleClassifier,
        simple_entity_masker: SimpleEntityMasker,
        role_aware_masker: RoleAwareMasker,
        max_samples: int | None = None,
    ):
        self.dataset_adapter = dataset_adapter
        self.entity_extractor = entity_extractor
        self.role_classifier = role_classifier
        self.simple_entity_masker = simple_entity_masker
        self.role_aware_masker = role_aware_masker
        self.max_samples = max_samples

    def create_masked_rows(self) -> list[dict]:
        samples = self._limit_samples(self.dataset_adapter.load_train())

        return [self._create_masked_row(sample) for sample in samples]

    def _create_masked_row(self, sample: TextSample) -> dict:
        original_text = sample.source_text
        entities = self.entity_extractor.extract(original_text)
        role_mentions = self.role_classifier.classify_roles(
            original_text,
            entities,
        )

        meta = sample.meta or {}

        return {
            "id": sample.id,
            "task_name": sample.task_name,
            "label": sample.label,
            "fallacy_exists": meta.get("fallacy_exists"),
            "fallacy_type": meta.get("fallacy_type"),
            "resembles_fallacy": meta.get("resembles_fallacy"),
            "original_text": original_text,
            "simple_masked_text": self.simple_entity_masker.mask(
                original_text,
                entities,
            ),
            "role_aware_masked_text": self.role_aware_masker.mask(
                original_text,
                role_mentions,
            ),
            "entities": [asdict(entity) for entity in entities],
            "entity_roles": [asdict(mention) for mention in role_mentions],
        }

    def _limit_samples(self, samples: Sequence[TextSample]) -> list[TextSample]:
        if self.max_samples is None:
            return list(samples)

        if self.max_samples <= 0:
            return []

        return list(samples[:self.max_samples])
