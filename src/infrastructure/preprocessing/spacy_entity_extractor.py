import spacy

from src.domain.entities import EntityMention
from src.domain.interfaces import EntityExtractor


class SpacyEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str, include_labels: list[str] | None = None):
        self.model_name = model_name
        self.include_labels = set(include_labels) if include_labels else None
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> list[EntityMention]:
        if not text:
            return []

        doc = self.nlp(text)

        return [
            EntityMention(
                text=entity.text,
                label=entity.label_,
                start_char=int(entity.start_char),
                end_char=int(entity.end_char),
            )
            for entity in doc.ents
            if self._should_include(entity.label_)
        ]

    def _should_include(self, label: str) -> bool:
        if self.include_labels is None:
            return True

        return label in self.include_labels
