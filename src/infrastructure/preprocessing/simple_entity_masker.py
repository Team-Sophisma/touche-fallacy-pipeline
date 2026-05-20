from src.domain.entities import EntityMention


class SimpleEntityMasker:
    def mask(self, text: str, entities: list[EntityMention]) -> str:
        masked_text = text

        for entity in sorted(entities, key=lambda item: item.start_char, reverse=True):
            masked_text = (
                masked_text[:entity.start_char]
                + f"[{entity.label}]"
                + masked_text[entity.end_char:]
            )

        return masked_text
