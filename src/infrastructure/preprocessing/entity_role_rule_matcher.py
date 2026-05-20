import re

from src.domain.entities import EntityMention, RoleMention


class EntityRoleRuleMatcher:
    def __init__(
        self,
        known_entities: dict[str, list[str]] | None = None,
        context_keywords: dict[str, list[str]] | None = None,
        context_window_chars: int = 120,
        score: float = 1.0,
    ):
        self.known_entity_to_role = self._build_known_entity_map(known_entities or {})
        self.context_keywords = context_keywords or {}
        self.context_window_chars = context_window_chars
        self.score = score

    def match(self, text: str, entity: EntityMention) -> RoleMention | None:
        role = self._match_known_entity(entity)
        if role is None:
            role = self._match_context(text, entity)

        if role is None:
            return None

        return RoleMention(
            text=entity.text,
            original_label=entity.label,
            role=role,
            score=self.score,
            start_char=entity.start_char,
            end_char=entity.end_char,
        )

    def _match_known_entity(self, entity: EntityMention) -> str | None:
        normalized = self._normalize_entity_text(entity.text)
        return self.known_entity_to_role.get(normalized)

    def _match_context(self, text: str, entity: EntityMention) -> str | None:
        context = self._context_window(text, entity).lower()

        for role, keywords in self.context_keywords.items():
            if any(self._keyword_matches(context, keyword) for keyword in keywords):
                return role

        return None

    def _context_window(self, text: str, entity: EntityMention) -> str:
        start = max(0, entity.start_char - self.context_window_chars)
        end = min(len(text), entity.end_char + self.context_window_chars)
        return text[start:end]

    def _keyword_matches(self, context: str, keyword: str) -> bool:
        if not keyword.isalnum():
            return keyword.lower() in context

        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        return re.search(pattern, context) is not None

    def _build_known_entity_map(
        self,
        known_entities: dict[str, list[str]],
    ) -> dict[str, str]:
        known_entity_to_role = {}

        for role, entities in known_entities.items():
            for entity in entities:
                known_entity_to_role[self._normalize_entity_text(entity)] = role

        return known_entity_to_role

    def _normalize_entity_text(self, value: str) -> str:
        normalized = value.strip().lower()
        normalized = re.sub(r"['’]s$", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized
