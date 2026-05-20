from src.domain.entities import EntityMention, RoleMention
from src.domain.interfaces import EntityRoleClassifier
from src.infrastructure.preprocessing.entity_role_rule_matcher import (
    EntityRoleRuleMatcher,
)


class HybridEntityRoleClassifier(EntityRoleClassifier):
    def __init__(
        self,
        rule_matcher: EntityRoleRuleMatcher,
        fallback_classifier: EntityRoleClassifier,
    ):
        self.rule_matcher = rule_matcher
        self.fallback_classifier = fallback_classifier

    def classify_roles(
        self,
        text: str,
        entities: list[EntityMention],
    ) -> list[RoleMention]:
        if not entities:
            return []

        role_mentions_by_index: dict[int, RoleMention] = {}
        unresolved_entities = []
        unresolved_indexes = []

        for index, entity in enumerate(entities):
            rule_match = self.rule_matcher.match(text, entity)
            if rule_match is not None:
                role_mentions_by_index[index] = rule_match
                continue

            unresolved_indexes.append(index)
            unresolved_entities.append(entity)

        fallback_mentions = self.fallback_classifier.classify_roles(
            text,
            unresolved_entities,
        )
        for index, mention in zip(unresolved_indexes, fallback_mentions):
            role_mentions_by_index[index] = mention

        return [
            role_mentions_by_index[index]
            for index in range(len(entities))
        ]
