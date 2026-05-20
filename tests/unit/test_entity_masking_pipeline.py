import json

import pandas as pd

from src.application.services.entity_masking_service import EntityMaskingService
from src.domain.entities import EntityMention, RoleMention, TextSample
from src.infrastructure.persistence.masked_dataset_writer import MaskedDatasetWriter
from src.infrastructure.preprocessing.entity_role_rule_matcher import (
    EntityRoleRuleMatcher,
)
from src.infrastructure.preprocessing.hybrid_entity_role_classifier import (
    HybridEntityRoleClassifier,
)
from src.infrastructure.preprocessing.role_aware_masker import RoleAwareMasker
from src.infrastructure.preprocessing.simple_entity_masker import SimpleEntityMasker
from src.infrastructure.preprocessing.zero_shot_role_classifier import (
    ZeroShotRoleClassifier,
)


class FakeDatasetAdapter:
    def load_train(self) -> list[TextSample]:
        return [
            TextSample(
                id="sample-1",
                source_text="Einstein said this, so it must be true.",
                argument_text="Einstein said this, so it must be true.",
                label="authority",
                task_name="fallacy_type",
                meta={
                    "fallacy_exists": 1,
                    "fallacy_type": "authority",
                    "resembles_fallacy": "authority",
                },
            )
        ]

    def load_test(self) -> list[TextSample]:
        return []


class FakeEntityExtractor:
    def extract(self, text: str) -> list[EntityMention]:
        return [
            EntityMention(
                text="Einstein",
                label="PERSON",
                start_char=0,
                end_char=8,
            )
        ]


class FakeRoleClassifier:
    def classify_roles(
        self,
        text: str,
        entities: list[EntityMention],
    ) -> list[RoleMention]:
        return [
            RoleMention(
                text=entities[0].text,
                original_label=entities[0].label,
                role="SCIENTIFIC_AUTHORITY",
                score=0.95,
                start_char=entities[0].start_char,
                end_char=entities[0].end_char,
            )
        ]


class FakeFallbackRoleClassifier:
    def classify_roles(
        self,
        text: str,
        entities: list[EntityMention],
    ) -> list[RoleMention]:
        return [
            RoleMention(
                text=entity.text,
                original_label=entity.label,
                role=entity.label,
                score=0.1,
                start_char=entity.start_char,
                end_char=entity.end_char,
            )
            for entity in entities
        ]


def test_maskers_replace_spans_from_end_to_start():
    entities = [
        EntityMention("Jacob", "PERSON", 10, 15),
        EntityMention("Einstein", "PERSON", 0, 8),
    ]
    roles = [
        RoleMention("Jacob", "PERSON", "PERSONAL_SOURCE", 0.91, 10, 15),
        RoleMention("Einstein", "PERSON", "SCIENTIFIC_AUTHORITY", 0.97, 0, 8),
    ]
    text = "Einstein, Jacob"

    assert SimpleEntityMasker().mask(text, entities) == "[PERSON], [PERSON]"
    assert (
        RoleAwareMasker().mask(text, roles)
        == "[SCIENTIFIC_AUTHORITY], [PERSONAL_SOURCE]"
    )


def test_entity_masking_service_creates_expected_row():
    service = EntityMaskingService(
        dataset_adapter=FakeDatasetAdapter(),
        entity_extractor=FakeEntityExtractor(),
        role_classifier=FakeRoleClassifier(),
        simple_entity_masker=SimpleEntityMasker(),
        role_aware_masker=RoleAwareMasker(),
        max_samples=20,
    )

    rows = service.create_masked_rows()

    assert rows[0]["simple_masked_text"] == "[PERSON] said this, so it must be true."
    assert (
        rows[0]["role_aware_masked_text"]
        == "[SCIENTIFIC_AUTHORITY] said this, so it must be true."
    )
    assert rows[0]["fallacy_type"] == "authority"
    assert rows[0]["entities"][0]["label"] == "PERSON"


def test_masked_dataset_writer_serializes_nested_columns(tmp_path):
    output_path = tmp_path / "masked.csv"
    rows = [{
        "id": "sample-1",
        "entities": [{"text": "Einstein"}],
        "entity_roles": [{"role": "SCIENTIFIC_AUTHORITY"}],
    }]

    MaskedDatasetWriter().write_csv(rows, str(output_path))

    df = pd.read_csv(output_path)
    assert json.loads(df.loc[0, "entities"]) == [{"text": "Einstein"}]
    assert json.loads(df.loc[0, "entity_roles"]) == [
        {"role": "SCIENTIFIC_AUTHORITY"}
    ]


def test_zero_shot_role_classifier_resolves_configured_device():
    assert ZeroShotRoleClassifier._resolve_device("cpu") == -1
    assert ZeroShotRoleClassifier._resolve_device("0") == 0


def test_zero_shot_role_classifier_uses_other_as_fallback_only():
    classifier = ZeroShotRoleClassifier.__new__(ZeroShotRoleClassifier)

    mapping = classifier._build_candidate_label_to_role(
        roles=["SCIENTIFIC_AUTHORITY", "OTHER"],
        role_descriptions={
            "SCIENTIFIC_AUTHORITY": "scientific or technical expert authority",
        },
        fallback_role="OTHER",
    )

    assert mapping == {
        "scientific or technical expert authority": "SCIENTIFIC_AUTHORITY",
    }
    assert "OTHER" not in mapping


def test_zero_shot_role_classifier_filters_candidates_by_ner_label():
    classifier = ZeroShotRoleClassifier.__new__(ZeroShotRoleClassifier)
    classifier.candidate_label_to_role = {
        "scientific authority": "SCIENTIFIC_AUTHORITY",
        "causal event": "CAUSAL_CHAIN_EVENT",
        "personal source": "PERSONAL_SOURCE",
    }
    classifier.role_to_candidate_label = {
        role: label
        for label, role in classifier.candidate_label_to_role.items()
    }
    classifier.ner_role_candidates = {
        "PERSON": ["SCIENTIFIC_AUTHORITY", "PERSONAL_SOURCE"],
    }

    labels = classifier._candidate_labels_for_entity(
        EntityMention("Einstein", "PERSON", 0, 8)
    )

    assert labels == ["scientific authority", "personal source"]


def test_zero_shot_role_classifier_can_fallback_to_ner_label():
    classifier = ZeroShotRoleClassifier.__new__(ZeroShotRoleClassifier)
    classifier.fallback_to_ner_label = True
    classifier.fallback_role = "OTHER"

    mention = classifier._fallback_mention(
        EntityMention("Mark Twain", "PERSON", 0, 10),
        score=0.12,
    )

    assert mention.role == "PERSON"


def test_entity_role_rule_matcher_matches_known_entity():
    matcher = EntityRoleRuleMatcher(
        known_entities={
            "CELEBRITY_OR_PUBLIC_FIGURE": ["Mark Twain"],
        },
    )

    mention = matcher.match(
        "Mark Twain said this.",
        EntityMention("Mark Twain", "PERSON", 0, 10),
    )

    assert mention.role == "CELEBRITY_OR_PUBLIC_FIGURE"
    assert mention.score == 1.0


def test_hybrid_classifier_uses_rules_before_fallback():
    classifier = HybridEntityRoleClassifier(
        rule_matcher=EntityRoleRuleMatcher(
            known_entities={
                "INSTITUTIONAL_AUTHORITY": ["BBC"],
            },
        ),
        fallback_classifier=FakeFallbackRoleClassifier(),
    )
    entities = [
        EntityMention("BBC", "ORG", 0, 3),
        EntityMention("Jacob", "PERSON", 14, 19),
    ]

    mentions = classifier.classify_roles("BBC quoted Jacob.", entities)

    assert [mention.role for mention in mentions] == [
        "INSTITUTIONAL_AUTHORITY",
        "PERSON",
    ]
