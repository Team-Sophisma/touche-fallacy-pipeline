import yaml

from src.application.services.entity_masking_service import EntityMaskingService
from src.infrastructure.datasets.touche_fallacy_adapter import ToucheFallacyAdapter
from src.infrastructure.persistence.masked_dataset_writer import MaskedDatasetWriter
from src.infrastructure.preprocessing.entity_role_rule_matcher import (
    EntityRoleRuleMatcher,
)
from src.infrastructure.preprocessing.hybrid_entity_role_classifier import (
    HybridEntityRoleClassifier,
)
from src.infrastructure.preprocessing.role_aware_masker import RoleAwareMasker
from src.infrastructure.preprocessing.simple_entity_masker import SimpleEntityMasker
from src.infrastructure.preprocessing.spacy_entity_extractor import SpacyEntityExtractor
from src.infrastructure.preprocessing.zero_shot_role_classifier import (
    ZeroShotRoleClassifier,
)


def main() -> None:
    dataset_config = _load_yaml("configs/dataset/touche_fallacy.yaml")
    masking_config = _load_yaml("configs/preprocessing/entity_masking.yaml")

    dataset_adapter = ToucheFallacyAdapter(
        train_path=dataset_config["paths"]["train"],
        test_path=dataset_config["paths"]["test"],
        task_name=masking_config["dataset"]["task_name"],
        text_variant=masking_config["dataset"]["text_variant"],
        argument_variant=masking_config["dataset"]["argument_variant"],
    )

    zero_shot_classifier = ZeroShotRoleClassifier(
        model_name=masking_config["zero_shot"]["model_name"],
        roles=masking_config["roles"],
        confidence_threshold=masking_config["zero_shot"]["confidence_threshold"],
        batch_size=masking_config["zero_shot"].get("batch_size", 8),
        device=masking_config["zero_shot"].get("device", "auto"),
        role_descriptions=masking_config.get("role_descriptions", {}),
        ner_role_candidates=masking_config.get("ner_role_candidates", {}),
        fallback_to_ner_label=masking_config["zero_shot"].get(
            "fallback_to_ner_label",
            False,
        ),
        context_window_chars=masking_config["zero_shot"].get("context_window_chars"),
    )
    rule_config = masking_config.get("role_rules", {})
    role_classifier = HybridEntityRoleClassifier(
        rule_matcher=EntityRoleRuleMatcher(
            known_entities=rule_config.get("known_entities", {}),
            context_keywords=rule_config.get("context_keywords", {}),
            context_window_chars=rule_config.get("context_window_chars", 120),
            score=rule_config.get("score", 1.0),
        ),
        fallback_classifier=zero_shot_classifier,
    )

    service = EntityMaskingService(
        dataset_adapter=dataset_adapter,
        entity_extractor=SpacyEntityExtractor(
            model_name=masking_config["spacy"]["model_name"],
            include_labels=masking_config["spacy"].get("include_labels"),
        ),
        role_classifier=role_classifier,
        simple_entity_masker=SimpleEntityMasker(),
        role_aware_masker=RoleAwareMasker(),
        max_samples=masking_config.get("processing", {}).get("max_samples"),
    )

    rows = service.create_masked_rows()
    MaskedDatasetWriter().write_csv(
        rows,
        masking_config["outputs"]["masked_train_csv"],
    )

    print(f"Entity masking completed successfully. Rows written: {len(rows)}")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
