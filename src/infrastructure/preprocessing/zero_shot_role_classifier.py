import torch
from transformers import pipeline

from src.domain.entities import EntityMention, RoleMention
from src.domain.interfaces import EntityRoleClassifier


class ZeroShotRoleClassifier(EntityRoleClassifier):
    def __init__(
        self,
        model_name: str,
        roles: list[str],
        confidence_threshold: float,
        batch_size: int = 8,
        device: str | int = "auto",
        role_descriptions: dict[str, str] | None = None,
        ner_role_candidates: dict[str, list[str]] | None = None,
        fallback_role: str = "OTHER",
        fallback_to_ner_label: bool = False,
        context_window_chars: int | None = None,
    ):
        self.model_name = model_name
        self.roles = roles
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.fallback_role = fallback_role
        self.fallback_to_ner_label = fallback_to_ner_label
        self.context_window_chars = context_window_chars
        self.candidate_label_to_role = self._build_candidate_label_to_role(
            roles=roles,
            role_descriptions=role_descriptions or {},
            fallback_role=fallback_role,
        )
        self.role_to_candidate_label = {
            role: candidate_label
            for candidate_label, role in self.candidate_label_to_role.items()
        }
        self.ner_role_candidates = ner_role_candidates or {}
        self.device = self._resolve_device(device)
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
        )

    def classify_roles(
        self,
        text: str,
        entities: list[EntityMention],
    ) -> list[RoleMention]:
        if not entities:
            return []

        results_by_index = {}
        fallback_mentions = {}
        grouped_prompts: dict[tuple[str, ...], list[tuple[int, str]]] = {}

        for index, entity in enumerate(entities):
            candidate_labels = tuple(self._candidate_labels_for_entity(entity))
            if not candidate_labels:
                fallback_mentions[index] = self._fallback_mention(entity, score=0.0)
                continue

            grouped_prompts.setdefault(candidate_labels, []).append(
                (index, self._build_prompt(text, entity))
            )

        for candidate_labels, indexed_prompts in grouped_prompts.items():
            results = self.classifier(
                [prompt for _, prompt in indexed_prompts],
                candidate_labels=list(candidate_labels),
                multi_label=False,
                batch_size=self.batch_size,
            )
            if isinstance(results, dict):
                results = [results]

            for (index, _), result in zip(indexed_prompts, results):
                results_by_index[index] = result

        role_mentions = []
        for index, entity in enumerate(entities):
            if index in fallback_mentions:
                role_mentions.append(fallback_mentions[index])
                continue

            result = results_by_index[index]
            candidate_label = str(result["labels"][0])
            role = self.candidate_label_to_role[candidate_label]
            score = float(result["scores"][0])
            if score < self.confidence_threshold:
                role = self._fallback_role_for_entity(entity)

            role_mentions.append(
                RoleMention(
                    text=entity.text,
                    original_label=entity.label,
                    role=role,
                    score=score,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                )
            )

        return role_mentions

    def _candidate_labels_for_entity(self, entity: EntityMention) -> list[str]:
        allowed_roles = self.ner_role_candidates.get(entity.label)
        if not allowed_roles:
            return list(self.candidate_label_to_role.keys())

        return [
            self.role_to_candidate_label[role]
            for role in allowed_roles
            if role in self.role_to_candidate_label
        ]

    def _fallback_mention(self, entity: EntityMention, score: float) -> RoleMention:
        return RoleMention(
            text=entity.text,
            original_label=entity.label,
            role=self._fallback_role_for_entity(entity),
            score=score,
            start_char=entity.start_char,
            end_char=entity.end_char,
        )

    def _build_prompt(self, text: str, entity: EntityMention) -> str:
        context = self._context_for_entity(text, entity)

        return (
            f"Full argument text: {text}\n\n"
            f"Nearby entity context: {context}\n\n"
            f"Entity phrase: {entity.text}\n"
            f"Original NER label: {entity.label}\n\n"
            "Classify the role of this entity in the argument."
        )

    def _context_for_entity(self, text: str, entity: EntityMention) -> str:
        if self.context_window_chars is None:
            return text

        start = max(0, entity.start_char - self.context_window_chars)
        end = min(len(text), entity.end_char + self.context_window_chars)
        return text[start:end]

    def _fallback_role_for_entity(self, entity: EntityMention) -> str:
        if self.fallback_to_ner_label:
            return entity.label

        return self.fallback_role

    def _build_candidate_label_to_role(
        self,
        roles: list[str],
        role_descriptions: dict[str, str],
        fallback_role: str,
    ) -> dict[str, str]:
        candidate_label_to_role = {}

        for role in roles:
            if role == fallback_role:
                continue

            candidate_label = role_descriptions.get(role, self._humanize_role(role))
            candidate_label_to_role[candidate_label] = role

        if not candidate_label_to_role:
            raise ValueError("At least one non-fallback role must be configured.")

        return candidate_label_to_role

    @staticmethod
    def _humanize_role(role: str) -> str:
        return role.lower().replace("_", " ")

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
