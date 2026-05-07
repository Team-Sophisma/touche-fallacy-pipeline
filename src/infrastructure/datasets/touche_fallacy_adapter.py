
from typing import Optional

from src.domain.entities import TextSample
from src.domain.interfaces import DatasetAdapter
from src.infrastructure.datasets.jsonl_adapter import JsonlAdapter


class ToucheFallacyAdapter(DatasetAdapter):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        task_name: str,
        text_variant: str = "text_base",
        argument_variant: str = "argument_base",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.task_name = task_name
        self.text_variant = text_variant
        self.argument_variant = argument_variant

    def load_train(self) -> list[TextSample]:
        return [
            self._to_text_sample(item, is_train=True)
            for item in JsonlAdapter(self.train_path).read_items()
        ]

    def load_test(self) -> list[TextSample]:
        return [
            self._to_text_sample(item, is_train=False)
            for item in JsonlAdapter(self.test_path).read_items()
        ]

    def _to_text_sample(self, item: dict, is_train: bool) -> TextSample:
        argument_data = item.get(self.argument_variant, {}) or {}

        return TextSample(
            id=item["id"],
            source_text=item.get(self.text_variant, ""),
            argument_text=self._build_argument_text(argument_data),
            label=self._extract_label(item) if is_train else None,
            task_name=self.task_name,
            meta=self._build_meta(item),
        )

    def _extract_label(self, item: dict) -> Optional[str]:
        if self.task_name == "fallacy_binary":
            return str(item.get("fallacy_exists"))

        if self.task_name == "fallacy_type":
            return item.get("fallacy_type")

        if self.task_name == "resembles_fallacy":
            return item.get("resembles_fallacy")

        if self.task_name == "argument_goal":
            return (item.get("classification") or {}).get("argument_goal")

        if self.task_name == "argument_basis":
            return (item.get("classification") or {}).get("argument_basis")

        raise ValueError(f"Unknown task_name: {self.task_name}")

    def _build_argument_text(self, argument_data: dict) -> str:
        claim = argument_data.get("claim", "")
        supports = argument_data.get("supports", [])

        supports_text = " ".join(supports)

        return f"Claim: {claim}\nSupports: {supports_text}".strip()

    def _build_meta(self, item: dict) -> dict:
        classification = item.get("classification") or {}
        argument_base = item.get("argument_base") or {}
        argument_enhanced = item.get("argument_enhanced") or {}

        return {
            "title": item.get("text_raw_title"),
            "parent_text": item.get("text_raw_parent"),
            "text_raw": item.get("text_raw"),
            "text_base": item.get("text_base"),
            "text_enhanced": item.get("text_enhanced"),

            "base_claim": argument_base.get("claim"),
            "base_supports": argument_base.get("supports", []),

            "enhanced_claim": argument_enhanced.get("claim"),
            "enhanced_supports": argument_enhanced.get("supports", []),

            "fallacy_exists": item.get("fallacy_exists"),
            "fallacy_type": item.get("fallacy_type"),
            "resembles_fallacy": item.get("resembles_fallacy"),

            "argument_goal": classification.get("argument_goal"),
            "argument_basis": classification.get("argument_basis"),
        }