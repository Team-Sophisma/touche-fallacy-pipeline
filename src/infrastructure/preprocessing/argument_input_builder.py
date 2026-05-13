from src.domain.interfaces import InputBuilder
from src.infrastructure.preprocessing.text_normalizer import TextNormalizer


class ArgumentInputBuilder(InputBuilder):
    def __init__(
        self,
        text_field: str,
        argument_field: str,
        normalizer: TextNormalizer,
        include_title: bool = False,
        include_parent: bool = False,
    ):
        self.text_field = text_field
        self.argument_field = argument_field
        self.normalizer = normalizer
        self.include_title = include_title
        self.include_parent = include_parent

    def build(self, item: dict) -> str:
        sections = []

        if self.include_title:
            self._append_section(sections, "Title", item.get("text_raw_title"))

        if self.include_parent:
            self._append_section(sections, "Parent", item.get("text_raw_parent"))

        self._append_section(sections, "Text", item.get(self.text_field))

        argument = item.get(self.argument_field) or {}
        self._append_section(sections, "Claim", argument.get("claim"))

        supports = self._build_supports_text(argument.get("supports", []))
        self._append_section(sections, "Supports", supports)

        return "\n\n".join(sections)

    def _append_section(
        self,
        sections: list[str],
        title: str,
        value: object,
    ) -> None:
        normalized = self.normalizer.normalize(value)
        if normalized:
            sections.append(f"{title}: {normalized}")

    def _build_supports_text(self, supports: object) -> str:
        if not isinstance(supports, list):
            return self.normalizer.normalize(supports)

        normalized_supports = [
            self.normalizer.normalize(support)
            for support in supports
        ]
        normalized_supports = [
            support for support in normalized_supports
            if support
        ]

        return " [SEP] ".join(normalized_supports)
