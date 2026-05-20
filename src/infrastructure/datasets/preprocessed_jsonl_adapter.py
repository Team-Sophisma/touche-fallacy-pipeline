from src.domain.entities import TextSample
from src.infrastructure.datasets.jsonl_adapter import JsonlAdapter


class PreprocessedJsonlAdapter:
    """
    Loads already preprocessed JSONL splits for EDA-only analysis.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[TextSample]:
        return [
            self._to_text_sample(item)
            for item in JsonlAdapter(self.file_path).read_items()
        ]

    def _to_text_sample(self, item: dict) -> TextSample:
        label = item.get("label")
        text = item.get("text", "")

        return TextSample(
            id=item["id"],
            source_text=text,
            argument_text=text,
            label=label,
            task_name=item.get("task", ""),
            meta={
                "fallacy_type": label,
                "source_id": (item.get("meta") or {}).get("source_id"),
                "tag": item.get("tag"),
            },
        )
