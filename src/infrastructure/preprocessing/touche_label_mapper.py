from src.domain.interfaces import LabelMapper


class ToucheLabelMapper(LabelMapper):
    FALLACY_TYPE_LABELS = {
        "authority": "authority",
        "blackwhite": "black-white",
        "black-white": "black-white",
        "hasty_generalization": "hasty_generalization",
        "natural": "natural",
        "population": "population",
        "slippery_slope": "slippery_slope",
        "tradition": "tradition",
        "worse_problems": "worse_problems",
    }

    def map_label(self, item: dict, task_name: str) -> str | None:
        if task_name == "fallacy_detection":
            return self._map_fallacy_detection(item)

        if task_name == "fallacy_classification":
            return self._map_fallacy_classification(item)

        if task_name == "scheme_classification":
            return self._map_scheme_classification(item)

        raise ValueError(f"Unknown preprocessing task: {task_name}")

    def should_use_for_training(self, item: dict, task_name: str) -> bool:
        if task_name == "fallacy_detection":
            return item.get("fallacy_exists") in {0, 1, "0", "1"}

        if task_name == "fallacy_classification":
            return self._is_fallacious(item) and self.map_label(item, task_name) is not None

        if task_name == "scheme_classification":
            return not self._is_fallacious(item) and self.map_label(item, task_name) is not None

        raise ValueError(f"Unknown preprocessing task: {task_name}")

    def _map_fallacy_detection(self, item: dict) -> str | None:
        value = item.get("fallacy_exists")
        if value in {1, "1"}:
            return "fallacy"
        if value in {0, "0"}:
            return "non-fallacy"
        return None

    def _map_fallacy_classification(self, item: dict) -> str | None:
        label = item.get("fallacy_type")
        if label is None:
            return None

        return self.FALLACY_TYPE_LABELS.get(str(label))

    def _map_scheme_classification(self, item: dict) -> str | None:
        classification = item.get("classification") or {}
        argument_goal = classification.get("argument_goal")
        argument_basis = classification.get("argument_basis")

        if not argument_goal or not argument_basis:
            return None

        return f"{argument_goal}-{argument_basis}"

    def _is_fallacious(self, item: dict) -> bool:
        return item.get("fallacy_exists") in {1, "1"}
