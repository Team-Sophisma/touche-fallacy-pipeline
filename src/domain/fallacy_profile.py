from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FallacyProfile:
    fallacy_type: str
    sample_count: int
    common_keywords: list[str] = field(default_factory=list)
    representative_examples: list[dict[str, Any]] = field(default_factory=list)
    hard_negatives: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
