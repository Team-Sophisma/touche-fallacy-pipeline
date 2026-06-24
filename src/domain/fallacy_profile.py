from dataclasses import dataclass, field
from typing import Any


@dataclass
class FallacyArchitecture:
    """
    Conceptual description of one fallacy type.
    Used for interpretation and EDA reporting.
    """

    fallacy_type: str
    display_name: str
    definition: str
    claim_pattern: str
    support_pattern: str
    hidden_reasoning_bridge: str
    failure_point: str
    valid_lookalike: str
    common_cues: list[str] = field(default_factory=list)
    confusion_risks: list[str] = field(default_factory=list)


@dataclass
class FallacyProfile:
    """
    EDA profile for one fallacy type.
    Combines conceptual explanation, dataset statistics, and examples.
    """

    fallacy_type: str
    display_name: str

    total_fallacious: int
    total_valid_lookalikes: int

    definition: str
    claim_pattern: str
    support_pattern: str
    hidden_reasoning_bridge: str
    failure_point: str
    valid_lookalike: str

    common_cues: list[str] = field(default_factory=list)
    confusion_risks: list[str] = field(default_factory=list)

    top_keywords_fallacious: list[tuple[str, int]] = field(default_factory=list)
    top_keywords_valid_lookalikes: list[tuple[str, int]] = field(default_factory=list)

    fallacious_scheme_distribution: dict[str, int] = field(default_factory=dict)
    valid_lookalike_scheme_distribution: dict[str, int] = field(default_factory=dict)

    fallacious_argument_goal_distribution: dict[str, int] = field(default_factory=dict)
    valid_lookalike_argument_goal_distribution: dict[str, int] = field(default_factory=dict)

    fallacious_argument_basis_distribution: dict[str, int] = field(default_factory=dict)
    valid_lookalike_argument_basis_distribution: dict[str, int] = field(default_factory=dict)

    avg_fallacious_source_words: float = 0.0
    avg_valid_lookalike_source_words: float = 0.0

    avg_fallacious_support_count: float = 0.0
    avg_valid_lookalike_support_count: float = 0.0

    representative_fallacious_examples: list[dict[str, Any]] = field(default_factory=list)
    representative_valid_lookalikes: list[dict[str, Any]] = field(default_factory=list)

    interpretation_notes: list[str] = field(default_factory=list)