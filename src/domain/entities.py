from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class TextSample:
    id: str
    source_text: str
    argument_text: str
    label: Optional[str]
    task_name: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessedSample:
    id: str
    task: str
    text: str
    label: Optional[str]
    tag: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityMention:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RoleMention:
    text: str
    original_label: str
    role: str
    score: float
    start_char: int
    end_char: int


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
