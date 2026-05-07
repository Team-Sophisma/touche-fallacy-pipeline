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