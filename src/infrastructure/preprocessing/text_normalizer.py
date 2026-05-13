import re
from typing import Any


class TextNormalizer:
    def normalize(self, value: Any) -> str:
        if value is None:
            return ""

        text = str(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
