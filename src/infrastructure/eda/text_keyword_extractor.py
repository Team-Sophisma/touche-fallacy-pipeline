from collections import Counter
import re
from typing import Iterable


class TextKeywordExtractor:
    _stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "will", "with", "you", "your",
    }

    def extract(self, texts: Iterable[str], top_n: int = 20) -> list[str]:
        counter: Counter[str] = Counter()

        for text in texts:
            words = re.findall(r"[A-Za-z][A-Za-z']+", str(text or "").lower())
            counter.update(word for word in words if word not in self._stopwords)

        return [word for word, _ in counter.most_common(top_n)]
