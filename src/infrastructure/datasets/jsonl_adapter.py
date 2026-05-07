import json
from pathlib import Path
from typing import Iterator


class JsonlAdapter:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def read_items(self) -> Iterator[dict]:
        with self.file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)