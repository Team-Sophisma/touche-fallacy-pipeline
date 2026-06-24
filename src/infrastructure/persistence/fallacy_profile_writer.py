import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.fallacy_profile import FallacyProfile


class FallacyProfileWriter:
    """
    Writes deep fallacy EDA outputs to Markdown, JSON, and CSV.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_dataframe_csv(self, df: pd.DataFrame, filename: str) -> None:
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def write_json(self, data: dict, filename: str) -> None:
        output_path = self.output_dir / filename
        output_path.write_text(
            json.dumps(self._to_jsonable(data), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_profiles_json(
        self,
        profiles: list[FallacyProfile],
        filename: str = "fallacy_profiles.json",
    ) -> None:
        output_path = self.output_dir / filename

        data = [self._to_jsonable(asdict(profile)) for profile in profiles]

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_profiles_markdown(
        self,
        profiles: list[FallacyProfile],
        filename: str = "fallacy_profiles.md",
    ) -> None:
        output_path = self.output_dir / filename

        lines = []
        lines.append("# Fallacy Architecture EDA Report")
        lines.append("")
        lines.append(
            "This report summarizes the reasoning architecture of each fallacy type, "
            "including fallacious examples and valid lookalikes."
        )
        lines.append("")

        for profile in profiles:
            lines.extend(self._profile_to_markdown(profile))

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def write_architecture_matrix(
        self,
        matrix_df: pd.DataFrame,
        filename: str = "fallacy_architecture_matrix.csv",
    ) -> None:
        output_path = self.output_dir / filename
        matrix_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def write_examples_per_fallacy(
        self,
        profiles: list[FallacyProfile],
        examples_dir_name: str = "examples",
    ) -> None:
        examples_dir = self.output_dir / examples_dir_name
        examples_dir.mkdir(parents=True, exist_ok=True)
        expected_filenames = {
            f"{profile.fallacy_type}_examples.md"
            for profile in profiles
        }

        for stale_path in examples_dir.glob("*_examples.md"):
            if stale_path.name not in expected_filenames:
                stale_path.unlink()

        for profile in profiles:
            output_path = examples_dir / f"{profile.fallacy_type}_examples.md"

            lines = []
            lines.append(f"# {profile.display_name} Examples")
            lines.append("")

            lines.append("## Fallacious examples")
            lines.append("")
            for example in profile.representative_fallacious_examples:
                lines.extend(self._example_to_markdown(example))

            lines.append("")
            lines.append("## Valid lookalike examples")
            lines.append("")
            for example in profile.representative_valid_lookalikes:
                lines.extend(self._example_to_markdown(example))

            output_path.write_text("\n".join(lines), encoding="utf-8")

    def _profile_to_markdown(self, profile: FallacyProfile) -> list[str]:
        lines = []

        lines.append(f"## {profile.display_name} (`{profile.fallacy_type}`)")
        lines.append("")

        lines.append("### Definition")
        lines.append(profile.definition)
        lines.append("")

        lines.append("### Argument architecture")
        lines.append(f"- **Claim pattern:** {profile.claim_pattern}")
        lines.append(f"- **Support pattern:** {profile.support_pattern}")
        lines.append(f"- **Hidden reasoning bridge:** {profile.hidden_reasoning_bridge}")
        lines.append(f"- **Failure point:** {profile.failure_point}")
        lines.append(f"- **Valid lookalike:** {profile.valid_lookalike}")
        lines.append("")

        lines.append("### Dataset statistics")
        lines.append(f"- Fallacious examples: {profile.total_fallacious}")
        lines.append(f"- Valid lookalikes: {profile.total_valid_lookalikes}")
        lines.append(
            f"- Average fallacious source words: "
            f"{profile.avg_fallacious_source_words:.2f}"
        )
        lines.append(
            f"- Average valid lookalike source words: "
            f"{profile.avg_valid_lookalike_source_words:.2f}"
        )
        lines.append(
            f"- Average fallacious support count: "
            f"{profile.avg_fallacious_support_count:.2f}"
        )
        lines.append(
            f"- Average valid lookalike support count: "
            f"{profile.avg_valid_lookalike_support_count:.2f}"
        )
        lines.append("")

        lines.append("### Common cues")
        lines.append(", ".join(profile.common_cues) if profile.common_cues else "None")
        lines.append("")

        lines.append("### Confusion risks")
        lines.append(
            ", ".join(profile.confusion_risks)
            if profile.confusion_risks
            else "None"
        )
        lines.append("")

        lines.append("### Fallacious scheme distribution")
        lines.append(self._dict_to_bullets(profile.fallacious_scheme_distribution))
        lines.append("")

        lines.append("### Valid lookalike scheme distribution")
        lines.append(self._dict_to_bullets(profile.valid_lookalike_scheme_distribution))
        lines.append("")

        lines.append("### Top keywords in fallacious examples")
        lines.append(self._keyword_list_to_text(profile.top_keywords_fallacious))
        lines.append("")

        lines.append("### Top keywords in valid lookalikes")
        lines.append(self._keyword_list_to_text(profile.top_keywords_valid_lookalikes))
        lines.append("")

        lines.append("### Interpretation notes")
        for note in profile.interpretation_notes:
            lines.append(f"- {note}")

        lines.append("")

        return lines

    def _example_to_markdown(self, example: dict[str, Any]) -> list[str]:
        lines = []

        lines.append(f"### ID: `{example.get('id')}`")
        lines.append("")
        lines.append(f"- **fallacy_exists:** {example.get('fallacy_exists')}")
        lines.append(f"- **fallacy_type:** {example.get('fallacy_type')}")
        lines.append(f"- **resembles_fallacy:** {example.get('resembles_fallacy')}")
        lines.append(f"- **argument_scheme:** {example.get('argument_scheme')}")
        lines.append("")
        lines.append("**Source text:**")
        lines.append("")
        lines.append(self._truncate(example.get("source_text"), max_chars=1200))
        lines.append("")
        lines.append("**Claim:**")
        lines.append("")
        lines.append(self._truncate(example.get("claim"), max_chars=600))
        lines.append("")
        lines.append("**Supports:**")
        lines.append("")
        lines.append(self._truncate(example.get("supports"), max_chars=900))
        lines.append("")

        return lines

    def _dict_to_bullets(self, value: dict[str, int]) -> str:
        if not value:
            return "None"

        return "\n".join(f"- {key}: {count}" for key, count in value.items())

    def _keyword_list_to_text(self, keywords: list[tuple[str, int]]) -> str:
        if not keywords:
            return "None"

        return ", ".join(f"{word} ({count})" for word, count in keywords)

    def _truncate(self, value: Any, max_chars: int = 1000) -> str:
        if value is None:
            return ""

        text = str(value)

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "..."

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]

        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]

        if hasattr(value, "item"):
            return value.item()

        return value
