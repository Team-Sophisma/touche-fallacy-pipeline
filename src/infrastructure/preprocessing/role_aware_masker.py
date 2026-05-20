from src.domain.entities import RoleMention


class RoleAwareMasker:
    def mask(self, text: str, role_mentions: list[RoleMention]) -> str:
        masked_text = text

        for mention in sorted(role_mentions, key=lambda item: item.start_char, reverse=True):
            masked_text = (
                masked_text[:mention.start_char]
                + f"[{mention.role}]"
                + masked_text[mention.end_char:]
            )

        return masked_text
