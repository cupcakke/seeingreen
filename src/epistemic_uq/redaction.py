import re
from collections.abc import Iterable


PATTERNS = {
    "email": re.compile(r"(?<![\w.-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
    "ip": re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"),
}


class Redactor:
    def __init__(self, enabled: Iterable[str]) -> None:
        self._patterns = [PATTERNS[name] for name in enabled if name in PATTERNS]

    def redact(self, text: str) -> str:
        result = text
        for pattern in self._patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def redact_mapping(self, value):
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self.redact_mapping(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_mapping(item) for item in value)
        if isinstance(value, dict):
            return {key: self.redact_mapping(item) for key, item in value.items()}
        return value
