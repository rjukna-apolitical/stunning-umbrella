import re
from typing import Optional, Set


def normalize_locale(locale: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", locale.lower()).strip("_")


def build_locale_metadata(entry_raw: dict, fields_to_include: Optional[Set[str]] = None) -> dict:
    out = {}
    fields = entry_raw.get("fields", {})
    for field_name, localized_values in fields.items():
        if fields_to_include and field_name not in fields_to_include:
            continue
        if not isinstance(localized_values, dict):
            continue
        for locale, value in localized_values.items():
            if value is None:
                continue
            loc = normalize_locale(locale)
            key = f"{field_name}_{loc}"
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
            elif isinstance(value, list):
                cleaned = [v for v in value if isinstance(v, (str, int, float, bool))]
                if cleaned:
                    out[key] = cleaned
            else:
                out[key] = str(value)
    return out
