from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .config import MODELS_DIR


FAMILY_REGISTRY_PATH = MODELS_DIR / "family_registry.json"


@lru_cache(maxsize=1)
def load_family_registry() -> dict[str, Any]:
    return json.loads(FAMILY_REGISTRY_PATH.read_text(encoding="utf-8"))


def family_for_name(value: str) -> str:
    lowered = (value or "").lower()
    if not lowered:
        return "unknown"
    registry = load_family_registry()["families"]
    for family, config in registry.items():
        keywords = config.get("keywords", [])
        if any(keyword in lowered for keyword in keywords):
            return family
    return "unknown"


def family_config(family: str) -> dict[str, Any]:
    registry = load_family_registry()["families"]
    return registry.get(family, registry["unknown"])


def alternative_families(exclude_family: str) -> list[str]:
    registry = load_family_registry()
    return [family for family in registry.get("alternative_order", []) if family != exclude_family]
