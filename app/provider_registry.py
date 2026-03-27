from __future__ import annotations

import json
from functools import lru_cache
from difflib import SequenceMatcher
from typing import Any

from .config import MODELS_DIR


FAMILY_REGISTRY_PATH = MODELS_DIR / "family_registry.json"
CANONICAL_MODEL_REGISTRY_PATH = MODELS_DIR / "canonical_model_registry.json"


@lru_cache(maxsize=1)
def load_family_registry() -> dict[str, Any]:
    return json.loads(FAMILY_REGISTRY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_canonical_model_registry() -> dict[str, Any]:
    return json.loads(CANONICAL_MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))


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


def canonical_models_for_family(family: str) -> list[dict[str, Any]]:
    registry = load_canonical_model_registry()["families"]
    family_data = registry.get(family, registry.get("unknown", {}))
    models = family_data.get("models", [])
    return [model for model in models if isinstance(model, dict) and model.get("id")]


def canonical_model_ids_for_family(family: str) -> list[str]:
    return [model["id"] for model in canonical_models_for_family(family)]


def preferred_defaults_for_family(family: str) -> list[str]:
    registry = load_canonical_model_registry()["families"]
    family_data = registry.get(family, registry.get("unknown", {}))
    defaults = family_data.get("preferred_defaults", [])
    return [value for value in defaults if isinstance(value, str) and value]


def canonical_model_record_for_name(family: str, value: str) -> dict[str, Any] | None:
    claimed = (value or "").strip().lower()
    if not claimed:
        return None

    best_match: dict[str, Any] | None = None
    best_score = 0.0
    for model in canonical_models_for_family(family):
        model_id = model["id"]
        alias_candidates = [model_id, *model.get("aliases", [])]
        score = max(_alias_similarity(claimed, alias) for alias in alias_candidates if isinstance(alias, str))
        if score > best_score:
            best_score = score
            best_match = model
    if best_score >= 0.72:
        return best_match
    return None


def sibling_distance(family: str, claimed_model: str, candidate_model: str) -> float:
    if not claimed_model or not candidate_model:
        return 1.0
    if claimed_model.strip().lower() == candidate_model.strip().lower():
        return 0.0

    claimed_record = canonical_model_record_for_name(family, claimed_model)
    candidate_record = canonical_model_record_for_name(family, candidate_model)
    if claimed_record and candidate_record:
        generation_penalty = 0.0
        claimed_generation = claimed_record.get("generation")
        candidate_generation = candidate_record.get("generation")
        if isinstance(claimed_generation, (int, float)) and isinstance(candidate_generation, (int, float)):
            generation_penalty = min(0.45, abs(float(claimed_generation) - float(candidate_generation)) * 0.22)

        series_penalty = 0.0
        claimed_series = str(claimed_record.get("series") or "").strip().lower()
        candidate_series = str(candidate_record.get("series") or "").strip().lower()
        if claimed_series and candidate_series and claimed_series != candidate_series:
            series_penalty = 0.18

        tier_penalty = _tier_penalty(claimed_record.get("tier"), candidate_record.get("tier"))
        return min(1.0, round(generation_penalty + series_penalty + tier_penalty, 4))

    return min(
        1.0,
        round(1.0 - _alias_similarity(claimed_model.strip().lower(), candidate_model.strip().lower()), 4),
    )


def closest_canonical_models(family: str, claimed_model: str, limit: int = 3) -> list[str]:
    claimed = (claimed_model or "").strip().lower()
    if not claimed:
        defaults = preferred_defaults_for_family(family)
        return defaults[:limit]

    scored: list[tuple[float, str]] = []
    for model in canonical_models_for_family(family):
        model_id = model["id"]
        alias_candidates = [model_id, *model.get("aliases", [])]
        score = max(_alias_similarity(claimed, alias) for alias in alias_candidates if isinstance(alias, str))
        scored.append((score, model_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    ordered = [model_id for score, model_id in scored if score > 0]
    if len(ordered) < limit:
        for default_id in preferred_defaults_for_family(family):
            if default_id not in ordered:
                ordered.append(default_id)
    return ordered[:limit]


def resolve_dialect(base_url: str, provider_hint: str, claimed_model: str) -> str:
    url = (base_url or "").strip().lower()
    family = family_for_name(provider_hint)
    if family == "unknown":
        family = family_for_name(claimed_model)
    if family == "unknown":
        family = family_for_name(url)

    if url.startswith("mock://"):
        mock_family = url.split("mock://", 1)[1].split("/", 1)[0]
        if mock_family in {"anthropic", "minimax"}:
            return "anthropic_messages"
        if mock_family == "gemini":
            return "gemini_generate_content"
        if family in {"anthropic", "minimax"}:
            return "anthropic_messages"
        if family == "gemini":
            return "gemini_generate_content"
        return "openai_chat_completions"

    if "generativelanguage.googleapis.com" in url and "/openai/" not in url:
        return "gemini_generate_content"
    if "api.anthropic.com" in url:
        return "anthropic_messages"

    family_dialects = family_config(family).get("dialects", [])
    if family_dialects:
        return family_dialects[0]
    return "openai_chat_completions"


def _alias_similarity(claimed: str, alias: str) -> float:
    normalized_alias = alias.strip().lower()
    if not normalized_alias:
        return 0.0
    if claimed == normalized_alias:
        return 1.0
    if claimed in normalized_alias or normalized_alias in claimed:
        return 0.94
    compact_claimed = _compact_token(claimed)
    compact_alias = _compact_token(normalized_alias)
    if compact_claimed == compact_alias:
        return 0.98
    if compact_claimed and (compact_claimed in compact_alias or compact_alias in compact_claimed):
        return 0.90
    return SequenceMatcher(None, compact_claimed, compact_alias).ratio()


def _tier_penalty(left: Any, right: Any) -> float:
    left_value = str(left or "").strip().lower()
    right_value = str(right or "").strip().lower()
    if not left_value or not right_value:
        return 0.16
    if left_value == right_value:
        return 0.0

    tier_groups = [
        {"flagship", "pro", "opus"},
        {"sonnet", "flash", "base"},
        {"mini", "lite", "haiku"},
        {"nano"},
    ]
    left_index = next((index for index, group in enumerate(tier_groups) if left_value in group), None)
    right_index = next((index for index, group in enumerate(tier_groups) if right_value in group), None)
    if left_index is None or right_index is None:
        return 0.18
    return min(0.36, abs(left_index - right_index) * 0.12)


def _compact_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
