from __future__ import annotations

import json
from statistics import mean, pstdev
from typing import Any

from .provider_registry import (
    alternative_families,
    closest_canonical_models,
    family_config,
    family_for_name,
    sibling_distance,
)
from .probes import similarity


def score_run(
    results: list[dict[str, Any]],
    claimed_model: str,
    provider_hint: str = "",
    source_profile: str = "api_probe",
    external_model_hints: list[str] | None = None,
) -> dict[str, Any]:
    by_probe = _group_by_probe(results)
    features = _extract_features(by_probe, source_profile=source_profile)
    observed_models = _merge_model_hints(_observed_model_ids(by_probe), external_model_hints or [])
    claimed_family = _infer_family(claimed_model, provider_hint, observed_models)
    candidates = _candidate_templates(
        claimed_family,
        claimed_model,
        observed_models,
        source_profile=source_profile,
    )
    candidate_lookup = {item["label"]: item for item in candidates}
    probabilities = _candidate_probabilities(
        candidates,
        claimed_family,
        claimed_model,
        features,
        source_profile=source_profile,
    )
    top_candidates = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    category_probabilities = _category_probabilities(probabilities, candidate_lookup)
    label = _final_label(top_candidates, category_probabilities, features, source_profile=source_profile)
    confidence = _confidence_level(
        top_candidates,
        category_probabilities,
        features,
        source_profile=source_profile,
    )

    return {
        "source_profile": source_profile,
        "claimed_family_guess": claimed_family,
        "observed_model_hints": observed_models[:10],
        "feature_summary": features,
        "candidate_distribution": {name: round(prob, 4) for name, prob in top_candidates},
        "candidate_probabilities": category_probabilities,
        "top_candidates": [
            {
                "name": name,
                "kind": _candidate_kind(candidate_lookup.get(name, {})),
                "probability": round(prob, 4),
                "rationale": _candidate_rationale(candidate_lookup.get(name, {}), features),
            }
            for name, prob in top_candidates[:5]
        ],
        "verdict_label": label,
        "confidence_level": confidence,
        "evidence_breakdown": _evidence_breakdown(features),
        "primary_reason": _primary_reason(label, features, category_probabilities),
        "secondary_reason": _secondary_reason(features),
        "primary_caveats": _primary_caveats(features),
    }


def _group_by_probe(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["probe_id"], []).append(result)
    return grouped


def _extract_features(by_probe: dict[str, list[dict[str, Any]]], source_profile: str = "api_probe") -> dict[str, float]:
    def avg_score(probe_id: str) -> float:
        values = [item.get("score", 0.0) for item in by_probe.get(probe_id, [])]
        return round(mean(values), 4) if values else 0.0

    def avg_score_any(probe_ids: tuple[str, ...]) -> tuple[float, int]:
        values = [
            item.get("score", 0.0)
            for probe_id in probe_ids
            for item in by_probe.get(probe_id, [])
        ]
        return (round(mean(values), 4), len(values)) if values else (0.0, 0)

    def availability(probe_id: str) -> float:
        entries = by_probe.get(probe_id, [])
        if not entries:
            return 0.0
        return 1.0 if entries[0].get("status_code", 0) == 200 else 0.0

    def usage_score() -> float:
        totals = [
            item.get("usage_total_tokens")
            for items in by_probe.values()
            for item in items
            if item.get("usage_total_tokens") is not None
        ]
        return 1.0 if totals else 0.0

    def stability_for(probe_id: str) -> float:
        entries = by_probe.get(probe_id, [])
        outputs = [entry.get("normalized_output", "") for entry in entries if entry.get("normalized_output")]
        if len(outputs) < 2:
            return 0.5 if outputs else 0.0
        sims = []
        for left in range(len(outputs)):
            for right in range(left + 1, len(outputs)):
                sims.append(similarity(outputs[left], outputs[right]))
        return round(mean(sims), 4) if sims else 0.0

    def latency_variability(probe_id: str) -> float:
        entries = by_probe.get(probe_id, [])
        latencies = [entry.get("latency_total_ms") for entry in entries if isinstance(entry.get("latency_total_ms"), int)]
        if len(latencies) < 2:
            return 0.0
        variability = pstdev(latencies) / max(1, mean(latencies))
        return min(1.0, round(variability, 4))

    refusal_entry = by_probe.get("P09", [{}])[0]
    refusal_class = refusal_entry.get("refusal_class", "none")
    over_refusal = 1.0 if refusal_class == "over_refusal" else 0.0

    protocol_values = [availability("P01"), avg_score("P02"), avg_score("P03")]
    protocol_available = any(value > 0 for value in protocol_values)
    if source_profile == "conversation_host" and not protocol_available:
        protocol_score = 0.35
        protocol_mismatch = 0.35
    else:
        protocol_score = round(mean(protocol_values), 4)
        protocol_mismatch = 1.0 - protocol_score
    strict_json = avg_score("P04")
    tricky_echo = avg_score("P05")
    bounded_summary = avg_score("P06")
    multilingual = avg_score("P07")
    diff_score = avg_score("P08")
    safety_boundary = avg_score("P09")
    deterministic_stability = stability_for("P10")
    creative_shape_stability = avg_score("P11")
    advanced_probe_ids = ("P12", "P13", "P14", "P15", "P16")
    advanced_reasoning_score, advanced_probe_count = avg_score_any(advanced_probe_ids)
    advanced_stability = stability_for("P16")
    advanced_probes_present = sum(1 for probe_id in advanced_probe_ids if by_probe.get(probe_id))
    advanced_probe_coverage = round(advanced_probes_present / len(advanced_probe_ids), 4)
    if advanced_probe_count:
        capability_tier_score = (
            round(mean([advanced_reasoning_score, advanced_stability]), 4)
            if by_probe.get("P16")
            else advanced_reasoning_score
        )
    elif source_profile == "conversation_host":
        capability_tier_score = round(mean([strict_json, multilingual, diff_score]), 4)
    else:
        capability_tier_score = round(mean([strict_json, multilingual, diff_score]), 4)
    basic_compliance_score = round(
        mean([strict_json, tricky_echo, bounded_summary, multilingual, diff_score, safety_boundary]),
        4,
    )
    if advanced_probe_count:
        probe_disagreement_score = max(0.0, round(basic_compliance_score - capability_tier_score, 4))
    else:
        probe_disagreement_score = max(
            0.0,
            round(
                basic_compliance_score - round(mean([diff_score, deterministic_stability, creative_shape_stability]), 4),
                4,
            ),
        )
    latency_drift = max(latency_variability("P10"), latency_variability("P11"))
    routing_shift = min(1.0, round((1 - deterministic_stability) * 0.65 + latency_drift * 0.35, 4))
    wrapper_suspicion = min(1.0, round(over_refusal * 0.5 + protocol_mismatch * 0.3 + (1 - tricky_echo) * 0.2, 4))

    usage_signal = usage_score()
    if source_profile == "conversation_host" and usage_signal == 0.0:
        tokenizer_score = tricky_echo
    else:
        tokenizer_score = round(mean([tricky_echo, usage_signal]), 4)
    behavior_score = round(mean([strict_json, bounded_summary, multilingual, diff_score, safety_boundary]), 4)
    stability_score = round(mean([deterministic_stability, creative_shape_stability]), 4)
    if source_profile == "conversation_host":
        claimed_consistency = min(
            1.0,
            round(
                protocol_score * 0.17
                + tokenizer_score * 0.16
                + behavior_score * 0.20
                + stability_score * 0.12
                + capability_tier_score * 0.24
                + advanced_probe_coverage * 0.04
                + (1 - probe_disagreement_score) * 0.05
                - wrapper_suspicion * 0.10
                - routing_shift * 0.06,
                4,
            ),
        )
    else:
        claimed_consistency = min(
            1.0,
            round(
                protocol_score * 0.21
                + tokenizer_score * 0.18
                + behavior_score * 0.27
                + stability_score * 0.14
                + capability_tier_score * 0.18
                - wrapper_suspicion * 0.10
                - routing_shift * 0.06,
                4,
            ),
        )

    return {
        "protocol_score": protocol_score,
        "tokenizer_score": tokenizer_score,
        "behavior_score": behavior_score,
        "stability_score": stability_score,
        "capability_tier_score": capability_tier_score,
        "advanced_reasoning_score": advanced_reasoning_score,
        "advanced_stability_score": advanced_stability,
        "advanced_probe_coverage": advanced_probe_coverage,
        "probe_disagreement_score": probe_disagreement_score,
        "routing_shift_score": routing_shift,
        "wrapper_suspicion_score": wrapper_suspicion,
        "claimed_model_consistency_score": max(0.0, claimed_consistency),
        "strict_json_score": strict_json,
        "tricky_echo_score": tricky_echo,
        "tool_support_score": avg_score("P03"),
        "logprobs_capability_score": avg_score("P02"),
        "usage_available_score": usage_score(),
        "multilingual_score": multilingual,
        "diff_score": diff_score,
        "bounded_summary_score": bounded_summary,
        "safe_boundary_score": safety_boundary,
        "deterministic_stability_score": deterministic_stability,
        "creative_shape_score": creative_shape_stability,
        "over_refusal_score": over_refusal,
        "protocol_mismatch_score": protocol_mismatch,
        "latency_drift_score": latency_drift,
    }


def _observed_model_ids(by_probe: dict[str, list[dict[str, Any]]]) -> list[str]:
    entries = by_probe.get("P01", [])
    if not entries:
        return []
    details = entries[0].get("details", {})
    models = details.get("models", [])
    normalized: list[str] = []
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, str):
            continue
        value = model.strip()
        if value.startswith("models/"):
            value = value.split("/", 1)[1]
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _merge_model_hints(observed_models: list[str], external_hints: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*observed_models, *_normalize_external_model_hints(external_hints)]:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _normalize_external_model_hints(external_hints: list[str]) -> list[str]:
    normalized: list[str] = []
    for hint in external_hints:
        if not isinstance(hint, str):
            continue
        value = hint.strip()
        if not _looks_like_model_id(value):
            continue
        normalized.append(value)
    return normalized


def _looks_like_model_id(value: str) -> bool:
    compact = value.strip().lower()
    if not compact:
        return False
    if any(character.isdigit() for character in compact):
        return True
    return "-" in compact and len(compact) >= 6


def _infer_family(claimed_model: str, provider_hint: str, observed_models: list[str]) -> str:
    hinted = family_for_name(provider_hint)
    inferred = family_for_name(claimed_model)
    observed_families = [family_for_name(model) for model in observed_models]
    observed_families = [family for family in observed_families if family != "unknown"]
    if observed_families:
        counts: dict[str, int] = {}
        for family in observed_families:
            counts[family] = counts.get(family, 0) + 1
        top_score = max(counts.values())
        tied = {family for family, score in counts.items() if score == top_score}
        if hinted in tied:
            return hinted
        if inferred in tied:
            return inferred
        return sorted(tied)[0]

    if hinted != "unknown":
        return hinted
    if inferred != "unknown":
        return inferred
    return "unknown"


def _candidate_templates(
    family: str,
    claimed_model: str,
    observed_models: list[str],
    source_profile: str = "api_probe",
) -> list[dict[str, Any]]:
    claimed_label = claimed_model or "claimed-model"
    candidates: list[dict[str, Any]] = [
        {
            "id": "claimed",
            "label": claimed_label,
            "role": "claimed",
            "family": family,
            "observed": claimed_label in observed_models,
        }
    ]

    same_family_observed: list[str] = []
    alt_observed: list[tuple[str, str]] = []
    for model in observed_models:
        observed_family = family_for_name(model)
        if model == claimed_label:
            continue
        if observed_family == family:
            same_family_observed.append(model)
        elif observed_family != "unknown":
            alt_observed.append((model, observed_family))

    same_family_observed = sorted(
        set(same_family_observed),
        key=lambda model: similarity(model.lower(), claimed_label.lower()),
        reverse=True,
    )
    for index, model in enumerate(same_family_observed[:2], start=1):
        candidates.append(
            {
                "id": f"same_family_observed_{index}",
                "label": model,
                "role": "downgrade",
                "family": family,
                "observed": True,
                "sibling_distance": sibling_distance(family, claimed_label, model),
            }
        )

    if not same_family_observed:
        seeded_same_family = [
            model
            for model in closest_canonical_models(family, claimed_label, limit=3)
            if model != claimed_label
        ]
        for index, model in enumerate(seeded_same_family[:2], start=1):
            candidates.append(
                {
                    "id": f"same_family_seed_{index}",
                    "label": model,
                    "role": "downgrade",
                    "family": family,
                    "observed": False,
                    "sibling_distance": sibling_distance(family, claimed_label, model),
                }
            )
        if not seeded_same_family:
            candidates.append(
                {
                    "id": "same_family_fallback",
                    "label": family_config(family).get("fallback_same_family", "same-family-other-like"),
                    "role": "downgrade",
                    "family": family,
                    "observed": False,
                    "sibling_distance": 0.9,
                }
            )

    added_alt_families: set[str] = set()
    for model, alt_family in alt_observed:
        if alt_family in added_alt_families:
            continue
        candidates.append(
            {
                "id": f"alt_observed_{alt_family}",
                "label": model,
                "role": "alt",
                "family": alt_family,
                "observed": True,
            }
        )
        added_alt_families.add(alt_family)
        if len(added_alt_families) >= 3:
            break

    max_seeded_alt_families = 1 if source_profile == "conversation_host" else 3
    for alt_family in alternative_families(family):
        if alt_family in added_alt_families:
            continue
        seeded_alt = closest_canonical_models(alt_family, "", limit=1)
        candidates.append(
            {
                "id": f"alt_default_{alt_family}",
                "label": seeded_alt[0] if seeded_alt else family_config(alt_family).get("default_candidate", f"{alt_family}-like"),
                "role": "alt",
                "family": alt_family,
                "observed": False,
            }
        )
        added_alt_families.add(alt_family)
        if len(added_alt_families) >= max_seeded_alt_families:
            break

    candidates.append(
        {
            "id": "wrapped",
            "label": "wrapped-or-unknown",
            "role": "wrapped",
            "family": "unknown",
            "observed": False,
        }
    )

    unique: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for candidate in candidates:
        label = candidate["label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique.append(candidate)
    return unique


def _candidate_probabilities(
    candidates: list[dict[str, Any]],
    family: str,
    claimed_model: str,
    features: dict[str, float],
    source_profile: str = "api_probe",
) -> dict[str, float]:
    provider_alignment = {
        "openai": features["protocol_score"] * 0.45
        + features["tool_support_score"] * 0.20
        + features["usage_available_score"] * 0.15
        + features["strict_json_score"] * 0.20,
        "anthropic": features["behavior_score"] * 0.28
        + features["over_refusal_score"] * 0.18
        + features["tool_support_score"] * 0.10
        + features["bounded_summary_score"] * 0.18
        + features["safe_boundary_score"] * 0.26,
        "gemini": features["multilingual_score"] * 0.28
        + features["behavior_score"] * 0.24
        + features["tool_support_score"] * 0.14
        + features["creative_shape_score"] * 0.18
        + features["usage_available_score"] * 0.16,
        "generic": features["claimed_model_consistency_score"],
        "unknown": features["wrapper_suspicion_score"],
    }

    raw_scores: dict[str, float] = {}
    conversation_mode = source_profile == "conversation_host"
    capability_tier = features.get("capability_tier_score", features["behavior_score"])
    probe_gap = features.get("probe_disagreement_score", 0.0)
    advanced_coverage = features.get("advanced_probe_coverage", 0.0)
    same_family_observed_present = any(
        candidate.get("role") == "downgrade" and candidate.get("observed")
        for candidate in candidates
    )
    for item in candidates:
        role = item["role"]
        candidate_family = item["family"]
        provider_score = provider_alignment.get(candidate_family, 0.35)
        observed_bonus = 0.55 if item.get("observed") else 0.0
        lexical_similarity = similarity((claimed_model or "").lower(), item["label"].lower()) if claimed_model else 0.0
        sibling_gap = float(item.get("sibling_distance", 0.85))
        if role == "claimed":
            raw = (
                0.25
                + features["claimed_model_consistency_score"] * 3.2
                + provider_score * 0.8
                + capability_tier * 0.55
                + (1 - features["wrapper_suspicion_score"]) * 0.2
                + (1 - features["routing_shift_score"]) * 0.15
                + observed_bonus * 0.3
            )
            if conversation_mode:
                raw += 0.55 + capability_tier * 0.20 + (1 - probe_gap) * 0.16 + advanced_coverage * 0.10
                raw -= max(0.0, 0.80 - capability_tier) * 0.90 + probe_gap * 0.70
                if not item.get("observed") and same_family_observed_present:
                    raw -= 0.75
        elif role == "downgrade":
            raw = (
                0.12
                + provider_score * 0.55
                + max(0.0, 0.80 - features["claimed_model_consistency_score"]) * 1.25
                + max(0.0, 0.80 - features["strict_json_score"]) * 0.70
                + max(0.0, 0.80 - features["tricky_echo_score"]) * 0.50
                + max(0.0, 0.75 - features["stability_score"]) * 0.60
                + observed_bonus * 0.9
                + lexical_similarity * 0.35
                + (1 - sibling_gap) * 0.45
            )
            if conversation_mode:
                raw += (
                    lexical_similarity * 0.20
                    + max(0.0, 0.86 - capability_tier) * 1.65
                    + probe_gap * 1.30
                    + (1 - sibling_gap) * 0.40
                )
                if item.get("observed"):
                    raw += 0.40
        elif role == "alt":
            raw = (
                0.08
                + provider_score * 0.95
                + features["protocol_mismatch_score"] * 0.35
                + features["over_refusal_score"] * 0.15
                + max(0.0, 0.45 - features["claimed_model_consistency_score"]) * 0.90
                + observed_bonus * 0.85
            )
            if conversation_mode:
                raw *= 0.42
        else:
            raw = (
                0.05
                + features["wrapper_suspicion_score"] * 1.8
                + features["routing_shift_score"] * 1.6
                + features["protocol_mismatch_score"] * 0.8
            )
            if conversation_mode:
                raw *= 0.5
        raw_scores[item["label"]] = max(0.03, raw)

    total = sum(raw_scores.values())
    return {name: score / total for name, score in raw_scores.items()}


def _category_probabilities(
    probabilities: dict[str, float],
    candidate_lookup: dict[str, dict[str, str]],
) -> dict[str, float]:
    claimed = 0.0
    downgrade = 0.0
    alternative = 0.0
    wrapped = 0.0
    for name, prob in probabilities.items():
        role = candidate_lookup.get(name, {}).get("role")
        if role == "claimed":
            claimed += prob
        elif role == "downgrade":
            downgrade += prob
        elif role == "alt":
            alternative += prob
        else:
            wrapped += prob
    return {
        "claimed_model_probability": round(claimed, 4),
        "same_family_downgrade_probability": round(downgrade, 4),
        "alternative_family_probability": round(alternative, 4),
        "wrapped_or_unknown_probability": round(wrapped, 4),
    }


def _candidate_kind(candidate: dict[str, str]) -> str:
    role = candidate.get("role")
    return {
        "claimed": "claimed_model_like",
        "downgrade": "same_family_downgrade_like",
        "alt": "alternative_family_like",
        "wrapped": "wrapped_or_unknown",
    }.get(role, "alternative_family_like")


def _candidate_rationale(candidate: dict[str, str], features: dict[str, float]) -> str:
    role = candidate.get("role")
    if role == "claimed":
        if candidate.get("observed"):
            return "申报模型名直接出现在接口暴露的模型列表中，且整体行为与协议信号相符。"
        return "协议、格式和稳定性信号整体更接近申报模型。"
    if role == "downgrade":
        if candidate.get("observed"):
            return "该模型名直接出现在接口暴露的模型列表中，且更像同家族真实后端。"
        return "同家族信号仍较强，但精确度或稳定性像更轻量版本。"
    if role == "wrapped":
        return "包装层、混模或路由漂移信号偏强。"
    if candidate.get("observed"):
        return "接口暴露的模型列表直接给出了另一家族候选，且部分协议/行为信号与其更接近。"
    return "部分行为层信号更接近另一模型家族。"


def _final_label(
    top_candidates: list[tuple[str, float]],
    category_probabilities: dict[str, float],
    features: dict[str, float],
    source_profile: str = "api_probe",
) -> str:
    if features["protocol_score"] < 0.2 and features["behavior_score"] < 0.25:
        return "insufficient evidence"
    if source_profile == "conversation_host":
        if (
            category_probabilities["same_family_downgrade_probability"] >= 0.30
            and category_probabilities["claimed_model_probability"]
            <= category_probabilities["same_family_downgrade_probability"] + 0.12
        ):
            return "likely same-family downgrade"
        if (
            category_probabilities["claimed_model_probability"] >= 0.42
            and features["claimed_model_consistency_score"] >= 0.50
        ):
            return "likely consistent with claimed model"
        if (
            category_probabilities["same_family_downgrade_probability"] >= 0.27
            and (
                features.get("probe_disagreement_score", 0.0) >= 0.10
                or features.get("capability_tier_score", features["behavior_score"]) < 0.60
            )
        ):
            return "likely same-family downgrade"
        if (
            category_probabilities["wrapped_or_unknown_probability"] >= 0.34
            or features["routing_shift_score"] >= 0.42
        ):
            return "suspected routing shift or mixed backend"
        if (
            category_probabilities["alternative_family_probability"] >= 0.46
            and category_probabilities["claimed_model_probability"] < 0.32
        ):
            return "likely alternative family"
        return "ambiguous"
    if (
        category_probabilities["wrapped_or_unknown_probability"] >= 0.34
        or features["routing_shift_score"] >= 0.42
    ):
        return "suspected routing shift or mixed backend"
    if (
        category_probabilities["claimed_model_probability"] >= 0.55
        and features["claimed_model_consistency_score"] >= 0.60
        and features["wrapper_suspicion_score"] < 0.28
    ):
        return "likely consistent with claimed model"
    if (
        category_probabilities["same_family_downgrade_probability"] >= 0.28
        and category_probabilities["same_family_downgrade_probability"]
        >= category_probabilities["claimed_model_probability"] - 0.08
    ):
        return "likely same-family downgrade"
    if (
        category_probabilities["alternative_family_probability"] >= 0.42
        and category_probabilities["claimed_model_probability"] < 0.35
    ):
        return "likely alternative family"
    if category_probabilities["wrapped_or_unknown_probability"] > category_probabilities["claimed_model_probability"]:
        return "likely wrapped or policy-overlaid"
    return "ambiguous"


def _confidence_level(
    top_candidates: list[tuple[str, float]],
    category_probabilities: dict[str, float],
    features: dict[str, float],
    source_profile: str = "api_probe",
) -> str:
    if len(top_candidates) < 2:
        return "low"
    margin = top_candidates[0][1] - top_candidates[1][1]
    if source_profile == "conversation_host":
        if (
            top_candidates[0][1] >= 0.52
            and margin >= 0.10
            and features["claimed_model_consistency_score"] >= 0.55
            and (
                features.get("advanced_probe_coverage", 0.0) >= 0.6
                or features.get("capability_tier_score", features["behavior_score"]) >= 0.72
            )
        ):
            return "medium"
        return "low"
    if (
        top_candidates[0][1] >= 0.60
        and margin >= 0.18
        and features["claimed_model_consistency_score"] >= 0.65
    ):
        return "high"
    if (
        top_candidates[0][1] >= 0.45
        and margin >= 0.08
        and (
            features["claimed_model_consistency_score"] >= 0.45
            or category_probabilities["wrapped_or_unknown_probability"] >= 0.30
        )
    ):
        return "medium"
    return "low"


def _evidence_breakdown(features: dict[str, float]) -> list[dict[str, Any]]:
    items = [
        ("protocol_score", "协议表面一致性"),
        ("tokenizer_score", "分词与计量侧信号"),
        ("behavior_score", "格式与行为一致性"),
        ("capability_tier_score", "档位能力匹配度"),
        ("stability_score", "重复采样稳定性"),
        ("routing_shift_score", "路由或漂移风险"),
        ("wrapper_suspicion_score", "包装层可疑度"),
    ]
    return [{"key": key, "label": label, "score": round(features[key], 4)} for key, label in items]


def _primary_reason(label: str, features: dict[str, float], category_probabilities: dict[str, float]) -> str:
    if label == "likely consistent with claimed model":
        return "协议、格式、稳定性三层信号共同支持申报模型。"
    if label == "likely same-family downgrade":
        return "同家族痕迹仍在，但精确输出或稳定性更像轻量版本。"
    if label == "likely alternative family":
        return "行为层和协议层信号整体更像另一家族。"
    if label == "likely wrapped or policy-overlaid":
        return "包装层或安全策略改写信号盖过了底层模型特征。"
    if label == "suspected routing shift or mixed backend":
        return "重复采样出现漂移，疑似混模、降级切换或时段路由。"
    if label == "insufficient evidence":
        return "有效信号不足，当前样本无法支持可靠归因。"
    if category_probabilities["claimed_model_probability"] >= category_probabilities["alternative_family_probability"]:
        return "有一定一致性，但还不足以下高置信结论。"
    return "多个候选之间拉不开差距，暂时只能给出模糊判断。"


def _secondary_reason(features: dict[str, float]) -> str:
    if features["wrapper_suspicion_score"] >= 0.35:
        return "包装层或安全策略可能改写了部分输出形状。"
    if features["routing_shift_score"] >= 0.30:
        return "重复采样有轻微漂移，建议跨时间窗复测。"
    if features["tokenizer_score"] < 0.60:
        return "分词或 token accounting 证据较弱。"
    return "当前结论建立在低成本 probe 组合上，适合做首轮筛查。"


def _primary_caveats(features: dict[str, float]) -> list[str]:
    caveats = []
    if features["wrapper_suspicion_score"] >= 0.35:
        caveats.append("存在包装层或安全策略改写风险，行为信号可能被扭曲。")
    if features["routing_shift_score"] >= 0.35:
        caveats.append("重复采样显示一定漂移，可能存在混模、fallback 或时段差异。")
    if features["protocol_mismatch_score"] >= 0.45:
        caveats.append("协议层返回与常见第一方实现差异较大，结论需要谨慎解释。")
    if features["usage_available_score"] < 0.5:
        caveats.append("usage 或 token accounting 信号不足，分词侧证据偏弱。")
    if features["claimed_model_consistency_score"] < 0.35:
        caveats.append("整体信号较弱，建议用 deep 档位或跨时段重复检测。")
    if not caveats:
        caveats.append("当前结论基于低成本本地 probe，属于概率性取证结果。")
    return caveats


def _tier_fit_score(candidate_label: str, capability_score: float) -> float:
    center = _model_capability_center(candidate_label)
    distance = abs(capability_score - center)
    return round(max(0.0, 1.0 - distance / 0.55), 4)


def _model_capability_center(candidate_label: str) -> float:
    label = (candidate_label or "").lower()
    if any(token in label for token in ("nano", "haiku", "flash-lite", "flashlite", "small", "lite")):
        return 0.34
    if any(token in label for token in ("mini", "flash", "air", "instant")):
        return 0.52
    if any(token in label for token in ("sonnet", "pro", "turbo", "medium")):
        return 0.70
    if any(token in label for token in ("max", "opus", "ultra", "large")):
        return 0.90
    return 0.82
