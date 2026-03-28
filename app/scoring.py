from __future__ import annotations

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
    observed_models = _observed_model_ids(by_probe)
    weak_model_hints = _normalize_external_model_hints(external_model_hints or [])
    claimed_family = _infer_family(claimed_model, provider_hint)
    candidates = _candidate_templates(
        claimed_family,
        claimed_model,
        observed_models,
        weak_model_hints,
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
    hypothesis_ranking = _hypothesis_ranking(category_probabilities)
    label = _final_label(top_candidates, category_probabilities, features, source_profile=source_profile)
    confidence = _confidence_level(
        top_candidates,
        category_probabilities,
        features,
        source_profile=source_profile,
    )
    model_candidate_ranking = [
        {
            "name": name,
            "kind": _candidate_kind(candidate_lookup.get(name, {})),
            "probability": round(prob, 4),
            "rationale": _candidate_rationale(candidate_lookup.get(name, {}), features),
            "hint_source": candidate_lookup.get(name, {}).get("hint_source", "seed"),
        }
        for name, prob in top_candidates
    ]

    return {
        "source_profile": source_profile,
        "claimed_family_guess": claimed_family,
        "observed_model_hints": observed_models[:10],
        "weak_model_hints": weak_model_hints[:10],
        "feature_summary": features,
        "candidate_distribution": {name: round(prob, 4) for name, prob in top_candidates},
        "candidate_probabilities": category_probabilities,
        "hypothesis_ranking": hypothesis_ranking,
        "model_candidate_ranking": model_candidate_ranking,
        "top_candidates": model_candidate_ranking[:5],
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


def _normalize_external_model_hints(external_hints: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for hint in external_hints:
        if not isinstance(hint, str):
            continue
        value = hint.strip()
        if not _looks_like_model_id(value) or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _looks_like_model_id(value: str) -> bool:
    compact = value.strip().lower()
    if not compact:
        return False
    if any(character.isdigit() for character in compact):
        return True
    return "-" in compact and len(compact) >= 6


def _infer_family(claimed_model: str, provider_hint: str) -> str:
    hinted = family_for_name(provider_hint)
    inferred = family_for_name(claimed_model)
    if hinted != "unknown":
        return hinted
    if inferred != "unknown":
        return inferred
    return "unknown"


def _candidate_templates(
    family: str,
    claimed_model: str,
    observed_models: list[str],
    weak_model_hints: list[str],
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
            "hint_source": "claimed_input",
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
                "hint_source": "catalog_observed",
            }
        )

    same_family_weak_hints = sorted(
        {
            model
            for model in weak_model_hints
            if model != claimed_label and family_for_name(model) == family
        },
        key=lambda model: similarity(model.lower(), claimed_label.lower()),
        reverse=True,
    )
    for index, model in enumerate(same_family_weak_hints[:1], start=1):
        candidates.append(
            {
                "id": f"same_family_hint_{index}",
                "label": model,
                "role": "downgrade",
                "family": family,
                "observed": False,
                "sibling_distance": sibling_distance(family, claimed_label, model),
                "hint_source": "weak_self_report",
            }
        )

    seeded_same_family = [
        model
        for model in closest_canonical_models(family, claimed_label, limit=3)
        if model != claimed_label and model not in same_family_weak_hints and model not in same_family_observed
    ]
    if not same_family_observed:
        for index, model in enumerate(seeded_same_family[:2], start=1):
            candidates.append(
                {
                    "id": f"same_family_seed_{index}",
                    "label": model,
                    "role": "downgrade",
                    "family": family,
                    "observed": False,
                    "sibling_distance": sibling_distance(family, claimed_label, model),
                    "hint_source": "seed",
                }
            )
        if not seeded_same_family and not same_family_weak_hints:
            candidates.append(
                {
                    "id": "same_family_fallback",
                    "label": family_config(family).get("fallback_same_family", "same-family-other-like"),
                    "role": "downgrade",
                    "family": family,
                    "observed": False,
                    "sibling_distance": 0.9,
                    "hint_source": "fallback",
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
                "hint_source": "catalog_observed",
            }
        )
        added_alt_families.add(alt_family)
        if len(added_alt_families) >= 3:
            break

    for model in weak_model_hints:
        alt_family = family_for_name(model)
        if alt_family in {family, "unknown"} or alt_family in added_alt_families:
            continue
        candidates.append(
            {
                "id": f"alt_hint_{alt_family}",
                "label": model,
                "role": "alt",
                "family": alt_family,
                "observed": False,
                "hint_source": "weak_self_report",
            }
        )
        added_alt_families.add(alt_family)
        break

    has_weak_alt_hint = any(
        family_for_name(model) not in {family, "unknown"}
        for model in weak_model_hints
    )
    if alt_observed or has_weak_alt_hint:
        max_seeded_alt_families = 1 if source_profile == "conversation_host" else 2
    elif same_family_observed or same_family_weak_hints:
        max_seeded_alt_families = 0 if source_profile == "conversation_host" else 1
    else:
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
                "hint_source": "seed",
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
            "hint_source": "derived",
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
    claimed_tier_fit = _tier_fit_score(claimed_model, capability_tier)
    probe_gap = features.get("probe_disagreement_score", 0.0)
    advanced_coverage = features.get("advanced_probe_coverage", 0.0)
    same_family_catalog_present = any(
        candidate.get("role") == "downgrade" and candidate.get("hint_source") == "catalog_observed"
        for candidate in candidates
    )

    for item in candidates:
        role = item["role"]
        candidate_family = item["family"]
        provider_score = provider_alignment.get(candidate_family, 0.35)
        hint_bonus = _hint_bonus(item)
        lexical_similarity = similarity((claimed_model or "").lower(), item["label"].lower()) if claimed_model else 0.0
        sibling_gap = float(item.get("sibling_distance", 0.85))
        tier_fit = _tier_fit_score(item["label"], capability_tier)
        tier_advantage = max(0.0, tier_fit - claimed_tier_fit)

        if role == "claimed":
            raw = (
                0.20
                + features["claimed_model_consistency_score"] * 3.25
                + provider_score * 0.75
                + tier_fit * 0.35
                + (1 - features["wrapper_suspicion_score"]) * 0.18
                + (1 - features["routing_shift_score"]) * 0.12
                + hint_bonus * 0.08
            )
            if conversation_mode:
                raw += 0.45 + capability_tier * 0.18 + (1 - probe_gap) * 0.16 + advanced_coverage * 0.12
                raw -= max(0.0, 0.80 - capability_tier) * 0.92 + probe_gap * 0.72
                if not item.get("observed") and same_family_catalog_present:
                    raw -= 0.55
        elif role == "downgrade":
            raw = (
                0.10
                + provider_score * 0.45
                + max(0.0, 0.82 - features["claimed_model_consistency_score"]) * 1.15
                + max(0.0, 0.78 - features["stability_score"]) * 0.35
                + tier_fit * 0.85
                + tier_advantage * 0.95
                + lexical_similarity * 0.28
                + (1 - sibling_gap) * 0.40
                + hint_bonus * 0.20
            )
            if conversation_mode:
                raw += (
                    max(0.0, 0.86 - capability_tier) * 1.45
                    + probe_gap * 1.20
                    + advanced_coverage * 0.08
                )
                if item.get("hint_source") == "catalog_observed":
                    raw += 0.08
        elif role == "alt":
            raw = (
                0.08
                + provider_score * 0.85
                + features["protocol_mismatch_score"] * 0.35
                + features["over_refusal_score"] * 0.15
                + max(0.0, 0.45 - features["claimed_model_consistency_score"]) * 1.00
                + tier_fit * 0.20
                + hint_bonus * 0.18
            )
            if conversation_mode:
                raw *= 0.45
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


def _hint_bonus(candidate: dict[str, Any]) -> float:
    source = candidate.get("hint_source", "seed")
    if source == "catalog_observed":
        return 0.12
    if source == "weak_self_report":
        return 0.04
    if source == "claimed_input":
        return 0.02
    return 0.0


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


def _hypothesis_ranking(category_probabilities: dict[str, float]) -> list[dict[str, Any]]:
    items = [
        (
            "claimed_model_probability",
            "claimed-model-consistent",
            "Signals remain most consistent with the claimed model.",
        ),
        (
            "same_family_downgrade_probability",
            "same-family-downgrade",
            "Signals suggest the same family, but likely a weaker sibling or fallback path.",
        ),
        (
            "alternative_family_probability",
            "alternative-family",
            "Signals suggest another model family may be behind the endpoint.",
        ),
        (
            "wrapped_or_unknown_probability",
            "wrapped-or-unknown",
            "Signals suggest a wrapper, routing layer, or insufficiently attributable backend.",
        ),
    ]
    ranked = sorted(items, key=lambda item: category_probabilities[item[0]], reverse=True)
    return [
        {
            "id": key,
            "label": label,
            "probability": round(category_probabilities[key], 4),
            "rationale": rationale,
        }
        for key, label, rationale in ranked
    ]


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
    source = candidate.get("hint_source", "seed")
    if role == "claimed":
        if source == "catalog_observed":
            return "The claimed model also appears in the exposed catalog, but that metadata is treated as weak supporting evidence only."
        return "Protocol, behavior, and stability signals lean toward the claimed model."
    if role == "downgrade":
        if source == "catalog_observed":
            return "A same-family sibling appears in the exposed catalog, but the main evidence comes from weaker capability or stability fit."
        if source == "weak_self_report":
            return "A self-reported sibling hint exists, but it is treated as weak metadata and only lightly nudges ranking."
        return "Same-family evidence remains strong, but capability or stability looks more like a lighter sibling."
    if role == "wrapped":
        return "Wrapper, routing drift, or mixed-backend signals are stronger than any single-model match."
    if source == "catalog_observed":
        return "An alternative-family model appears in the exposed catalog, but that catalog evidence is not trusted as a strong identity proof."
    if source == "weak_self_report":
        return "An alternative-family self-report exists, but it is treated as weak metadata only."
    return "Some behavior or protocol evidence fits another model family better than the claimed one."


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
        ("protocol_score", "Protocol surface fit"),
        ("tokenizer_score", "Tokenizer and accounting fit"),
        ("behavior_score", "Behavioral fit"),
        ("capability_tier_score", "Capability-tier fit"),
        ("stability_score", "Repeatability and stability"),
        ("routing_shift_score", "Routing-shift risk"),
        ("wrapper_suspicion_score", "Wrapper suspicion"),
    ]
    return [{"key": key, "label": label, "score": round(features[key], 4)} for key, label in items]


def _primary_reason(label: str, features: dict[str, float], category_probabilities: dict[str, float]) -> str:
    if label == "likely consistent with claimed model":
        return "Protocol, behavior, and stability signals jointly support the claimed model."
    if label == "likely same-family downgrade":
        return "The family signature is still present, but the endpoint behaves more like a weaker sibling."
    if label == "likely alternative family":
        return "Behavior and protocol signals fit another family better than the claimed one."
    if label == "likely wrapped or policy-overlaid":
        return "A wrapper or policy layer appears to be distorting the underlying model signals."
    if label == "suspected routing shift or mixed backend":
        return "Repeated probes show drift, suggesting mixed routing, fallback, or time-varying backend selection."
    if label == "insufficient evidence":
        return "The current probe set did not produce enough reliable evidence for attribution."
    if category_probabilities["claimed_model_probability"] >= category_probabilities["alternative_family_probability"]:
        return "Some evidence leans toward the claim, but not enough for a high-confidence match."
    return "Multiple explanations remain close, so the current result is still ambiguous."


def _secondary_reason(features: dict[str, float]) -> str:
    if features["wrapper_suspicion_score"] >= 0.35:
        return "A wrapper or policy layer may be rewriting part of the output shape."
    if features["routing_shift_score"] >= 0.30:
        return "Repeated probes show drift; re-running across time windows would improve confidence."
    if features["tokenizer_score"] < 0.60:
        return "Tokenizer and token-accounting evidence is still relatively weak."
    return "This conclusion comes from a low-cost probe set and should be treated as screening evidence."


def _primary_caveats(features: dict[str, float]) -> list[str]:
    caveats = []
    if features["wrapper_suspicion_score"] >= 0.35:
        caveats.append("A wrapper or policy layer may be distorting the behavioral signals.")
    if features["routing_shift_score"] >= 0.35:
        caveats.append("Repeated probes show drift, which may indicate mixed backends, fallback, or time-window variance.")
    if features["protocol_mismatch_score"] >= 0.45:
        caveats.append("Protocol-level behavior differs from common first-party implementations and should be interpreted carefully.")
    if features["usage_available_score"] < 0.5:
        caveats.append("Usage and token-accounting evidence is limited, so tokenizer evidence remains weak.")
    if features["claimed_model_consistency_score"] < 0.35:
        caveats.append("Overall evidence is weak; a deeper run or repeated measurements would help.")
    if not caveats:
        caveats.append("This result is based on low-cost local probes and should be treated as probabilistic evidence.")
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
