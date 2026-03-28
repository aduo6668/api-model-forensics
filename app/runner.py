from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from . import api_client
from .config import BUDGET_PROFILES, PRIVACY_MODE, PRIVACY_UI_TEXT, SCHEMA_VERSION, TOOL_VERSION
from .probes import ProbeExecutionContext, execute_probe, load_probes
from .report import generate_pdf_report
from .scoring import score_run
from .storage import create_run_paths, write_json


ProgressCallback = Callable[[str], None]


def run_analysis(
    base_url: str,
    api_key: str,
    claimed_model: str,
    provider_hint: str = "",
    mode: str = "standard",
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = datetime.now()
    target_info = api_client.describe_target(base_url, provider_hint=provider_hint, claimed_model=claimed_model)
    normalized_base_url = target_info["normalized_base_url"]
    ctx = ProbeExecutionContext(
        base_url=normalized_base_url,
        api_key=api_key,
        claimed_model=claimed_model,
        provider_hint=provider_hint,
        mode=mode,
    )

    run_paths = create_run_paths(claimed_model)
    probes = load_probes()
    results: list[dict[str, Any]] = []

    for probe in probes:
        repeats = _repeat_count(probe["id"], mode, probe.get("default_repeats", 1))
        for repeat_index in range(1, repeats + 1):
            if progress_cb:
                progress_cb(f"Running {probe['id']} ({probe['name']}) [{repeat_index}/{repeats}] ...")
            results.append(execute_probe(probe, ctx, repeat_index=repeat_index))

    summary = score_run(results, claimed_model=claimed_model, provider_hint=provider_hint)
    finished_at = datetime.now()
    run_meta = {
        "run_id": run_paths["run_id"],
        "generated_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "target_base_url": normalized_base_url,
        "resolved_chat_endpoint": target_info["resolved_chat_endpoint"],
        "resolved_models_endpoint": target_info["resolved_models_endpoint"],
        "claimed_model": claimed_model,
        "provider_hint": provider_hint,
        "adapter_name": target_info["adapter_name"],
        "dialect": target_info["dialect"],
        "mode": mode,
        "budget_profile": BUDGET_PROFILES[mode],
        "privacy_note": PRIVACY_UI_TEXT,
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "privacy_mode": PRIVACY_MODE,
        "status": "success",
    }

    normalized_outputs = [
        {
            "probe_id": item["probe_id"],
            "repeat_index": item["repeat_index"],
            "normalized_output": item["normalized_output"],
            "score": item["score"],
            "signal_tags": item.get("signal_tags", []),
        }
        for item in results
    ]
    request_log_path = run_paths["run_dir"] / "request_log.json"
    summary_path = run_paths["run_dir"] / "summary.json"
    write_json(request_log_path, {"run_meta": run_meta, "results": results})
    write_json(summary_path, {"run_meta": run_meta, "summary": summary})
    write_json(run_paths["normalized_outputs_path"], normalized_outputs)

    pdf_path = generate_pdf_report(run_paths["report_path"], summary, results, run_meta)
    report_payload = _build_report_payload(
        run_meta=run_meta,
        base_url_input=base_url,
        claimed_model=claimed_model,
        provider_hint=provider_hint,
        mode=mode,
        results=results,
        summary=summary,
        summary_json_path=str(summary_path),
        run_log_path=str(request_log_path),
        normalized_output_path=str(run_paths["normalized_outputs_path"]),
        report_pdf_path=str(pdf_path),
    )
    write_json(run_paths["report_json_path"], report_payload)

    return {
        "run_meta": run_meta,
        "summary": summary,
        "results": results,
        "run_dir": str(run_paths["run_dir"]),
        "summary_json": str(summary_path),
        "report_json": str(run_paths["report_json_path"]),
        "report_pdf": str(pdf_path),
        "normalized_outputs_json": str(run_paths["normalized_outputs_path"]),
    }


def _repeat_count(probe_id: str, mode: str, default_repeats: int) -> int:
    if mode == "fast":
        if probe_id in {"P10", "P11"}:
            return 2
        return 1
    if mode == "deep":
        if probe_id in {"P10", "P11"}:
            return max(default_repeats, 4)
        if probe_id in {"P03", "P08"}:
            return 2
        return default_repeats
    return default_repeats


def _build_report_payload(
    run_meta: dict[str, Any],
    base_url_input: str,
    claimed_model: str,
    provider_hint: str,
    mode: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    summary_json_path: str,
    run_log_path: str,
    normalized_output_path: str,
    report_pdf_path: str,
) -> dict[str, Any]:
    prompt_tokens = sum(item.get("usage_prompt_tokens") or 0 for item in results)
    completion_tokens = sum(item.get("usage_completion_tokens") or 0 for item in results)
    total_tokens = sum(item.get("usage_total_tokens") or 0 for item in results)
    unique_probes = {item["probe_id"] for item in results}
    group_counts: dict[str, int] = {}
    for item in results:
        group_counts[item["probe_group"]] = group_counts.get(item["probe_group"], 0) + 1

    caveat_objects = [
        {
            "category": _classify_caveat(message),
            "message": message,
            "impact": _caveat_impact(message),
        }
        for message in summary["primary_caveats"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "run_metadata": {
            "run_id": run_meta["run_id"],
            "started_at": run_meta["started_at"],
            "finished_at": run_meta["finished_at"],
            "schema_version": SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "mode": mode,
            "privacy_mode": PRIVACY_MODE,
            "target_base_url": run_meta["target_base_url"],
            "resolved_chat_endpoint": run_meta["resolved_chat_endpoint"],
            "resolved_models_endpoint": run_meta.get("resolved_models_endpoint"),
            "adapter_name": run_meta.get("adapter_name"),
            "dialect": run_meta.get("dialect"),
            "claimed_model": claimed_model,
            "claimed_provider_guess": summary["claimed_family_guess"],
            "claimed_snapshot_guess": None,
            "token_budget_target": BUDGET_PROFILES[mode]["max_total_tokens"],
            "token_budget_estimated": BUDGET_PROFILES[mode]["estimated_total_tokens"],
            "token_budget_used": total_tokens,
            "status": "success",
        },
        "input_summary": {
            "api_address": base_url_input,
            "api_key_source": "user-input",
            "claimed_model_input": claimed_model,
            "claimed_provider_input": provider_hint,
            "claimed_family_input": summary["claimed_family_guess"],
            "test_budget_input": mode,
        },
        "probe_summary": {
            "probe_set_version": "v0",
            "probe_groups": [
                {
                    "group_id": key,
                    "name": key,
                    "probe_count": value,
                    "repeat_count": max(0, value - 1),
                    "purpose": _group_purpose(key),
                }
                for key, value in sorted(group_counts.items())
            ],
            "probe_count_total": len(results),
            "probe_count_unique": len(unique_probes),
            "repeat_count_total": max(0, len(results) - len(unique_probes)),
            "retries_total": 0,
            "estimated_input_tokens": BUDGET_PROFILES[mode]["estimated_total_tokens"],
            "estimated_output_tokens": None,
            "actual_input_tokens": prompt_tokens,
            "actual_output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "evidence_layers": _build_evidence_layers(summary, results),
        "decision_hypotheses": {
            "ranking": summary.get("hypothesis_ranking", []),
            **summary["candidate_probabilities"],
        },
        "model_candidates": {
            "ranking": summary.get("model_candidate_ranking", []),
            "observed_catalog_hints": summary.get("observed_model_hints", []),
            "weak_model_hints": summary.get("weak_model_hints", []),
        },
        "candidate_probabilities": {
            **summary["candidate_probabilities"],
            "top_candidates": summary["top_candidates"],
        },
        "decision": {
            "label": summary["verdict_label"],
            "confidence_level": summary["confidence_level"],
            "primary_reason": summary["primary_reason"],
            "secondary_reason": summary["secondary_reason"],
            "evidence_strength": _evidence_strength(summary["confidence_level"]),
        },
        "caveats": caveat_objects,
        "privacy_note": {
            "storage_model": PRIVACY_MODE,
            "cloud_transfer": "none by default",
            "api_key_handling": "user-controlled",
            "user_control": "all outputs stay on the user's machine",
            "disclaimer": "probabilistic evidence only, not absolute proof",
        },
        "artifacts": {
            "report_pdf_path": report_pdf_path,
            "json_result_path": summary_json_path,
            "run_log_path": run_log_path,
            "normalized_output_path": normalized_output_path,
        },
        "appendix": {
            "probe_details": results,
        },
    }


def _build_evidence_layers(summary: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    feature_summary = summary["feature_summary"]
    layer_map = {
        "protocol": ("protocol_score", ["protocol"]),
        "tokenizer_accounting": ("tokenizer_score", ["tokenizer"]),
        "behavior": ("behavior_score", ["behavior"]),
        "stability_routing": ("stability_score", ["stability", "routing"]),
        "wrapper_policy": ("wrapper_suspicion_score", ["wrapper"]),
    }
    payload: dict[str, Any] = {}
    for layer_name, (feature_key, tags) in layer_map.items():
        supporting = [
            f"{item['probe_id']}#{item['repeat_index']}"
            for item in results
            if any(tag in item.get("signal_tags", []) for tag in tags) and item.get("score", 0.0) >= 0.7
        ]
        weak = [
            f"{item['probe_id']}#{item['repeat_index']}"
            for item in results
            if any(tag in item.get("signal_tags", []) for tag in tags) and item.get("score", 0.0) < 0.5
        ]
        payload[layer_name] = {
            "layer_score": round(feature_summary.get(feature_key, 0.0), 4),
            "supporting_probes": supporting[:8],
            "weak_probes": weak[:8],
            "main_signals": [_layer_signal(layer_name, feature_summary)],
            "confounders": _layer_confounders(layer_name, feature_summary),
            "interpretation": _layer_interpretation(layer_name, feature_summary),
            "confidence": _layer_confidence(feature_summary.get(feature_key, 0.0)),
        }
    return payload


def _group_purpose(group_name: str) -> str:
    return {
        "protocol": "surface compatibility and metadata shape",
        "exactness": "strict format and exact echo behavior",
        "behavior": "multilingual, refusal, and code-edit behavior",
        "stability": "repeatability and routing drift hints",
    }.get(group_name, "general probes")


def _layer_signal(layer_name: str, features: dict[str, float]) -> str:
    if layer_name == "protocol":
        return f"Protocol surface fit {features['protocol_score']:.3f}"
    if layer_name == "tokenizer_accounting":
        return f"Tokenizer/accounting fit {features['tokenizer_score']:.3f}"
    if layer_name == "behavior":
        return f"Behavioral fit {features['behavior_score']:.3f}"
    if layer_name == "stability_routing":
        return f"Stability {features['stability_score']:.3f}; routing risk {features['routing_shift_score']:.3f}"
    return f"Wrapper suspicion {features['wrapper_suspicion_score']:.3f}"


def _layer_confounders(layer_name: str, features: dict[str, float]) -> list[str]:
    confounders: list[str] = []
    if layer_name == "tokenizer_accounting" and features["usage_available_score"] < 0.5:
        confounders.append("usage fields are missing or incomplete")
    if layer_name == "stability_routing" and features["routing_shift_score"] >= 0.35:
        confounders.append("repeated probes show noticeable drift")
    if layer_name == "wrapper_policy" and features["wrapper_suspicion_score"] >= 0.35:
        confounders.append("a wrapper or policy layer may be rewriting outputs")
    return confounders


def _layer_interpretation(layer_name: str, features: dict[str, float]) -> str:
    if layer_name == "protocol":
        return "Higher scores mean the endpoint surface looks more like a stable first-party implementation."
    if layer_name == "tokenizer_accounting":
        return "Higher scores mean echo behavior and usage accounting are more internally consistent."
    if layer_name == "behavior":
        return "Higher scores mean structured output, rewriting, and multilingual behavior are more consistent."
    if layer_name == "stability_routing":
        return "Higher scores mean repeated probes stay stable; lower scores suggest routing drift or mixed backends."
    return "Higher scores do not mean better quality; they mean wrapper suspicion is stronger."


def _layer_confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _evidence_strength(confidence_level: str) -> str:
    return {
        "high": "strong",
        "medium": "moderate",
        "low": "weak",
    }.get(confidence_level, "weak")


def _classify_caveat(message: str) -> str:
    lowered = message.lower()
    if "wrapper" in lowered or "policy" in lowered:
        return "wrapper"
    if "drift" in lowered or "mixed backend" in lowered or "fallback" in lowered:
        return "routing"
    if "protocol" in lowered:
        return "endpoint_compatibility"
    if "usage" in lowered or "token accounting" in lowered or "token-accounting" in lowered:
        return "token_accounting"
    return "low_signal"


def _caveat_impact(message: str) -> str:
    lowered = message.lower()
    if "drift" in lowered or "mixed backend" in lowered:
        return "reduces confidence in a single-model judgment"
    if "wrapper" in lowered or "policy" in lowered:
        return "may distort underlying family signals"
    if "protocol" in lowered:
        return "surface compatibility may be misleading"
    return "suggests re-running with deeper or repeated probes"
