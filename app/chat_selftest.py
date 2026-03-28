from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import MODELS_DIR
from .probes import parse_probe_result
from .provider_registry import family_for_name
from .scoring import score_run


CHAT_SELF_TEST_PACK_PATHS = {
    "conversation-selftest-v1": MODELS_DIR / "conversation_selftest_pack_v1.json",
    "conversation-selftest-v2": MODELS_DIR / "conversation_selftest_pack_v2.json",
}
LATEST_CHAT_SELF_TEST_PACK_ID = "conversation-selftest-v2"


def load_chat_self_test_pack(pack_id: str = "") -> dict[str, Any]:
    resolved_pack_id = pack_id.strip() or LATEST_CHAT_SELF_TEST_PACK_ID
    pack_path = CHAT_SELF_TEST_PACK_PATHS.get(resolved_pack_id, CHAT_SELF_TEST_PACK_PATHS[LATEST_CHAT_SELF_TEST_PACK_ID])
    return json.loads(pack_path.read_text(encoding="utf-8"))


def build_chat_self_test_pack(claimed_model: str, provider_hint: str = "") -> dict[str, Any]:
    pack = load_chat_self_test_pack()
    transcript_cases: list[dict[str, Any]] = []
    for probe in pack["probes"]:
        repeat_count = max(1, int(probe.get("repeat_count", 1)))
        for repeat_index in range(1, repeat_count + 1):
            transcript_cases.append(
                {
                    "probe_id": probe["id"],
                    "title": probe["name"],
                    "parser": probe["parser"],
                    "repeat_index": repeat_index,
                    "prompt_text": probe["prompt_text"],
                    "response_text": "",
                    "metadata": {},
                }
            )

    payload = {
        "pack_id": pack["protocol_version"],
        "claimed_model": claimed_model,
        "provider_hint": provider_hint,
        "source_kind": "conversation_host",
        "notes": pack.get("notes", []),
        "usage_notes": pack.get("instructions", []),
        "probes": pack["probes"],
        "transcript_schema": {
            "protocol_version": pack["protocol_version"],
            "pack_id": pack["protocol_version"],
            "claimed_model": claimed_model,
            "claimed_provider_hint": provider_hint,
            "source_kind": "conversation_host",
            "host_context": {
                "host_name": "",
                "surface_name": "",
                "claimed_runtime_label": "",
            },
            "collected_at": "",
            "cases": transcript_cases,
        },
    }
    return {
        **payload,
        "text": render_chat_self_test_pack_text(payload),
    }


def render_chat_self_test_pack_text(payload: dict[str, Any]) -> str:
    lines = [
        "API Model Forensics Direct Chat Self-Test Pack",
        f"Pack: {payload['pack_id']}",
        f"Claimed model: {payload.get('claimed_model') or 'unknown'}",
        f"Provider hint: {payload.get('provider_hint') or 'unknown'}",
        "",
        "Usage notes:",
    ]
    for note in payload.get("usage_notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "Prompts:"])
    for probe in payload.get("probes", []):
        repeat_count = max(1, int(probe.get("repeat_count", 1)))
        repeat_note = f" x{repeat_count}" if repeat_count > 1 else ""
        lines.append(f"- {probe['id']} {probe['name']}{repeat_note}")
        lines.append(f"  parser: {probe['parser']}")
        lines.append(f"  prompt: {probe['prompt_text']}")
    lines.extend(
        [
            "",
            "Transcript schema:",
            json.dumps(payload["transcript_schema"], ensure_ascii=False, indent=2),
            "",
            "Scoring command:",
            "python -m app.cli --score-chat-self-test path\\to\\transcript.json --model <claimed-model> --format text",
        ]
    )
    return "\n".join(lines) + "\n"


def load_chat_self_test_transcript(path: str | Path) -> dict[str, Any]:
    transcript_path = Path(path)
    return json.loads(transcript_path.read_text(encoding="utf-8-sig"))


def score_chat_self_test_transcript(
    transcript: dict[str, Any],
    claimed_model: str = "",
    provider_hint: str = "",
) -> dict[str, Any]:
    transcript_pack_id = str(transcript.get("pack_id") or transcript.get("protocol_version") or "").strip()
    pack = load_chat_self_test_pack(transcript_pack_id)
    probe_specs = {probe["id"]: probe for probe in pack["probes"]}
    transcript_case_lookup = _transcript_case_lookup(
        transcript.get("cases") or _cases_from_responses(transcript.get("responses", []))
    )
    expected_cases = _expected_cases(pack["probes"])
    self_report = _empty_self_report()
    missing_cases: list[str] = []
    results: list[dict[str, Any]] = []

    for expected_case in expected_cases:
        probe_id = expected_case["probe_id"]
        probe_spec = probe_specs[probe_id]
        case = transcript_case_lookup.get((probe_id, expected_case["repeat_index"]), expected_case)
        response_text = case.get("response_text", case.get("response", ""))
        if probe_id == "C01":
            if not str(response_text).strip():
                missing_cases.append(f"{probe_id}#{case.get('repeat_index', 1)}")
            self_report = _parse_self_report_case(response_text)
            continue
        if not str(response_text).strip():
            missing_cases.append(f"{probe_id}#{case.get('repeat_index', 1)}")
            continue
        parser_payload = _parse_transcript_case(probe_spec, response_text)
        results.append(_result_from_case(probe_spec, case, response_text, parser_payload))

    effective_claimed_model = (
        claimed_model
        or transcript.get("claimed_model")
        or _self_report_field(self_report, "model_guess")
        or ""
    )
    effective_provider_hint = (
        provider_hint
        or transcript.get("claimed_provider_hint")
        or _self_report_field(self_report, "family_guess")
        or _self_report_field(self_report, "provider_guess")
        or ""
    )
    summary = score_run(
        results,
        claimed_model=effective_claimed_model,
        provider_hint=effective_provider_hint,
        source_profile="conversation_host",
        external_model_hints=self_report["hints"],
    )
    summary = _decorate_summary(summary, self_report=self_report, missing_cases=missing_cases)

    payload = {
        "suite": pack["protocol_version"],
        "claimed_model": effective_claimed_model,
        "provider_hint": effective_provider_hint,
        "runtime": {
            "adapter_name": "ConversationHostAdapter",
            "dialect": "conversation_transcript",
            "source_profile": "conversation_host",
        },
        "verdict_label": summary["conversation_verdict_label"],
        "base_verdict_label": summary["verdict_label"],
        "confidence_level": summary["confidence_level"],
        "candidate_probabilities": summary["candidate_probabilities"],
        "hypothesis_ranking": summary.get("hypothesis_ranking", []),
        "model_candidate_ranking": summary.get("model_candidate_ranking", []),
        "top_candidates": summary["top_candidates"],
        "observed_model_hints": summary.get("observed_model_hints", []),
        "weak_model_hints": summary.get("weak_model_hints", []),
        "self_report_hints": self_report["hints"],
        "feature_summary": summary["feature_summary"],
        "primary_reason": summary["primary_reason"],
        "secondary_reason": summary["secondary_reason"],
        "caveats": summary["primary_caveats"],
        "completion": {
            "expected_responses": len(expected_cases),
            "received_responses": len(expected_cases) - len(missing_cases),
            "completion_rate": round(
                (len(expected_cases) - len(missing_cases)) / max(1, len(expected_cases)),
                4,
            ),
        },
        "transcript": {
            "cases_total": len(expected_cases),
            "cases_scored": len(results),
            "missing_cases": missing_cases,
        },
        "responses": results,
    }
    return {
        **payload,
        "text": render_chat_self_test_score_text(payload),
    }


def render_chat_self_test_score_text(payload: dict[str, Any]) -> str:
    completion = payload["completion"]
    lines = [
        "API Model Forensics Direct Chat Self-Test",
        f"Suite: {payload['suite']}",
        f"Claimed model: {payload.get('claimed_model') or 'unknown'}",
        f"Verdict: {payload['verdict_label']}",
        f"Base verdict: {payload['base_verdict_label']}",
        f"Confidence: {payload['confidence_level']}",
        "",
        "Completion:",
        f"- Received: {completion['received_responses']}/{completion['expected_responses']}",
        f"- Completion rate: {completion['completion_rate']:.1%}",
        "",
        "Decision hypotheses:",
    ]
    for item in payload.get("hypothesis_ranking", []):
        lines.append(f"- {item['label']}: {item['probability']:.1%}")
    lines.extend(["", "Model candidates:"])
    for item in payload.get("model_candidate_ranking", [])[:5]:
        lines.append(f"- {item['name']}: {item['probability']:.1%} ({item['kind']})")
    if payload.get("self_report_hints"):
        lines.extend(["", "Self-report hints (weak metadata):"])
        for hint in payload["self_report_hints"]:
            lines.append(f"- {hint}")
    if payload.get("observed_model_hints"):
        lines.extend(["", "Observed catalog hints:"])
        for hint in payload["observed_model_hints"][:5]:
            lines.append(f"- {hint}")
    if payload.get("weak_model_hints"):
        lines.extend(["", "Weak model hints used for candidate narrowing:"])
        for hint in payload["weak_model_hints"][:5]:
            lines.append(f"- {hint}")
    if payload["transcript"]["missing_cases"]:
        lines.extend(["", "Missing cases:"])
        for item in payload["transcript"]["missing_cases"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Primary reason: {payload['primary_reason']}",
            f"Secondary reason: {payload['secondary_reason']}",
            "",
            "Caveats:",
        ]
    )
    for item in payload.get("caveats", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _parse_transcript_case(probe_spec: dict[str, Any], response_text: str) -> dict[str, Any]:
    fake_result = {
        "message_content": response_text,
        "status_code": 200,
        "latency_total_ms": None,
        "ttft_ms": None,
        "stream_chunk_count": 0,
        "usage": {},
        "finish_reason": "stop",
        "tool_calls": [],
        "logprobs": None,
        "raw_text": response_text,
        "adapter_name": "ConversationHostAdapter",
        "dialect": "conversation_transcript",
        "resolved_endpoint": "conversation://host",
    }
    return parse_probe_result(probe_spec, fake_result)


def _result_from_case(
    probe_spec: dict[str, Any],
    case: dict[str, Any],
    response_text: str,
    parser_payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = case.get("metadata", {}) or {}
    return {
        "probe_id": probe_spec["id"],
        "probe_name": probe_spec["name"],
        "probe_group": probe_spec["group"],
        "repeat_index": int(case.get("repeat_index", 1)),
        "prompt_id": f"{probe_spec['id'].lower()}_chat_selftest_v1",
        "request_payload": {"prompt_text": probe_spec["prompt_text"]},
        "status_code": 200,
        "error_shape": None,
        "latency_total_ms": metadata.get("latency_ms"),
        "ttft_ms": None,
        "stream_chunk_count": 0,
        "inter_token_times_ms": [],
        "usage_prompt_tokens": None,
        "usage_completion_tokens": None,
        "usage_total_tokens": None,
        "finish_reason": "stop",
        "raw_output": response_text,
        "normalized_output": parser_payload.get("normalized_output", response_text),
        "parse_success": parser_payload.get("parse_success", False),
        "score": parser_payload.get("score", 0.0),
        "structured_score": parser_payload.get("structured_score", parser_payload.get("score", 0.0)),
        "exact_match_score": parser_payload.get("exact_match_score"),
        "details": parser_payload.get("details", {}),
        "refusal_class": parser_payload.get("refusal_class", "none"),
        "signal_tags": parser_payload.get("signal_tags", [probe_spec["group"]]),
        "notes": parser_payload.get("notes", ""),
        "system_fingerprint": None,
        "tool_calls": [],
        "logprobs": None,
        "raw_response_text": response_text,
        "adapter_name": "ConversationHostAdapter",
        "dialect": "conversation_transcript",
        "resolved_endpoint": "conversation://host",
    }


def _parse_self_report_case(response_text: str) -> dict[str, Any]:
    raw = (response_text or "").strip()
    if not raw:
        return _empty_self_report()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "payload": {},
            "hints": [],
        }
    hints: list[str] = []
    model_guess = parsed.get("model_guess")
    if isinstance(model_guess, str) and model_guess.strip():
        hints.append(model_guess.strip())

    for key in ("family_guess", "provider_guess"):
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = family_for_name(value.strip())
        hints.append(normalized if normalized != "unknown" else value.strip())
    return {
        "valid": True,
        "payload": parsed,
        "hints": _unique(hints),
    }


def _decorate_summary(
    summary: dict[str, Any],
    self_report: dict[str, Any],
    missing_cases: list[str],
) -> dict[str, Any]:
    summary["observed_model_hints"] = _unique(summary.get("observed_model_hints", []))[:10]
    summary["weak_model_hints"] = _unique([*self_report["hints"], *summary.get("weak_model_hints", [])])[:10]
    summary["confidence_level"] = _cap_confidence(summary.get("confidence_level", "low"))
    summary["conversation_verdict_label"] = _conversation_verdict(summary)
    summary["primary_reason"] = (
        "Behavior-level probes from the current chat host were scored locally and compared against the claimed model family."
    )
    summary["secondary_reason"] = (
        "This self-test path is useful for screening and employer-side verification, but it is weaker than the full API audit path."
    )
    caveats = [
        "Direct-chat self-test cannot inspect /models, usage accounting, transport fingerprints, or native tool-call wire format.",
        "Self-reported identity is weak evidence only and may be inaccurate.",
        *summary.get("primary_caveats", []),
    ]
    if missing_cases:
        caveats.append("Some transcript cases were missing, so the self-test used a partial evidence set.")
    summary["primary_caveats"] = _unique(caveats)
    return summary


def _conversation_verdict(summary: dict[str, Any]) -> str:
    probs = summary["candidate_probabilities"]
    features = summary["feature_summary"]
    if probs["wrapped_or_unknown_probability"] >= 0.38 or features["routing_shift_score"] >= 0.42:
        return "direct-chat self-test suggests possible routing overlay or mismatch"
    if (
        probs["same_family_downgrade_probability"] >= 0.30
        and probs["claimed_model_probability"] <= probs["same_family_downgrade_probability"] + 0.12
    ):
        return "direct-chat self-test suggests same-family variant or downgrade"
    if probs["claimed_model_probability"] >= 0.45 and features["claimed_model_consistency_score"] >= 0.52:
        return "direct-chat self-test leans consistent with claimed model"
    if probs["same_family_downgrade_probability"] >= 0.28:
        return "direct-chat self-test suggests same-family variant or downgrade"
    if probs["alternative_family_probability"] >= 0.38 and probs["claimed_model_probability"] < 0.35:
        return "direct-chat self-test suggests an alternative family"
    return "direct-chat self-test is ambiguous"


def _cap_confidence(level: str) -> str:
    if level == "high":
        return "medium"
    return level or "low"


def _empty_self_report() -> dict[str, Any]:
    return {
        "valid": False,
        "payload": {},
        "hints": [],
    }


def _self_report_field(self_report: dict[str, Any], key: str) -> str:
    payload = self_report.get("payload", {})
    value = payload.get(key) if isinstance(payload, dict) else None
    return value.strip() if isinstance(value, str) else ""


def _expected_cases(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for probe in probes:
        repeat_count = max(1, int(probe.get("repeat_count", 1)))
        for repeat_index in range(1, repeat_count + 1):
            expected.append(
                {
                    "probe_id": probe["id"],
                    "title": probe["name"],
                    "parser": probe["parser"],
                    "repeat_index": repeat_index,
                    "prompt_text": probe["prompt_text"],
                    "response_text": "",
                    "metadata": {},
                }
            )
    return expected


def _transcript_case_lookup(cases: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        probe_id = str(case.get("probe_id", "")).strip()
        if not probe_id:
            continue
        try:
            repeat_index = int(case.get("repeat_index", 1))
        except (TypeError, ValueError):
            repeat_index = 1
        lookup[(probe_id, repeat_index)] = case
    return lookup


def _unique(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _cases_from_responses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        probe_id = str(item.get("probe_id", "")).strip()
        if not probe_id:
            continue
        cases.append(
            {
                "probe_id": probe_id,
                "title": item.get("title", probe_id),
                "parser": item.get("parser", ""),
                "repeat_index": item.get("repeat_index", 1),
                "prompt_text": item.get("prompt_text", ""),
                "response_text": item.get("response_text", item.get("response", "")),
                "metadata": item.get("metadata", {}),
            }
        )
    return cases
