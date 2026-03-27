from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runner import run_analysis


@dataclass(frozen=True)
class SelfTestCase:
    case_id: str
    title: str
    base_url: str
    claimed_model: str
    provider_hint: str = ""
    expected_adapter: str = ""
    expected_dialect: str = ""
    expected_top_candidate: str = ""
    expected_verdicts: tuple[str, ...] = field(default_factory=tuple)
    min_claimed_probability: float = 0.0
    min_same_family_probability: float = 0.0
    min_alt_or_wrapped_probability: float = 0.0
    required_observed_hints: tuple[str, ...] = field(default_factory=tuple)


SELF_TEST_CASES: list[SelfTestCase] = [
    SelfTestCase(
        case_id="openai-primary",
        title="OpenAI-like primary candidate stays on top",
        base_url="mock://openai",
        claimed_model="gpt-4o",
        expected_adapter="OpenAICompatibleAdapter",
        expected_dialect="openai_chat_completions",
        expected_top_candidate="gpt-4o",
        expected_verdicts=("ambiguous", "likely consistent with claimed model"),
        min_claimed_probability=0.40,
        required_observed_hints=("gpt-4o",),
    ),
    SelfTestCase(
        case_id="openai-downgrade",
        title="OpenAI-like downgrade pressure is detectable",
        base_url="mock://openai",
        claimed_model="gpt-5.4",
        expected_adapter="OpenAICompatibleAdapter",
        expected_dialect="openai_chat_completions",
        expected_verdicts=("likely same-family downgrade", "ambiguous"),
        min_same_family_probability=0.30,
        required_observed_hints=("gpt-4o", "gpt-4o-mini"),
    ),
    SelfTestCase(
        case_id="anthropic-primary",
        title="Anthropic adapter and family ranking stay stable",
        base_url="mock://anthropic",
        claimed_model="claude-3-5-sonnet",
        expected_adapter="AnthropicMessagesAdapter",
        expected_dialect="anthropic_messages",
        expected_top_candidate="claude-3-5-sonnet",
        expected_verdicts=("ambiguous", "likely consistent with claimed model"),
        min_claimed_probability=0.25,
        required_observed_hints=("claude-3-5-sonnet",),
    ),
    SelfTestCase(
        case_id="gemini-primary",
        title="Gemini adapter keeps claimed candidate on top",
        base_url="mock://gemini",
        claimed_model="gemini-2.5-pro",
        expected_adapter="GeminiGenerateContentAdapter",
        expected_dialect="gemini_generate_content",
        expected_top_candidate="gemini-2.5-pro",
        expected_verdicts=("ambiguous", "likely consistent with claimed model"),
        min_claimed_probability=0.30,
        required_observed_hints=("gemini-2.5-pro",),
    ),
    SelfTestCase(
        case_id="mixed-signal",
        title="Mixed catalog produces cross-family ambiguity",
        base_url="mock://mixed",
        claimed_model="gpt-5.4",
        expected_adapter="OpenAICompatibleAdapter",
        expected_dialect="openai_chat_completions",
        expected_verdicts=("ambiguous", "suspected routing shift or mixed backend"),
        min_alt_or_wrapped_probability=0.40,
        required_observed_hints=("claude-3-5-sonnet", "gemini-2.5-flash"),
    ),
]


def run_self_test_suite(mode: str = "fast") -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for case in SELF_TEST_CASES:
        result = run_analysis(
            base_url=case.base_url,
            api_key="self-test-key",
            claimed_model=case.claimed_model,
            provider_hint=case.provider_hint,
            mode=mode,
        )
        evaluation = _evaluate_case(case, result)
        case_payload = {
            "case_id": case.case_id,
            "title": case.title,
            "claimed_model": case.claimed_model,
            "base_url": case.base_url,
            "passed": evaluation["passed"],
            "checks": evaluation["checks"],
            "runtime": {
                "adapter_name": result["run_meta"].get("adapter_name"),
                "dialect": result["run_meta"].get("dialect"),
            },
            "decision": {
                "verdict_label": result["summary"]["verdict_label"],
                "confidence_level": result["summary"]["confidence_level"],
                "candidate_probabilities": result["summary"]["candidate_probabilities"],
                "top_candidates": result["summary"]["top_candidates"][:4],
                "observed_model_hints": result["summary"].get("observed_model_hints", [])[:8],
            },
        }
        cases.append(case_payload)
        if evaluation["passed"]:
            passed += 1
        else:
            failed += 1

    return {
        "suite": "detector-self-test-v1",
        "mode": mode,
        "passed": passed,
        "failed": failed,
        "total": len(cases),
        "ok": failed == 0,
        "cases": cases,
    }


def render_self_test_text(payload: dict[str, Any]) -> str:
    lines = [
        "API Model Forensics Self-Test",
        f"Suite: {payload['suite']}",
        f"Mode: {payload['mode']}",
        f"Passed: {payload['passed']}/{payload['total']}",
        f"Failed: {payload['failed']}",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"[{'PASS' if case['passed'] else 'FAIL'}] {case['case_id']} - {case['title']}")
        lines.append(f"  claimed_model: {case['claimed_model']}")
        lines.append(f"  verdict: {case['decision']['verdict_label']} ({case['decision']['confidence_level']})")
        lines.append(
            f"  runtime: {case['runtime']['adapter_name']} / {case['runtime']['dialect']}"
        )
        top_candidates = case["decision"]["top_candidates"]
        if top_candidates:
            lines.append(f"  top_candidate: {top_candidates[0]['name']} ({top_candidates[0]['probability']:.1%})")
        failed_checks = [check for check in case["checks"] if not check["passed"]]
        if failed_checks:
            for check in failed_checks:
                lines.append(f"  - FAIL: {check['name']} -> {check['message']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _evaluate_case(case: SelfTestCase, result: dict[str, Any]) -> dict[str, Any]:
    run_meta = result["run_meta"]
    summary = result["summary"]
    candidate_probabilities = summary["candidate_probabilities"]
    top_candidates = summary["top_candidates"]
    observed_hints = summary.get("observed_model_hints", [])

    checks = [
        _check_equals("adapter_name", run_meta.get("adapter_name"), case.expected_adapter),
        _check_equals("dialect", run_meta.get("dialect"), case.expected_dialect),
        _check_top_candidate(top_candidates, case.expected_top_candidate),
        _check_verdict(summary["verdict_label"], case.expected_verdicts),
        _check_minimum(
            "claimed_model_probability",
            candidate_probabilities["claimed_model_probability"],
            case.min_claimed_probability,
        ),
        _check_minimum(
            "same_family_downgrade_probability",
            candidate_probabilities["same_family_downgrade_probability"],
            case.min_same_family_probability,
        ),
        _check_minimum(
            "alt_plus_wrapped_probability",
            candidate_probabilities["alternative_family_probability"] + candidate_probabilities["wrapped_or_unknown_probability"],
            case.min_alt_or_wrapped_probability,
        ),
        _check_hints(observed_hints, case.required_observed_hints),
    ]
    relevant_checks = [check for check in checks if check is not None]
    return {
        "passed": all(check["passed"] for check in relevant_checks),
        "checks": relevant_checks,
    }


def _check_equals(name: str, actual: Any, expected: str) -> dict[str, Any] | None:
    if not expected:
        return None
    passed = actual == expected
    return {
        "name": name,
        "passed": passed,
        "message": f"expected={expected}, actual={actual}",
    }


def _check_top_candidate(top_candidates: list[dict[str, Any]], expected: str) -> dict[str, Any] | None:
    if not expected:
        return None
    actual = top_candidates[0]["name"] if top_candidates else ""
    passed = actual == expected
    return {
        "name": "top_candidate",
        "passed": passed,
        "message": f"expected={expected}, actual={actual}",
    }


def _check_verdict(actual: str, expected_verdicts: tuple[str, ...]) -> dict[str, Any] | None:
    if not expected_verdicts:
        return None
    passed = actual in expected_verdicts
    return {
        "name": "verdict_label",
        "passed": passed,
        "message": f"expected one of {list(expected_verdicts)}, actual={actual}",
    }


def _check_minimum(name: str, actual: float, threshold: float) -> dict[str, Any] | None:
    if threshold <= 0:
        return None
    passed = actual >= threshold
    return {
        "name": name,
        "passed": passed,
        "message": f"expected >= {threshold:.4f}, actual={actual:.4f}",
    }


def _check_hints(observed_hints: list[str], required_hints: tuple[str, ...]) -> dict[str, Any] | None:
    if not required_hints:
        return None
    missing = [hint for hint in required_hints if hint not in observed_hints]
    return {
        "name": "observed_model_hints",
        "passed": not missing,
        "message": "missing hints: " + ", ".join(missing) if missing else "all required hints present",
    }
