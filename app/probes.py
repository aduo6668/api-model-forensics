from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from . import api_client
from .config import MODELS_DIR


@dataclass
class ProbeExecutionContext:
    base_url: str
    api_key: str
    claimed_model: str
    provider_hint: str
    mode: str


def load_probes() -> list[dict[str, Any]]:
    probe_path = MODELS_DIR / "probes.json"
    return json.loads(probe_path.read_text(encoding="utf-8"))


def execute_probe(probe: dict[str, Any], ctx: ProbeExecutionContext, repeat_index: int = 1) -> dict[str, Any]:
    probe_type = probe["type"]
    if probe_type == "models_list":
        result = api_client.list_models(
            ctx.base_url,
            ctx.api_key,
            provider_hint=ctx.provider_hint,
            claimed_model=ctx.claimed_model,
        )
        models = result.get("models", [])
        parser = {
            "score": 1.0 if models else 0.0,
            "structured_score": 1.0 if models else 0.0,
            "exact_match_score": None,
            "details": {"models_count": len(models), "models": models[:10]},
            "signal_tags": ["protocol"],
        }
        return _base_result(probe, repeat_index, parser, result)

    payload = {
        "model": ctx.claimed_model,
        "messages": probe.get("messages", []),
        "temperature": probe.get("temperature", 0.2),
        "max_tokens": probe.get("max_tokens", 96),
    }

    if probe.get("tools"):
        payload["tools"] = probe["tools"]

    for key, value in probe.get("extras", {}).items():
        payload[key] = value

    stream = bool(probe.get("stream", False))
    if stream:
        payload["stream"] = True
    result = api_client.post_chat(
        ctx.base_url,
        ctx.api_key,
        payload,
        probe["id"],
        stream=stream,
        provider_hint=ctx.provider_hint,
        claimed_model=ctx.claimed_model,
    )
    parser = parse_probe_result(probe, result)
    return _base_result(probe, repeat_index, parser, result, payload)


def _base_result(
    probe: dict[str, Any],
    repeat_index: int,
    parser: dict[str, Any],
    result: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = parser.get("score", 0.0)
    return {
        "probe_id": probe["id"],
        "probe_name": probe["name"],
        "probe_group": probe["group"],
        "repeat_index": repeat_index,
        "prompt_id": f"{probe['id'].lower()}_v1",
        "request_payload": payload or {},
        "status_code": result.get("status_code", 0),
        "error_shape": result.get("error"),
        "latency_total_ms": result.get("latency_total_ms"),
        "ttft_ms": result.get("ttft_ms"),
        "stream_chunk_count": result.get("stream_chunk_count", 0),
        "inter_token_times_ms": result.get("inter_token_times_ms", []),
        "usage_prompt_tokens": _usage_value(result, "prompt_tokens"),
        "usage_completion_tokens": _usage_value(result, "completion_tokens"),
        "usage_total_tokens": _usage_value(result, "total_tokens"),
        "finish_reason": result.get("finish_reason"),
        "raw_output": result.get("message_content", ""),
        "normalized_output": parser.get("normalized_output", result.get("message_content", "")),
        "parse_success": parser.get("parse_success", False),
        "score": score,
        "structured_score": parser.get("structured_score", score),
        "exact_match_score": parser.get("exact_match_score"),
        "details": parser.get("details", {}),
        "refusal_class": parser.get("refusal_class", "none"),
        "signal_tags": parser.get("signal_tags", [probe["group"]]),
        "notes": parser.get("notes", ""),
        "system_fingerprint": result.get("system_fingerprint"),
        "tool_calls": result.get("tool_calls", []),
        "logprobs": result.get("logprobs"),
        "raw_response_text": result.get("raw_text", ""),
        "adapter_name": result.get("adapter_name"),
        "dialect": result.get("dialect"),
        "resolved_endpoint": result.get("resolved_endpoint"),
    }


def _usage_value(result: dict[str, Any], key: str) -> int | None:
    usage = result.get("usage") or {}
    value = usage.get(key)
    if isinstance(value, int):
        return value
    return None


def parse_probe_result(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    parser_name = probe.get("parser", "plain_text")
    return {
        "strict_json": _parse_strict_json,
        "json_assertions": _parse_json_assertions,
        "echo_exact": _parse_echo_exact,
        "two_bullets_cn": _parse_two_bullets_cn,
        "transform_json": _parse_transform_json,
        "unified_diff": _parse_unified_diff,
        "safe_rewrite": _parse_safe_rewrite,
        "title_json": _parse_title_json,
        "tool_call": _parse_tool_call,
        "logprobs_capability": _parse_logprobs_capability,
    }.get(parser_name, _parse_plain_text)(probe, result)


def _parse_plain_text(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    score = 1.0 if text else 0.0
    return {
        "parse_success": bool(text),
        "score": score,
        "structured_score": score,
        "exact_match_score": None,
        "normalized_output": text,
        "details": {},
        "signal_tags": [probe.get("group", "behavior")],
    }


def _parse_strict_json(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    expected = probe.get("expected_json")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_success": False,
            "score": 0.0,
            "structured_score": 0.0,
            "exact_match_score": 0.0,
            "normalized_output": text,
            "details": {"reason": "invalid_json"},
            "signal_tags": ["behavior", "protocol"],
        }

    if parsed == expected:
        score = 1.0
    elif isinstance(parsed, dict) and isinstance(expected, dict):
        overlap = sum(1 for key in expected if parsed.get(key) == expected[key])
        score = overlap / max(1, len(expected))
    else:
        score = 0.3
    return {
        "parse_success": True,
        "score": round(score, 3),
        "structured_score": round(score, 3),
        "exact_match_score": 1.0 if parsed == expected else round(score, 3),
        "normalized_output": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        "details": {"parsed": parsed},
        "signal_tags": ["behavior", "protocol"],
    }


def _parse_json_assertions(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    assertions = probe.get("assertions", [])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_success": False,
            "score": 0.0,
            "structured_score": 0.0,
            "exact_match_score": 0.0,
            "normalized_output": text,
            "details": {"reason": "invalid_json"},
            "signal_tags": ["behavior", "capability"],
        }

    checks: list[dict[str, Any]] = []
    passed = 0
    for assertion in assertions:
        path = str(assertion.get("path", "")).strip()
        expected = assertion.get("equals")
        actual = _json_path_value(parsed, path)
        ok = actual == expected
        checks.append(
            {
                "path": path,
                "expected": expected,
                "actual": actual,
                "passed": ok,
            }
        )
        if ok:
            passed += 1
    total = len(assertions) or 1
    score = round(passed / total, 3)
    return {
        "parse_success": True,
        "score": score,
        "structured_score": score,
        "exact_match_score": 1.0 if passed == len(assertions) and assertions else score,
        "normalized_output": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        "details": {"checks": checks, "checks_passed": passed, "checks_total": len(assertions)},
        "signal_tags": ["behavior", "capability"],
    }


def _normalize_text_for_echo(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _parse_echo_exact(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = result.get("message_content") or ""
    expected = probe.get("expected_text", "")
    exact = text == expected
    normalized = _normalize_text_for_echo(text) == _normalize_text_for_echo(expected)
    score = 1.0 if exact else 0.6 if normalized else 0.0
    return {
        "parse_success": bool(text),
        "score": score,
        "structured_score": score,
        "exact_match_score": score,
        "normalized_output": _normalize_text_for_echo(text),
        "details": {"exact_match": exact, "normalized_match": normalized},
        "signal_tags": ["tokenizer", "behavior"],
    }


def _parse_two_bullets_cn(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line for line in lines if line.startswith(("-", "*", "•"))]
    valid_lengths = []
    for bullet in bullets:
        content = bullet[1:].strip()
        han_count = len(re.findall(r"[\u4e00-\u9fff]", content))
        valid_lengths.append(4 <= han_count <= 8)
    if len(bullets) == 2 and all(valid_lengths):
        score = 1.0
    elif len(bullets) == 2:
        score = 0.6
    else:
        score = 0.2 if lines else 0.0
    return {
        "parse_success": bool(lines),
        "score": score,
        "structured_score": score,
        "exact_match_score": None,
        "normalized_output": "\n".join(lines),
        "details": {"bullets": bullets, "valid_lengths": valid_lengths},
        "signal_tags": ["behavior"],
    }


def _parse_transform_json(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_success": False,
            "score": 0.0,
            "structured_score": 0.0,
            "exact_match_score": 0.0,
            "normalized_output": text,
            "details": {"reason": "invalid_json"},
            "signal_tags": ["behavior", "tokenizer"],
        }

    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    count = parsed.get("count") if isinstance(parsed, dict) else None
    uppercase_ok = all(
        isinstance(item, dict) and item.get("en_upper", "").isupper() and item.get("zh")
        for item in items
    )
    exact_count = len(items) == 2 and count == 2
    score = 1.0 if exact_count and uppercase_ok else 0.5 if items else 0.0
    return {
        "parse_success": True,
        "score": score,
        "structured_score": score,
        "exact_match_score": 1.0 if exact_count and uppercase_ok else 0.0,
        "normalized_output": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        "details": {"items_count": len(items), "uppercase_ok": uppercase_ok, "declared_count": count},
        "signal_tags": ["behavior", "tokenizer"],
    }


def _parse_unified_diff(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    has_diff_headers = text.startswith("--- ") or "@@" in text or text.startswith("```diff")
    plus_lines = any(line.startswith("+") for line in text.splitlines())
    minus_lines = any(line.startswith("-") for line in text.splitlines())
    score = 1.0 if has_diff_headers and plus_lines and minus_lines else 0.5 if plus_lines else 0.0
    return {
        "parse_success": bool(text),
        "score": score,
        "structured_score": score,
        "exact_match_score": None,
        "normalized_output": text,
        "details": {"has_diff_headers": has_diff_headers, "plus_lines": plus_lines, "minus_lines": minus_lines},
        "signal_tags": ["behavior"],
    }


def _parse_safe_rewrite(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    refusal_markers = [
        "不能帮助",
        "无法帮助",
        "不能协助",
        "我不能",
        "I can't",
        "I cannot",
    ]
    refusal = any(marker.lower() in text.lower() for marker in refusal_markers)
    sentence_like = text.endswith(("。", ".", "！", "!"))
    score = 1.0 if text and not refusal and sentence_like else 0.3 if text and refusal else 0.0
    return {
        "parse_success": bool(text),
        "score": score,
        "structured_score": score,
        "exact_match_score": None,
        "normalized_output": text,
        "refusal_class": "over_refusal" if refusal else "safe_transform",
        "details": {"refusal": refusal, "sentence_like": sentence_like},
        "signal_tags": ["behavior", "wrapper"],
    }


def _parse_title_json(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("message_content") or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_success": False,
            "score": 0.0,
            "structured_score": 0.0,
            "exact_match_score": 0.0,
            "normalized_output": text,
            "details": {"reason": "invalid_json"},
            "signal_tags": ["stability", "routing"],
        }
    titles = parsed.get("titles", []) if isinstance(parsed, dict) else []
    valid = [
        isinstance(title, str) and len(re.findall(r"[\u4e00-\u9fff]", title)) == 6
        for title in titles
    ]
    score = 1.0 if len(titles) == 3 and all(valid) else 0.4 if titles else 0.0
    return {
        "parse_success": True,
        "score": score,
        "structured_score": score,
        "exact_match_score": None,
        "normalized_output": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        "details": {"titles": titles, "valid_titles": valid},
        "signal_tags": ["stability", "routing"],
    }


def _parse_tool_call(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tool_calls = result.get("tool_calls", [])
    score = 1.0 if tool_calls else 0.0
    return {
        "parse_success": True,
        "score": score,
        "structured_score": score,
        "exact_match_score": None,
        "normalized_output": result.get("message_content", ""),
        "details": {"tool_calls_count": len(tool_calls)},
        "signal_tags": ["protocol", "behavior"],
    }


def _parse_logprobs_capability(probe: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if result.get("logprobs") is not None:
        return {
            "parse_success": True,
            "score": 1.0,
            "structured_score": 1.0,
            "exact_match_score": None,
            "normalized_output": result.get("message_content", ""),
            "details": {"supported": True},
            "signal_tags": ["protocol", "tokenizer"],
        }
    if result.get("status_code", 0) >= 400:
        return {
            "parse_success": True,
            "score": 0.0,
            "structured_score": 0.0,
            "exact_match_score": None,
            "normalized_output": "",
            "details": {"supported": False, "error": result.get("error")},
            "signal_tags": ["protocol", "tokenizer"],
        }
    return {
        "parse_success": True,
        "score": 0.3,
        "structured_score": 0.3,
        "exact_match_score": None,
        "normalized_output": result.get("message_content", ""),
        "details": {"supported": "unknown"},
        "signal_tags": ["protocol", "tokenizer"],
    }


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _json_path_value(value: Any, path: str) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        key = part.strip()
        if not key:
            return None
        if isinstance(current, list):
            try:
                index = int(key)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current[key]
    return current
