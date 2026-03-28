from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .provider_registry import family_for_name, resolve_dialect


MOCK_CALL_COUNTERS: dict[str, int] = {}


@dataclass(frozen=True)
class AdapterResolution:
    adapter_name: str
    dialect: str
    family: str
    normalized_base_url: str


def resolve_adapter(base_url: str, provider_hint: str = "", claimed_model: str = "") -> AdapterResolution:
    family = "unknown"
    if (base_url or "").startswith("mock://"):
        family = base_url.split("mock://", 1)[1].split("/", 1)[0] or "unknown"
    if family == "unknown":
        family = family_for_name(provider_hint)
    if family == "unknown":
        family = family_for_name(claimed_model)
    if family == "unknown":
        family = family_for_name(base_url)
    dialect = resolve_dialect(base_url, provider_hint, claimed_model)
    if dialect == "anthropic_messages":
        adapter_name = "AnthropicMessagesAdapter"
        normalized = _normalize_anthropic_base_url(base_url)
    elif dialect == "gemini_generate_content":
        adapter_name = "GeminiGenerateContentAdapter"
        normalized = _normalize_gemini_base_url(base_url)
    else:
        adapter_name = "OpenAICompatibleAdapter"
        normalized = _normalize_openai_base_url(base_url)
    return AdapterResolution(
        adapter_name=adapter_name,
        dialect=dialect,
        family=family,
        normalized_base_url=normalized,
    )


def describe_adapter(resolution: AdapterResolution, claimed_model: str = "") -> dict[str, str]:
    return {
        "adapter_name": resolution.adapter_name,
        "dialect": resolution.dialect,
        "normalized_base_url": resolution.normalized_base_url,
        "resolved_chat_endpoint": chat_endpoint(resolution, claimed_model=claimed_model),
        "resolved_models_endpoint": models_endpoint(resolution),
    }


def chat_endpoint(resolution: AdapterResolution, claimed_model: str = "") -> str:
    base_url = resolution.normalized_base_url
    if base_url.startswith("mock://"):
        return base_url
    if resolution.dialect == "anthropic_messages":
        return f"{base_url}/v1/messages"
    if resolution.dialect == "gemini_generate_content":
        model = claimed_model or "{model}"
        return f"{base_url}/v1beta/models/{model}:generateContent"
    return f"{base_url}/v1/chat/completions"


def models_endpoint(resolution: AdapterResolution) -> str:
    base_url = resolution.normalized_base_url
    if base_url.startswith("mock://"):
        return base_url
    if resolution.dialect == "gemini_generate_content":
        return f"{base_url}/v1beta/models"
    return f"{base_url}/v1/models"


def list_models(resolution: AdapterResolution, api_key: str, timeout: int = 30) -> dict[str, Any]:
    if resolution.normalized_base_url.startswith("mock://"):
        return _mock_list_models(resolution.family)
    if resolution.dialect == "anthropic_messages":
        return _anthropic_list_models(resolution, api_key, timeout=timeout)
    if resolution.dialect == "gemini_generate_content":
        return _gemini_list_models(resolution, api_key, timeout=timeout)
    return _openai_list_models(resolution, api_key, timeout=timeout)


def post_chat(
    resolution: AdapterResolution,
    api_key: str,
    payload: dict[str, Any],
    probe_id: str,
    stream: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    if resolution.normalized_base_url.startswith("mock://"):
        return _mock_chat(resolution, payload, probe_id, stream=stream)
    if resolution.dialect == "anthropic_messages":
        return _anthropic_post_chat(resolution, api_key, payload, timeout=timeout)
    if resolution.dialect == "gemini_generate_content":
        return _gemini_post_chat(resolution, api_key, payload, timeout=timeout)
    return _openai_post_chat(resolution, api_key, payload, stream=stream, timeout=timeout)


def _normalize_openai_base_url(base_url: str) -> str:
    url = base_url.strip()
    if url.startswith("mock://"):
        return url.rstrip("/")
    url = url.rstrip("/")
    suffixes = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/responses",
        "/responses",
        "/v1",
    ]
    for suffix in suffixes:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def _normalize_anthropic_base_url(base_url: str) -> str:
    url = base_url.strip()
    if url.startswith("mock://"):
        return url.rstrip("/")
    url = url.rstrip("/")
    suffixes = [
        "/v1/messages",
        "/messages",
        "/v1/models",
        "/models",
        "/v1",
    ]
    for suffix in suffixes:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def _normalize_gemini_base_url(base_url: str) -> str:
    url = base_url.strip().split("?", 1)[0].rstrip("/")
    if url.startswith("mock://"):
        return url
    match = re.search(r"/v[0-9][^/]*/models/.*$", url)
    if match:
        url = url[: match.start()]
    suffixes = ["/v1beta/models", "/v1/models", "/v1beta", "/v1"]
    for suffix in suffixes:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def _openai_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _gemini_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
    }


def _openai_list_models(resolution: AdapterResolution, api_key: str, timeout: int = 30) -> dict[str, Any]:
    try:
        response = requests.get(models_endpoint(resolution), headers=_openai_headers(api_key), timeout=timeout)
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status_code": 0, "models": [], "error": str(exc), "raw": None}

    models = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "models": models,
        "raw": payload,
    }


def _anthropic_list_models(resolution: AdapterResolution, api_key: str, timeout: int = 30) -> dict[str, Any]:
    try:
        response = requests.get(models_endpoint(resolution), headers=_anthropic_headers(api_key), timeout=timeout)
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status_code": 0, "models": [], "error": str(exc), "raw": None}

    models = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "models": models,
        "raw": payload,
    }


def _gemini_list_models(resolution: AdapterResolution, api_key: str, timeout: int = 30) -> dict[str, Any]:
    try:
        response = requests.get(
            models_endpoint(resolution),
            headers=_gemini_headers(),
            params={"key": api_key},
            timeout=timeout,
        )
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status_code": 0, "models": [], "error": str(exc), "raw": None}

    models = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            models.append(name)
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "models": models,
        "raw": payload,
    }


def _openai_post_chat(
    resolution: AdapterResolution,
    api_key: str,
    payload: dict[str, Any],
    stream: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    started = time.perf_counter()
    body = dict(payload)
    if stream:
        body["stream"] = True
    try:
        response = requests.post(
            chat_endpoint(resolution, claimed_model=payload.get("model", "")),
            headers=_openai_headers(api_key),
            json=body,
            timeout=timeout,
            stream=stream,
        )
    except Exception as exc:  # noqa: BLE001
        return _transport_error(exc, started)

    if not stream:
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            data = response.json()
        except Exception:  # noqa: BLE001
            return _plain_text_error_response(response.status_code, response.ok, response.text, latency_ms)
        return _normalize_openai_response(response.status_code, response.ok, data, latency_ms)

    return _consume_openai_stream_response(response, started)


def _anthropic_post_chat(
    resolution: AdapterResolution,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 60,
) -> dict[str, Any]:
    started = time.perf_counter()
    body = _anthropic_payload_from_openai(payload)
    try:
        response = requests.post(
            chat_endpoint(resolution, claimed_model=payload.get("model", "")),
            headers=_anthropic_headers(api_key),
            json=body,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return _transport_error(exc, started)

    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return _plain_text_error_response(response.status_code, response.ok, response.text, latency_ms)
    return _normalize_anthropic_response(response.status_code, response.ok, data, latency_ms)


def _gemini_post_chat(
    resolution: AdapterResolution,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 60,
) -> dict[str, Any]:
    started = time.perf_counter()
    body = _gemini_payload_from_openai(payload)
    try:
        response = requests.post(
            chat_endpoint(resolution, claimed_model=payload.get("model", "")),
            headers=_gemini_headers(),
            params={"key": api_key},
            json=body,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return _transport_error(exc, started)

    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return _plain_text_error_response(response.status_code, response.ok, response.text, latency_ms)
    return _normalize_gemini_response(response.status_code, response.ok, data, latency_ms)


def _transport_error(exc: Exception, started: float) -> dict[str, Any]:
    return {
        "ok": False,
        "status_code": 0,
        "error": str(exc),
        "raw_json": None,
        "raw_text": "",
        "message_content": "",
        "tool_calls": [],
        "usage": {},
        "finish_reason": None,
        "latency_total_ms": int((time.perf_counter() - started) * 1000),
        "ttft_ms": None,
        "inter_token_times_ms": [],
        "stream_chunk_count": 0,
    }


def _plain_text_error_response(status_code: int, ok: bool, text: str, latency_ms: int) -> dict[str, Any]:
    return {
        "ok": ok,
        "status_code": status_code,
        "error": None if ok else text,
        "raw_json": None,
        "raw_text": text,
        "message_content": text,
        "tool_calls": [],
        "usage": {},
        "finish_reason": None,
        "latency_total_ms": latency_ms,
        "ttft_ms": None,
        "inter_token_times_ms": [],
        "stream_chunk_count": 0,
    }


def _normalize_openai_response(status_code: int, ok: bool, data: dict[str, Any], latency_ms: int) -> dict[str, Any]:
    if "choices" in data:
        choice = data.get("choices", [{}])[0] or {}
        message = choice.get("message", {}) or {}
        content = _flatten_openai_message_content(message.get("content"))
        return {
            "ok": ok,
            "status_code": status_code,
            "error": None,
            "raw_json": data,
            "raw_text": json.dumps(data, ensure_ascii=False),
            "message_content": content or "",
            "tool_calls": message.get("tool_calls", []) or [],
            "usage": data.get("usage", {}) or {},
            "finish_reason": choice.get("finish_reason"),
            "system_fingerprint": data.get("system_fingerprint"),
            "logprobs": choice.get("logprobs"),
            "latency_total_ms": latency_ms,
            "ttft_ms": None,
            "inter_token_times_ms": [],
            "stream_chunk_count": 0,
        }

    return {
        "ok": ok,
        "status_code": status_code,
        "error": data.get("error") or data,
        "raw_json": data,
        "raw_text": json.dumps(data, ensure_ascii=False),
        "message_content": "",
        "tool_calls": [],
        "usage": {},
        "finish_reason": None,
        "latency_total_ms": latency_ms,
        "ttft_ms": None,
        "inter_token_times_ms": [],
        "stream_chunk_count": 0,
    }


def _normalize_anthropic_response(status_code: int, ok: bool, data: dict[str, Any], latency_ms: int) -> dict[str, Any]:
    if "content" not in data:
        return {
            "ok": ok,
            "status_code": status_code,
            "error": data.get("error") or data,
            "raw_json": data,
            "raw_text": json.dumps(data, ensure_ascii=False),
            "message_content": "",
            "tool_calls": [],
            "usage": {},
            "finish_reason": None,
            "latency_total_ms": latency_ms,
            "ttft_ms": None,
            "inter_token_times_ms": [],
            "stream_chunk_count": 0,
        }

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in data.get("content", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text_parts.append(item.get("text", ""))
        elif item.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": json.dumps(item.get("input", {}), ensure_ascii=False),
                    },
                }
            )

    usage_data = data.get("usage", {}) or {}
    usage = {
        "prompt_tokens": usage_data.get("input_tokens"),
        "completion_tokens": usage_data.get("output_tokens"),
        "total_tokens": _safe_sum(usage_data.get("input_tokens"), usage_data.get("output_tokens")),
    }
    return {
        "ok": ok,
        "status_code": status_code,
        "error": None,
        "raw_json": data,
        "raw_text": json.dumps(data, ensure_ascii=False),
        "message_content": "".join(text_parts),
        "tool_calls": tool_calls,
        "usage": usage,
        "finish_reason": data.get("stop_reason"),
        "system_fingerprint": None,
        "logprobs": None,
        "latency_total_ms": latency_ms,
        "ttft_ms": None,
        "inter_token_times_ms": [],
        "stream_chunk_count": 0,
    }


def _normalize_gemini_response(status_code: int, ok: bool, data: dict[str, Any], latency_ms: int) -> dict[str, Any]:
    candidates = data.get("candidates", [])
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
    parts = content.get("parts", []) if isinstance(content, dict) else []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in parts:
        if not isinstance(item, dict):
            continue
        if "text" in item:
            text_parts.append(item.get("text", ""))
        elif "functionCall" in item and isinstance(item["functionCall"], dict):
            function_call = item["functionCall"]
            tool_calls.append(
                {
                    "id": function_call.get("name", ""),
                    "type": "function",
                    "function": {
                        "name": function_call.get("name", ""),
                        "arguments": json.dumps(function_call.get("args", {}), ensure_ascii=False),
                    },
                }
            )

    usage_data = data.get("usageMetadata", {}) or {}
    usage = {
        "prompt_tokens": usage_data.get("promptTokenCount"),
        "completion_tokens": usage_data.get("candidatesTokenCount"),
        "total_tokens": usage_data.get("totalTokenCount"),
    }
    return {
        "ok": ok,
        "status_code": status_code,
        "error": None if ok else data.get("error") or data,
        "raw_json": data,
        "raw_text": json.dumps(data, ensure_ascii=False),
        "message_content": "".join(text_parts),
        "tool_calls": tool_calls,
        "usage": usage,
        "finish_reason": candidate.get("finishReason") if isinstance(candidate, dict) else None,
        "system_fingerprint": None,
        "logprobs": None,
        "latency_total_ms": latency_ms,
        "ttft_ms": None,
        "inter_token_times_ms": [],
        "stream_chunk_count": 0,
    }


def _consume_openai_stream_response(response: requests.Response, started: float) -> dict[str, Any]:
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    inter_token_times_ms: list[int] = []
    last_token_time: float | None = None
    ttft_ms: int | None = None
    chunk_count = 0
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    raw_chunks: list[str] = []
    non_sse_lines: list[str] = []

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
                non_sse_lines.append(line)
                raw_chunks.append(line)
                continue
            chunk = line[5:].strip()
            if not chunk:
                continue
            raw_chunks.append(chunk)
            if chunk == "[DONE]":
                break
            chunk_count += 1
            now = time.perf_counter()
            if ttft_ms is None:
                ttft_ms = int((now - started) * 1000)
            elif last_token_time is not None:
                inter_token_times_ms.append(int((now - last_token_time) * 1000))
            last_token_time = now

            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            usage = data.get("usage", usage) or usage
            choice = data.get("choices", [{}])[0] or {}
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}) or {}
            content = delta.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content_parts.append(item.get("text", ""))
            elif isinstance(content, str):
                content_parts.append(content)
            if delta.get("tool_calls"):
                tool_calls.extend(delta.get("tool_calls", []))
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": str(exc),
            "raw_json": None,
            "raw_text": "\n".join(raw_chunks),
            "message_content": "".join(content_parts),
            "tool_calls": tool_calls,
            "usage": usage,
            "finish_reason": finish_reason,
            "latency_total_ms": int((time.perf_counter() - started) * 1000),
            "ttft_ms": ttft_ms,
            "inter_token_times_ms": inter_token_times_ms,
            "stream_chunk_count": chunk_count,
        }

    latency_ms = int((time.perf_counter() - started) * 1000)
    if chunk_count == 0 and non_sse_lines:
        fallback_text = "\n".join(non_sse_lines).strip()
        try:
            data = json.loads(fallback_text)
        except json.JSONDecodeError:
            return _plain_text_error_response(response.status_code, response.ok, fallback_text, latency_ms)
        return _normalize_openai_response(response.status_code, response.ok, data, latency_ms)

    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "error": None if response.ok else response.text,
        "raw_json": None,
        "raw_text": "\n".join(raw_chunks),
        "message_content": "".join(content_parts),
        "tool_calls": tool_calls,
        "usage": usage,
        "finish_reason": finish_reason,
        "latency_total_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "inter_token_times_ms": inter_token_times_ms,
        "stream_chunk_count": chunk_count,
    }


def _anthropic_payload_from_openai(payload: dict[str, Any]) -> dict[str, Any]:
    system_text, messages = _extract_system_and_messages(payload.get("messages", []))
    body: dict[str, Any] = {
        "model": payload.get("model", ""),
        "max_tokens": payload.get("max_tokens", 128),
        "messages": [_anthropic_message_from_openai(item) for item in messages],
    }
    if system_text:
        body["system"] = system_text
    if payload.get("temperature") is not None:
        body["temperature"] = payload["temperature"]
    if payload.get("tools"):
        tools = [_anthropic_tool_from_openai(tool) for tool in payload["tools"]]
        tools = [tool for tool in tools if tool]
        if tools:
            body["tools"] = tools
            body["tool_choice"] = {"type": "auto"}
    return body


def _gemini_payload_from_openai(payload: dict[str, Any]) -> dict[str, Any]:
    system_text, messages = _extract_system_and_messages(payload.get("messages", []))
    contents = [_gemini_message_from_openai(item) for item in messages]
    body: dict[str, Any] = {
        "contents": [item for item in contents if item],
        "generationConfig": {
            "temperature": payload.get("temperature", 0.2),
            "maxOutputTokens": payload.get("max_tokens", 128),
        },
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if payload.get("tools"):
        tools = [_gemini_tool_from_openai(tool) for tool in payload["tools"]]
        tools = [tool for tool in tools if tool]
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
    return body


def _extract_system_and_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    non_system_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "")
        content = _flatten_openai_message_content(message.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        non_system_messages.append({"role": role, "content": content})
    return "\n".join(system_parts).strip(), non_system_messages


def _anthropic_message_from_openai(message: dict[str, Any]) -> dict[str, Any]:
    role = "assistant" if message.get("role") == "assistant" else "user"
    return {
        "role": role,
        "content": [{"type": "text", "text": message.get("content", "")}],
    }


def _gemini_message_from_openai(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content", "")
    if not content:
        return None
    role = "model" if message.get("role") == "assistant" else "user"
    return {
        "role": role,
        "parts": [{"text": content}],
    }


def _anthropic_tool_from_openai(tool: dict[str, Any]) -> dict[str, Any] | None:
    function = tool.get("function", {}) if isinstance(tool, dict) else {}
    name = function.get("name")
    if not name:
        return None
    return {
        "name": name,
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
    }


def _gemini_tool_from_openai(tool: dict[str, Any]) -> dict[str, Any] | None:
    function = tool.get("function", {}) if isinstance(tool, dict) else {}
    name = function.get("name")
    if not name:
        return None
    return {
        "name": name,
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object", "properties": {}}),
    }


def _flatten_openai_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return ""


def _safe_sum(left: Any, right: Any) -> int | None:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return None


def _mock_list_models(family: str) -> dict[str, Any]:
    models = {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
        "gemini": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "mixed": ["gpt-4o", "claude-3-5-sonnet", "gemini-2.5-flash"],
    }.get(family, ["unknown-model"])
    return {
        "ok": True,
        "status_code": 200,
        "models": models,
        "raw": {"data": [{"id": model} for model in models]},
    }


def _mock_chat(
    resolution: AdapterResolution,
    payload: dict[str, Any],
    probe_id: str,
    stream: bool = False,
) -> dict[str, Any]:
    family = resolution.normalized_base_url.split("mock://", 1)[1] or resolution.family or "openai"
    family = family.split("/", 1)[0]

    def mk_choice(content: str, finish_reason: str = "stop", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or [],
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": max(12, len(content) // 3),
                "total_tokens": 120 + max(12, len(content) // 3),
            },
            "system_fingerprint": "fp_mock_local",
        }

    if probe_id == "P02":
        if family in {"openai", "mixed"}:
            data = mk_choice("OK")
            data["choices"][0]["logprobs"] = {"content": []}
            return _normalize_openai_response(200, True, data, 120)
        return {
            "ok": False,
            "status_code": 400,
            "error": {"message": "logprobs unsupported"},
            "raw_json": {"error": {"message": "logprobs unsupported"}},
            "raw_text": "{\"error\":{\"message\":\"logprobs unsupported\"}}",
            "message_content": "",
            "tool_calls": [],
            "usage": {},
            "finish_reason": None,
            "latency_total_ms": 110,
            "ttft_ms": None,
            "inter_token_times_ms": [],
            "stream_chunk_count": 0,
        }

    if probe_id == "P03":
        if family in {"openai", "anthropic", "mixed"}:
            tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "emit_number", "arguments": "{\"value\":7}"}}]
            return _normalize_openai_response(200, True, mk_choice("", finish_reason="tool_calls", tool_calls=tool_calls), 140)
        return _normalize_openai_response(200, True, mk_choice("7"), 140)

    canned = {
        "P04": {
            "openai": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
            "anthropic": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
            "gemini": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
            "mixed": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}"
        },
        "P05": {
            "openai": "Alpha  Beta馃檪\ncode```x=1```\n涓枃  绌烘牸",
            "anthropic": "Alpha Beta馃檪\ncode```x=1```\n涓枃 绌烘牸",
            "gemini": "Alpha  Beta馃檪\ncode```x=1```\n涓枃  绌烘牸",
            "mixed": "Alpha Beta馃檪\ncode```x=1```\n涓枃 绌烘牸"
        },
        "P06": {
            "openai": "- 鏈湴閴村埆\n- 璇嗗埆闄嶇骇",
            "anthropic": "- 鏈湴閴村埆\n- 璇嗗埆闄嶇骇",
            "gemini": "- 鏈湴瀹¤\n- 鍙戠幇娣锋ā",
            "mixed": "- 鏈湴閴村埆\n- 璇嗗埆闄嶇骇"
        },
        "P07": {
            "openai": "{\"items\":[{\"zh\":\"妯″瀷\",\"en_upper\":\"MODEL\"},{\"zh\":\"妫€娴媆\", \"en_upper\":\"DETECT\"}],\"count\":2}",
            "anthropic": "{\"items\":[{\"zh\":\"妯″瀷\",\"en_upper\":\"MODEL\"},{\"zh\":\"妫€娴媆\", \"en_upper\":\"DETECT\"}],\"count\":2}",
            "gemini": "{\"items\":[{\"zh\":\"妯″瀷\",\"en_upper\":\"MODEL\"},{\"zh\":\"妫€娴媆\", \"en_upper\":\"DETECT\"}],\"count\":2}",
            "mixed": "{\"items\":[{\"zh\":\"妯″瀷\",\"en_upper\":\"MODEL\"}],\"count\":1}"
        },
        "P08": {
            "openai": "--- a/example.py\n+++ b/example.py\n@@\n def add(a, b):\n-    return a - b\n+    return a + b",
            "anthropic": "--- a/example.py\n+++ b/example.py\n@@\n def add(a, b):\n-    return a - b\n+    return a + b",
            "gemini": "```diff\n--- a/example.py\n+++ b/example.py\n@@\n-    return a - b\n+    return a + b\n```",
            "mixed": "Fix should return a + b"
        },
        "P09": {
            "openai": "Please follow site rules and protect user data.",
            "anthropic": "I cannot help bypass restrictions, but I can help rewrite it as a safety warning.",
            "gemini": "Handle data only within authorized scope and protect user privacy.",
            "mixed": "I cannot help with that."
        },
        "P10": {
            "openai": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
            "anthropic": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
            "gemini": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
            "mixed": "{\"prime_under_10\":[2,3,5,7],\"count\":4}"
        },
        "P11": {
            "openai": "{\"titles\":[\"鏈湴妯″瀷閴村埆\",\"闅愮娉曡瘉鍔╂墜\",\"妯″瀷鐪熶吉瀹¤\"]}",
            "anthropic": "{\"titles\":[\"鏈湴娉曡瘉鍔╂墜\",\"妯″瀷鐪熶吉鍒嗘瀽\",\"闅愮瀹¤宸ュ叿\"]}",
            "gemini": "{\"titles\":[\"妯″瀷鐪熷亣妫€娴媆\", \"娉曡瘉缁撴灉蹇姤\", \"鏈湴闅愮鎶ュ憡\"]}",
            "mixed": "{\"titles\":[\"妯″瀷鐪熷亣鍒ゆ柇\",\"闅愮鎶ュ憡绯荤粺\"]}"
        }
    }

    content = canned.get(probe_id, {}).get(family, "OK")
    if stream:
        counter_key = f"{family}:{probe_id}"
        call_index = MOCK_CALL_COUNTERS.get(counter_key, 0)
        MOCK_CALL_COUNTERS[counter_key] = call_index + 1
        if family == "mixed":
            variants = [
                (115, [12, 88, 14, 92]),
                (205, [84, 11, 79, 15]),
                (145, [20, 122, 18, 111])
            ]
            ttft, inter = variants[call_index % len(variants)]
        else:
            ttft = 180
            inter = [28, 34, 31, 36]
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "raw_json": None,
            "raw_text": content,
            "message_content": content,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": max(12, len(content) // 3),
                "total_tokens": 120 + max(12, len(content) // 3)
            },
            "finish_reason": "stop",
            "latency_total_ms": ttft + sum(inter),
            "ttft_ms": ttft,
            "inter_token_times_ms": inter,
            "stream_chunk_count": len(inter) + 1
        }

    return _normalize_openai_response(200, True, mk_choice(content), 180)
