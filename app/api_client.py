from __future__ import annotations

import json
import time
from typing import Any

import requests


MOCK_CALL_COUNTERS: dict[str, int] = {}


def normalize_base_url(base_url: str) -> str:
    url = base_url.strip()
    if url.startswith("mock://"):
        return url.rstrip("/")

    url = url.rstrip("/")
    suffixes = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1",
    ]
    for suffix in suffixes:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def chat_endpoint(base_url: str) -> str:
    if base_url.startswith("mock://"):
        return base_url
    return f"{base_url}/v1/chat/completions"


def models_endpoint(base_url: str) -> str:
    if base_url.startswith("mock://"):
        return base_url
    return f"{base_url}/v1/models"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def list_models(base_url: str, api_key: str, timeout: int = 30) -> dict[str, Any]:
    if base_url.startswith("mock://"):
        family = base_url.split("mock://", 1)[1] or "openai"
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

    try:
        response = requests.get(models_endpoint(base_url), headers=_headers(api_key), timeout=timeout)
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


def post_chat(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    probe_id: str,
    stream: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    if base_url.startswith("mock://"):
        return _mock_chat(base_url, payload, probe_id, stream=stream)

    started = time.perf_counter()
    try:
        response = requests.post(
            chat_endpoint(base_url),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
            stream=stream,
        )
    except Exception as exc:  # noqa: BLE001
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

    if not stream:
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            data = response.json()
        except Exception:  # noqa: BLE001
            text = response.text
            return {
                "ok": response.ok,
                "status_code": response.status_code,
                "error": None if response.ok else text,
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
        return _normalize_nonstream_response(response.status_code, response.ok, data, latency_ms)

    return _consume_stream_response(response, started)


def _normalize_nonstream_response(status_code: int, ok: bool, data: dict[str, Any], latency_ms: int) -> dict[str, Any]:
    if "choices" in data:
        choice = data.get("choices", [{}])[0] or {}
        message = choice.get("message", {}) or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = "\n".join(parts)
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


def _consume_stream_response(response: requests.Response, started: float) -> dict[str, Any]:
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    inter_token_times_ms: list[int] = []
    last_token_time: float | None = None
    ttft_ms: int | None = None
    chunk_count = 0
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    raw_chunks: list[str] = []

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
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
        "latency_total_ms": int((time.perf_counter() - started) * 1000),
        "ttft_ms": ttft_ms,
        "inter_token_times_ms": inter_token_times_ms,
        "stream_chunk_count": chunk_count,
    }


def _mock_chat(base_url: str, payload: dict[str, Any], probe_id: str, stream: bool = False) -> dict[str, Any]:
    family = base_url.split("mock://", 1)[1] or "openai"
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
            return _normalize_nonstream_response(200, True, data, 120)
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
            return _normalize_nonstream_response(200, True, mk_choice("", finish_reason="tool_calls", tool_calls=tool_calls), 140)
        return _normalize_nonstream_response(200, True, mk_choice("7"), 140)

    canned = {
        "P04": {
            "openai": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
            "anthropic": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
            "gemini": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
            "mixed": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
        },
        "P05": {
            "openai": "Alpha  Beta🙂\ncode```x=1```\n中文  空格",
            "anthropic": "Alpha Beta🙂\ncode```x=1```\n中文 空格",
            "gemini": "Alpha  Beta🙂\ncode```x=1```\n中文  空格",
            "mixed": "Alpha Beta🙂\ncode```x=1```\n中文 空格",
        },
        "P06": {
            "openai": "- 本地鉴别\n- 识别降级",
            "anthropic": "- 本地鉴别\n- 识别降级",
            "gemini": "- 本地审计\n- 发现混模",
            "mixed": "- 本地鉴别\n- 识别降级",
        },
        "P07": {
            "openai": "{\"items\":[{\"zh\":\"模型\",\"en_upper\":\"MODEL\"},{\"zh\":\"检测\",\"en_upper\":\"DETECT\"}],\"count\":2}",
            "anthropic": "{\"items\":[{\"zh\":\"模型\",\"en_upper\":\"MODEL\"},{\"zh\":\"检测\",\"en_upper\":\"DETECT\"}],\"count\":2}",
            "gemini": "{\"items\":[{\"zh\":\"模型\",\"en_upper\":\"MODEL\"},{\"zh\":\"检测\",\"en_upper\":\"DETECT\"}],\"count\":2}",
            "mixed": "{\"items\":[{\"zh\":\"模型\",\"en_upper\":\"MODEL\"}],\"count\":1}",
        },
        "P08": {
            "openai": "--- a/example.py\n+++ b/example.py\n@@\n def add(a, b):\n-    return a - b\n+    return a + b",
            "anthropic": "--- a/example.py\n+++ b/example.py\n@@\n def add(a, b):\n-    return a - b\n+    return a + b",
            "gemini": "```diff\n--- a/example.py\n+++ b/example.py\n@@\n-    return a - b\n+    return a + b\n```",
            "mixed": "修复后应返回 a + b",
        },
        "P09": {
            "openai": "请遵守网站规则，避免越权抓取或滥用用户数据。",
            "anthropic": "我不能帮助绕过限制，但可以提醒你遵守规则并保护用户数据。",
            "gemini": "请在合法授权范围内处理数据并保护用户隐私。",
            "mixed": "我不能帮助你这样做。",
        },
        "P10": {
            "openai": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
            "anthropic": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
            "gemini": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
            "mixed": "{\"prime_under_10\":[2,3,5,7],\"count\":4}",
        },
        "P11": {
            "openai": "{\"titles\":[\"本地模型鉴别\",\"隐私法证助手\",\"模型真伪审计\"]}",
            "anthropic": "{\"titles\":[\"本地法证助手\",\"模型真伪分析\",\"隐私审计工具\"]}",
            "gemini": "{\"titles\":[\"模型真假检测\",\"法证结果快报\",\"本地隐私报告\"]}",
            "mixed": "{\"titles\":[\"模型真假判断\",\"隐私报告系统\"]}",
        },
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
                (145, [20, 122, 18, 111]),
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
                "total_tokens": 120 + max(12, len(content) // 3),
            },
            "finish_reason": "stop",
            "latency_total_ms": ttft + sum(inter),
            "ttft_ms": ttft,
            "inter_token_times_ms": inter,
            "stream_chunk_count": len(inter) + 1,
        }

    return _normalize_nonstream_response(200, True, mk_choice(content), 180)
