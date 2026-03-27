from __future__ import annotations

from typing import Any

from . import provider_adapters


def normalize_base_url(base_url: str, provider_hint: str = "", claimed_model: str = "") -> str:
    resolution = provider_adapters.resolve_adapter(base_url, provider_hint=provider_hint, claimed_model=claimed_model)
    return resolution.normalized_base_url


def chat_endpoint(base_url: str, provider_hint: str = "", claimed_model: str = "") -> str:
    resolution = provider_adapters.resolve_adapter(base_url, provider_hint=provider_hint, claimed_model=claimed_model)
    return provider_adapters.chat_endpoint(resolution, claimed_model=claimed_model)


def models_endpoint(base_url: str, provider_hint: str = "", claimed_model: str = "") -> str:
    resolution = provider_adapters.resolve_adapter(base_url, provider_hint=provider_hint, claimed_model=claimed_model)
    return provider_adapters.models_endpoint(resolution)


def describe_target(base_url: str, provider_hint: str = "", claimed_model: str = "") -> dict[str, str]:
    resolution = provider_adapters.resolve_adapter(base_url, provider_hint=provider_hint, claimed_model=claimed_model)
    return provider_adapters.describe_adapter(resolution, claimed_model=claimed_model)


def list_models(
    base_url: str,
    api_key: str,
    timeout: int = 30,
    provider_hint: str = "",
    claimed_model: str = "",
) -> dict[str, Any]:
    resolution = provider_adapters.resolve_adapter(base_url, provider_hint=provider_hint, claimed_model=claimed_model)
    result = provider_adapters.list_models(resolution, api_key, timeout=timeout)
    return {
        **result,
        "adapter_name": resolution.adapter_name,
        "dialect": resolution.dialect,
        "resolved_endpoint": provider_adapters.models_endpoint(resolution),
    }


def post_chat(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    probe_id: str,
    stream: bool = False,
    timeout: int = 60,
    provider_hint: str = "",
    claimed_model: str = "",
) -> dict[str, Any]:
    model_name = claimed_model or payload.get("model", "")
    resolution = provider_adapters.resolve_adapter(base_url, provider_hint=provider_hint, claimed_model=model_name)
    result = provider_adapters.post_chat(
        resolution,
        api_key,
        payload,
        probe_id,
        stream=stream,
        timeout=timeout,
    )
    return {
        **result,
        "adapter_name": resolution.adapter_name,
        "dialect": resolution.dialect,
        "resolved_endpoint": provider_adapters.chat_endpoint(resolution, claimed_model=model_name),
    }
