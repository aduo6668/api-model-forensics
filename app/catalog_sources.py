from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from .config import CATALOGS_DIR, ensure_runtime_dirs


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_catalog(timeout: int = 60) -> dict[str, Any]:
    response = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    return {
        "source": "openrouter",
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_url": OPENROUTER_MODELS_URL,
        "model_count": len(models),
        "models": models,
    }


def simplify_openrouter_catalog(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        architecture = item.get("architecture", {}) if isinstance(item.get("architecture"), dict) else {}
        top_provider = item.get("top_provider", {}) if isinstance(item.get("top_provider"), dict) else {}
        rows.append(
            {
                "id": item.get("id"),
                "canonical_slug": item.get("canonical_slug"),
                "name": item.get("name"),
                "description": item.get("description"),
                "context_length": item.get("context_length"),
                "tokenizer": architecture.get("tokenizer"),
                "modality": architecture.get("modality"),
                "input_modalities": architecture.get("input_modalities"),
                "output_modalities": architecture.get("output_modalities"),
                "top_provider_context_length": top_provider.get("context_length"),
                "top_provider_max_completion_tokens": top_provider.get("max_completion_tokens"),
                "is_moderated": top_provider.get("is_moderated"),
                "pricing": item.get("pricing"),
                "created": item.get("created"),
                "provider_family": _provider_family(item.get("id", "")),
            }
        )
    return rows


def save_openrouter_catalog(payload: dict[str, Any], simplified_rows: list[dict[str, Any]]) -> dict[str, str]:
    ensure_runtime_dirs()
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raw_path = CATALOGS_DIR / f"openrouter-catalog-{stamp}.json"
    summary_path = CATALOGS_DIR / f"openrouter-catalog-{stamp}-summary.json"
    markdown_path = CATALOGS_DIR / f"openrouter-catalog-{stamp}-summary.md"

    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(simplified_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown_summary(payload, simplified_rows), encoding="utf-8")

    latest_raw = CATALOGS_DIR / "openrouter-catalog-latest.json"
    latest_summary = CATALOGS_DIR / "openrouter-catalog-latest-summary.json"
    latest_markdown = CATALOGS_DIR / "openrouter-catalog-latest-summary.md"
    latest_raw.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_markdown.write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "raw_json": str(raw_path),
        "summary_json": str(summary_path),
        "summary_md": str(markdown_path),
        "latest_raw_json": str(latest_raw),
        "latest_summary_json": str(latest_summary),
        "latest_summary_md": str(latest_markdown),
    }


def _provider_family(model_id: str) -> str:
    prefix = (model_id or "").split("/", 1)[0].strip().lower()
    return prefix or "unknown"


def _render_markdown_summary(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# OpenRouter Catalog Snapshot",
        "",
        f"- Fetched at: {payload.get('fetched_at', '')}",
        f"- Source URL: {payload.get('source_url', OPENROUTER_MODELS_URL)}",
        f"- Model count: {payload.get('model_count', len(rows))}",
        "",
        "## First 40 Models",
        "",
        "| ID | Name | Context | Tokenizer | Description |",
        "|---|---|---:|---|---|",
    ]
    for item in rows[:40]:
        description = (item.get("description") or "").replace("\n", " ").strip()
        if len(description) > 140:
            description = description[:137] + "..."
        lines.append(
            f"| {item.get('id','')} | {item.get('name','')} | {item.get('context_length','')} | {item.get('tokenizer','')} | {description} |"
        )
    return "\n".join(lines) + "\n"
