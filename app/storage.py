from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import REPORTS_DIR, RUNS_DIR, ensure_runtime_dirs


def create_run_paths(claimed_model: str) -> dict[str, Path]:
    ensure_runtime_dirs()
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S-%f")
    slug = _slugify(claimed_model) or "model"
    run_dir = RUNS_DIR / f"{stamp}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{stamp}-{slug}-report.pdf"
    report_json_path = run_dir / "report.json"
    normalized_outputs_path = run_dir / "normalized_outputs.json"
    return {
        "run_id": f"{stamp}-{slug}",
        "run_dir": run_dir,
        "report_path": report_path,
        "report_json_path": report_json_path,
        "normalized_outputs_path": normalized_outputs_path,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    return "-".join(part for part in cleaned.split("-") if part)[:48]
