from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import BUDGET_PROFILES, default_runtime_settings, ensure_runtime_dirs
from .runner import run_analysis


def build_parser() -> argparse.ArgumentParser:
    defaults = default_runtime_settings()
    parser = argparse.ArgumentParser(
        prog="api-model-forensics",
        description="Run API Model Forensics from the command line.",
    )
    parser.add_argument(
        "--base-url",
        default=defaults["base_url"],
        help="Target API base URL. Defaults to the local .env value when available.",
    )
    parser.add_argument(
        "--api-key",
        default=defaults["api_key"],
        help="API key for the target endpoint. Defaults to the local .env value when available.",
    )
    parser.add_argument(
        "--model",
        default=defaults["model"],
        help="Claimed model name. Defaults to the local .env value when available.",
    )
    parser.add_argument(
        "--provider-hint",
        default=defaults["provider_hint"],
        help="Optional provider hint, such as openai / anthropic / gemini.",
    )
    parser.add_argument(
        "--mode",
        choices=list(BUDGET_PROFILES.keys()),
        default=defaults["mode"],
        help="Probe budget profile.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Stdout format. Use json for AI/tool callers.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation when --format json is used.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print per-probe progress to stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.base_url or not args.api_key or not args.model:
        parser.error("--base-url, --api-key, and --model are required.")

    ensure_runtime_dirs()
    progress_cb = _stderr_progress if args.show_progress else None
    result = run_analysis(
        base_url=args.base_url,
        api_key=args.api_key,
        claimed_model=args.model,
        provider_hint=args.provider_hint,
        mode=args.mode,
        progress_cb=progress_cb,
    )
    payload = _build_cli_payload(args, result)

    if args.format == "json":
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=args.indent)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_render_text(payload))
        if not payload.get("text", "").endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _stderr_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _build_cli_payload(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    return {
        "input": {
            "base_url": args.base_url,
            "claimed_model": args.model,
            "provider_hint": args.provider_hint,
            "mode": args.mode,
        },
        "decision": {
            "verdict_label": summary["verdict_label"],
            "confidence_level": summary["confidence_level"],
            "observed_model_hints": summary.get("observed_model_hints", []),
            "candidate_probabilities": summary["candidate_probabilities"],
            "top_candidates": summary["top_candidates"],
            "primary_reason": summary["primary_reason"],
            "secondary_reason": summary["secondary_reason"],
            "primary_caveats": summary["primary_caveats"],
            "evidence_breakdown": summary["evidence_breakdown"],
        },
        "artifacts": {
            "run_dir": result["run_dir"],
            "summary_json": result["summary_json"],
            "report_json": result["report_json"],
            "report_pdf": result["report_pdf"],
            "normalized_outputs_json": result["normalized_outputs_json"],
        },
        "text": _render_text(
            {
                "input": {
                    "base_url": args.base_url,
                    "claimed_model": args.model,
                    "provider_hint": args.provider_hint,
                    "mode": args.mode,
                },
                "decision": {
                    "verdict_label": summary["verdict_label"],
                    "confidence_level": summary["confidence_level"],
                    "candidate_probabilities": summary["candidate_probabilities"],
                    "top_candidates": summary["top_candidates"],
                    "primary_reason": summary["primary_reason"],
                    "secondary_reason": summary["secondary_reason"],
                    "primary_caveats": summary["primary_caveats"],
                    "evidence_breakdown": summary["evidence_breakdown"],
                },
                "artifacts": {
                    "run_dir": result["run_dir"],
                    "summary_json": result["summary_json"],
                    "report_json": result["report_json"],
                    "report_pdf": result["report_pdf"],
                    "normalized_outputs_json": result["normalized_outputs_json"],
                },
            }
        ),
    }


def _render_text(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    probs = decision["candidate_probabilities"]
    lines = [
        "API Model Forensics CLI",
        f"Model: {payload['input']['claimed_model']}",
        f"Mode: {payload['input']['mode']}",
        f"Verdict: {decision['verdict_label']}",
        f"Confidence: {decision['confidence_level']}",
        "",
        "Probabilities:",
        f"- Claimed model: {probs['claimed_model_probability']:.1%}",
        f"- Same-family downgrade: {probs['same_family_downgrade_probability']:.1%}",
        f"- Alternative family: {probs['alternative_family_probability']:.1%}",
        f"- Wrapped or unknown: {probs['wrapped_or_unknown_probability']:.1%}",
        "",
        "Top candidates:",
    ]
    for item in decision["top_candidates"]:
        lines.append(f"- {item['name']}: {item['probability']:.1%} ({item['kind']})")
    observed_hints = decision.get("observed_model_hints", [])
    if observed_hints:
        lines.extend(["", "Observed model hints:"])
        for hint in observed_hints[:5]:
            lines.append(f"- {hint}")
    lines.extend(
        [
            "",
            f"Primary reason: {decision['primary_reason']}",
            f"Secondary reason: {decision['secondary_reason']}",
            "",
            "Artifacts:",
            f"- summary_json: {payload['artifacts']['summary_json']}",
            f"- report_json: {payload['artifacts']['report_json']}",
            f"- report_pdf: {payload['artifacts']['report_pdf']}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
