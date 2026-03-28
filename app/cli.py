from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .chat_selftest import (
    build_chat_self_test_pack,
    load_chat_self_test_transcript,
    render_chat_self_test_pack_text,
    render_chat_self_test_score_text,
    score_chat_self_test_transcript,
)
from .config import BUDGET_PROFILES, default_runtime_settings, ensure_runtime_dirs
from .runner import run_analysis
from .selftest import render_self_test_text, run_self_test_suite


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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in offline detector regression suite. No external API is required.",
    )
    parser.add_argument(
        "--emit-chat-self-test-pack",
        action="store_true",
        help="Emit the direct-chat self-test prompt pack for testing the currently active conversational AI.",
    )
    parser.add_argument(
        "--score-chat-self-test",
        help="Score a JSON transcript captured from the direct-chat self-test pack.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime_dirs()
    _validate_mode_args(parser, args)

    if args.self_test:
        payload = run_self_test_suite(mode=args.mode)
        _write_stdout(payload, output_format=args.format, indent=args.indent, text_renderer=render_self_test_text)
        return 0

    if args.emit_chat_self_test_pack:
        payload = build_chat_self_test_pack(claimed_model=args.model, provider_hint=args.provider_hint)
        _write_stdout(payload, output_format=args.format, indent=args.indent, text_renderer=render_chat_self_test_pack_text)
        return 0

    if args.score_chat_self_test:
        transcript = load_chat_self_test_transcript(args.score_chat_self_test)
        claimed_model_override = args.model if "--model" in raw_args else ""
        provider_hint_override = args.provider_hint if "--provider-hint" in raw_args else ""
        payload = score_chat_self_test_transcript(
            transcript,
            claimed_model=claimed_model_override,
            provider_hint=provider_hint_override,
        )
        _write_stdout(
            payload,
            output_format=args.format,
            indent=args.indent,
            text_renderer=render_chat_self_test_score_text,
        )
        return 0

    if not args.base_url or not args.api_key or not args.model:
        parser.error("--base-url, --api-key, and --model are required.")

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
    _write_stdout(payload, output_format=args.format, indent=args.indent, text_renderer=_render_text)
    return 0


def _write_stdout(
    payload: dict[str, Any],
    output_format: str,
    indent: int,
    text_renderer: Any,
) -> None:
    if output_format == "json":
        _safe_stdout_write(json.dumps(payload, ensure_ascii=False, indent=indent) + "\n")
        return
    _safe_stdout_write(text_renderer(payload))


def _safe_stdout_write(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(text.encode("utf-8"))
        else:
            fallback = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8",
                errors="replace",
            )
            sys.stdout.write(fallback)


def _validate_mode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    selected = sum(
        bool(option)
        for option in (
            args.self_test,
            args.emit_chat_self_test_pack,
            args.score_chat_self_test,
        )
    )
    if selected > 1:
        parser.error("--self-test, --emit-chat-self-test-pack, and --score-chat-self-test are mutually exclusive.")


def _stderr_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _build_cli_payload(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    run_meta = result["run_meta"]
    text_payload = {
        "input": {
            "base_url": args.base_url,
            "claimed_model": args.model,
            "provider_hint": args.provider_hint,
            "mode": args.mode,
        },
        "runtime": {
            "adapter_name": run_meta.get("adapter_name"),
            "dialect": run_meta.get("dialect"),
            "resolved_chat_endpoint": run_meta.get("resolved_chat_endpoint"),
            "resolved_models_endpoint": run_meta.get("resolved_models_endpoint"),
        },
        "decision": {
            "verdict_label": summary["verdict_label"],
            "confidence_level": summary["confidence_level"],
            "observed_model_hints": summary.get("observed_model_hints", []),
            "weak_model_hints": summary.get("weak_model_hints", []),
            "candidate_probabilities": summary["candidate_probabilities"],
            "hypothesis_ranking": summary.get("hypothesis_ranking", []),
            "model_candidate_ranking": summary.get("model_candidate_ranking", []),
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
    return {
        **text_payload,
        "text": _render_text(text_payload),
    }


def _render_text(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    runtime = payload.get("runtime", {})
    lines = [
        "API Model Forensics CLI",
        f"Model: {payload['input']['claimed_model']}",
        f"Mode: {payload['input']['mode']}",
        f"Adapter: {runtime.get('adapter_name', 'unknown')}",
        f"Dialect: {runtime.get('dialect', 'unknown')}",
        f"Verdict: {decision['verdict_label']}",
        f"Confidence: {decision['confidence_level']}",
        "",
        "Decision hypotheses:",
    ]
    for item in decision.get("hypothesis_ranking", []):
        lines.append(f"- {item['label']}: {item['probability']:.1%}")
    lines.extend(["", "Model candidates:"])
    for item in decision.get("model_candidate_ranking", [])[:5]:
        lines.append(f"- {item['name']}: {item['probability']:.1%} ({item['kind']})")
    observed_hints = decision.get("observed_model_hints", [])
    if observed_hints:
        lines.extend(["", "Observed catalog hints:"])
        for hint in observed_hints[:5]:
            lines.append(f"- {hint}")
    weak_hints = decision.get("weak_model_hints", [])
    if weak_hints:
        lines.extend(["", "Weak self-claimed hints:"])
        for hint in weak_hints[:5]:
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
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
