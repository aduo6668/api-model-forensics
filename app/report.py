from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import fitz

from .config import REPORT_FOOTER_TEXT, resolve_font_path


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 48
LINE_HEIGHT = 18


def generate_pdf_report(output_path: Path, summary: dict[str, Any], results: list[dict[str, Any]], run_meta: dict[str, Any]) -> Path:
    doc = fitz.open()
    font_path = resolve_font_path()
    font_kwargs = {"fontfile": str(font_path), "fontname": "custom"} if font_path else {"fontname": "helv"}

    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN
    y = _write_heading(page, y, "API Model Forensics Report", 20, **font_kwargs)
    y = _write_paragraph(page, y, f"Claimed model: {run_meta['claimed_model']}", **font_kwargs)
    y = _write_paragraph(page, y, f"Target API: {run_meta['target_base_url']}", **font_kwargs)
    y = _write_paragraph(page, y, f"Mode: {run_meta['mode']}", **font_kwargs)
    y = _write_paragraph(page, y, f"Run ID: {run_meta['run_id']}", **font_kwargs)
    y = _write_paragraph(page, y, f"Verdict: {summary['verdict_label']}", **font_kwargs)
    y = _write_paragraph(page, y, f"Confidence: {summary['confidence_level']}", **font_kwargs)
    y += 8

    y = _write_heading(page, y, "Probability Summary", 14, **font_kwargs)
    y = _write_paragraph(page, y, f"- Claimed model: {summary['candidate_probabilities']['claimed_model_probability']:.1%}", **font_kwargs)
    y = _write_paragraph(page, y, f"- Same-family downgrade: {summary['candidate_probabilities']['same_family_downgrade_probability']:.1%}", **font_kwargs)
    y = _write_paragraph(page, y, f"- Alternative family: {summary['candidate_probabilities']['alternative_family_probability']:.1%}", **font_kwargs)
    y = _write_paragraph(page, y, f"- Wrapped / unknown: {summary['candidate_probabilities']['wrapped_or_unknown_probability']:.1%}", **font_kwargs)

    y += 8
    y = _write_heading(page, y, "Top Candidates", 14, **font_kwargs)
    for item in summary["top_candidates"]:
        y = _write_paragraph(page, y, f"- {item['name']}: {item['probability']:.1%} ({item['kind']})", **font_kwargs)
        y = _write_paragraph(page, y, f"  rationale: {item['rationale']}", fontsize=10, **font_kwargs)

    y += 8
    y = _write_heading(page, y, "Evidence Breakdown", 14, **font_kwargs)
    for evidence in summary["evidence_breakdown"]:
        y = _write_paragraph(page, y, f"- {evidence['label']}: {evidence['score']:.3f}", **font_kwargs)

    y += 8
    y = _write_heading(page, y, "Decision Notes", 14, **font_kwargs)
    y = _write_paragraph(page, y, f"- Primary: {summary['primary_reason']}", **font_kwargs)
    y = _write_paragraph(page, y, f"- Secondary: {summary['secondary_reason']}", **font_kwargs)

    y += 8
    y = _write_heading(page, y, "Primary Caveats", 14, **font_kwargs)
    for caveat in summary["primary_caveats"]:
        y = _write_paragraph(page, y, f"- {caveat}", **font_kwargs)

    _write_footer(page, REPORT_FOOTER_TEXT, **font_kwargs)

    page2 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y2 = MARGIN
    y2 = _write_heading(page2, y2, "Probe Results", 18, **font_kwargs)
    for result in results:
        line = (
            f"[{result['probe_id']}] {result['probe_name']} | group={result['probe_group']} | "
            f"score={result['score']:.3f} | latency={result.get('latency_total_ms')}ms | "
            f"finish={result.get('finish_reason')}"
        )
        y2 = _write_paragraph(page2, y2, line, **font_kwargs)
        if result.get("details"):
            y2 = _write_paragraph(page2, y2, f"  details: {result['details']}", fontsize=10, **font_kwargs)
        if y2 > PAGE_HEIGHT - 120:
            _write_footer(page2, REPORT_FOOTER_TEXT, **font_kwargs)
            page2 = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            y2 = MARGIN

    _write_footer(page2, REPORT_FOOTER_TEXT, **font_kwargs)
    doc.save(output_path)
    doc.close()
    return output_path


def _write_heading(page: fitz.Page, y: int, text: str, fontsize: int, **font_kwargs: Any) -> int:
    page.insert_text((MARGIN, y), text, fontsize=fontsize, **font_kwargs)
    return y + int(fontsize * 1.8)


def _write_paragraph(page: fitz.Page, y: int, text: str, fontsize: int = 11, **font_kwargs: Any) -> int:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width=68, break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped or [""])
    for line in lines:
        page.insert_text((MARGIN, y), line, fontsize=fontsize, **font_kwargs)
        y += LINE_HEIGHT if fontsize >= 11 else 15
    return y + 2


def _write_footer(page: fitz.Page, text: str, **font_kwargs: Any) -> None:
    lines = textwrap.wrap(text, width=80, break_long_words=False, replace_whitespace=False)
    y = PAGE_HEIGHT - 64
    for line in lines:
        page.insert_text((MARGIN, y), line, fontsize=9, **font_kwargs)
        y += 12
