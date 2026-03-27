# API Model Forensics Maintenance Plan

This document defines how the public repository evolves without losing stability, reproducibility, or local-only privacy.

## Goals

- Keep the existing detection stack usable while new providers and model families are added.
- Make incremental onboarding cheap enough that we can adapt to new model releases without reworking the whole system.
- Preserve compatibility for existing users, baselines, reports, and CLI automation.
- Keep all sensitive or user-owned data local by default.

## Maintenance Model

The project uses three layers of change:

- `Stable core`
  - `runner`, `scoring`, `storage`, and report generation.
  - These should change slowly and only when a structural bug or schema change requires it.
- `Extension layer`
  - provider adapters, family registry entries, probe specs, and baseline datasets.
  - These are the main place for growth.
- `Protocol layer`
  - direct-chat self-test packs, transcript schema, and skill-facing orchestration contracts.
  - Keep this separate from API transport auditing so host-AI workflows can evolve without destabilizing the main detector.
- `Local-only layer`
  - `.env`, run outputs, local baselines, paper corpora, and user testing notes.
  - These never need to be committed to the public repository.

## Maintenance Windows

Recommended cadence:

- `Weekly window`
  - Small maintenance pass for provider quirks, probe tuning, and regression checks.
  - Suggested time: one short slot per week, preferably during a low-traffic period.
- `Monthly window`
  - Registry refresh, new model-family review, baseline review, and deprecation review.
  - Use this window for additive updates rather than redesign.
- `Hotfix window`
  - Open only for breakage in CLI, GUI launch, core scoring, or report generation.
  - Hotfixes should be minimal and should not bundle unrelated feature work.

Maintenance windows should be short, predictable, and explicit. The goal is to avoid long-lived unreviewed drift in the public repo.

## Incremental Onboarding

When a new vendor or model family appears, follow this sequence:

1. Add a registry entry for the new family.
2. Add or extend a provider adapter skeleton if the API dialect differs.
3. Define the observable signals worth tracking.
4. Add a small probe set or reuse existing probes with provider-specific mapping.
5. Collect a local baseline from the first-party or best available reference endpoint.
6. Compare the new traces against existing families.
7. Update the README and any public-facing usage notes if the workflow changes.

Keep the first version small. The goal is to identify the family and obvious downgrade or routing anomalies first, then improve exact-model ranking later.

Suggested onboarding checklist:

- Start from the smallest observable change that still distinguishes the new family.
- Prefer additive registry entries over rewriting old ones.
- Add aliases only when they are supported by repeatable traces or official naming.
- Keep any newly discovered quirks in the adapter or registry, not in ad hoc notes.
- Ship the first pass behind a maintenance window so the regression suite can be re-run immediately.

## Direct-Chat Self-Test Maintenance

The direct-chat self-test layer should evolve conservatively.

- Reuse parser-backed probes from `app/models/probes.json` when possible instead of inventing a second parallel prompt library.
- Keep host-AI prompt packs and transcript scoring in a dedicated protocol module.
- Treat self-reported model names as weak evidence.
- Do not merge direct-chat verdicts into API verdicts unless a shared evidence model is validated with baselines.
- When the protocol changes, version the pack id and keep old transcript scoring paths readable for at least one maintenance cycle.

## Baseline And Regression

Every meaningful change to probes, registry, adapters, or scoring should be checked against a small baseline suite.

Baseline rules:

- Keep a canonical baseline set for each supported family.
- Re-run the same probe suite after every non-trivial change.
- Compare the new outputs to prior outputs, not just to human intuition.
- Track both confidence and top-candidate drift.
- Treat `ambiguous` as a valid result when the evidence is noisy.
- Prefer reference baselines from the best available first-party endpoint, then keep a local comparison set for future changes.

Regression gates:

- CLI still prints valid JSON when `--format json` is used.
- GUI still launches and runs a probe suite.
- PDF report still renders.
- Existing family rankings do not regress unexpectedly.
- Local-only files are still ignored.
- Token usage remains within the current budget profile unless there is a documented reason to expand the probe set.

## Registry Update Rules

The registry is the source of truth for candidate families and their observable traits.

Update principles:

- Prefer additive changes over destructive edits.
- Treat `provider`, `family`, `canonical ids`, `aliases`, `dialect`, and `known quirks` as explicit fields.
- Keep observed data separate from guessed data.
- Preserve old names until the deprecation window expires.
- Do not collapse unrelated vendors into a single generic family just because they share OpenAI-compatible transport.

When a new model is found inside an existing family, add it as a new canonical candidate or alias entry instead of overwriting the family entry.

## Compatibility And Deprecation

Compatibility is part of the product.

- Existing CLI flags should remain valid unless there is a strong reason to change them.
- Existing report schema fields should be kept stable whenever possible.
- New provider-specific fields should be additive.
- If a feature needs to be removed, mark it deprecated for at least two release cycles before deletion.
- If a provider changes a field name or response shape, keep a compatibility shim before removing the old path.

Recommended deprecation flow:

1. Mark the feature or field as deprecated in docs.
2. Keep the old behavior working.
3. Add the new behavior in parallel.
4. Collect one or more regression runs.
5. Remove the old path only after the replacement is stable.

## Local-Only Data Policy

The following data must stay local unless the user explicitly decides otherwise:

- API keys and environment overrides.
- Raw request and response logs.
- Run folders in `outputs/runs/`.
- Generated PDF reports in `outputs/reports/`.
- Local baseline traces.
- Paper PDFs and extracted text corpora.
- Internal design notes and experiment memos.

The public repository should only contain the product code, public documentation, and lightweight reference files that are safe to share.

Local-only storage is part of the product promise, not just a convenience. If data is not needed for public collaboration or user-facing output, it should stay on the user machine.

## Release Checklist

Before publishing a new public release, verify:

- CLI and GUI both still work.
- New provider or family entries are documented.
- Baseline comparisons have been re-run.
- Deprecations are documented.
- No local-only data was added accidentally.
- README still reflects the current user workflow.
- Any new provider or family entry has a matching regression note or baseline refresh entry.

## Practical Rule

If a change helps future maintenance but does not help the current user path, keep it small or defer it. The project should stay easy to extend, but not become hard to ship.
