# Direct Chat Self-Test Protocol

This protocol is for testing the currently active conversational AI directly, without sending requests to an external API target.

## Why This Exists

The main detector in this repository audits an API endpoint. That covers transport shape, exposed model ids, token accounting, and routing clues.

Sometimes the question is different:

- "What model is this chat actually running on right now?"
- "Did the relay claim one model name but hand me another?"
- "Can I make the current AI answer a standard probe pack and save the transcript?"

This protocol exists for that second layer.

## What It Does

The protocol exports a versioned prompt pack and a transcript schema.

A caller, script, or future skill can:

1. Ask the currently active AI to answer each probe.
2. Save the exact raw answers in the transcript JSON format.
3. Feed that transcript back into the CLI scorer.

The scorer then reports:

- candidate probabilities and top candidates
- exactness / behavior / stability-derived evidence
- capability-tier evidence from higher-constraint probes
- self-reported provider/model hints
- a limited verdict based on direct-chat evidence only

## What It Does Not Do

This protocol does not inspect:

- `/models`
- token usage accounting
- native tool-call wire format
- provider-specific transport quirks

Because of that, it should be treated as supporting evidence, not as final proof of exact base-model identity.

## CLI Interfaces

Export the prompt pack:

```powershell
python -m app.cli --emit-chat-self-test-pack --model gpt-5.4 --format json
```

Score a captured transcript:

```powershell
python -m app.cli --score-chat-self-test path\to\transcript.json --model gpt-5.4 --format text
```

## Transcript Schema

Minimal shape:

```json
{
  "protocol_version": "conversation-selftest-v2",
  "pack_id": "conversation-selftest-v2",
  "claimed_model": "gpt-5.4",
  "claimed_provider_hint": "openai",
  "source_kind": "conversation_host",
  "host_context": {
    "host_name": "codex",
    "surface_name": "desktop-chat",
    "claimed_runtime_label": "gpt-5.4"
  },
  "collected_at": "2026-03-27 19:30:00",
  "cases": [
    {
      "probe_id": "C01",
      "title": "Self-Report Hint Probe",
      "parser": "self_report_json",
      "repeat_index": 1,
      "prompt_text": "Return valid JSON only with keys provider_guess, family_guess, model_guess, confidence. Use null when unsure. Do not add any extra text.",
      "response_text": "{\"provider_guess\":\"openai\",\"family_guess\":\"openai\",\"model_guess\":\"gpt-5.4\",\"confidence\":\"medium\"}",
      "metadata": {}
    },
    {
      "probe_id": "P04",
      "title": "Strict JSON Probe",
      "parser": "strict_json",
      "repeat_index": 1,
      "prompt_text": "Return exactly this JSON object and nothing else: {\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
      "response_text": "{\"status\":\"ok\",\"digits\":[1,2,3],\"lang\":\"zh\"}",
      "metadata": {}
    }
  ]
}
```

Compatibility note:

- `conversation-selftest-v2` is the current default pack emitted by the CLI.
- The scorer still accepts legacy `conversation-selftest-v1` transcripts.
- The scorer treats `cases` as the canonical transcript field.
- For easier skill integration, it also accepts a flat `responses` array when each item includes `probe_id`, `repeat_index`, and `response` or `response_text`.

## Stable Extension Points

These are the intended long-term boundaries:

- `app/models/conversation_selftest_pack_v1.json`
  - legacy direct-chat pack kept for backward compatibility
- `app/models/conversation_selftest_pack_v2.json`
  - current direct-chat pack with stronger same-family tier probes
- `app/models/probes.json`
  - reusable parser-backed probe specs used by both API probing and direct-chat scoring
- `app/chat_selftest.py`
  - direct-chat pack export and transcript scoring
- `app/selftest.py`
  - offline detector regression suite
- `app/cli.py`
  - public entrypoints only

Keep the direct-chat protocol separate from API runner logic unless a real shared scoring primitive is proven stable.

## Future Skill Integration

A future skill should:

1. Export the pack or embed the same probe order.
2. Run probes against the current host AI.
3. Preserve exact raw answers.
4. Emit transcript JSON in the schema above.
5. Optionally call the CLI scorer to produce a final summary.

That keeps the protocol public, the scorer local, and the host-specific orchestration outside the stable detector core.
