# Fork 02 — Polylogue semantic transcript launch slice

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, particularly `polylogue-ap7`. Implement the smallest high-impact semantic transcript slice that makes Polylogue visibly different from a chat viewer.

## Goal

A reader should see agent work, not serialized provider payloads. The launch slice must render at least these constructs through one shared intermediate contract:

1. shell command and structural result;
2. file edit or patch;
3. lineage/subagent relationship;
4. generic unknown-tool fallback.

Terminal and web surfaces must consume the same semantic card model rather than independently guessing from provider JSON.

## Owned scope

Own the semantic rendering contract, renderer registry/intermediate model, terminal renderer, focused web renderer integration, and their snapshots/tests. Avoid broad changes to ingestion, provider parsers, query grammar, storage, or demo fixture definitions unless a tiny fixture addition is indispensable and documented.

Coordinate with the demo-fixture lane by publishing the exact card inputs you need. Do not edit the same fixture files unless explicitly necessary.

## Design requirements

- Derive cards from normalized content blocks/actions, not prose regexes.
- Preserve exact evidence refs and source-order information.
- Render failure from structural fields such as nonzero exit status or provider `is_error`.
- Distinguish call, result, duration, status, and omitted output.
- Fold large output without hiding the existence or size of omitted material.
- Show file paths under the repository’s privacy/rendering policy.
- A lineage card must link parent/fork/subagent refs without treating domain topology as derivation provenance.
- Unknown tools must remain fully inspectable through a generic typed fallback.
- Terminal and HTML output must be deterministic for snapshots.
- The semantic model must be serializable so future MCP/web clients can consume it.

## Launch acceptance story

The deterministic “Receipts” session should render approximately as:

```text
$ pytest tests/missing_test.py
FAILED · exit 4 · 0.8s
ERROR: file or directory not found: tests/missing_test.py

Assistant response
I hit an error and need the missing path corrected before continuing.
```

The key claim is not styling. It is that the result is grounded in a typed tool-result block and retains a resolvable ref.

## Validation

- focused contract and snapshot tests;
- one cross-surface parity test over the same semantic model;
- unknown-tool fallback test;
- structural-failure test where prose does not contain the word “error”;
- output-folding boundary test;
- relevant MyPy/Ruff checks;
- one generated public-safe screenshot or text artifact.

## Deliverables

Produce a patch, semantic-card schema/example packet, before/after screenshots or terminal captures, test output, and a short note describing the next card types that remain outside the launch slice.

Favor a coherent vertical slice over broad partial support.
