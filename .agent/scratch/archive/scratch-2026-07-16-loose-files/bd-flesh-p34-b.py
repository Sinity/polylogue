#!/usr/bin/env python3
"""Flesh out thin P3/P4 beads, batch B: importers, DSL, research/audit. Target fields verified empty. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:110] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:250])


def label(bid, *labs):
    for lab in labs:
        subprocess.run(["bd", "label", "add", bid, lab], capture_output=True, text=True)


bd(
    "update",
    "polylogue-611",
    "--design",
    "The plumbing already reserves the seat (verified live): Origin.GROK_EXPORT in core/enums.py:50, "
    "source family grok-export in core/sources.py:126, provider_identity maps xai->grok. Missing: a "
    "detector in sources/dispatch.py detect_provider() inserted at the tightness level the format "
    "deserves (likely loose dict-key check alongside chatgpt/claude-web — get a real export first, "
    "do NOT guess the shape) + a parser module sources/grok.py + fixtures under tests. Precondition: "
    "obtain a Grok GDPR/export sample; encode its actual shape as a Pydantic record check if tight "
    "enough, else dict-key. Detector-order trap: an earlier looser parser can claim its records.",
    "--acceptance",
    "A real Grok export ingests end-to-end (sessions/messages/blocks with native ids); detector "
    "fixture proves no other parser claims it and it claims no other fixture; parser props test "
    "added (protected file family). Verify: devtools test -k grok + the detector-matrix test.",
)
label("polylogue-611", "horizon:mid")

bd(
    "update",
    "polylogue-fs1.8",
    "--design",
    "Browser-capture adapter for Nous Chat (chat.nousresearch.com): a site adapter in "
    "browser-extension/ (MV3; existing chatgpt-dom adapter is the template) mapping the DOM/fetch "
    "shapes to the capture envelope, + origin routing so captured payloads land as hermes-session "
    "or a dedicated origin (decide with the fs1 epic owner — Hermes state.db import (fs1.1, done) "
    "may make DOM capture redundant except for web-only usage).",
    "--acceptance",
    "A live Nous Chat session captures end-to-end into the archive with correct origin + native "
    "ids; duplicate-vs-state.db ingest is reconciled (content-hash idempotency, no double "
    "sessions). Verify: manual capture run + devtools test -k capture.",
)
label("polylogue-fs1.8", "horizon:vision")

bd(
    "update",
    "polylogue-wmj",
    "--design",
    "Export lane: archive sessions -> OTel GenAI semantic-convention spans (gen_ai.* attributes) "
    "so LangSmith/Langfuse/Phoenix-class consumers can read Polylogue evidence. Mapping: session -> "
    "trace, message/tool block pairs -> spans (actions view gives tool_use<->tool_result pairing), "
    "cost/token fields -> gen_ai.usage.*. Emit OTLP-file/JSONL first (no live exporter dependency); "
    "reuse the D07 doctrine: internal schema first, adapters second (fs1.10). Note ops.db already "
    "has an otlp table family — check before adding new plumbing.",
    "--acceptance",
    "polylogue export --format otel-genai-jsonl produces spans that pass an OTel GenAI "
    "semantic-convention validator for a sample session set; tool pairs land as parent/child "
    "spans; cost attributes present where evidence exists. Verify: validator run + devtools test "
    "-k otel.",
)
label("polylogue-wmj", "horizon:vision")

bd(
    "update",
    "polylogue-0dz",
    "--design",
    "Huge exports (multi-GiB Claude Code JSONL) already stream on INGEST; the READ side "
    "(read --all/session dumps, web payloads) still materializes whole sessions. Add a chunked "
    "read-package layout: manifest + segment files at block ranges, byte-budgeted, so surfaces "
    "can page. Anchors: polylogue/surfaces/payloads.py (payload assembly), read_view_handlers.py "
    "(CLI), daemon/http.py session routes (web paging params exist? verify). Fits the "
    "CompactProjectionSpec family (fnm) — a layout, not a new subsystem.",
    "--acceptance",
    "Reading the largest live session streams in bounded memory (measure RSS before/after); "
    "manifest + segments round-trip to identical content; web/CLI consume the same layout. "
    "Verify: devtools test -k package + an RSS spot-check on the known-largest session.",
)
label("polylogue-0dz", "horizon:mid")

bd(
    "update",
    "polylogue-pf1",
    "--design",
    "Twins: async lane storage/sqlite/async_sqlite*.py vs sync lane storage/sqlite/archive_tiers/. "
    "Method: extract per-lane surface inventories (method name, SQL statements touched, tables "
    "written) via AST + SQL-string parse; diff into three classes — identical, intentionally "
    "async-only/sync-only, DIVERGENT (same table+intent, different SQL/semantics). Divergent rows "
    "become bugs; the artifact becomes a regression fixture so new divergence fails a test "
    "(the standing STORAGE TWINS trap made mechanical). Feeds a7xr (consolidation epic).",
    "--acceptance",
    "A committed twin-diff artifact classifies every write-path method; zero unexplained "
    "divergences (each is fixed or has an explicit rationale row); a test regenerates the diff "
    "and fails on new divergence. Verify: devtools test -k twin.",
)
label("polylogue-pf1", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.14",
    "--design",
    "polylogue/api/archive.py is 5391 lines (verified). Method: enumerate its public methods; for "
    "each, classify consumer (CLI/MCP/daemon/tests-only/none) via rg call census; map each to its "
    "repository-mixin home (storage/repository/ is already 10 mixins). Output: a decomposition "
    "table — keep-on-facade / move-to-mixin / deprecate — that 4822 (SDK boundary) consumes as its "
    "curation input. This is the measurement half; 4822 owns the cut.",
    "--acceptance",
    "Committed table covers 100% of the facade's public methods with consumer counts + verdicts; "
    "tests-only and zero-consumer methods explicitly listed (candidates for deletion). Verify: the "
    "census script re-run is clean vs the table.",
)
label("polylogue-9e5.14", "horizon:frontier")

bd(
    "update",
    "polylogue-fnm.8",
    "--design",
    "Anchor: archive/query/expression.py — pipeline stages are hand-parsed OUTSIDE the Lark "
    "grammar (split on |, _parse_pipeline_unit_source ~L1949), so logical: needs no grammar "
    "change if implemented as a predicate-expansion pass; if it becomes a TERMINAL containing "
    "':', it must slot above FIELD_CLAUSE.4 priority or it is eaten as a field clause (standing "
    "LALR trap). Semantics: logical:SESSION expands to the lineage closure (parent-prefix + "
    "divergent tails via session_links recomposition) BEFORE SQL lowering, and read paths "
    "dedupe replayed prefixes (the 2/5-read-paths composition gap #2470 is the cautionary tale).",
    "--acceptance",
    "logical:REF in any predicate position returns the recomposed logical session set; replayed "
    "prefix messages are not double-counted in downstream aggregates; grammar tests cover the "
    "terminal-priority trap. Verify: devtools test -k 'logical or lineage' + one live fork-family "
    "query.",
)
label("polylogue-fnm.8", "horizon:mid")

bd(
    "update",
    "polylogue-fnm.9",
    "--design",
    "Pipeline-as-subquery: allow a pipeline result to feed an outer expression (sessions where id "
    "in (<pipeline>) or from result-set/query refs per rxdo.6). Since stages are hand-parsed "
    "outside Lark (expression.py split-on-|), subquery composition is an AST-level substitution: "
    "lower the inner pipeline to a SQL CTE or materialized id-set, then bind as an operand. "
    "Decide with rxdo.2/rxdo.6: content-addressed query identity may make this 'from query:HASH' "
    "instead of inline nesting — prefer the ref form, it gets provenance for free.",
    "--acceptance",
    "An inner pipeline's session/unit set is consumable as an outer predicate operand (inline or "
    "via query:/result-set: ref); provenance records the composition; no quadratic re-execution "
    "(inner runs once). Verify: devtools test -k subquery + explain output showing the CTE/ref.",
)
label("polylogue-fnm.9", "horizon:vision")

print("--- batch B done")
