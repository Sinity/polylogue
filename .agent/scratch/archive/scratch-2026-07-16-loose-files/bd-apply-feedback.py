#!/usr/bin/env python3
"""Apply verified gpt-pro-feedback corrections + DR2 report digests. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    line = (r.stdout or r.stderr).strip().splitlines()
    print(("OK  " if r.returncode == 0 else "FAIL") + " bd " + " ".join(args[:2]), "|", line[0][:120] if line else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:300])


# --- 1. cpf.6: corrected temporal diagnosis (verified live 2026-07-06) ---
bd(
    "update",
    "polylogue-cpf.6",
    "--title",
    "Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit",
    "--description",
    "(1) core/dates.py:37 sets RELATIVE_BASE = datetime.now(tz=utc) PER CALL inside parse_date "
    "(verified live 2026-07-06 — the earlier 'frozen at import' claim was wrong; a long-lived daemon "
    "does NOT drift). The real defect: relative-date parsing has no clock seam, so frozen_clock cannot "
    "reach it, since:7d is untestable deterministically, and query-time now() is uncontrolled. "
    "Fix = route parse_date + query lowering through a core/clock.py seam. "
    "(2) sort_key_ms COALESCE(...,0): SOME read paths epoch-pin timeless sessions; others handle NULL "
    "explicitly — this needs a targeted audit of every ordering/window path (classify each: fixed / safe / "
    "intentionally synthetic), not a blanket claim. Timeless sessions excluded from lower-bounded timed "
    "windows by default but reachable via include_timeless with explicit time_confidence. "
    "The wider four-time-kinds doctrine lives in the cpf epic; this bead is the clock seam + the audit + "
    "the two concrete fixes.",
    "--acceptance",
    "parse_date and query lowering accept an injected clock; since:7d under frozen_clock is deterministic "
    "and shifts only with the injected clock; no direct datetime.now in query-time parsing outside the seam "
    "(lint or grep gate); audit table enumerates every sort_key_ms/COALESCE ordering+window path with a "
    "fixed/safe/synthetic verdict; timeless sessions appear with time_confidence=synthetic instead of "
    "vanishing or pinning to 1970. Verify: focused date/query tests + the audit artifact.",
)

# --- 2. l4kf.2: replace stale raw_id/source_path hazard with real union semantics ---
bd(
    "update",
    "polylogue-l4kf.2",
    "--description",
    "WHY: local-first peers (second machine, trusted collaborator) should discover and exchange archive "
    "slices without a cloud service. ENABLES: cross-machine sync (durable tiers only: source content-hash "
    "union is idempotent+commutative; user assertions natural-key LWW; derived rebuilds on the peer), "
    "selective sharing, and the CIF envelope as the wire format. "
    "SYNC HAZARDS (corrected 2026-07-06 against live source — raw_id is already the blob content hash, "
    "acquisition_records.py:make_raw_record, so the old 'raw_id embeds source_path' claim is stale): "
    "(a) acquisition provenance must be a multimap — same bytes acquired on two machines = one raw blob "
    "identity, N acquisition observations (machine, source_path, mtime); "
    "(b) session identity can still collide via origin:native_id (incl. non-injective origin mapping) — "
    "different bytes with the same origin:native_id across machines must produce an explicit "
    "conflict/quarantine state, never a silent overwrite. "
    "Vision-tier: no full AC until the export origin lands and a second machine exists in the loop, but "
    "the two fixtures above are the acceptance sketch.",
)

# --- 3. 37t.15: priority bump + rationale ---
bd("update", "polylogue-37t.15", "--priority", "1")
bd(
    "update",
    "polylogue-37t.15",
    "--append-notes",
    "2026-07-06 priority P2->P1 (gpt-pro feedback concurs with internal read): nearly every frontier lane "
    "assumes agent-authored content cannot become operator-grade memory by accident — context scheduler "
    "(37t.11), recall (37t.20), distillery (37t.21), standing-query findings (rxdo.5), annotation import "
    "(rxdo.7), coordination/blackboard writes. The invariant lives INSIDE upsert_assertion (one chokepoint): "
    "author_kind != user => status=CANDIDATE + context_policy.inject=false; terminal judged rows must not be "
    "resurrected by later agent writes. Blocks-edges added accordingly.",
)

# --- 4. at44: guardrail against flat global KV ---
bd(
    "update",
    "polylogue-at44",
    "--append-notes",
    "2026-07-06 guardrail (gpt-pro feedback, accepted): even the liveness slice must not create a free-form "
    "global KV — define a typed registry of allowed setting keys from day one (subscription_tier first), "
    "partition deployment secrets OUT (they stay env/agenix, never user.db), and leave scope layering "
    "(global/repo/origin/surface) + winning-layer resolver explain to the w8db epic as designed. The failure "
    "mode to avoid: user_settings reborn as an untyped junk drawer, recreating the dead-table problem "
    "one level up.",
)

# --- 5. 4822: reword away from stale async-only/130-methods framing ---
bd(
    "update",
    "polylogue-4822",
    "--description",
    "Problem: downstream consumers (Lynchpin is the live example — raw sqlite + reimplemented models + a "
    "stale FROM conversations query) bypass the Python facade because the public boundary is broad, "
    "unversioned, and unstable — every internal module is reachable and nothing distinguishes supported "
    "surface from implementation detail. (Reworded 2026-07-06: the earlier framing 'async-only facade, "
    "130 methods' overstated method count and misplaced the gap — sync-vs-async is secondary; the missing "
    "thing is a small, stable, versioned boundary.) Extract polylogue.sdk (curated ~20-verb surface) + "
    "polylogue.models (frozen DTO re-exports), a sync wrapper over the async core, schema pin-and-warn, "
    "and a layering lint. Named risk: do NOT freeze origin vocabulary mid-retirement (provider->origin in "
    "progress) — the SDK speaks origin, gates provider behind the transitional shim.",
    "--acceptance",
    "Explicit public __all__ on polylogue.sdk + polylogue.models; stable DTO namespace with frozen models; "
    "capability/schema-version check API (consumer can ask: does this archive support X, which index "
    "version); SDK covers Lynchpin usage and Lynchpin drops its raw-sqlite path; layering lint forbids "
    "internal imports; examples import only the public namespace. Verify: SDK contract tests + layering lint.",
)

# --- 6. t46.8: shadow telemetry before deletion ---
bd(
    "update",
    "polylogue-t46.8",
    "--acceptance",
    "Verb set + resources + prompts cover every retired tool proven by goldens; EXPECTED_TOOL_NAMES shrinks "
    "with equivalence evidence per deletion; discovery tests + contracts regenerated; no capability "
    "regression reported by the golden suite. SHADOW TELEMETRY GATE (added 2026-07-06): before any tool "
    "deletion, a shadow-mode window records per-tool called-count by client/harness, mapped replacement "
    "verb/resource/prompt, golden parity status, and last-seen timestamp — deletion order follows observed "
    "compatibility, not design purity alone (MCP clients may have prompts/learned behavior keyed to old "
    "tool names). Verify: golden equivalence suite + the shadow-usage report artifact.",
)

# --- 7. fnm.14: DTO separation note ---
bd(
    "update",
    "polylogue-fnm.14",
    "--append-notes",
    "2026-07-06 boundary note (gpt-pro feedback, accepted): keep CorpusCompactionPack and ContextImage as "
    "SEPARATE top-level DTOs — compile_context answers 'what should an agent get to continue/review' "
    "(current decisions, open loops); find|compact answers 'best evidence digest for an external analyst' "
    "(representative evidence, contradictions, error-fix paths, drop manifest). Scoring differs; sharing the "
    "top-level object would blur both. Shared helpers are fine: token estimation, refs, omission accounting, "
    "segment rendering. This distinction is the R and D flywheel hinge: compact pack -> external model -> "
    "browser capture -> imported annotations -> next pack.",
)

print("--- batch 1 done")
