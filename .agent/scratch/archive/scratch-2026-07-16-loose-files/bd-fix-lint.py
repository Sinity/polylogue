#!/usr/bin/env python3
"""Fix bead-lint findings: H4 design fields + E1/E2 epic gaps. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:130] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:300])


# ---------- H4: design fields naming exact code surfaces ----------
bd(
    "update",
    "polylogue-kwsb.1",
    "--design",
    "All three holes live in polylogue/daemon/http.py: the Origin check exists only on the POST path "
    "(~L1305 headers.get Origin, skipped when absent) while GET routes (_static_get_routes ~L228, "
    "_parameterized_get_routes ~L257) have no Host/Origin gate at all — that is the DNS-rebinding read "
    "hole. Fix shape: one request-admission gate applied to EVERY route before dispatch — Host allowlist "
    "(127.0.0.1/localhost + configured), Origin required-and-matched on state-changing routes, "
    "capability token for the browser-capture receiver POSTs (_authenticated_post_routes ~L321 is the "
    "seam), and a spool-size governor in the receiver path (polylogue/daemon/browser_capture.py + spool "
    "writer) so a hostile page cannot disk-fill. Pitfall: the dev-loop and MCP localhost clients must "
    "keep working — gate by route class, not blanket; add regression tests per hole (rebinding GET, "
    "absent-Origin POST, spool flood).",
)

bd(
    "update",
    "polylogue-9e5.29",
    "--design",
    "Anchor files: polylogue/insights/rigor.py (RigorContract ~L45, RigorVersionField ~L37), "
    "polylogue/insights/audit.py (insight_rigor_audit surface), polylogue/insights/confidence.py. "
    "Add a field-level RigorFieldContract: for each quantitative field of an insight payload declare "
    "provenance class (counted/derived/estimated), the evidence query or reducer that grounds it, and "
    "nullable_when_ungrounded=True so an empty backing frame renders None/uncovered — never 0.0. "
    "Wire: registry descriptors (insights/registry.py) declare field contracts; the rigor audit "
    "enumerates fields lacking contracts; renderers treat None as uncovered, not zero. Start with the "
    "worst offenders: any field the audit currently shows emitting 0.0 over empty rows. Pitfall from "
    "notes: field paths must resolve to block+json-path+reducer+denominator or the bytes-resolution "
    "product promise narrows to block granularity — fold that dimension into the contract design.",
)

bd(
    "update",
    "polylogue-1xc.12",
    "--design",
    "Anchor files: polylogue/storage/sqlite/archive_tiers/index.py ~L307-318 (messages_fts_ai/ad/au "
    "triggers; threads_fts ~L449+), polylogue/daemon/fts_startup.py (startup repair), "
    "polylogue/storage/archive_readiness.py (current boolean readiness). Keystone identity: "
    "messages_fts.rowid == blocks.rowid == docsize.id, and SQLite ROWID REUSE means a ghost FTS row can "
    "silently attach to an unrelated reinserted block — drift checks must therefore compare block_id, "
    "not rowid existence. Deliverables: (1) drift GAUGES (counts of missing/ghost/mismatched rows) "
    "surfaced through readiness instead of a boolean; (2) metamorphic trigger-coherence tests: apply "
    "arbitrary insert/delete/update block mutations, assert FTS row set converges to exactly the "
    "search_text-bearing blocks (Hypothesis stateful fits; tests/infra strategies exist). Pitfall: FTS "
    "is contentless (content='') — you cannot SELECT text back; compare via docsize/rowid+blocks join.",
)

bd(
    "update",
    "polylogue-212.7",
    "--design",
    "Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, "
    "CURATED_CATALOG.md as the manifest seed). Contract: every demo directory gains PROMPT.md "
    "(executable instructions a coding agent runs cold) and emits an identical Demo Finding Packet: "
    "finding.yaml (five-part provenance stanza per 3tl.4), rendered artifact, and the exact "
    "reproduction commands. Build a registry manifest (extend CURATED_CATALOG.md or a demos.yaml) "
    "listing id, claim, packet path, substrate features exercised, last-regenerated. A prompt runner "
    "(thin script or devtools lab command) executes one demo prompt end-to-end and validates packet "
    "shape. Pitfall: demos run against the LIVE archive — packet outputs must be private-data-audited "
    "before any publication lane (3tl.4 owns publishing).",
)

bd(
    "update",
    "polylogue-212.8",
    "--design",
    "Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is "
    "not_supported. Pick the tempting claim: minute-by-minute multi-source operator reconstruction "
    "(needs modalities the archive lacks). The packet lists missing modalities, missing refs, and the "
    "exact query/evidence that WOULD support it, using the same finding.yaml shape. Anchor: "
    ".agent/demos/<new-dir>/ + the insight_rigor_audit surface to enumerate what evidence exists vs "
    "required. The success criterion is the refusal being specific, not vague: every missing item names "
    "the unit/table/modality that would have to exist.",
)

bd(
    "update",
    "polylogue-3tl.15",
    "--design",
    "Target: README.md (skeptic section) + docs site page. Ground in ONE regenerated finding from "
    ".agent/demos (agent-forensics or claim-vs-evidence packet) — the card shows the same question "
    "answered by grep over ~/.claude/projects vs by polylogue query: paired tool_use/tool_result via "
    "the actions view, tool_result_is_error/exit_code failure predicates, lineage recomposition "
    "(prefix-sharing means grep double-counts replayed parents), cost attribution, and the material_"
    "origin authoredness split that grep cannot see. Keep it one page, every number citing the packet. "
    "Blocked-by nothing once a current packet exists; regenerate via the 212.7 runner when it lands.",
)

# ---------- E1/E2: epic membership + descriptions ----------
bd(
    "update",
    "polylogue-jlme",
    "--description",
    "WHY: the MV3 browser-capture extension is a load-bearing acquisition surface (live chat capture "
    "feeding the spool/receiver) but had no owning epic: reliability, capture-state visibility, and "
    "in-page presence were scattered. ENABLES: trustworthy always-on capture (the R and D flywheel "
    "assumes browser sessions land in the archive without operator babysitting). MEMBER BEADS (grouped "
    "by design ref, not id-prefix): polylogue-3v1 (reliability/status surface), polylogue-3v1.1 "
    "(concurrent-instance safety), polylogue-90y (in-page overlay presence). Epic closes when the "
    "member set is closed and a capture-reliability finding exists (spool loss rate over a week of "
    "live use).",
)

bd(
    "update",
    "polylogue-w8db",
    "--description",
    "WHY: configuration semantics are scattered (env vars, hardcoded defaults, dead user_settings "
    "table) and nothing distinguishes deployment config from runtime preference from learned default. "
    "ENABLES: at44 liveness slice (subscription_tier), verb-behavior prefs, reading prefs, learned "
    "defaults via the judgment gate. MEMBER BEADS (grouped by design ref): polylogue-y4c (doctrine "
    "spine: prefs table in user.db + resolution order + Nix surface), polylogue-3xx (verb-behavior/ops "
    "prefs), polylogue-y8w (reading prefs), polylogue-6kh (query-scope prefs), polylogue-1jc (learned "
    "defaults as judged candidates). SEQUENCE: y4c first (defines table + resolver), bundles fill "
    "lanes, 1jc last. Guardrail shared with at44: typed key registry from day one, no free-form global "
    "KV; secrets stay out of user.db.",
)

bd(
    "update",
    "polylogue-9l5.18",
    "--append-notes",
    "2026-07-06 decomposition contract: this epic decomposes on claim — each of the six units "
    "(entity-mention, world-effect, verification-run, project, topic-cluster, cross-origin-thread) "
    "becomes a child bead inheriting its TABLE-vs-VIEW decision + extraction gating from this "
    "description; claiming agent creates the child, lifts the relevant desc slice into it, and "
    "executes per the enrich-on-claim convention. Do not implement units directly against this epic.",
)

# 7aw is one coherent capability, not a container - retype epic -> feature
bd("update", "polylogue-7aw", "--type", "feature")

print("--- lint fixes done")
