# Decision Brief — every murky/decision-shaped item, with my calls

Date: 2026-07-08. Prepared for one-sitting operator review.
Rule followed: I state **my decision** everywhere; alternatives appear only
where I am genuinely unsure mine is strictly better. Ratifying this brief
means each bead gets its decision recorded and execution proceeds without
further adjudication.

Legend: ✅ = already decided in the bead, I ratify and add nothing material;
🔷 = I made the call here; ⚠️ = genuinely unsure, alternative stated.

---

## 1. Architecture decisions

### 1.1 bby.11 — Webui v2 stack ✅ (ratify: TypeScript + Preact + Vite)

The design field's rationale is correct and complete: Preact = React idioms
(deepest agent training data) at 4KB; Vite dev-proxy; typed client generated
from the OpenAPI render; committed `dist/` so node never enters the deploy
chain; strangler migration at `/app`; command palette + deep-link routing as
foundation. I add three riders:

- **Testing is scaffold AC**: the 1ilk plan (vitest component lane per-PR,
  playwright e2e + visual regression on master/nightly) merges into the
  scaffold acceptance, not a follow-up.
- **Styling**: the scaffold's `lib/tokens.css` is *generated* from
  `theme.py` (9xuk), not hand-written — one token authority from day one.
- **Landing order** from the notes stands: block-hash substrate (svfj) and
  citation verifier precede the force-graph/report ambitions; scaffold +
  list/reader parity is the first PR series.

This is the highest-leverage ratification in the brief: it unblocks 1ilk,
bby.9/.10/.15, d4zk UX surfaces, semantic renderers, and gqx's widget feed.

### 1.2 4p1 — One read algebra ✅ (ratify; execution is now cheap)

Doctrine: every read surface lowers to QuerySpec × ProjectionSpec ×
RenderSpec; MCP tools / analyze modes / CLI views / web routes are **named
presets** (a registered (Q,P,R) triple); a new read affordance must be
expressible as a preset or the algebra gains the capability first; writes/ops
explicitly out of scope. Note: the conformance-inventory deliverable already
has a strong first draft (`.agent/scratch/new-gpt-pro/one-read-contract-cut.report.md`,
~40 anchored rows + 11 contract gaps + ordered collapse plan) — execution is
re-verifying anchors against master and landing the doc + manifest. 4p1.1
(daemon fast path through `from_params`) is not a decision, just overdue
dedup; execute as specced.

### 1.3 hiu — Storage twins ✅ (ratify direction B: sync core, async adapter)

Already operator-delegated and decided 2026-07-03; the rationale (aiosqlite
is a thread-queue fiction; sync store is the battle-tested 8.9k-line
implementation; per-thread read pool under WAL beats one serialized thread)
is correct. Sequencing stands: pf1 divergence reconciliation first, adapter
at repository-method granularity, mixin-by-mixin migration, ingest-throughput
gate per step. One consequence I lock in: **the runtime sync/async
differential harness idea stays rejected** — hiu makes it moot.

### 1.4 dx1 — Daemon HTTP substrate 🔷⚠️ (my call: migrate to ASGI —
Starlette + uvicorn — via the evidence-first ramp, with presumption to
proceed)

The bead frames three options (full ASGI / hybrid / stay + extract typed
routing). My decision: run the one-route-family prototype as designed, but
with a **presumption of migration** unless the probe finds a hard blocker
(latency/RSS regression, extension-receiver incompatibility). Reasons:

- Webui v2 makes push non-negotiable ("SSE-live everything", bby.4 tailing).
  Hand-rolled threaded SSE at 45+ routes is where the next bby.7-class bug
  family lives; Starlette gives typed params, middleware (auth/gzip/CORS for
  the extension), and native SSE/streaming.
- The agent-builder argument from bby.11 applies verbatim: Starlette/uvicorn
  is the deepest server-side training-data vein; a bespoke typed-routing
  layer over BaseHTTPRequestHandler is a private framework agents must learn.
- The zero-web-dep posture was right for a loopback archive daemon; the
  cockpit ambition changed the requirements. Two mature, local-only deps is
  the honest price.

⚠️ Genuine uncertainty: the "stay + typed route layer" option is defensible —
the current server *works at production load*, and migration risk across 45
routes + SPA + extension receiver is real. If you weight operational
stability over the cockpit timeline, choose the typed-layer option and
revisit at H-gate. I would not: it spends effort building the worst third of
a framework and still lacks push. Migration shape if ratified: new routes
(20d.1 fast path, webui v2 API) land on ASGI first; existing families move
one-per-PR behind unchanged contracts (/metrics, /healthz byte-stable);
old server dies when route count hits zero.

### 1.5 ze5 — user.db vocabulary ✅ (ratify the design's recommendation)

Four-class lens over the unified assertions table (epistemic | curation |
workspace | comms), class-appropriate surface nouns, table name unchanged;
additive migrations for relations (supersedes/contradicts/refines),
append-only revision history, optional confidence on epistemic kinds; judge
queue surfaces contradiction pairs. Explicit rejection of per-class table
split stands (hash boundary + audit surface are load-bearing). My one
refinement: the epistemic surface noun is **"notes"** externally and
"records" in API vocabulary — "assertion" survives only as the storage/enum
term, never user-facing.

### 1.6 ca4 — DuckDB OLAP ✅ (ratify the decision matrix; N-horizon, no pull-forward)

Probe as designed (attach live index.db read-only via sqlite_scanner, 5 real
heavy queries, parity + concurrency checks); adopt dual-lowering only at
10x+ wins, stay SQLite-only below 3x, and DuckDB is never a write path or
required dep. Between 3–10x: stay SQLite-only — the operational surface of a
second engine needs a decisive win, not a moderate one.

### 1.7 fie — Scaling doctrine 🔷 (my call: lever order fixed now, probes confirm)

Keep-everything is settled policy. My decision on lever order, independent of
probe outcomes: (1) **blob zstd (83u.5) proceeds regardless** — ~36GB→5-8GB
at zero loss is pure win with no doctrine tension; (2) hot/cold FTS sharding
second, only if FTS rebuild is what actually degrades; (3) **incremental
index maintenance is rejected by default** — it strains the fresh-first
derived-tier doctrine that keeps rebuild trustworthy (hjwr differential,
b5l blue-green). If the 10x-scale probe shows unacceptable rebuild windows,
the answer is blue-green rebuild windows (b5l) + sharding, not abandoning
rebuild-from-source semantics. The bead's flagged tension is thereby
resolved in favor of the doctrine.

### 1.8 c36 — mypyc probe ✅ (ratify: park until 20d.15, expect close-as-no)

The analysis in the bead is honest and almost certainly right (bulk time is
C-library-bound; predicted end-to-end ≤1.15x). One py-spy profile after
parallel parse lands; close-as-no with the profile attached unless transform
exceeds ~40% of wall.

### 1.9 e6ja — Zero-tests-pre-merge hole 🔷 (my call: option C now, A on trigger)

Attestation-first: the pre-push hook records verify tier + head; a CI check
makes "tests ran locally" legible instead of assumed. Zero CI compute,
honest, immediate. Escalation trigger pre-committed: **two post-merge
regressions within 30 days** → add option A (changed-package heuristic job,
10-min cap, non-required at first). Option B (testmon-in-CI) stays rejected
until 27rb (deadlock root-cause) lands.

---

## 2. Interaction/UX decisions (the two direction beads)

### 2.1 tjx1 — Aesthetics direction (sign-off requested)

The direction doc (`.agent/reports/aesthetics-direction-2026-07-08.md`) is
the decision: forensic-instrument visual language, color-is-semantic-or-
absent, unknown-as-first-class-visual-state, transcript-as-hero, theme.py as
enforced token generator, web-shell palette canonicalized. Children 9xuk /
bkzv / 37km / dbiv execute it. Nothing in it is speculative enough to
warrant alternatives.

### 2.2 occ5 — Post-query interaction design 🔷 (design questions resolved here)

1. **Result-set handles: yes, always.** Every query-mode output ends with a
   one-line footer: `result-set <short-id> · N sessions · query <hash-short>`
   (rxdo.3 provides the ids; until it lands, the footer prints the canonical
   query hash only). `@last` resolves to the most recent result-set of the
   current workspace; `from result-set:<id>` (rxdo.6) is the durable form.
2. **Presentation: both, by TTY.** Printed next-action lines always (they are
   the agent affordance and are copy-pasteable); the fzf picker (jnj.11
   pattern) additionally offered only on interactive TTY. FORCE_PLAIN
   suppresses the picker, never the printed affordances.
3. **Affordance source: the existing action_affordances registry** — CLI
   footer lines, MCP affordance payloads (post-rsad, opt-in), and web
   next-action chips all render the same registry entries. No second
   registry.
4. **Chaining grammar: `then` stays as-is.** Result-set-carrying pipelines
   arrive exclusively through the DSL `from` operand, not new CLI syntax —
   one grammar owns composition (fnm doctrine).
5. **Per-verb follow-through map** (the doc deliverable): find → narrow /
   open-Nth / mark / save-as-named-query / compact; read → next/prev in
   result order, lineage parent/children, extract refs; analyze → every
   aggregate row is a drillable cohort handle; mark/select → echo what
   changed + one-step undo affordance; continue → compose the harness
   invocation (37t.8/2n6 boundary).

---

## 3. Operator-gate items (I take them)

- **ptx (browser-capture posting)** ✅ — un-gate was already your decision;
  execution order inside the bead stands (0mu newest-wins fix verified
  first, then land branch, then attachments). It waits for its H-gate; no
  pull-forward — capture reliability (jlme) outranks posting.
- **37t.4 (SessionStart preamble)** ✅ — fully specced (source-aware,
  600-token cap, relevance gate, escape hatch, self-instrumenting). Execute
  at D-gate as arm-B instrumentation for cfk. No changes.
- **lio (cross-repo devloop contract)** ✅ — execute as specced; the only
  call is timing, and it should ride the next sinex-side session.
- **bfv (advisory hooks)** ✅ — the restraint doctrine (high-confidence
  matches only, fail-open <20ms, every advisory logged with refs, one-per-
  tool-call budget) is the right spec. Sequence after d1y + 20d.1/20d.12 as
  designed. The scheduler owns cooldowns (per the coherence note).
- **212.9 (Fable-as-Foreman demo)** ✅ — proceed via the rxdo.7 annotation
  loop when available; interim Task-block queries are fine for the private
  packet. The privacy gate is inherent: no public artifact without your
  review — that's the only operator moment, and it comes at the end.
- **1vpm.2 (episode unit)** — the "operator" references are runtime workflow
  (confirm/split/reject as assertions), not a pending decision. Nothing to
  adjudicate.

---

## 4. Spec-needed items — specs written

### 4.1 rsad — MCP agent ergonomics 🔷 (P2; full spec, all six fixes)

1. **Affordances become opt-in.** Default responses carry zero affordance
   boilerplate; a `capabilities` MCP resource serves the catalog once;
   `include_affordances=true` opts a call in. (Registry unchanged — occ5
   uses the same entries.)
2. **Response budget envelope.** One clamp at the callback layer
   (`json_payload`): responses over budget (default ~25KB) degrade to
   metadata + continuation handle + explicit next-action refs, never
   truncated JSON. This is the AC's "refused or summarized" rule, applied
   uniformly rather than per-tool.
3. **get_messages** gains `max_chars_per_message` (honored for all read
   shapes) and `excerpt` mode (head+tail windows around matches).
4. **get_session_summary summarizes.** It returns the session_profiles
   digest (counts, phases, top tools, first/last user words) — the honest
   materialized summary that already exists. No rename needed once the name
   is true.
5. **Errors enumerate.** QuerySpec-class errors return valid values for the
   offending field (sort keys, origins, kinds). One error-shaping helper,
   applied at the same callback layer.
6. **list dedup + search diagnostics.** Lists deduplicate by session_id with
   `match_count` preserved; zero-hit searches return per-term hit counts so
   AND-tokenization failures explain themselves.

### 4.2 2n6 — Harness remote-control ✅ (spec ratified, stays N-horizon)

Native-resume-first (claude --resume / Codex AppServer), one-shot invocation
composition (37t.8 owns the mapping), kitty control as actuation floor, MCP
exposure with every control action archived. Parked until 37t.8 exists;
capabilities re-verified at build time.

### 4.3 lu1 — Ambient theming ✅ (reconciled with 9xuk)

Terminal: semantic use of the terminal's own 16-color palette (pywal
propagates free). Webui: themes are flat token-file swaps on the 9xuk
substrate; 2–3 curated + prefers-color-scheme; live-editable file is the
floor, settings-panel editor the stretch. Recordings pin their theme.

### 4.4 gqx — Desktop presence spike ✅ (build order: a, b, then c)

Widget feed (SSE consumer) and `polylogue://` URL handler first — both
afternoon-sized once SSE + deep links exist; kitty session-correlation
kitten third (verify the env-join trick); notifications last and only
capture-gap/hook-liveness. Sinnix-consumer posture stands.

### 4.5 2jj — IssueBench ✅ (vision; park until fs1.10 + cfk machinery)

No changes; the leakage-gate requirement is the load-bearing part.

---

## 5. What ratification unblocks, in order

1. **bby.11 scaffold** → 1ilk, bby.9/.10/.15, d4zk, gqx(a,b), semantic
   renderers. Single biggest planning unblock.
2. **4p1 doc PR** (draft inventory exists) → jnj.3/.2/.4, fnm.1/.11, 5wp,
   7le/ap7 all get their stated invariant.
3. **rsad spec** → the MCP surface stops fighting agents; occ5 and the
   annotation loop (rxdo.7) both land on a usable surface.
4. **dx1 ASGI ramp** → 20d.1 fast-path endpoints and webui v2 API land on
   the winning substrate instead of accreting on the old one.
5. **e6ja attestation check** → closes the zero-tests-pre-merge legibility
   hole for near-zero cost.
6. tjx1/occ5 sign-off → the four aesthetics beads + occ5 implementation
   children become claimable.

Items that deliberately stay parked: ca4, c36, 2n6, 2jj, gqx (N-horizon,
each with its wake condition recorded above).

---

## Addendum: RATIFIED 2026-07-08 (operator, in-session)

All calls above approved. Recorded per-bead (ratification notes on bby.11,
4p1, hiu, dx1, ze5, ca4, fie, c36, e6ja, ptx, 37t.4, lio, bfv, 2n6, lu1,
gqx, 2jj, 212.9; rsad design field now carries the six-fix spec; occ5
design questions resolved in notes; tjx1 closed with sign-off).

### What ratification unlocks — the next design frontier

**Immediately executable, no design left:** e6ja option C (attestation
check), 4p1 doc PR (draft inventory exists), 4p1.1 fast-path dedup, rsad
six fixes, hiu step 0 (pf1 divergence reconciliation), 83u.5 blob zstd.

**Newly designable (was blocked on these decisions):**

1. **The cockpit view-spec pack** — bby.11 ratified means every v2 view is
   now spec-writable against a known substrate (Preact components, generated
   API client, SSE module, 9xuk tokens, bkzv vocabulary, occ5 affordances):
   bby.9 mission control, bby.10 timeline/firehose, bby.15 pinboard/basket,
   the judge-queue view, d4zk variant UX, bby.12 replay, ap7 semantic
   renderers. One batched design session can write these coherently — the
   largest formerly-unplannable area in the beadset.
2. **Typed route registry** — dx1 (ASGI) implies the daemon route table
   becomes a declare-once registry the OpenAPI render generates FROM; design
   that registry before the first migrated family so 20d.1 endpoints and
   webui v2 API land on it.
3. **user.db v4→v5 migration design** — ze5 (relations, revisions,
   confidence) + rxdo.2 (queries/result_sets/query_edges) now share one
   60i5-batched window; the combined DDL + migration doc is writable today.
4. **rxdo.3 minimal slice** — occ5's result-set footer needs the run/relation
   schema; designing it now keeps occ5 implementation and rxdo compatible.
5. **C-gate collapse execution** — with 4p1 doctrine landed, jnj.3/.2/.4,
   fnm.1/.11, 5wp become mechanical preset-registration slices in the
   inventory's stated order.

**Recommended batching (per the batch-execution protocol):** three design
sessions — (a) cockpit pack, (b) read-contract pack (4p1 doc + occ5
direction + rsad implementation), (c) storage/migration pack (v5 window +
rxdo.3 + zstd phasing). After 27rb lands, revisit e6ja escalation and
testmon-in-CI.
