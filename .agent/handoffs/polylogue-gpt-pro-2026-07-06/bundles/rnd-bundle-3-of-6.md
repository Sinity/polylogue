# Polylogue R&D bundle — file 3 of 6

> Preamble below is the MASTER SYNTHESIS, identical across all 6 files.
> The unique payload of THIS file is the 16 subagent outputs after the PAYLOAD divider.
> Files 1-6 together hold every substantial subagent output + the brainstorm corpus, size-balanced.

=====================================================================
  PART 1 - MASTER SYNTHESIS (preamble, same in all 6 files)
=====================================================================

---
created: "2026-07-05"
purpose: "Integrated master synthesis of the ~92-agent Polylogue R&D run (5 waves): convergent design themes, confirmed bugs, per-subsystem implementation specs, GPT-pro prompt fan-out, and the critical path. Written under context pressure to survive compaction."
status: "active"
project: "polylogue"
---

# Polylogue R&D — Master Synthesis (2026-07-05)

## 0. What this is and where the detail lives

A five-wave parallel R&D run (~92 background agents) over the Polylogue archive, driven from a
GPT-pro webui design session (`chatgpt-export:6a49af33…`, auto-captured via the browser extension).

- **Wave 1 (8):** wide ideation — product/analytics/self-referential/ecosystem/content/trust/UX/moonshots.
- **Wave 2 (14):** deeper ideation + first specs — queries-as-objects, measure-catalog, self-capture loop,
  activation, incident, temporal, economics, search, ingestion, proactive, missing-units, triage, episode, meta-quality.
- **Wave 3 (16):** new dimensions + 4 red-teams — whole-product, API surface, distribution, docs, workflows,
  observability, config, releases, **red-team ×4** (vision / honesty / security / cost), a11y, MCP, perf, demos.
- **Wave 4 (12 impl-specs):** measure-registry, context-scheduler, notification-policy, query/finding-objects,
  ingest-fidelity, compaction-event, content-variant, triage-inbox, delegation, episode, recursive-safety, temporal.
- **Wave 5 (42 impl-specs):** everything not yet spec'd + the red-team fixes + missing-units + RL/eval + 2 synthesis lanes.

**Full detail per agent** is in `/tmp/claude-1000/-realm-project-polylogue/<session>/tasks/*.output` (one JSONL per agent —
read a single file if you need one spec verbatim; do NOT bulk-read).
**Idea corpus (~170 deduped):** `.agent/scratch/rnd-brainstorm-2026-07-05.md`.
**GPT-pro prompt files (11 of 66):** `/realm/inbox/tmp/`.
**Sinnix fix already committed** (`f6de855`, not pushed): `mcp-polylogue` launcher strips the cloud-lane
`POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-archive` leak so the recall MCP hits the real archive.

---

## 0.1 Provenance & confidence (read before trusting a citation)

This doc has **two groundedness tiers**, because a context-compaction fired mid-run and replaced the full subagent
*source outputs* for waves 1–4 (and the earliest wave-5 specs) with a summary:

- **Source-grounded (high confidence):** the §3 wave-5 entries and the §2b "later wave-5" bugs — these were written
  with the subagents' full `<result>` bodies in context, so their file:line claims are first-hand.
- **Summary-derived (verify before acting):** the §1 themes, the §2 confirmed-bug list, and the §3 **wave-4** entries —
  these were distilled from the pre-compaction summary, *not* re-read from the source `tasks/*.output` files this
  session. The claims are faithful at the level of *what* the bug/mechanism is; the specific **file:line citations are
  unverified-since-compaction**. The 50 source docs survive on disk at `<session>/tasks/*.output` and are the authority.

Practical rule: this doc is the **index and the reasoning**; before you touch code on any specific `file:line`, open that
spec's task-output file and confirm the anchor. Nothing is lost — it's a re-read away — but don't treat a summary-tier
citation as verified.

## 1. The convergent design themes (load-bearing — every spec assumes these)

1. **Queries + findings as first-class stored objects.** `query:<hash>` ObjectRef keyed on the *lowered,
   macro-expanded* AST (so equivalent queries collapse, mirroring content-hash idempotency); `result_sets`
   snapshots in the **derived** index tier (Merkle root over sorted member keys, `hash(query_id ‖ corpus_version)`);
   a `query_edges` DAG populated from the set-algebra EXPLAIN nodes; **`AssertionKind.FINDING`** reusing the
   existing pathology candidate→`judge_assertion_candidate`→promote lifecycle *verbatim*; a `StandingQueryStage`
   in the DaemonConverger; **findings-as-tests** (a promoted finding with `value.expected` becomes a re-runnable
   invariant). Query versioning via the existing `supersedes_json`.

2. **The measure algebra.** A measure = `⟨reducer(column) over unit-frame × grouping × window × comparison × uncertainty⟩`
   — a cartesian product, not hand-written `analyze` modes. Each measure declares a construct-validity registry row
   (construct, formula, correct denominator/unit-frame, evidence tier, top-2 confounds, coverage precondition).
   Uncertainty layer: Wilson for proportions, bootstrap CI for skewed median/pXX, a coverage gate. `count` is the
   trivial measure so there is **one** aggregate path. **`SampleFrame.LOGICAL_SESSION` is the default frame** —
   this is what structurally kills the ~32% lineage cost double-count.

3. **The recursive-safety gate** (coherent subsystem, not a flag — matters *because* the browser extension
   auto-captures the operator's own R&D). Mechanisms: agent-authored assertions default `candidate` + `inject:false`;
   a **closed-loop laundering gate** (a claim whose evidence resolves *only* to other agent-authored sessions →
   quarantine until an external-grounded citation or human judgment breaks the cycle); the inject gate requires
   `author_kind='user'` (promotion via `judge_assertion_candidate` flips it); never inject/alert/mine on
   `material_origin ∈ {generated_context_pack, runtime_context, runtime_protocol, tool_result}`; provenance-cycle
   quarantine reusing `TopologyEdgeStatus.quarantined`; auto-downgrade findings on evidence-content-hash drift.

4. **Content-hash citation anchors.** Block identity today is `message_id:position` — position shifts on re-ingest
   and fork-replay. The fix (webui/CIF specs): add `blocks.block_content_hash` (derived, materializer-populated),
   resolve anchors against the *composed* lineage transcript by hash not offset, three states
   (`exact` / `drifted` / `deleted` / `quarantined`). This is the atom the cockpit, findings, and CIF all stand on.

5. **The episode unit.** A logical task spanning sessions/tools/time, a tier *above* within-provider lineage,
   edges thematic/temporal not identity-shared. 4-signal scorer (repo × repo-conditioned-time-kernel × embedding ×
   shared-artifact) with a **hard false-merge floor** (a strong structural corroborator — shared SHA / error
   fingerprint / same-repo+time — required before any `corroborated+` tier; thematic-only stays `candidate`).

6. **Corpus-compaction (`find <q> | compact`).** Token-budgeted, decision-dense digest for feeding an external LLM
   (the R&D-flywheel enabler): drop tool-spam via `material_origin`, keep human turns + decisions + error→fix +
   outcomes, dedup lineage prefixes, fit a context window, emit a fidelity drop-manifest.

7. **Construct-validity is executable everywhere.** Coverage-preconditions that *refuse rather than fabricate*;
   evidence-tier footnotes on every number; the `unknown` bucket never folded into a denominator; text-mined
   fields tagged `text_derived`/unverified in the payload contract, not just the bundle renderer.

8. **"Wire what exists" beats "build new."** The daemon already has a 5-backend notification fan-out (carries only
   ops alerts); `score_components` already carries the per-lane RRF decomposition; postmortem-bundle machinery,
   lease-safe blob GC, and the coordination envelope + degradation ladder are already shipped. Most wins are routing
   an existing pipe to a new signal, not new infrastructure.

9. **Durability-keyed schema discipline (the recurring "how").** *Almost nothing touches the durable tier.* New
   `AssertionKind` values are schema-free `TEXT` (no migration). Derived index/embeddings changes edit canonical DDL +
   bump version + rebuild (`ops reset --index && polylogued run`), **never an upgrade helper** (`devtools lab policy
   schema-versioning` rejects them). Durable source/user changes need a numbered additive migration + a **verified
   backup manifest** + one `PRAGMA user_version` step. Every new AssertionKind/enum is embedded in `render openapi` +
   `render cli-output-schemas`; every new module breaks the topology projection — regenerate and grep `render all
   --check` for `out of sync` (the tail line lies).

---

## 2. Confirmed bugs (red-team, file:line'd, actionable NOW)

### Cost/lineage (from the cost red-team + cost-correctness spec)
- **Day/week summaries sum PHYSICAL sessions** (`session_summaries.py:98-104`) → ~32% lineage double-count of
  cost/duration/messages/words. Fix: logical-grain rollups (one representative per `logical_session_id`).
- **Exact provider-cost path is DEAD** — `pricing.py:615` `_session_level_estimate` returns `None`;
  `session_reported_costs` is written but read by nothing → every subscription-cost figure is silently $0. Fix: wire the reader.
- **Missing credit rates for Opus 4.7/4.8** (`subscription_pricing.py:84-105` only has 4-5/4-6) →
  `get_credit_rate("claude-opus-4-8")=None` → credit_cost=0 for the most-used model. Fix: add rates + a test that
  `MODEL_CREDIT_RATES ⊇ curated Anthropic keys`; lock `output = 5×input`.
- **Subscription $/credit hardcodes Pro tier** (`cost_compute.py:133` `/21_700_000*20.0`). Fix: parametrize by
  `user_settings` subscription tier (assertion `subscription_tier`).
- **`cost_enrichment.py:54` downgrades EXACT→catalog.** Fix: strength lattice, never demote provider_reported.
- **`wall_duration_ms` summed across parallel subagents** (>86.4M ms/day representable). Fix: interval-union.
- **Codex disjoint-lane double-count** (the 7.69× class, fixed once in `3938bc6c2`, unguarded). Fix: assert
  `cached ⊆ input`, `reasoning ⊆ output` at parse.

### Honesty (from the honesty red-team + honesty-audit spec)
- **`_RIGOR_MATRIX` covers only 5 of 11 number-bearing products** — cost/coverage/tool/debt silently skipped
  (`audit.py:188` iterates contracts, not the registry). Fix: iterate the registry, emit `coverage_status=uncovered`.
- **`transforms.py` text-mines** commit SHAs / decisions / caveats / test-pass-counts from prose into forensic
  bundles while "no regex over prose" only holds for the exit-code axis. Fix: tag `text_derived` in the payload model.
- **`classify_aggregate_hwm_source` launders provenance to `provider_ts`** (`temporal_source.py:97`). Fix: propagate
  the *weakest* source.
- **Attachment referenced-vs-stored bytes** reported as retrievable (`#2468`; 967 acquired vs 6,414 unfetched,
  ~13.6 GB referenced-never-stored). Fix: split `referenced_bytes` from `stored_bytes` on every surface.

### Security (from the security red-team + security-hardening spec)
- **DNS-rebinding reads the whole archive**: no `Host` check on GET routes; `Origin` checked only on POST and skipped
  when absent (`daemon/http.py`). Fix: central `Host`/`Origin` allowlist middleware before dispatch.
- **Browser-capture receiver has no auth on loopback** → any local process POSTs forged captures into the spool.
  Fix: auto-mint a `0600` receiver token; `hmac.compare_digest`; restrict `?access_token=` to the SSE route; spool quota.
- **`reset.py` tombstones before the `--yes`/preview gate** (jnj.5). Fix: one shared mutation-audit contract with a
  dry-run preview across excise/reset/delete/MCP-admin.

### Live bugs (from the temporal spec)
- **`RELATIVE_BASE = datetime.now()` at import** (`core/dates.py:37`) → `since:7d` anchors to process-start,
  unpatchable by `frozen_clock`. Fix: route through a `core/clock.py` seam.
- **`sort_key_ms = COALESCE(...,0)`** vanishes timeless sessions from `since:` windows and pins them at 1970 in sorts.
  Fix: half-open `[since,until)`, explicit `IS NULL` handling, surfaced as `time_confidence=synthetic`.

---

### Live bugs found by the later wave-5 specs (not in §2 above — all file:line'd in their task outputs)
- **`ops doctor cleanup_orphans` can delete an in-flight leased blob** (blob-GC spec, risk R1). `run_blob_gc` is
  lease/ref/generation-safe, but `BlobStore.detect_orphans`/`cleanup_orphans` (the `ops doctor` path) compares disk
  against **only `raw_sessions.raw_id`** — ignores `pending_blob_refs`, `blob_refs`, and the generation-age gate. This
  is the real "#818 orphan-detection bug." Fix: make `cleanup_orphans` lease+generation-aware or hard-gate it behind
  `run_blob_gc`. Not optional.
- **`blackboard_post` lets an agent write `author_kind=agent` that lands `status=ACTIVE`** (annotation-recipe spec) —
  a recursive-safety hole: an agent's own claim can self-inject as authoritative. Fix: one `coerce_agent_authored`
  chokepoint inside `upsert_assertion` forcing all non-`user` authors → `CANDIDATE` + `inject:false` (never resurrecting
  a terminal-judged row). This is the load-bearing invariant recall/recipe/distillery/goal/decision all assume.
- **Identity collision *beneath* the origin-collapse aggregate bug** (provider→origin spec). Beyond the §1 aggregate
  double-count: `session_id = origin || ':' || native_id` means a `gemini-export` and a `drive-takeout` session sharing a
  `native_id` are **already one physical row** — undetectable, un-splittable even by reparse. The `source_family` column
  + `lossy_grouping` marker fixes *aggregation* honesty, not identity; file the identity half as separate durable scope.
- **`user_settings` table is dead** (config-engine spec): declared in `user.py` + migration `004`, but **zero read/write
  helpers exist** — unwired, empty. Unify into `assertions` as `AssertionKind.setting` (drop the table via copy-forward).
- **`session_phases` deliberately has no `kind`** (units-D grounding correction) — the intent-classified equivalent is
  `session_work_events`, which **already exists as durable index rows**. Re-adding `kind` reverts a construct decision;
  `phase-segment` is a DSL projection over work-events, not a new table.

## 3. Spec inventory (implementation-grade; each ships DDL + tier/regime + algorithms + tests + a 4–8 bead breakdown)

Format: **name** — tier/regime · key mechanism · headline risk.

### Wave-4 (all read in full)
- **measure-registry** — index/no-DDL (compute-on-read) · the 5-tuple algebra + 16 measures as registry rows +
  coverage-gate that *refuses* rather than fabricates + Wilson/bootstrap uncertainty · risk: any surface bypassing
  the single aggregate path escapes the gate. 8 beads.
- **query/finding-objects** — index v25 (query_defs/query_edges/result_sets/result_set_members) + user.db
  `FINDING` enum (no migration) · content-addressed query id, StandingQueryStage, findings-as-tests · risk:
  canonical-form drift orphans every result_set + finding evidence_ref. 6 beads. **Depends on set-algebra (fnm.13).**
- **recursive-safety** — user.db v5 (+`provenance_state` reusing TopologyEdgeStatus, +`safety_json`; additive migration) ·
  closed-loop laundering CTE over the evidence graph, `author_kind='user'` inject gate, drift auto-downgrade on the
  *durable* content-hash · risk: too-tight quarantines legit agent R&D (operator_command counts as grounding). 6 beads (37t.13-18).
- **context-scheduler** — user.db JSON-only (programmable `context_policy_json` windows) + ops.db ledger + read-access log ·
  ranking = staleness-decay × topic-proximity × attention; trust-class as slice-1-not-retrofit · risk: trust retrofit = injection hole. 8 beads (37t.11.a-h).
- **compaction-event** — source.db v3 (durable, `compaction_snapshots` + blob) + index `compaction_loss` · loss-forensics
  diff (retained/lost/transformed over file-path/tool-outcome/marked-decision/cited-ref), honest degradation on
  JSONL-boundary snapshots, loss-record survives the *next* compaction · risk: PreCompact payload may lack the assembled context. 7 beads (gjg.1-7).
- **episode** — index v25 (episodes/episode_members/episode_edges/produced_refs) + embeddings v2 (session_embeddings) +
  user.db `EPISODE_{CONFIRM,SPLIT,REJECT}` · 4-signal scorer + hard false-merge floor + anchor-keyed deterministic id ·
  risk: embedding cosine conflates thematic-similarity with same-task. 7 beads.
- **delegation** — index v25 (`delegations` VIEW + `primary_model_*` cols) · delegation-yield measure + delegation-card +
  the Fable `.polydemo`; `result_status` derives only from `actions.is_error`/exit_code (`unknown` never in the ROI denominator) ·
  risk: provider-vocab leak (model→family must project origin-vocab). 6 beads.
- **content-variant** — user.db v5 (content_variants/variant_nodes/variant_alignments; additive migration) + index
  `block_language_facts` · mechanical-vs-generative provenance axis, coverage>0 write invariant, `source_content_hash`
  staleness (never auto-repaint), dark-matter rendering · risk: false-stale storms from fingerprint drift; durable/derived
  orphaning during rebuild. 6 beads.
- **triage-inbox** — index v25 (`worth_reviewing_score` + breakdown + gate on session_profiles) + user.db `TRIAGED`
  (no migration) · time-invariant score materialized, inverted-U staleness applied at read, `WHERE NOT EXISTS triaged`
  empties the inbox, snooze-with-wake · risk: materialized staleness goes stale; gates hide instead of resolve. 6 beads.
- **ingest-fidelity** — source.db v3 (+`parser_fingerprint`, +`decode_failure_class`) + index `raw_fidelity` · byte-fidelity
  ratio (band per origin, not absolute), round-trip reconstruction (structural equality is the real bar), unparsed-key
  census, misclassification tripwire, `parser_fingerprint`-driven reprocess-on-improvement · risk: ratio misread as a bug. 8 beads (F1-F8).
- **temporal** — index-only + `core/clock.py` seam · four typed time-kinds, `sort_key_provenance`, skew-band **quantization**
  (not pairwise — transitivity), `[since,until)`, tz-unknown-by-default, backfill-never-moves-sort_key; fixes the two live
  bugs · risk: de-generating sort_key_ms reorders history/breaks cursors. 8 beads (cpf.*).
- **notification-policy** — user.db `NOTIFICATION_POLICY` (no migration) + ops.db ledger · routes CONTENT signals through
  the existing fan-out (`Notice` severity + `content` family), standing queries as producers, "you're repeating a past
  mistake" nudge, token-bucket fatigue control, no-self-alert · risk: fatigue defeats adoption → ship ledger-first. 8 beads (notify.1-8).

### Wave-5 (read in full so far)
- **blue-green index rebuild (b5l)** — durable pointer file + ops mirror · generation-suffixed `index.gN.db` + atomic
  pointer swap (<100ms) + delta-replay + lease-safe reaping; kills the 20-40min degraded window · risk: delta-replay race
  at swap boundary. 7 beads (b5l.1-7).
- **cross-machine sync** — sync only durable tiers · source.db content-hash union (idempotent+commutative), user.db
  assertion natural-key LWW merge, rebuild derived on peer, `.well-known/ai-sessions` manifest · risk: `raw_id` contains
  machine-local `source_path` → same session gets two ids across machines. 6 beads.
- **python-sdk** — `polylogue.sdk` + `polylogue.models` · ~20-method curated surface, sync facade (the async-only facade
  is why lynchpin bypasses via raw sqlite + reimplemented models + a stale `FROM conversations` query), frozen model
  re-exports, schema pin-and-warn, layering lint · risk: freezing origin vocab mid-retirement. 8 beads.
- **vec0-ANN** — embeddings v2 · int8-primary + f32-rerank vec0 (4× smaller scan), origin partition-key + centroid IVF
  prefilter, multi-probe batching to kill the 20-scan session-similarity fan-out, **recall@k eval from fork/resume
  lineage pairs (free positive labels)** · risk: quantization silently degrades recall at real scale. 6 beads (mhx.6.*).
- **mcp-collapse** — ~130 tools → **9 verbs** (`query`/`get`/`explain`/`context`/`correlate`/`coordinate`/`assert`/`retract`/`maintenance`) ·
  one `query(expression)` over the DSL absorbs ~40 read tools; resources/subscribe + `list_changed`; saved-views as
  dynamic prompts; `assert`/`retract` over the unified assertions table · risk: silent capability loss (per-tool
  equivalence goldens before deletion). 8 beads.
- **second-brain graph** — index v25 (entities/entity_mentions/entity_topics + entity_backlinks VIEW) · structural-vs-
  candidate mention split (recursive-safety-gated prose mining), bare-`#N` repo-scoped, topic co-occurrence clustering ·
  risk: prose-mining fabrication feedback loop (self-archiving). 6 beads.
- **session-replay** — ops.db (disposable) replay_runs/replay_steps · phase-1 DRY plan from the `actions` stream (no
  execution), phase-2 gated worktree re-execution with network-deny + frozen-clock + fail-closed classification ·
  risk: false anchors (only `explicit_ref`/`origin_reported` SHAs gate a checkout); silent step incompleteness. 8 beads (repro-1..8).
- **security-hardening** — runtime+config only, no DB migration · central Host/Origin gate, auto-minted receiver token,
  `hmac.compare_digest`, spool governor · risk: breaking the same-origin web shell (allowlist must admit its Host). 7 beads.
- **polish-fts** — index v25 (`unicode61 remove_diacritics 2` + a `pl_fold` write+query symmetry for the precomposed
  `ł`/`Ł` that `remove_diacritics` *cannot* fold) + `block_prose_lang` + trigram fallback lane · risk: query/index folding
  drift = silent recall loss. 6-8 beads.
- **attachment-bytes** — no schema change (columns already honest) · project `acquisition_status`+`blob_hash` through the
  4 read paths that drop them; two-sum accounting (`referenced` vs `stored`) everywhere + a lint forbidding bare byte
  totals; census (83u.6) + classifier (83u.4) + citation-anchor three-state resolver · risk: `size_bytes` compat alias
  stays the leak (prefer rename so mypy surfaces call sites). 6 beads.
- **own-export/CIF** — index additive + Origin enum · `polylogue-export` origin whose parser reconstructs the *embedded*
  origin so `import(export(A))` is a content-hash **no-op** (free correctness invariant + federation primitive);
  `content_hash_algo` pinned in the envelope · risk: hash-algorithm drift breaks federation. 6 beads.
- **cost-correctness** — index v25 + user.db tier assertion · the precedence lattice (provider_reported ≻ catalog_priced ≻
  heuristic, never downgrade EXACT), logical-grain rollups, per-model credit rates, tier-parametrized $/credit, disjoint-
  lane guard, wall interval-union · risk: logical-representative selection is owned by 4ts. 6 beads (f2qv.*).
- **honesty-audit** — index derived + pure contracts · field-level `RigorFieldContract` + `ProvenanceClass`, iterate the
  *registry* (emit `uncovered`), number-over-empty gate, weakest-source propagation, text_derived tags, referenced-vs-
  stored split, a `devtools lab policy insight-honesty` CI gate · risk: `nullable_when_ungrounded` breaks byte-compat
  consumers (deliberate; gate behind materializer-version bump). 7 beads.
- **zstd/tiering economics** — source v3 (blob_placement/blob_dicts/blob_tombstones/frozen_segments) · address stays
  `SHA(uncompressed)` forever; zstd per-origin trained dicts (36GB→4-7GB), access-temperature cold shard, embeddings
  int8, yearly-epoch freeze, single `blob-compact` walk, per-origin bytes/session SLO breaker · risk: any read/verify
  path that forgets to decompress silently breaks the backup verifier. 8 beads.
- **product-positioning** (plan) · keep "system of record for AI work" as the category anchor, lead with the
  **flight-recorder** hook + the **24.1% silent-proceed** hero finding (published, reproducible, construct-valid);
  honesty-benchmark is the *launch campaign* not the product identity; hold all memory-uplift claims capability-phrased
  until `cfk` reports (the `jxe` pilot closed *negative*).
- **secret-redaction/forget** — source v3 (`excision_tombstones`) + ops audit + user.db `SECRET_CANDIDATE` · scan-at-read
  overlay (never persists the value), judgment-queue, hash-tombstone excision, one mutation-audit contract, converger
  reconciliation invariant, **idempotency resurrection guard** (re-ingesting the original source can't resurrect) ·
  risk: parse-drift asymmetry vs the content-hash key → block-granularity + span-text-hash probe. 8 beads.
- **adoption/install** — ops.db v2 (hook_liveness/doctor_snapshots), live-computed adoption · `polylogue install`
  (idempotent all-hook wiring, match-by-command-substring so foreign hooks never clobbered) + `polylogue doctor`
  (liveness heartbeat + 5-way "why-zero-usage" diagnosis with a *relevance control*) + PreCompact recall + SessionStart
  brief · risk: adoption metric false-alarms → the signal gets ignored (the very failure it exists to prevent). 8 beads.
- **webui evidence-cockpit (bby.11)** — index `blocks.block_content_hash` (the re-ingest-stable anchor) + user.db
  `citation_anchor`/`evidence_report`/`report_basket` (no migration) · TS+Preact SPA hydrating a server-rendered
  Markdown/HTML twin (progressive enhancement), evidence-basket→report, integrity verifier, force-directed lineage graph ·
  risk: duplicated-block ambiguity (acompact replays a hash N×). 8 beads.
- **s7ae coordination** — user.db v5 (`coordination_message`/`coordination_ack` + a virtual `expires_at_ms` generated
  column) · scoped message bus on the assertions table, `query:<hash>` live-query refs (notepad→task bus), coordination
  as a `ContextSource` (37t.11), the two-agent same-repo proof harness · risk: hard dependency on unbuilt 37t.11 (the
  advisory leg is the injection surface). Envelope/degradation-ladder already shipped. 8 beads (s7ae.3/.5).
- **demo/proof engine** — additive · `.polydemo` executable format (frontmatter budget + product-primitive CLI steps +
  content-addressed `finding_id` = hash(claim+metric+anchor+sorted-refs+corpus-datasheet-hash)) + round-trip evidence-ref
  resolver gate + construct-validity pre-render tripwire + refusal manifest + demo-as-CI-test (finding_id drift breaks
  the build); reframes the 60KB `claim_vs_evidence.py` as the first `.polydemo` on the sru.1 DSL · risk: compositionality
  erosion (steps shelling to python recreate the monolith — parser accepts only `polylogue …` argv). 7 beads (B1-B7).

### Wave-5 — the remaining ~22 (ALL now complete; full detail in task files). Sharpest finding each surfaced:
- **provider→origin** — `source_family` column + data-driven `lossy_grouping` marker (fires only when a projected origin
  merges ≥2 families); identity-collision residual filed separately (see §2b). One projection chokepoint or surfaces drift.
- **FTS-coherence** — the `messages_fts.rowid == blocks.rowid == docsize.id` identity is the keystone; metamorphic law
  (any block op-sequence ⇒ 0 missing/excess) + O(1) ledger-only drift gauge; rowid-reuse must also check `block_id`.
- **search-relevance** — session resolution lives in **two** paths (`hybrid_sessions` + `archive_execution`); collapse +
  `variant_count` must land in one shared helper or lineage siblings duplicate (the #2470 class). `AssertionKind.RELEVANCE` LtR corpus, nDCG gate.
- **migration-fuzz** — the chain is **sparse** (source starts `002`, user `004`); fuzz `start_version` must map to real
  historical DDL or the contiguous-chain check legitimately raises. CHECK↔`Literal` drift tripwire via site-count assert.
- **config-engine** — 5-layer *deployment* resolver already exists (`config.py`) but has no `db` layer; add a
  scope(global→repo→origin→surface)×actor(operator→agent→harness)×override(flag>env>file>db>default) resolver over
  settings-as-assertions; learned-defaults self-exclude `config`/`judge` telemetry.
- **observability/SLOs** — one measure-sample stream (`slo_samples`, ops.db) + reducers (level/quantile/slope/eta/burn);
  the honesty keystone is the **idle-vs-stalled verdict** (backlog>0 is a defect only when not draining AND work offered);
  `ingest_latency` must scope to live-tail origins (bulk-export excluded by construction).
- **units A** — entity-mention=**TABLE** (regex extraction, structural-vs-candidate trust axis); world-effect=**VIEW**
  (cause from `observed_event.evidence_refs_json[0]`→`actions`); verification-run=**VIEW** (pass from exit_code keystone).
- **units B** — turn-pair=**VIEW** (material_origin adjacency); artifact=VIEW→table (distinct from `files`/`raw_artifacts`
  name-collision); **correction-edge MUST be a runtime query method, not a stored VIEW** — SQLite forbids a persistent
  VIEW over the ATTACHed `user_tier`. Correction anchors are mostly `session:`-coarse (limits error-rate-per-tool).
- **units C** — topic-cluster (HDBSCAN+kmeans fallback, content-addressed ids for timeline stability); **project is the
  only durable-tier unit** (user.db v5, gizmo_id extraction prereq, segment-aligned cwd match); cross-origin-thread
  (differ-origin ∩ ¬lineage ∩ hard-signal floor + hub-merge guard).
- **units D** — phase-segment projects over existing work-events (no new table); goal/decision are construct-gated
  candidates via the **existing** candidate→judge state machine; the recovery-digest incident is the shared regression test.
- **RL/eval (fs1.5)** — pure read projection to Atropos JSONL; **verify the live Atropos schema first** (round-trip via
  `jsonl2html.py`); corrections are **session-scoped** so reward-model reports AUC *with base-rate + n*; `PROMPT_EVAL` kind exists.
- **cross-project recall MCP** — text-hint→vector via existing `VectorProvider.query` (needs live Voyage key → honest FTS
  fallback); trust-class (OPERATOR/QUOTED/SYSTEM) is the laundering barrier, built in from slice 1, not retrofit.
- **spec-cards + TQI** — spec-card reproducibility gated on high-confidence commit attribution; TQI's `fragmentation_sub`
  is heuristic (phases aren't intent), lowest weight; ship strictly as reward-*shaping*, never sole reward (Goodhart).
- **model-drift observatory** — blocked on 9l5.7; intent-anchor validity is the weakest link (mandatory embed-coverage
  gate); a model upgrade re-keys the cohort, so changepoints are candidate + nearby-event, never causal.
- **read-access-log + attention** — ops.db is already multi-writer (per `daemon/events.py`); in-process debounce +
  decayed counter; **`context_inject` excluded from the attention signal** so the scheduler can't reinforce its own injections.
- **prompt-distillery** — recipes/prompts live in git-YAML (code under review), not user.db; distilled prompts are
  `PROMPT_TEMPLATE` candidates; A/B evaluator returns INSUFFICIENT_EVIDENCE below floor (never fabricates a win).
- **corpus-compaction** — `find <q> | compact` is a read-algebra point (Selection×CompactionProjection×COMPACT render);
  token proxy is word-count → apply a `0.72` BPE derate; outcomes from keystone fields only (anti-fabrication).
- **standing-query** — dedup by **stable finding-identity set**, not events → re-ingest storm is structurally impossible
  (same content-hash ⇒ same finding_id ⇒ already in membership); baseline-then-notify; self-trigger firewall excludes `notice.*`.
- **annotation-recipe** — the real completeness gap is **query-back**: assertions are MCP-`list`-only today, not in the
  `find` DSL; phase-1 is honest about that. Enforce recursive-safety at the single `upsert_assertion` chokepoint.

---

## 4. Cross-cutting implementation patterns (the recurring "how" — apply to every bead)

- **Prefer schema-free `AssertionKind` (TEXT, no CHECK)** for new user-tier vocabulary — no migration; but regenerate
  `render openapi` + `render cli-output-schemas` and add a `user_audit` surface entry, or discovery/audit tests fail.
- **Derived tier (index/embeddings) = edit canonical DDL + bump version + rebuild plan**, never an upgrade helper.
  Batch same-tier bumps from ready beads before triggering a live 38GB rebuild. **Blue-green (b5l) removes the downtime** —
  land it early and every subsequent index bump is a non-event.
- **Durable tier (source/user) = numbered additive migration + verified backup manifest + one `PRAGMA user_version` step.**
  The specs that need this: recursive-safety (user v5), content-variant (user v5), s7ae (user v5), compaction-event
  (source v3), ingest-fidelity (source v3), secret-redaction (source v3), zstd (source v3). **Consider batching the user
  v4→v5 and source v2→v3 bumps** so the live archive migrates once.
- **`LOGICAL_SESSION` grain is the default** for every measure/rollup/cost/analytics; `session_count` stays physical,
  `logical_session_count` is the distinct count. This is owned by `4ts` — gate cost-correctness B5 and episode cost
  rollup on it.
- **Every new module → regenerate topology projection** (`render topology-projection && topology-status`).
- **`ContextSource` protocol (37t.11) is the spine** for context-scheduler, notification advisories, coordination
  advisories, adoption SessionStart brief, and compaction re-grounding — build it first; its trust-class gate is a
  *security* invariant, not a nicety.
- **`frozen_clock` + `core/clock.py` seam** for all time-sensitive tests (the `verify-test-clock-hygiene` lint enforces it).

---

- **BATCH THE SCHEMA BUMPS — the single biggest operational insight across the specs.** Nearly every derived-tier spec
  bumps **index.db v24→v25** (units A/B/C/D, topic-cluster, entity-graph, spec-cards, cost-correctness, polish-FTS,
  query-objects, delegation, content-variant, honesty-audit, TQI, provider→origin source_family, drift…). Several durable
  specs bump **user.db v4→v5** (recursive-safety, content-variant, s7ae, config-engine, goal) and **source.db v2→v3**
  (compaction-event, ingest-fidelity, secret-redaction, zstd, blob-heartbeat). These MUST be batched: pick the coherent
  set of ready beads per tier, land them together, and rebuild the live 38 GB archive **once** (`ops reset --index` /
  the durable migration + backup manifest) — never per-isolated-addition. **Blue-green (b5l) removes the index-rebuild
  downtime, so land b5l first and every subsequent index bump is a non-event.**
- **Cross-tier VIEW constraint (recurring trap).** SQLite forbids a *persistent* `CREATE VIEW` in index.db that
  references the ATTACHed `user_tier` (user.db). Anything joining `assertions ↔ blocks/messages` (correction-edge,
  recall, annotation query-back) must be a **runtime query method** (like `query_assertions`) or a per-connection
  `TEMP VIEW` — never DDL. Guard it with a comment + a `devtools lab policy` check; a future contributor will try the VIEW.
- **Recursive-safety is one chokepoint, not per-path.** Enforce `author_kind != user ⇒ CANDIDATE + inject:false` inside
  `upsert_assertion` itself (never resurrecting a terminal-judged row), so the transform/pathology/goal/decision/recipe/
  distillery/recall writers and the `blackboard_post` agent-write path all inherit it. Enforcing it per-recipe leaves the
  `blackboard_post`-ACTIVE hole open. This is the QUOTED→OPERATOR promotion gate 37t.11's injection-security depends on.
- **Name-collision & registered-kind hygiene.** New units must avoid existing names (`raw_artifacts` = source-tier ingest
  taxonomy, not session artifacts; `threads` = lineage table, so cross-origin is `cross_origin_threads`). Every new
  `AssertionKind`/`ObjectRefKind` must be a *registered* kind (`insight:` not an invented prefix) and gets a `user_audit`
  surface entry, or the every-kind audit invariant fails.

## 5. GPT-pro prompt fan-out (66 prompts; 11 as files)

Full list + bodies in `.agent/scratch/rnd-brainstorm-2026-07-05.md` (A1-A33 design, D1-D33 deepresearch).
Files ready to paste in `/realm/inbox/tmp/`: `00-INDEX`, `A09-webui-cockpit`, `A11-corpus-compaction`, `A16-measure-registry`,
`A17-query-finding-objects`, `A18-recursive-safety`, `A20-compaction-event`, `A27-episode-merge`, `D01-competitive-landscape`,
`D02-local-llm-stack`, `D07-rl-eval-environment`. Standing depth contract (prepend to each): ground in the bundle (cite
file:line/bead), tag [evidence] vs [proposal], one defended recommendation + runner-up, three layers (today / near-term /
full direction), end with Open-questions + What's-missing, flag staleness, don't re-derive swarm2. Deepresearch lanes worth
running: competitive landscape, local-LLM+embedding stack, retrieval for long transcripts, harness+interchange-format
landscape, citation-anchor standard, cost/subscription models, RL/eval environments, agent-memory prior art, construct-
validity-as-product, verifiable-deletion, summary-faithfulness, persisted-queries-as-objects, big-graph/PE/ambient-theming.

**Flywheel correction:** the browser extension already auto-captures GPT-pro chats into the archive — there is **no
manual export/re-ingest step**. The only manual work is making each R&D batch a retrievable cohort (title stub / tag) and
distilling captured sessions into beads. Recursive-safety gates the distilled assertions.

---

## 6. The critical path (my synthesis; the prioritization agent will sharpen)

The honesty red-team's strongest argument: **~80% of realized value is "agent-memory + cost-visibility"; ~70% of the
backlog (insight museum, result-set algebra, lineage normalization, export/portfolio cluster) is orthogonal** and may be
over-engineering. Take it seriously. Ranked by (operator-value ÷ build-cost + risk):

**Tier 0 — fix the lies (days, mostly code-only, no new capability):** cost-correctness (dead exact path, missing Opus
credit rates, logical-grain double-count, tier $/credit), honesty-audit completeness, security DNS-rebinding + receiver
token, the two temporal bugs, attachment referenced-vs-stored honesty, jnj.5 mutation contract. These are pure credibility
— the brand is "every number resolves to bytes" and several numbers currently lie.

**Tier 1 — the load-bearing substrate (unblocks everything):** the `ContextSource`/37t.11 scheduler (its trust-class gate
is the recursive-safety spine); blue-green index rebuild (b5l — makes every later derived-tier bump downtime-free);
the measure-registry (turns analytics into a fill-the-registry job and absorbs the cost fixes via LOGICAL_SESSION grain).

**Tier 2 — the differentiated wedge (the launch story):** the claim-vs-evidence `.polydemo` + demo-as-CI-test + the
hero-finding README (positioning); adoption/install (`polylogue install` + `doctor` — the substrate is worthless if agents
don't use it); MCP-collapse to 9 verbs (the continuity surface an agent can actually navigate).

**Tier 3 — the convergent frontier (high-leverage, higher cost):** queries+findings-as-objects (needs set-algebra first);
the episode unit; recursive-safety subsystem; content-hash citation anchors + webui cockpit.

**Defer / question (the red-team's cut list):** the result-set algebra beyond what findings-as-objects needs; the insight
"museum" (behavioral analytics run once and screenshotted); the export/portfolio cluster; the composer TUI (swarm2) until
the daemon/thin-client is real. Build these only when a live query demands them.

**Decisions only the operator can make:** (1) product identity — OSS-personal-substrate vs product (positioning agent says
OSS-by-necessity, defer monetization); (2) whether to accept the durable user v4→v5 / source v2→v3 migrations as a batch
now; (3) whether "daemon required" (swarm2) is the target or daemonless stays first-class.

---

## 7. State / pointers

- **Sinnix:** `mcp-polylogue` archive-root-leak fix committed `f6de855` (NOT switched — dirty tree has unrelated
  uncommitted work; NOT pushed — push-after-deploy). Applying needs `nix develop --command switch` and will sweep the
  other uncommitted changes; that's the operator's call.
- **CLAUDE.md rewrite:** standalone ~4K-token version merged to master (#2546); AGENTS.md is a symlink; render-agents
  machinery removed. The `fix(readiness)` commit is #2547.
- **Master idea doc:** `.agent/scratch/rnd-brainstorm-2026-07-05.md`. **This synthesis:** `.agent/scratch/rnd-master-synthesis-2026-07-05.md`.
- **Prompt files:** `/realm/inbox/tmp/`. **Agent transcripts:** `<session>/tasks/*.output` (read one at a time).
- **Live archive gotcha:** MCP still points at `/tmp/polylogue-archive` this session (the sinnix fix isn't deployed yet) —
  for live-archive CLI use `export POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue`.

## 8. Synthesis lanes — critical-path & launch verdicts (sharpen §6; folded post-hoc, hence after §7)

**Critical-path lane.** The red-team's 80/20 is *directionally right but its proposed core is too small*. The honest minimal core is NOT "SessionEnd-copy + FTS + 3 views + 1 tool" (that's `llm`-logs+grep) — it keeps the three cheap differentiators grep can't reconstruct: **`material_origin`** (makes honest cost/user-word accounting possible), the **~10 real MCP continuity tools** (save_annotation/get_resume_brief/recall — not one search tool), and **content-hash idempotency + split-tier durability** (lets the archive rebuild without losing irreplaceable user.db). What IS orthogonal and deferrable: the 16-measure "museum," result-set set-algebra, the content-variants/translation epic (cut), CIF/federation, RL-env, temporal-navigation UI. Sequence: **Weeks 1–2** recursive-safety gate (now load-bearing *safety* since the archive auto-captures its own R&D) + the confirmed bug track + activation layer; **Weeks 3–4** t46 contract-thinning + construct-validity gate + cost-per-outcome (`terminal_state × session_costs`, one materializer) + blue-green rebuilds + attachment-honesty-half; **later, gated on operator** citation-anchors + compaction-pack + findings-as-objects. Additional confirmed bugs it pinned (beyond §2): **`1xc.11` convergence probes fail-*closed* to 'converged'** (a probe error silently suspends auto-convergence — highest severity), **`a7xr.1` ~9 sqlite3 connection leaks** (`with connect()` commits-not-closes), **`4ts.6` lineage truncation at `_MAX_LINEAGE_DEPTH=64` with no completeness flag in the read envelope**.

**Whole-product/launch lane.** Identity = **OSS-by-necessity personal substrate, NOT a product** (no users/roadmap/funnel — the operator's success metric is his own daily use; a product identity distorts toward a stranger's onboarding funnel and reads as *less* serious to the fundability/hiring audience that judges code+findings). The **entire distribution stack is already built and idle** — PyPI/Homebrew/GHCR/FlakeHub/Nix all wired, but **zero git tags ever fired, version frozen 0.1.0**, and install docs say "no packaged path." So the whole remaining game is **decision, not build**: (1) fire the first release tag; (2) prove the install matrix on a cold machine + reconcile the "not available" docs; (3) de-meta the README (strip "claims stay capability-phrased until measured…" meta-sentences, replace agent-coined jargon) with every retained claim demonstrable via `demo tour` or a cited finding; (4) regenerate ONE flagship live-cited finding on v24; publish screencast (already recorded). Launch staircase: release-cutover → quiet uvx one-liner + cold-reader pass → data-story essay (spine = **forensics on real data**, e.g. the 24.1% silent-proceed finding, NOT memory-uplift — `cfk`/`jxe` uplift pilot closed *negative*, so all memory claims stay capability-phrased) → one sharp thread into the AI-memory discourse with the s7ae two-agent demo attached. First-10-users = heavy coding-agent operators + local-first/self-hosters + Claude-Code/Codex power-users. Monetization: **defer explicitly** (patronage/grant/employment-leverage; build no signup/telemetry/tiers — they contradict local-first and weaken the competence signal). Do NOT build: separate docs site as a launch gate, Datasette exhibit, interchange-schema, multi-model leaderboard, any new capability for launch.

## 9. Next actions (when you pick up)
1. Convert Tier-0 fixes to execution-grade beads and start landing them (they're code-only, high-credibility, low-risk).
2. Batch the durable migrations decision before touching user.db/source.db (see §4 "BATCH THE SCHEMA BUMPS").
3. **Provenance caveat (read §0.1):** the file:line citations in §1/§2 and the wave-4 entries in §3 are distilled from the
   pre-compaction summary, not re-read from source this session. Before acting on any specific citation, re-read that
   spec's `tasks/*.output` to confirm the file:line — the task files are the authority, this doc is the index.


=====================================================================
  PART 2 - PAYLOAD: 16 subagent outputs (unique to file 3 of 6)
=====================================================================


---

## [a2a0ff0e0207773a7] SPEC — Queries & Findings as First-Class Objects

I have enough grounding. Confirmed: no `result_sets`/`query_edges`/`StandingQueryStage`/`AssertionKind.FINDING` exist yet — this is greenfield built on real seams. Here is the implementation-grade spec.

---

# SPEC — Queries & Findings as First-Class Objects

**Anchor beads:** `fnm.13` (set-algebra, design-ready), `fnm.9` (pipeline-as-subquery, enabling primitive), `4p1` (one read algebra), `4smp`/`rlsb` (variant identity key). **Depends on** the set-algebra layer landing first (it produces the EXPLAIN `set_op` nodes this spec's `query_edges` DAG consumes).

**Ground truth read:** `core/refs.py`, `core/enums.py:399` (AssertionKind), `storage/sqlite/archive_tiers/user_write.py` (candidate→judge→promote lifecycle), `storage/sqlite/archive_tiers/index.py` (INDEX_DDL, `INDEX_SCHEMA_VERSION=24`), `daemon/convergence.py` (ConvergenceStage), `daemon/convergence_stages.py:558` (`make_default_convergence_stages`), `docs/design/query-set-algebra.md`, `insights/pathology.py`.

---

## 1. Schema / DDL, tier assignment, regime

The design splits cleanly across three tiers by durability. **Nothing here needs a durable-tier migration** — the only persistent user state is FINDING assertions, and `user.db`'s `assertions.kind` is plain `TEXT` (no CHECK), so a new `AssertionKind` member is a code+enum change, not a schema bump.

### 1a. `index.db` (derived, rebuildable) — bump `INDEX_SCHEMA_VERSION` 24 → 25

Content-addressed query registry + result-set snapshots + the DAG. All derivable from `user.db` durable intent (saved/standing queries) + source, so it lives in the rebuildable tier. Add to `INDEX_DDL` in `archive_tiers/index.py`:

```sql
-- Content-addressed query object. Identity = hash of the CANONICAL,
-- macro-EXPANDED, lowered form (see §2a). Immutable: a changed query text
-- that lowers to a different canonical form is a *different* query object.
CREATE TABLE IF NOT EXISTS query_defs (
    query_hash      TEXT PRIMARY KEY,          -- "query:<sha256>" object_id (bare hash)
    grain           TEXT NOT NULL,             -- 'session' | 'message' | 'block'  (matches set-algebra §2.1)
    canonical_text  TEXT NOT NULL,             -- re-parseable canonical serialization
    ast_json        TEXT NOT NULL,             -- lowered ParsedQueryExpression / SessionQuerySpec, post-expansion
    has_set_ops     INTEGER NOT NULL DEFAULT 0,
    first_seen_ms   INTEGER NOT NULL
) STRICT;

-- The set-algebra DAG. One row per EXPLAIN set_op / subquery edge (§2c).
-- parent = the composite query; operand = a child query object.
CREATE TABLE IF NOT EXISTS query_edges (
    parent_hash     TEXT NOT NULL REFERENCES query_defs(query_hash) ON DELETE CASCADE,
    operand_hash    TEXT NOT NULL REFERENCES query_defs(query_hash) ON DELETE CASCADE,
    edge_kind       TEXT NOT NULL,             -- 'union'|'intersect'|'except'|'subquery'|'lift'
    operand_pos     INTEGER NOT NULL,          -- 0=left,1=right; ordinal for n-ary
    PRIMARY KEY (parent_hash, operand_pos, operand_hash)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_query_edges_operand ON query_edges(operand_hash);

-- A materialized snapshot of one query's result set at one corpus version.
-- Merkle root = sha256 over sorted member identity keys (+ ranks when rank-bearing).
CREATE TABLE IF NOT EXISTS result_sets (
    result_set_id   TEXT PRIMARY KEY,          -- deterministic: sha256(query_hash || corpus_version)
    query_hash      TEXT NOT NULL REFERENCES query_defs(query_hash) ON DELETE CASCADE,
    grain           TEXT NOT NULL,
    corpus_version  INTEGER NOT NULL,          -- monotonic archive content version at snapshot time
    merkle_root     TEXT NOT NULL,
    member_count    INTEGER NOT NULL,
    rank_policy     TEXT NOT NULL,             -- 'left'|'rrf'|'insertion' (mirrors set-algebra §3.1)
    computed_at_ms  INTEGER NOT NULL,
    UNIQUE (query_hash, corpus_version)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_result_sets_query ON result_sets(query_hash, corpus_version DESC);

-- Keyed, ranked members. Bounded by the per-operand row cap (set-algebra §6).
CREATE TABLE IF NOT EXISTS result_set_members (
    result_set_id   TEXT NOT NULL REFERENCES result_sets(result_set_id) ON DELETE CASCADE,
    member_key      TEXT NOT NULL,             -- session_id | "sid::mid[::vidx]" | "sid::mid::bidx"
    rank            INTEGER NOT NULL,
    sort_key        TEXT,                      -- NULL sorts last (mirrors occurred_at_ms IS NULL)
    PRIMARY KEY (result_set_id, member_key)
) STRICT;
```

`corpus_version` is a cheap monotonic derived from ingest activity — reuse the ops-tier ingest cursor high-water mark (or `MAX(sessions.updated_at_ms)` as fallback). It is the staleness axis for the StandingQueryStage: a snapshot with `corpus_version < current` is stale.

### 1b. `user.db` (durable, irreplaceable) — enum-only, **no DDL change**

- `core/enums.py`: add `FINDING = "finding"` to `AssertionKind` (after `PATHOLOGY`).
- FINDING assertions reuse the existing `assertions` table verbatim via `upsert_assertion`. A finding **targets** what it asserts about (`target_ref = session:… | query:<hash>`), **scopes** to its detector (`scope_ref = insight:<detector>@vN` or `query:<hash>` for standing-query findings), and carries `evidence_refs = [query:<hash>, …EvidenceRefs]`.
- **Standing-query definitions** are durable user intent → stored as a `SAVED_QUERY` assertion (existing kind) whose `value_json` gains `{"standing": true, "schedule_ms": …, "expected": {…}}`. No new kind, no DDL.

### 1c. `ops.db` (disposable) — standing-query run bookkeeping

```sql
CREATE TABLE IF NOT EXISTS standing_query_runs (
    query_hash        TEXT PRIMARY KEY,
    saved_view_ref    TEXT NOT NULL,           -- assertion:<id> of the SAVED_QUERY(standing)
    last_run_ms       INTEGER,
    last_result_set_id TEXT,
    last_corpus_version INTEGER,
    next_due_ms       INTEGER,
    consecutive_errors INTEGER NOT NULL DEFAULT 0
) STRICT;
```

### 1d. `core/refs.py` — new ObjectRef kind

Add `"query"` to the `ObjectRefKind` Literal (line 8) **and** the `_OBJECT_REF_KINDS` dict (line 43). `query:<sha256hash>` — no qualifiers. Findings need **no** new ref kind; a finding is `assertion:<id>` (its target/scope/evidence carry the `query:` ref). This keeps the ref surface minimal and matches how pathology/transform candidates already work.

**Regime classification:** `index.db` change is **additive-derived** → edit canonical `INDEX_DDL`, bump version, add a rebuild-plan line; `polylogue ops reset --index && polylogued run` (per `docs/internals.md` derived regime, enforced by `devtools lab policy schema-versioning` — do **not** write an upgrade helper). `user.db`/`ops.db` changes are additive and require no numbered migration (TEXT vocab / disposable tier).

---

## 2. Algorithms (pseudocode)

### 2a. Query identity — `query:<hash>` keyed on the expanded AST

```
def query_object_id(raw_expression: str) -> str:
    parsed   = _QUERY_PARSER.parse(raw_expression)        # expression.py Lark front door
    expanded = expand_macros(parsed)                       # fnm.12 @cohort → inline; identity if no macros
    lowered  = lower_to_spec(expanded)                     # SessionQuerySpec + pipeline stages, post _canonicalize_*
    canonical = canonical_serialize(lowered)               # sorted keys, NFC-normalized (reuse core/hashing.py rules)
    h = sha256(canonical.encode NFC).hexdigest()
    return h            # ObjectRef(kind="query", object_id=h)
```

Keying on the **lowered, expanded** form means `@arm_pack | intersect (@arm_raw)` and its manual inlining collide, and `auth and test` == `test and and auth`-after-canonicalization collide. Two textually different queries that mean the same thing share one `query:` object — the whole point of content-addressing.

### 2b. Register query + result-set snapshot (Merkle)

```
def snapshot(conn_index, query_hash, plan, corpus_version) -> result_set_id:
    upsert query_defs(query_hash, grain, canonical_text, ast_json, has_set_ops, first_seen_ms)  # idempotent
    members = plan_execution.run(plan)          # existing plan path; set-op stages already materialize keyed sets
    keyed   = [(identity_key(row, grain), rank, sort_key) for rank, row in enumerate(members)]
    keyed.sort(by member_key)                   # canonical order for Merkle
    merkle  = sha256( "\n".join(f"{k}\x00{r if rank_bearing else ''}" for k,r,_ in keyed) )
    rsid    = sha256(query_hash || ":" || corpus_version)
    if exists result_sets[rsid] and .merkle_root == merkle: return rsid   # idempotent no-op
    INSERT result_sets(rsid, query_hash, grain, corpus_version, merkle, len, rank_policy, now)
    INSERT result_set_members(rsid, ...keyed)
    return rsid
```

### 2c. Populate `query_edges` from the set-algebra EXPLAIN

The set-algebra layer (`fnm.13`) adds a `set_op` node to `plan_description.py` (design §6.5). Walk that plan tree:

```
def record_edges(conn, parent_hash, plan):
    for node in plan.walk():
        if node.type in {"set_op", "subquery"}:      # set_op has operands = sub-plans
            for pos, operand_plan in enumerate(node.operands):
                oh = query_object_id(operand_plan.canonical_text)
                register(operand_plan); snapshot(operand_plan, corpus_version)   # operands are query objects too
                upsert query_edges(parent_hash, oh, node.edge_kind, pos)
```

This makes the DAG a faithful projection of what EXPLAIN already shows (two sub-plans joined by a set-op node, never a cross join — set-algebra §6.5 AC).

### 2d. Finding lifecycle — reuse pathology candidate→judge→promote **verbatim**

FINDING slots into the exact machinery in `user_write.py`. Mirror `upsert_pathology_findings_as_assertions` (line 1060):

```
def upsert_query_findings_as_assertions(conn_user, query_hash, findings, now_ms):
    for f in findings:
        aid = deterministic_id(f"assertion-{FINDING}", query_hash, f.kind, str(f.detector_version), f.detail, *evidence)
        existing = read_assertion_envelope(conn, aid)
        if existing and existing.status != CANDIDATE: continue   # never re-downgrade a promoted finding
        upsert_assertion(conn,
            assertion_id=aid,
            scope_ref=f"query:{query_hash}",
            target_ref=f.target_ref or f"query:{query_hash}",
            key=f.kind,
            kind=AssertionKind.FINDING,
            value={"query_hash": query_hash, "expected": f.expected, ...},   # value.expected = invariant (§2e)
            body_text=f.detail,
            author_ref=f"query:{query_hash}", author_kind="query",
            evidence_refs=[f"query:{query_hash}", *f.evidence_refs],
            status=CANDIDATE, visibility=PRIVATE,
            context_policy={"inject": False, "promotion_required": True})
```

Promotion is the **existing** `judge_assertion_candidate` (line 1245) — accept/reject/defer/supersede, `_promote_candidate_assertion` clears `promotion_required`, records a `JUDGMENT` assertion, sets `supersedes`. Register `FINDING` in `ASSERTION_CLAIM_KINDS` (line 1520) so it flows through `list_assertion_candidates`/`list_assertion_candidate_reviews` unchanged.

### 2e. Findings-as-tests — `value.expected` → re-runnable invariant

A promoted FINDING with `value.expected` is a query invariant. The StandingQueryStage re-checks it:

```
def check_finding_invariant(finding, fresh_result_set):
    exp = finding.value["expected"]     # one of: {"merkle_root": h} | {"member_count": n} | {"contains":[keys]} | {"excludes":[keys]}
    match exp:
        "merkle_root": ok = fresh.merkle_root == exp.merkle_root
        "member_count": ok = fresh.member_count == exp.member_count
        "contains":     ok = set(exp.contains) <= fresh.member_keys
        "excludes":     ok = set(exp.excludes).isdisjoint(fresh.member_keys)
    if not ok:
        emit_regression(finding, fresh_result_set)   # §2f
```

### 2f. `StandingQueryStage` — DaemonConverger integration

New stage via `make_standing_query_stage(db_path)`, appended in `make_default_convergence_stages` (`convergence_stages.py:558`). `false_means_pending=True` (bounded work, backlog → convergence_debt), `cpu_bound=False` (it writes index + user + ops), session-scoped variant optional.

```
def check(path) -> bool:
    cv = current_corpus_version()
    for sq in load_standing_queries(conn_user):          # SAVED_QUERY where value.standing
        run = ops.standing_query_runs[sq.query_hash]
        if run is None or run.last_corpus_version < cv:   # snapshot stale vs corpus
            return True
    return False

def execute(path) -> bool:                                # bounded window (mirror embed stage caps)
    cv = current_corpus_version(); budget = STANDING_QUERY_MAX_PER_PASS
    stale = [sq for sq in load_standing_queries() if is_stale(sq, cv)]
    for sq in stale[:budget]:
        plan = compile_expression(sq.canonical_text)
        rsid = snapshot(conn_index, sq.query_hash, plan, cv)       # §2b
        record_edges(conn_index, sq.query_hash, plan)             # §2c
        prev = ops.last_result_set_id(sq.query_hash)
        diff = diff_result_sets(prev, rsid)                       # added/removed member_keys via Merkle short-circuit
        findings = derive_findings(sq, rsid, diff)                # standing-query findings (new members, invariant breaks)
        upsert_query_findings_as_assertions(conn_user, sq.query_hash, findings)
        for inv in active_findings_with_expected(sq.query_hash):  # §2e re-check promoted invariants
            check_finding_invariant(inv, load(rsid))
        ops.record_run(sq.query_hash, rsid, cv)
    return len(stale) <= budget      # False (pending) when backlog remains → convergence_debt retries
```

`emit_regression` writes a `CAVEAT`/`BLOCKER` FINDING candidate citing `query:<hash>` + the diff — surfaced through the existing candidate-review MCP/CLI surface. Merkle short-circuit: equal roots ⇒ no diff work.

---

## 3. Migration / rebuild plan

1. **`user.db`/`ops.db`:** no numbered migration. `FINDING` enum member is additive; `standing_query_runs` is a disposable-tier `CREATE TABLE IF NOT EXISTS`.
2. **`index.db` 24→25 (derived regime):** edit canonical `INDEX_DDL` (add the four tables), bump `INDEX_SCHEMA_VERSION=25` in `archive_tiers/index.py:36`. Add a line to the index rebuild plan. **No upgrade helper** — `devtools lab policy schema-versioning` rejects derived-tier upgrade paths. Deploy path: `polylogue ops reset --index && polylogued run`.
3. **`ops reset --index` interaction (the load-bearing correctness point):**
   - On reset, `query_defs`/`query_edges`/`result_sets`/`result_set_members` are **dropped and rebuilt from source** — as the derived tier must be.
   - Durable intent survives in `user.db`: **standing-query definitions** (`SAVED_QUERY` assertions) and **promoted FINDING assertions** (with their `value.expected` invariants and `evidence_refs=[query:<hash>]`).
   - On the next `polylogued run`, `StandingQueryStage.check` sees every standing query as stale (`run is None`), re-registers each `query_def`, re-materializes `result_sets`, and re-checks invariants. FINDING candidates re-emit idempotently by deterministic id; promoted findings are never re-downgraded (the `existing.status != CANDIDATE: continue` guard).
   - Ad-hoc (unsaved) query snapshots are lost by design — they are disposable. This is the same rebuild contract as FTS/embeddings.
4. **Regenerate generated surfaces:** new `AssertionKind` is embedded in `render openapi` + `render cli-output-schemas`; new `polylogue/` modules break the topology projection. Run `devtools render topology-projection && devtools render topology-status && devtools render openapi && devtools render cli-output-schemas && devtools render all --check` (grep for `out of sync` — the tail line lies).

---

## 4. Test strategy

- **Query identity (property):** `query_object_id` is stable under whitespace/macro-expansion/predicate-reordering that lowers to the same canonical form; distinct meanings ⇒ distinct hashes. Hypothesis strategy over the DSL grammar. (Protected-file-style, like `test_parsers_props.py`.)
- **Merkle determinism:** snapshot the same query at the same `corpus_version` twice ⇒ byte-identical `merkle_root` and idempotent no-op INSERT. Reorder-invariant (members sorted by key before hashing).
- **query_edges = EXPLAIN parity:** parametrized over `{union,intersect,except} × {fts,semantic,structural,macro}` (reuse the set-algebra §10 matrix) — assert `query_edges` rows exactly mirror the EXPLAIN `set_op` sub-plans (two operands, correct `edge_kind`/`operand_pos`), never a cross-join edge.
- **Finding lifecycle:** candidate→`judge_assertion_candidate(accept)`→promoted ACTIVE with `promotion_required` cleared + `JUDGMENT` row + `supersedes` set; reject leaves no active assertion; rebuild-idempotency (deterministic id; promoted finding not re-downgraded). Reuse `test_claim_vs_evidence.py` / `test_archive_tiers_user_write.py` patterns.
- **Findings-as-tests:** promote a FINDING with `expected.member_count=N`; mutate the corpus (add a matching session); StandingQueryStage flags the invariant break and emits a regression candidate citing `query:<hash>` + diff.
- **Standing-query convergence:** `check` returns True only when a standing query's snapshot `corpus_version < current`; `execute` with `false_means_pending=True` pushes over-budget backlog to `convergence_debt` and retries. Use `frozen_clock`.
- **ops reset --index survival (integration):** seed standing query + promoted finding → `ops reset --index` → `polylogued run` → assert `result_sets` re-materialized, finding still ACTIVE, invariant re-checked. This is the durability contract test.
- **Cross-surface parity:** the `fnm.11` matrix must route a `query:<hash>` resolve + finding-review through CLI/MCP/API/daemon identically (`resolve_ref`, `list_assertion_claims`).

Inner loop: `devtools test tests/unit/storage/test_archive_tiers_user_write.py -k finding` etc., never blanket directories.

---

## 5. Bead breakdown (propose under a new epic; parent chain `fnm` → this)

> Epic: **"Queries and findings as first-class, content-addressed objects"** — `relates-to` `fnm.13`/`fnm.9`/`4p1`. Sequenced: (1) blocks (2)(3); (3) blocks (4); (5) blocks (6).

1. **`query:` ObjectRef + content-addressed query identity.** Add `"query"` to `core/refs.py` Literal+dict; implement `query_object_id` keyed on lowered+macro-expanded canonical form; `resolve_ref` returns the `query_def`. *AC:* two DSL strings lowering to one canonical form share one `query:` id (property test); `resolve_ref("query:<h>")` returns canonical_text+grain+ast; render surfaces regenerated.
2. **`index.db` v25 DDL: query_defs / query_edges / result_sets / result_set_members + Merkle snapshot writer.** Bump `INDEX_SCHEMA_VERSION`; edit canonical `INDEX_DDL`; add rebuild-plan line; implement `snapshot()`. *AC:* `schema-versioning` policy green (no upgrade helper); Merkle deterministic + reorder-invariant + idempotent; `ops reset --index && polylogued run` recreates tables.
3. **`query_edges` DAG from the set-algebra EXPLAIN.** Consume `fnm.13`'s `set_op` plan nodes; `record_edges` walk. *AC:* the `{union,intersect,except}×{fts,semantic,structural,macro}` matrix produces edge rows exactly mirroring EXPLAIN sub-plans; no cross-join edge. **Blocked on `fnm.13` landing.**
4. **`AssertionKind.FINDING` + finding candidate→judge→promote reuse.** Add enum member; `upsert_query_findings_as_assertions`; register in `ASSERTION_CLAIM_KINDS`; regenerate openapi+cli-output-schemas. *AC:* finding flows through existing `judge_assertion_candidate`/candidate-review surfaces; promoted finding survives rebuild; deterministic-id idempotency.
5. **Query versioning via `supersedes`.** When a saved/standing query's canonical text changes → new `query:` object; the `SAVED_QUERY` assertion records `supersedes=[query:<old>]`; old findings marked superseded. *AC:* audit chain queryable; superseded findings excluded from active review; no orphaned invariants.
6. **`StandingQueryStage` in DaemonConverger + findings-as-tests.** `make_standing_query_stage` in `make_default_convergence_stages`; `corpus_version` staleness; `value.expected` invariant re-check; regression emission; `standing_query_runs` bookkeeping. *AC:* bounded-window `false_means_pending` backlog → convergence_debt; invariant break emits a regression candidate citing `query:<hash>`+diff; `frozen_clock` tests. **Blocked on (2)+(4).**

---

## 6. Top-3 risks

1. **Canonical-form drift breaks content-addressing (highest).** If `query_object_id`'s canonical serialization isn't perfectly stable across DSL/lowering/spec refactors, the same query gets a new hash — orphaning every result_set snapshot and every FINDING `evidence_ref` on the old hash, silently breaking findings-as-tests. **Mitigation:** hash the *lowered SessionQuerySpec* (already the canonicalization boundary via `_canonicalize_*`), not raw text; a golden-file property test pins the canonical bytes for a fixed corpus of queries so any drift fails CI; reuse `core/hashing.py`'s NFC/sentinel rules exactly. Treat a canonicalization change as a query-object epoch bump with a `supersedes` migration, mirroring the pathology `detector_version` discipline.
2. **`corpus_version` staleness is coarse → standing queries thrash or go stale.** A single global monotonic version means *any* ingest re-runs *every* standing query (thrash), while a too-lazy signal leaves invariants unchecked after relevant changes. **Mitigation:** scope staleness to the query's grain/lanes where cheap (e.g., only re-run when sessions matching the query's origin/repo filters changed), fall back to global version; run inside the existing bounded embed-style window with `false_means_pending` so cost is capped regardless; add a cursor-lag sample like the existing convergence telemetry.
3. **`ops reset --index` loses ad-hoc snapshots that a finding cited (correctness-of-evidence).** A FINDING (durable, user.db) can cite a `query:<hash>` whose `result_set` lived only in index.db and is gone post-reset until re-run. Between reset and the next `polylogued run`, `resolve_ref` on that result_set dangles. **Mitigation:** findings must cite the *query object* + `EvidenceRef`s (durable-resolvable), never a `result_set_id`; standing-query findings are re-materialized on next run; document that ad-hoc query snapshots are disposable (same contract as FTS/embeddings) and gate finding *promotion* on the query being saved/standing so no promoted invariant depends on a disposable snapshot.

**Key files to touch:** `core/refs.py`, `core/enums.py`, `storage/sqlite/archive_tiers/index.py`, `storage/sqlite/archive_tiers/user_write.py`, `archive/query/expression.py` + `plan_description.py`, `daemon/convergence_stages.py`, plus generated-surface regen. No durable-tier migration required.

---

## [aee6f764cca4c9ab4] Wave-5 Spec: Archive-Economics / Tiering / Compression Engine

I have full grounding. Here is the spec.

---

# Wave-5 Spec: Archive-Economics / Tiering / Compression Engine

**Horizon:** 10-year / 500 GB. **Anchor invariant (never broken):** every blob address stays `SHA-256(uncompressed_bytes)`. Compression, tiering, cold-sharding, and epoch-packing are all *physical-representation* concerns invisible to identity. Grounded in `blob_store.py`, `blob_gc.py`, `raw_retention.py`, `archive_tiers/{source,index,embeddings,ops,bootstrap}.py`, beads `polylogue-83u.5`/`83u.4`.

## Current substrate (verified)

- `BlobStore` (`blob_store.py`): sharded `{root}/{hash[:2]}/{hash[2:]}`, atomic tempfile+`os.replace`, `.blob.*` temps skipped by walks. `verify()`/`verify_all()` re-hash raw file bytes. `write_from_bytes/path/fileobj` return `(sha_hex, byte_count)`. **No compression today** — raw provider JSON stored verbatim (~36 GB).
- `run_blob_gc_report` (`blob_gc.py`): 4 safety invariants (DB-ref check across `raw_sessions`+`blob_refs` in current **and** `source.db`; active-lease `pending_blob_refs`; generation high-water + `MIN_AGE_S=60`; `max_batch`). Durable record = typed `gc_generations(reclaimed_count, reclaimed_bytes)`. `reclaimed_bytes` currently = physical `stat().st_size`. `sweep_orphaned_blob_leases` (`ORPHAN_LEASE_MAX_AGE_S=3600`).
- `raw_retention.py`: superseded live-snapshot compaction, `ROW_NUMBER() PARTITION BY (source_path, source_index)`, keep N full/append, guarded by source-file existence.
- Tiers (`bootstrap.py`): `DurabilityClass ∈ {irreplaceable, rebuildable, expensive_rebuild, human, disposable}`; `backup_required` flag. **source v2** (irreplaceable/backup), index v24 (rebuildable), **embeddings v1** (expensive_rebuild/backup, `vec0 float[1024]`), user v4 (human/backup), ops v1 (disposable). Durable tiers use numbered additive migrations behind a backup manifest; derived tiers rebuild.
- Index `attachments.acquisition_status ∈ {acquired, unavailable, unfetched}`, `blob_hash` honest-nullable. **zstandard not currently installed** (pure wheel exists per 83u.5).

---

## (1) Schema / DDL / tier / regime

Compression itself needs **no schema change** (self-identifying frames). Everything else is *additive to durable tiers* (source v2→v3, migration + backup manifest) or *disposable/derived*.

**Storage class is orthogonal to durability.** Introduce a `storage_class` axis on the physical layer only:

**source.db (v2→v3, additive migration `003_blob_economics.sql`, backup-gated):**
```sql
-- Per-origin trained dictionaries; needed to decompress dict-framed blobs, so backup_required.
CREATE TABLE IF NOT EXISTS blob_dicts (
    dict_id       INTEGER PRIMARY KEY,     -- matches zstd frame dictID
    origin        TEXT NOT NULL CHECK (...Origin...),
    trained_at_ms INTEGER NOT NULL,
    sample_count  INTEGER NOT NULL,
    dict_bytes    BLOB NOT NULL,           -- the dictionary itself (tens–hundreds KB)
    level_hint    INTEGER NOT NULL DEFAULT 3
) STRICT;

-- Physical placement + economics, keyed by the UNCHANGED content address.
CREATE TABLE IF NOT EXISTS blob_placement (
    blob_hash        BLOB PRIMARY KEY CHECK(length(blob_hash)=32),
    storage_class    TEXT NOT NULL DEFAULT 'hot'
                       CHECK(storage_class IN ('hot','cold','frozen','dropped')),
    codec            TEXT NOT NULL DEFAULT 'raw'
                       CHECK(codec IN ('raw','zstd','zstd_dict')),
    dict_id          INTEGER REFERENCES blob_dicts(dict_id),
    logical_bytes    INTEGER NOT NULL CHECK(logical_bytes>=0),   -- == old blob_size, SHA input length
    stored_bytes     INTEGER NOT NULL CHECK(stored_bytes>=0),    -- on-disk after codec
    segment_id       TEXT,                                       -- non-null iff storage_class='frozen'
    last_read_at_ms  INTEGER,                                    -- access-temperature signal
    reacquirability  TEXT NOT NULL DEFAULT 'unknown'
                       CHECK(reacquirability IN ('irreplaceable','reacquirable','unknown'))
) STRICT;

-- Citable tombstone: a dropped-but-re-derivable blob still resolves to evidence.
CREATE TABLE IF NOT EXISTS blob_tombstones (
    blob_hash      BLOB PRIMARY KEY CHECK(length(blob_hash)=32),
    origin         TEXT NOT NULL,
    source_path    TEXT,               -- where it can be re-acquired (~/.claude/...)
    logical_bytes  INTEGER NOT NULL,
    dropped_at_ms  INTEGER NOT NULL,
    drop_reason    TEXT NOT NULL,      -- 'slo_pressure' | 'cold_reacquirable' | ...
    reacquirability TEXT NOT NULL
) STRICT;

-- Yearly-epoch frozen segments (append-only concatenated packs).
CREATE TABLE IF NOT EXISTS frozen_segments (
    segment_id    TEXT PRIMARY KEY,     -- e.g. 'epoch-2024'
    epoch_year    INTEGER NOT NULL,
    segment_path  TEXT NOT NULL,        -- cold/frozen/epoch-2024.pack
    dict_id       INTEGER REFERENCES blob_dicts(dict_id),
    entry_count   INTEGER NOT NULL,
    logical_bytes INTEGER NOT NULL,
    stored_bytes  INTEGER NOT NULL,
    sealed_at_ms  INTEGER NOT NULL,
    checksum      BLOB NOT NULL         -- SHA-256 of the whole pack, verifies frozen integrity
) STRICT;
CREATE TABLE IF NOT EXISTS frozen_segment_entries (
    segment_id  TEXT NOT NULL REFERENCES frozen_segments(segment_id),
    blob_hash   BLOB NOT NULL CHECK(length(blob_hash)=32),
    offset      INTEGER NOT NULL,
    length      INTEGER NOT NULL,       -- stored (compressed) length within the pack
    PRIMARY KEY(blob_hash)
) STRICT;
```
Regime: **all of the above are source-tier durable** (they gate decompression / evidence integrity), so they go through the numbered-migration + verified-backup-manifest path (`DURABLE_MIGRATION_TIERS`, `migration_runner`). `blob_placement.last_read_at_ms` is a soft signal — acceptable to lose, but co-located for locality.

**ops.db (v1→v2, disposable — edit DDL directly, no migration chain):** retention/rollup tables:
```sql
CREATE TABLE storage_growth_samples (          -- feeds the ledger insight
    sampled_at_ms INTEGER NOT NULL, tier TEXT NOT NULL, origin TEXT,
    logical_bytes INTEGER NOT NULL, stored_bytes INTEGER NOT NULL,
    blob_count INTEGER NOT NULL, PRIMARY KEY(sampled_at_ms, tier, origin)
) STRICT;
CREATE TABLE blob_compact_runs (               -- one row per `blob-compact` walk
    run_id TEXT PRIMARY KEY, started_at_ms INTEGER, completed_at_ms INTEGER,
    verified INTEGER, recompressed INTEGER, reclaimed_bytes INTEGER,
    promoted_cold INTEGER, frozen INTEGER, dropped INTEGER, errors INTEGER
) STRICT;
```
Plus an ops-tier **rollup/prune** of high-cardinality disposable rows (`otlp_spans/telemetry`, `cursor_lag_samples`, `daemon_stage_events`, `storage_growth_samples`): keep raw ≤90 d, then daily rollup, then drop — this is the ops.db retention/rollup deliverable and stops the disposable tier from being a silent growth source over 10 years.

**embeddings.db (v1→v2, `expensive_rebuild` but rebuild-not-migrate regime):** add a quantized variant table beside the float32 `vec0`:
```sql
CREATE VIRTUAL TABLE message_embeddings_i8 USING vec0(embedding int8[1024]);   -- 4× smaller
-- optional binary lane for coarse recall:
CREATE VIRTUAL TABLE message_embeddings_bit USING vec0(embedding bit[1024]);   -- 32× smaller
```
Quantization is **derivable from float32** and float32 is itself re-derivable from source via Voyage re-embed; so schema bump = edit canonical DDL + rebuild plan, never an upgrade helper (`schema-versioning` policy rejects helpers on derived tiers).

---

## (2) Compress / tier / repack algorithms (pseudocode)

**Read path — codec sniffing (the one change that touches every reader):**
```
def read_blob(hash):
    place = blob_placement[hash]              # cheap; falls back to 'raw/hot' if absent
    if place.storage_class == 'frozen':
        seg, off, ln = frozen_segment_entries[hash]
        frame = read_range(frozen_segments[seg].segment_path, off, ln)
    elif place.storage_class == 'dropped':
        raise BlobDropped(tombstone=blob_tombstones[hash])   # citable, not silent
    else:
        frame = read_file(resolve_path(hash, place.storage_class))  # hot shard or cold/ shard
    # Self-identifying: sniff first 4 bytes regardless of metadata (metadata is an optimization,
    # frame magic is truth — survives a lost/rebuilt placement row).
    if frame[:4] == b'\x28\xB5\x2F\xFD':
        return zstd_decompress(frame, dicts=blob_dicts)   # dictID is inside the frame header
    return frame                                          # legacy raw
```
`resolve_path` checks the hot sharded path, then `cold/{hash[:2]}/{hash[2:]}`. `verify()`/`verify_all()` become **decompress-then-hash** (this is the "silently breaks if missed" trap called out in 83u.5).

**Ingest-time compression (fast lane, level 3):**
```
def store_blob(bytes):
    h = sha256(bytes)                          # ADDRESS = uncompressed hash, unchanged
    if exists(h): return h, len(bytes)
    if len(bytes) > 512 and not is_zstd(bytes):
        dict = current_dict_for(origin)        # may be None early
        frame = zstd_compress(bytes, level=3, dict=dict)
        codec = 'zstd_dict' if dict else 'zstd'
        write_atomic(hot_path(h), frame)
        record_placement(h, codec, dict_id, logical=len(bytes), stored=len(frame), 'hot')
    else:
        write_atomic(hot_path(h), bytes); record_placement(h, 'raw', logical=stored=len(bytes))
    return h, len(bytes)                        # return LOGICAL length (SHA input) — callers unchanged
```

**Single maintenance walk — `ops maintenance blob-compact` (verify → recompress → temperature → GC, one shard traversal):**
```
def blob_compact(limit=None, enable={verify,recompress,cold,freeze,drop,gc}):
    run = begin_blob_compact_run()
    for hash, path in walk_shards(hot, cold):          # reuses _candidate_blobs traversal
        if leased(hash): continue                      # honor pending_blob_refs (83u.5)
        frame = read_file(path)
        if 'verify' and sha256(decompress(frame)) != hash:   # integrity, decompress-then-hash
            run.errors += 1; log_corruption(hash); continue
        if 'recompress' and codec(frame)=='raw' and len(payload)>512:
            better = zstd_compress(payload, level=9, dict=dict_for(origin))
            atomic_replace(path, better); update_placement(stored=len(better))  # temp+rename, hash re-verified pre-swap
            run.recompressed += 1; run.reclaimed_bytes += saved
        temp = temperature(hash)                       # from last_read_at_ms
        if 'cold' and temp.age > 18mo and class=='hot':
            recompress(level=19); move(path -> cold_path(hash)); set_class('cold'); run.promoted_cold += 1
        if 'drop' and slo_pressure and reacquirability(hash)=='reacquirable' and source_still_live(hash):
            write_tombstone(hash, reason='slo_pressure'); unlink(path); set_class('dropped'); run.dropped += 1
    if 'gc': run_blob_gc_report(...)                   # SAME walk end — verify→recompress→GC in one job (83u.5 synergy)
    if 'freeze': freeze_cold_epochs()                  # see below
    finalize(run)   # -> ops.blob_compact_runs
```

**Per-origin dictionary training (one-shot / periodic):**
```
def train_dict(origin):
    samples = sample_raw_payloads(origin, n≈1000, decompressed)
    dict = zstd.train_dictionary(target_size=112KiB, samples)
    dict_id = insert blob_dicts(origin, dict_bytes=dict)   # dictID embedded in future frames
    # existing blobs are re-dict-compressed lazily by the next blob-compact recompress pass
```

**Yearly-epoch freeze (fully-cold years → frozen read-only segment):**
```
def freeze_cold_epochs():
    for year in years where ALL blobs(year).class=='cold' and year < now.year - 1:
        seg = open_append_pack(f'cold/frozen/epoch-{year}.pack')
        for hash in blobs(year) sorted:
            frame = zstd_compress(payload, level=19, dict=dict_for(year))
            off = seg.append(frame); insert frozen_segment_entries(hash, seg, off, len(frame))
            unlink(cold_path(hash)); set_class('frozen', segment_id)
        seg.seal(); insert frozen_segments(checksum=sha256(pack))   # whole-pack integrity
```
Frozen segments are immutable, single-fd, backup-friendly (one file per decade-year instead of millions of inodes), and read via `(offset,length)` range reads.

**Per-origin bytes/session SLO circuit-breaker (ingest guard):**
```
def slo_check(origin):
    ratio = stored_bytes(origin) / session_count(origin)
    if ratio > SLO[origin]:                      # e.g. codex 5 MB/session
        circuit_open(origin)                      # stop acquiring NEW raw for origin
        emit daemon_event('slo_breach', origin, ratio)
        # operator ack required to resume; blob-compact drop pass may auto-relieve if reacquirable
```

---

## (3) Rollout (address unchanged — no schema break)

1. **Add `zstandard` dependency** (pure wheel; no native build). Land read-path sniffing + `verify` decompress-then-hash **first**, defaulting all writes to `raw` — proves readers tolerate both codecs with zero blobs compressed. This is the safe reversible beachhead.
2. **source v2→v3 additive migration** (`blob_placement`, `blob_dicts`, `blob_tombstones`, `frozen_segments*`) behind a verified backup manifest. Backfill `blob_placement` from existing `raw_sessions(blob_hash, blob_size)` as `raw/hot/logical=stored=blob_size`.
3. **Enable ingest-time level-3** compression for new writes. Address is still `SHA(uncompressed)` so dedup, GC ref-checks, and `raw_sessions.raw_id`/`blob_hash` are untouched — a re-ingest of identical content still hits the existing blob.
4. **One-shot level-9 backpressure pass** via `ops maintenance blob-compact --recompress` (bounded `--limit`, resumable, lease-aware). Expected 5–10× on raw provider JSON (36 GB → ~4–7 GB).
5. Train per-origin dicts, then a second recompress wave upgrades `zstd`→`zstd_dict`.
6. Cold/freeze/drop passes and the SLO breaker enable last, gated by operator flags.

**Why no break:** `blob_size` semantics are preserved (logical). `stored_bytes` is *new* information, not a redefinition. `gc_generations.reclaimed_bytes` continues to mean physical bytes freed (correct — it already stats the on-disk file). The only behavioral redefinitions are `verify*` (must decompress) and `BlobStore.stats().total_bytes` (now physical, add a `logical_bytes` field) — both internal.

---

## (4) Test strategy

- **Round-trip identity property (the keystone):** `∀ payload: read_blob(store_blob(payload)) == payload` across `raw`, `zstd`, `zstd_dict`, `cold`, `frozen`. Hypothesis strategy over payload bytes incl. `>512`, `<512`, already-zstd, empty. This directly guards the "address = SHA of uncompressed" invariant.
- **`verify()` catches corruption post-compression:** flip a byte in a compressed frame → `verify` must fail (decompress-then-hash), not pass on the compressed bytes. Extends `tests/unit/storage/test_crud.py`/blob_integrity tests.
- **GC lease/ref safety unchanged under compression** — re-run the acquire→commit race test (`polylogue-8jg9.2` bead) against compressed blobs; leases still block, `reclaimed_bytes` reflects physical stored size.
- **Migration test:** source v2→v3 on a seeded pre-migration archive; backfilled `blob_placement` rows are consistent; `schema-versioning` policy passes.
- **`blob-compact` idempotency & resumability:** running twice with `--limit` produces no double-count; a SIGKILL mid-walk leaves a valid archive (atomic temp+rename per blob).
- **Frozen segment integrity:** seal → whole-pack `checksum` verifies; range-read of each entry decompresses to its address.
- **Tombstone resolution:** dropped reacquirable blob raises `BlobDropped` carrying a citable tombstone (source_path present); evidence/citation surfaces render it, don't 500.
- **Quantization recall regression:** int8/bit lanes retain acceptable top-k overlap vs float32 on the demo corpus (bounded recall floor, not exact).
- **SLO breaker:** synthetic origin over ratio opens the circuit and blocks new raw acquisition without touching other origins.
- Inner loop is testmon-affected (`devtools test tests/unit/storage/...`), not blanket runs.

---

## (5) Bead breakdown (children of `polylogue-83u`; `83u.5` becomes the umbrella)

1. **`blob-codec-readpath`** — add `zstandard` dep; read-path magic-sniff decompress; `verify`/`verify_all` decompress-then-hash; `BlobStore.stats` logical-vs-stored. **AC:** round-trip property green across raw+zstd; a compressed corrupt blob fails `verify`; zero blobs yet compressed; `devtools verify --quick` clean. *(directly satisfies 83u.5 read-path clause)*
2. **`blob-placement-schema`** — source v2→v3 migration: `blob_placement`+`blob_dicts`+`blob_tombstones`; backfill from `raw_sessions`. **AC:** migration runs under backup manifest; every existing blob has a `raw/hot` placement row with `logical==blob_size`; policy `schema-versioning` passes.
3. **`blob-compact-walk`** — the single `ops maintenance blob-compact` (verify→recompress→GC one traversal), level-3 ingest / level-9 one-shot, lease-aware, resumable, `--limit`; `ops.blob_compact_runs`. **AC:** 36 GB store compacts ≥5× on real archive; two runs idempotent; SIGKILL-safe; honors `pending_blob_refs`. *(satisfies 83u.5 migration clause)*
4. **`per-origin-dicts`** — dictionary training + `zstd_dict` codec + lazy dict-upgrade recompress. **AC:** dict-framed blobs decompress via embedded dictID; measured extra shrink over origin-uniform level-9 on ≥2 origins.
5. **`storage-growth-ledger`** — `storage_growth_samples` sampler + a new `InsightType` in `insights/registry.py` (logical vs stored, per tier/origin, growth rate, projected 500 GB date). **AC:** ledger renders in plaintext/JSON/MCP from one descriptor; numbers reconcile with `blob_placement` sums.
6. **`cold-tier-and-slo`** — access-temperature (`last_read_at_ms`), `cold/` shard promotion (>18 mo, level-19), reacquirability classification + citable drop-tombstones, per-origin bytes/session SLO circuit-breaker. **AC:** cold blob reads transparently; dropped reacquirable blob raises citable `BlobDropped`; SLO breach opens one origin's circuit only.
7. **`epoch-freeze`** — yearly frozen segment packs + `(offset,length)` manifest + whole-pack checksum verify. **AC:** a fully-cold year seals into one pack; every entry range-reads to its address; sealed pack is immutable and inode-count drops.
8. **`embeddings-quantization`** — int8 (+optional bit) `vec0` lanes; derived-tier rebuild plan (no migration helper). **AC:** i8 lane ~4× smaller; recall floor met on demo corpus; embeddings rebuild plan documented, `schema-versioning` passes. *(fold `83u.4` reference-debt classifier as a prerequisite audit so the compact walk starts from a clean reference graph)*

Sequence: 1→2→3 are the load-bearing spine (ship the 5–10× win); 4–8 are independent follow-ons.

---

## (6) Top-3 risks

1. **`verify`/backup trust silently breaking (highest).** `verify_all` is the backup verifier's honesty gate; if any read/verify path hashes *compressed* bytes it will either false-fail every blob or (worse) false-pass corruption. Mitigation: bead #1 lands read+verify decompression **before** any blob is compressed, and the magic-sniff is authoritative over the `blob_placement.codec` hint so a lost/rebuilt placement row can't misroute decoding. This is the exact "silently breaks if missed" hazard flagged in 83u.5.
2. **Dictionary availability = decompression dependency.** `zstd_dict` frames are undecodable without their dictionary; a lost dict bricks every blob that used it. Mitigation: `blob_dicts` lives in the **backup_required source tier**, dicts are append-only/never-deleted, dictID is embedded in each frame, and a fallback "recompress dict→plain-zstd" recovery pass exists. Frozen-segment dicts are pinned by `frozen_segments.dict_id`.
3. **Cold/frozen/drop path-resolution regressions in GC and evidence citation.** `blob_gc` walks only the hot shard tree and cross-checks `raw_sessions`/`blob_refs`; moving blobs to `cold/` or into frozen packs, or dropping-with-tombstone, can make GC misclassify a live blob as orphan or make an evidence citation dangle. Mitigation: GC and `_candidate_blobs` must treat `cold/` as an additional root and skip `storage_class∈{frozen,dropped}` (owned by the compact walk, not GC); dropped blobs resolve through `blob_tombstones` so citations degrade to "re-acquirable from `<source_path>`" instead of a hard miss. Re-run the acquire→commit race test (`83u.2`/`8jg9.2`) against every storage class.

**One-line thesis:** identity is `SHA-256(uncompressed)` forever; compression, temperature, freezing, and dropping are physical layers under an unchanged address, so 36 GB → single-digit GB and a credible 500 GB / 10-year path are reachable *without a single identity or query-surface break* — the only real danger is a read/verify path that forgets to decompress.

---

## [a8775376233da2cea] Spec: `find <query> | compact` — corpus-compaction projection

I have full grounding. Here is the implementation spec.

---

# Spec: `find <query> | compact` — corpus-compaction projection

## 0. Placement in the read algebra

Polylogue's read surface is already the tripartite algebra in `polylogue/surfaces/projection_spec.py`: `QueryProjectionSpec = SelectionSpec × ProjectionSpec × RenderSpec`. Corpus-compaction is a **new point in that algebra**, not a new subsystem:

- **Query** = a multi-session `SelectionSpec` (the `find <query>` selection; dozens–hundreds of sessions, `LOGICAL_SESSION` grain after lineage dedup).
- **Projection** = a *fixed preset* `CompactionProjection` — `BodyPolicy` beyond the existing four: keep human turns + decisions + error→fix + outcomes, drop tool-spam by `material_origin`.
- **Render** = a new `RenderFormat.COMPACT` — one token-budget-fitted, decision-dense digest + a fidelity drop-manifest, aimed at an *external* LLM (the GPT-pro R&D lane), not at terminal reading.

Surface wiring: `compact` is a new **session-source pipeline terminal action** alongside `rows`/`count`, i.e. `find <query> | compact` lowers to `SessionQueryPipeline(terminal=SessionQueryTerminalStage(action="compact", args=...))` (`polylogue/archive/query/expression.py:479`). It reuses the existing session-source pipeline plumbing; no new grammar branch beyond registering the terminal token.

### Contrast with `compile_context` / `build_context_image` (why this is a distinct product)

`polylogue/context/compiler.py` + `api/archive.py:compile_context` + `mcp/server_context_tools.py:build_context_image` already produce a `ContextImage`. They are **not** what `compact` is:

| Axis | `compile_context` (existing) | `compact` (new) |
|---|---|---|
| Scope | N *seed* sessions, one segment per session/view | corpus (whole `find` result), single fused digest |
| Selection unit | message *window*, tail-biased (`_budget_context_message_window`) | scored *blocks* across all sessions |
| Drop policy | positional (earlier/later omitted) + char clip | semantic: `material_origin` + decision/outcome scoring |
| Purpose | *continue/handoff* one thread (`ContextPurpose`) | *feed an external LLM to reason about the archive itself* |
| Dedup | per-session lineage composition only | corpus-wide lineage-prefix dedup across sessions |
| Fidelity report | `ContextOmission` list (per-input) | **drop-manifest** (what classes of material were dropped + recoverable refs) |

`compact` **reuses** the primitives (`_estimate_tokens`, `_clip_text_to_token_budget`, `ContextOmission`/`ContextSegment` shapes, `EvidenceRef` anchoring) but composes them at corpus grain. It is the R&D-flywheel enabler: it packages Polylogue's own archived sessions for the next model lane.

---

## 1. Projection spec + surface

New module `polylogue/insights/corpus_compaction.py` (insight-tier substrate; surfaces stay leaf adapters per the layering rule).

```python
class CompactionBudget(str, Enum):        # named token budgets
    SMALL = "60k"      # 60_000 tokens
    LARGE = "200k"     # 200_000 tokens
    # numeric override allowed via --budget N

class CompactionProjection(SurfacePayloadModel):
    """Fixed projection preset for corpus compaction (read-algebra Projection node)."""
    budget_tokens: int = 60_000
    keep_origins: tuple[MaterialOrigin, ...] = (          # what survives scoring floor
        MaterialOrigin.HUMAN_AUTHORED,
        MaterialOrigin.OPERATOR_COMMAND,
        MaterialOrigin.ASSISTANT_AUTHORED,
    )
    drop_origins: tuple[MaterialOrigin, ...] = (          # hard-drop (tool-spam)
        MaterialOrigin.TOOL_RESULT,
        MaterialOrigin.RUNTIME_PROTOCOL,
        MaterialOrigin.RUNTIME_CONTEXT,
        MaterialOrigin.GENERATED_CONTEXT_PACK,
    )
    per_session_floor_tokens: int = 200   # every kept session guaranteed a header + 1 decision
    include_drop_manifest: bool = True
    lineage_grain: Literal["logical_session","physical"] = "logical_session"

class CompactBlock(ArchiveInsightModel):   # the scored, selectable unit
    session_id: str
    message_id: str
    block_index: int
    material_origin: MaterialOrigin
    kind: Literal["human_turn","decision","error_fix","outcome","assistant_prose"]
    text: str
    token_estimate: int
    score: float
    evidence_ref: EvidenceRef              # citation anchor (reuses core/refs.py)

class CompactionDropManifest(ArchiveInsightModel):
    dropped_by_origin: dict[str, int]      # counts by MaterialOrigin
    dropped_by_reason: dict[str, int]      # {"budget": n, "tool_spam": n, "lineage_dup": n, "low_score": n}
    truncated_sessions: tuple[str, ...]
    recoverable_via: str = "polylogue read session:<id> --view transcript"

class CorpusCompaction(ArchiveInsightModel):
    projection: CompactionProjection
    logical_session_count: int
    physical_session_count: int
    blocks: tuple[CompactBlock, ...]
    token_estimate: int
    drop_manifest: CompactionDropManifest
    caveats: tuple[str, ...] = ()
```

**Surface exposure** (three adapters, one engine — mirrors `compile_context`):
- **CLI**: `find <query> | compact [--budget 60k|200k|N] [--format markdown|ndjson]`. Terminal action registered in `SessionTerminalAction`; renderer added to `RenderFormat.COMPACT` in `surfaces/projection_spec.py`; dispatch in `cli/query_verbs.py` next to the read-view path.
- **MCP**: new tool `compact_corpus(query, budget, ...)` in a `server_*` module — reuses the engine; requires `EXPECTED_TOOL_NAMES` + tool-contract update (per CLAUDE.md gotcha).
- **API**: `Polylogue.compact_corpus(spec: CompactionProjection, selection: SelectionSpec) -> CorpusCompaction` on `api/archive.py` beside `compile_context`.

**Render**: `polylogue/cli/read_views/` gains `compact.py` emitting markdown with per-session boundary markers and citation anchors:

```
## session:codex-session:abc123 — "batch deep-read defects" (2026-07-04, repo:polylogue)
[human] Fix the connection leak in ingest_batch.               ⟨codex-session:abc123:m12:0⟩
[decision] Chose lease+snapshot double-invariant over refcount ⟨codex-session:abc123:m48:2⟩
[error→fix] ResourceWarning conn leak → close in finally       ⟨codex-session:abc123:m51:1⟩
[outcome] 9 passed (keystone: exit_code=0)                     ⟨codex-session:abc123:m60:0⟩
--- 3 tool_result blocks, 1 lineage-dup prefix dropped ---
```

The `⟨…⟩` anchors are `EvidenceRef.format()` strings so a downstream LLM (or a follow-up `polylogue read`) can resolve any claim back to source.

---

## 2. Packing algorithms (pseudocode)

### 2a. Lineage-prefix dedup at corpus grain (`LOGICAL_SESSION`)

Reuses the iterative composition in `storage/sqlite/queries/message_query_reads.py` (`_prefix_sharing_edge`, `_MAX_LINEAGE_DEPTH`, visited-set cycle guard). Corpus twist: multiple *selected* sessions may share a physical prefix, so dedup once per logical lineage root.

```
function collapse_to_logical(selected_session_ids):
    # group physical sessions by lineage root using session_links prefix-sharing edges
    root_of = {}
    for sid in selected_session_ids:
        chain = walk_up_prefix_edges(sid)        # existing _prefix_sharing_edge loop
        root_of[sid] = chain[-1] if chain else sid
    logical = group_by(selected_session_ids, key=root_of)
    for root, members in logical:
        # keep the LONGEST tail member as the representative; its composed
        # transcript already contains the shared prefix exactly ONCE
        rep = argmax(members, key=composed_message_count)
        yield LogicalSession(root=root, representative=rep, absorbed=members - {rep})
    # dropped prefixes counted into drop_manifest.dropped_by_reason["lineage_dup"]
```

Guard: `visited` set + `_MAX_LINEAGE_DEPTH=1024` (inherit, do not re-implement) — quarantined/cyclic edges (`TopologyEdgeStatus.quarantined`) are treated as non-sharing.

### 2b. Block scoring (drop tool-spam, keep decision density)

```
function score_blocks(logical_session):
    composed = compose_transcript(rep)          # existing lineage recomposition
    for msg in composed:
        mo = MaterialOrigin.normalize(msg.material_origin)
        if mo in projection.drop_origins:
            drop(reason="tool_spam"); continue   # tool_result/runtime_* never scored
        for block in msg.blocks:
            kind, base = classify(block, msg)
            score = base * recency_decay(msg) * (1 + issue_ref_bonus(block))
            emit CompactBlock(kind, score, token_estimate=_estimate_tokens(block.text), ...)

function classify(block, msg):
    # reuse insights/transforms.py extractors — do NOT re-derive regexes
    if msg.material_origin == HUMAN_AUTHORED or OPERATOR_COMMAND: return ("human_turn", 1.0)
    if block in _extract_decision_candidates(...):                return ("decision", 0.95)
    if is_outcome(block):   # keystone: tool_result_is_error / tool_result_exit_code
                                                                  return ("outcome", 0.9)
    if is_error_fix_pair(block):                                  return ("error_fix", 0.85)
    if block.kind == THINKING:                                    return ("assistant_prose", 0.2)
    return ("assistant_prose", 0.4)
```

Outcome truth comes from the **keystone structured fields** (`blocks.tool_result_is_error`, `tool_result_exit_code`; `transforms.py:1764`), never regex-guessed from prose (#2482) — even though the *block* itself is a tool_result and is normally dropped, its outcome is *projected onto the preceding tool_use* as an `outcome` block, so the digest keeps "9 passed / exit 0" without keeping the raw output.

### 2c. Budget fitting (two-pass: floor then greedy)

```
function fit_budget(logical_sessions, budget):
    # Pass 1 — reserve a per-session floor so no session vanishes silently
    reserved = {}
    for ls in logical_sessions:
        header = session_header_block(ls)                 # boundary marker, always kept
        top = highest_scoring(ls.blocks, kinds={human_turn, decision})[:1]
        reserved[ls] = [header] + top
        if sum_tokens(reserved) > budget: 
            break  # too many sessions for even floors → truncate session set, record in manifest
    remaining = budget - sum_tokens(reserved.values())

    # Pass 2 — greedy global fill by score-density (score / token_estimate)
    pool = all_unreserved_blocks sorted by (score / token_estimate) desc
    for block in pool:
        if block.token_estimate <= remaining:
            keep(block); remaining -= block.token_estimate
        else:
            block.text = _clip_text_to_token_budget(block.text, remaining)  # existing helper
            if remaining > CLIP_FLOOR: keep(clipped); remaining = 0
            drop(reason="budget")
    return kept, drop_manifest
```

Token proxy = `_estimate_tokens` (word count, from `context/compiler.py`). **Note (risk 1)**: word-count underestimates real BPE tokens ~1.25–1.4×; the fitter applies a `budget * 0.72` safety derate for the `60k/200k` presets and records the derate in `caveats`.

### 2d. Drop manifest assembly

Every `drop(reason=...)` call increments `CompactionDropManifest` counters keyed by `MaterialOrigin` + reason. Truncated sessions list their `session:<id>` refs and the exact recovery command. This is the fidelity contract: the digest is lossy *by construction* but the loss is enumerated and reversible via `polylogue read`.

---

## 3. Migration

**No durable-tier schema change.** Everything compaction reads already exists in `index.db` v24: `messages.material_origin`, `blocks.tool_result_is_error/tool_result_exit_code`, `session_links` (lineage), FTS. Compaction is a **compute-on-read projection** — it materializes nothing.

Generated-surface regeneration required (per CLAUDE.md gotchas):
1. New module `polylogue/insights/corpus_compaction.py` + `cli/read_views/compact.py` → **regenerate topology projection**: `devtools render topology-projection && devtools render topology-status` (else `render all --check` fails). Commit `docs/plans/topology-target.yaml` + `docs/topology-status.md`.
2. New `RenderFormat.COMPACT` + `SessionTerminalAction` member → **regenerate** `devtools render cli-output-schemas` and `render openapi` (the enums are embedded).
3. New MCP tool → update `EXPECTED_TOOL_NAMES` + add a tool contract, then `render mcp-reference`.
4. New CLI verb-modifier → `render cli-reference` + refresh help/terminal snapshots (`tests/unit/cli/__snapshots__/*.ambr`).
5. `docs/search.md` / `docs/cli-reference.md`: document the `| compact` terminal and the drop-manifest contract.

If, and only if, corpus-compaction proves hot enough to precompute, a *later* PR may cache per-`LOGICAL_SESSION` scored-block sets in `index.db` (additive-derived: edit canonical DDL + rebuild plan, `ops reset --index && polylogued run`, never an upgrade helper). Keep this out of the first slice.

---

## 4. Test strategy

Testmon inner loop (`devtools test <file>`), never blanket. New tests:

- **Projection unit** (`tests/unit/insights/test_corpus_compaction.py`): `material_origin` drop set excludes tool_result/runtime rows; keep set retains human/decision/outcome; keystone outcome is projected onto the tool_use even though the tool_result block is dropped.
- **Budget-fit property** (`tests/property/`, Hypothesis over `SessionBuilder` corpora): invariants — (a) `token_estimate ≤ budget` always; (b) every kept logical session has ≥1 block (floor); (c) `kept ⊎ dropped == all scored blocks` (conservation — nothing vanishes unaccounted); (d) manifest counts sum to `dropped`.
- **Lineage dedup** (`tests/unit/`): a fork + its parent both selected → shared prefix appears exactly once; quarantined/cyclic edge does not double-count; reuse `frozen_clock` for recency scoring determinism.
- **Citation-anchor round-trip**: every `⟨ref⟩` in rendered markdown parses via `EvidenceRef.parse` and resolves to a real block (guards against fabricated anchors — the #2482 authorship-honesty lesson).
- **Golden render** (`syrupy`): boundary markers + drop-manifest footer stable; a dedicated `fix(test):` PR if snapshots churn.
- **Demo path** (private-data-free): `polylogue demo seed && find … | compact --budget 60k` over synthetic corpus, asserted non-empty digest + manifest.
- **CLI contract**: `find "x" | compact` parses to `SessionQueryTerminalStage(action="compact")`; bare `compact` without signalled query intent still raises `UsageError` (#1842 command floor).

Do **not** add tests that memorialize the enum additions themselves (rename/added-literal) — rely on mypy + the behavior tests above.

---

## 5. Bead breakdown (create with `bd`, do not implement here)

1. **`compact-projection-model`** — `CompactionProjection`/`CompactBlock`/`CorpusCompaction`/`CompactionDropManifest` Pydantic models in `insights/corpus_compaction.py`. *AC:* models typecheck under `mypy --strict`; `keep_origins`/`drop_origins` partition `MaterialOrigin` with no overlap; round-trips through `model_dump(mode="json")`.
2. **`compact-block-scoring`** — block classifier + scorer reusing `insights/transforms.py` decision/outcome extractors and keystone tool-result fields. *AC:* tool_result/runtime_* origins scored 0/dropped; outcome projected from keystone fields onto tool_use; unit tests green.
3. **`compact-lineage-dedup`** — corpus-grain `collapse_to_logical` over `session_links`, reusing `_prefix_sharing_edge`/visited-guard. *AC:* shared prefix counted once; cyclic/quarantined edges safe; property test (c) conservation holds.
4. **`compact-budget-fitter`** — two-pass floor+greedy fitter with `_estimate_tokens` proxy + safety derate + `_clip_text_to_token_budget`. *AC:* `token_estimate ≤ budget` invariant (Hypothesis); per-session floor honored; derate recorded in caveats.
5. **`compact-drop-manifest`** — manifest assembly + recovery-command emission. *AC:* counts reconcile with dropped blocks; truncated sessions list resolvable `session:` refs.
6. **`compact-cli-terminal`** — register `compact` in `SessionTerminalAction` + `RenderFormat.COMPACT`, dispatch in `cli/query_verbs.py`, renderer `cli/read_views/compact.py` with boundary markers + anchors. *AC:* `find "x" | compact --budget 60k` renders; `render cli-output-schemas`/`cli-reference`/topology regenerated + committed; help snapshots updated.
7. **`compact-mcp-tool`** — `compact_corpus` MCP tool + `EXPECTED_TOOL_NAMES` + contract + `render openapi`/`mcp-reference`. *AC:* tool-discovery tests green; contract present.
8. **`compact-docs-demo`** — `docs/search.md`/`cli-reference.md` entry, demo-path integration test, drop-manifest fidelity contract documented. *AC:* `polylogue demo … | compact` produces cited digest + manifest; `devtools verify` affected-set green.

Sequencing: 1→(2,3 parallel)→4→5→6→(7,8). Beads 2 and 3 are independent write-scopes; 6 is the shared-surface hotspot (serialize after 4/5).

---

## 6. Top-3 risks

1. **Token-proxy inaccuracy → budget overrun.** The whole codebase uses word-count (`_estimate_tokens`) as a token stand-in, but 60k/200k are *BPE* budgets for the target external model; word-count underestimates ~1.25–1.4×, so a naive fit overflows the real context window. *Mitigation:* fixed `0.72` derate on presets + record it in `caveats`; leave a follow-up bead to swap in a real tokenizer count if the external lane is BPE-sensitive. Do **not** silently claim "fits 200k".

2. **Fabricated decision/outcome claims (the #2482 / recovery-digest-fabrication trap).** Corpus-compaction is exactly the surface where regex-mined "decisions" and "9 passed" can become unfounded assertions fed to a downstream model that then treats them as ground truth. *Mitigation:* outcomes read *only* from keystone structured fields; every block carries a real `EvidenceRef` anchor validated by the round-trip test; `decision` kind reuses `_extract_decision_candidates` (already authorship-gated) rather than a fresh prose regex. Anchors make every claim reversible, which is the honesty contract.

3. **Lineage dedup collapsing legitimately distinct work.** Real Task subagents and independent resumes share physical prefixes but are *semantically distinct* (the #2467 lesson: 7383 real subagents are legit, not dups). Over-aggressive `collapse_to_logical` would drop genuine divergent tails and misreport corpus size. *Mitigation:* dedup only *shared prefix* messages (`inheritance='prefix-sharing'` with a real `branch_point_message_id`), always keep the longest-tail representative's divergent suffix, treat quarantined edges as non-sharing, and surface `logical_session_count` vs `physical_session_count` explicitly so the ratio is auditable.

---

**Grounding files** (all absolute): `/realm/project/polylogue/polylogue/surfaces/projection_spec.py` (read algebra), `/realm/project/polylogue/polylogue/core/enums.py:176` (`MaterialOrigin`), `/realm/project/polylogue/polylogue/context/compiler.py` + `/realm/project/polylogue/polylogue/api/archive.py:compile_context` (contrast + budget helpers `_budget_context_message_window`/`_clip_text_to_token_budget`/`_estimate_tokens`), `/realm/project/polylogue/polylogue/insights/transforms.py` (decision/outcome extraction substrate `compile_session_digest`, `_extract_decision_candidates`, keystone reader `transforms.py:1764`), `/realm/project/polylogue/polylogue/storage/sqlite/queries/message_query_reads.py:65` (lineage recomposition to reuse), `/realm/project/polylogue/polylogue/archive/query/expression.py:479` (session-source pipeline terminal to extend). New code lands in `/realm/project/polylogue/polylogue/insights/corpus_compaction.py` + `/realm/project/polylogue/polylogue/cli/read_views/compact.py`.

---

## [a0c6c2f1d5e06efd8] SPEC: Context Scheduler (polylogue-37t.11) — the single arbiter for agent context

I have complete grounding. Here is the implementation-grade spec.

---

# SPEC: Context Scheduler (polylogue-37t.11) — the single arbiter for agent context

**Implementation home:** `polylogue/context/scheduler.py` (new), hoisted above `polylogue/context/compiler.py` (`ContextImage`/`ContextSegment` stay the pure assembly primitive; the scheduler is the admission/ranking layer that feeds them). The SessionStart hook and every mid-session moment call **one** entrypoint: `schedule_context(moment, target_session, budget, sources) -> ScheduledContext`.

**Grounding confirmed in source:**
- `assertions` (user.db v4, durable) already carries `context_policy_json TEXT DEFAULT '{"inject":false}'`, `staleness_json TEXT`, `confidence REAL`, `author_kind TEXT DEFAULT 'user'` (`storage/sqlite/archive_tiers/user.py:12-31`). **The programmable-window schema needs no durable DDL change** — it is a JSON vocabulary extension of `AssertionContextPolicy` (`core/assertions.py:44`).
- Admission read point: `list_assertion_claims(..., context_inject=...)` filters in Python at `user_write.py:1585`; `ASSERTION_CLAIM_KINDS` at `:1520`.
- The recursive-safety gate already exists in embryo: candidate writers stamp `context_policy={"inject":False,"promotion_required":True}` with `author_kind` in `{"user","transform","detector"}` (`user_write.py:1052,1114`). Promotion IS the candidate→judged transition.
- No ledger/read-access table exists yet. ops.db (disposable, `OPS_SCHEMA_VERSION=1`) holds `daemon_stage_events`, `otlp_spans` — the correct home for high-write derived telemetry.
- Topic proximity: `RepositoryVectors.similarity_search()` returns `(conv_id, msg_id, distance)` cosine over vec0 (`repository/vectors/repository_vectors.py:111`).

---

## (1) Schema additions

### 1a. Programmable `context_policy_json` windows (user.db — **JSON vocab only, no DDL migration**)

Extend `AssertionContextPolicy.payload` with a closed window schema. Column stays `TEXT`; durable tier untouched (this is exactly why the column was made plain TEXT — CLAUDE.md invariant).

```jsonc
// context_policy_json
{
  "inject": true,                        // existing gate (unchanged)
  "promotion_required": false,           // existing recursive-safety flag (unchanged)
  "trust_class": "operator",             // NEW: derived, see §6-gate — operator|system|quoted
  "windows": [                           // NEW: programmable injection windows
    {
      "when": "session_start",           // session_start | on_topic | on_error
      "ttl_days": 30,                    // wall-clock expiry of THIS window's eligibility
      "max_injections": 5,              // lifetime cap across all sessions
      "cooldown_hours": 24,             // min gap between injections of this item
      "min_proximity": 0.55             // optional: on_topic gate threshold (cosine sim)
    }
  ]
}
```

`AssertionContextPolicy` gains typed accessors: `windows() -> tuple[InjectionWindow,...]`, `eligible_at(moment, now, proximity) -> bool`, `trust_class`. Absent `windows` ⇒ back-compat default `[{"when":"session_start","ttl_days":∞,"max_injections":∞,"cooldown_hours":0}]` whenever `inject:true`. Validation via a frozen dataclass `InjectionWindow` (mirrors `AssertionStaleness`); reject unknown `when` values at write time in `upsert_assertion`.

### 1b. Ledger + read-access-log tier (ops.db — **disposable, DDL edit + version bump 1→2**)

Two append-only tables in `archive_tiers/ops.py`. ops.db is disposable/derived: **no numbered migration chain, no backup manifest** — edit canonical DDL, bump `OPS_SCHEMA_VERSION`, schema-mismatch rebuilds the tier (`polylogue ops reset --ops`).

```sql
-- The "memory map": every scheduler allocation decision.
CREATE TABLE IF NOT EXISTS context_injection_ledger (
    ledger_id        TEXT PRIMARY KEY,   -- sha256(target_session|moment|item_ref|scheduled_at)[:16]
    target_session   TEXT NOT NULL,      -- session receiving the context
    moment           TEXT NOT NULL,      -- session_start|pre_compact_resume|mid_session_advisory|on_demand
    scheduled_at_ms  INTEGER NOT NULL,
    source_id        TEXT NOT NULL,      -- ContextSource.source_id
    item_ref         TEXT NOT NULL,      -- resolvable citation-anchor (ObjectRef/EvidenceRef)
    trust_class      TEXT NOT NULL,      -- operator|system|quoted
    priority_class   TEXT NOT NULL,      -- correctness|directives|recall|ambient
    disposition      TEXT NOT NULL,      -- included|degraded_ref|dropped
    drop_reason      TEXT,               -- budget|cooldown|dedup|expired|proximity|trust_gate
    score            REAL NOT NULL,      -- final composite rank
    score_json       TEXT NOT NULL,      -- {staleness, proximity, attention, class_weight}
    token_cost       INTEGER NOT NULL,
    budget_after     INTEGER NOT NULL    -- remaining moment budget after this decision
) STRICT;
CREATE INDEX IF NOT EXISTS idx_ledger_session_moment
  ON context_injection_ledger(target_session, moment, scheduled_at_ms);

-- Attention signal: which archived items an agent actually READ.
CREATE TABLE IF NOT EXISTS context_read_access (
    access_id     TEXT PRIMARY KEY,      -- sha256(reader_session|item_ref|accessed_at)[:16]
    item_ref      TEXT NOT NULL,         -- assertion/session/message ref read via MCP/CLI/resolve_ref
    reader_session TEXT,                 -- session that read it (nullable: operator-direct)
    surface       TEXT NOT NULL,         -- mcp|cli|resolve_ref|preamble_expand
    accessed_at_ms INTEGER NOT NULL,
    access_kind   TEXT NOT NULL          -- resolved|expanded|cited|search_hit
) STRICT;
CREATE INDEX IF NOT EXISTS idx_read_access_item
  ON context_read_access(item_ref, accessed_at_ms);
```

Read-access is written by the same MCP/CLI resolve paths that already exist (`resolve_ref`, `read`, `get_session`) — a thin `record_read_access()` call, daemon-owned like all writes. Attention = decayed count of read-access rows for an item's ref.

---

## (2) Ranking function + budget algorithm (pseudocode)

### Ranking — three-factor composite, per candidate item

```
score(item, now, current_embedding) =
    class_weight[item.priority_class]           # correctness=1.0 directives=0.7 recall=0.4 ambient=0.2
  * staleness_decay(item, now)                  # ∈ (0,1]
  * topic_proximity(item, current_embedding)    # ∈ [0,1], =1.0 for non-embeddable structural items
  * attention_boost(item, now)                  # ∈ [1, 2], read-recency amplifier

staleness_decay(item, now):
    # half-life from staleness_json.half_life_days, default per priority_class
    age_days = (now - item.updated_at) / DAY
    hl = item.staleness.half_life_days or DEFAULT_HL[item.priority_class]   # correctness=∞
    if item has hard ttl and age_days > ttl: return 0        # hard expiry ⇒ excluded upstream
    return 0.5 ** (age_days / hl)                            # correctness never decays (hl=∞ ⇒ 1.0)

topic_proximity(item, current_embedding):
    if item.embedding is None: return 1.0        # structural facts (identity, caveats) always on-topic
    if current_embedding is None: return 0.5     # neutral when seed has no embedding yet
    sim = 1 - cosine_distance(item.embedding, current_embedding)   # via similarity_search substrate
    return clamp(sim, 0, 1)

attention_boost(item, now):
    reads = read_access_rows(item.item_ref, since=now - ATTENTION_WINDOW)   # §1b table
    recency_weight = Σ 0.5**((now - r.accessed_at)/ATTENTION_HL) for r in reads
    return 1 + min(1.0, recency_weight)          # capped 2× — recently-read items resurface
```

### Budget — hard-cap ranked selection with class proportions + borrowing

```
schedule(moment, target_session, budget_tokens, sources):
    now = clock.now(); seed_emb = current_session_embedding(target_session)

    # 1. Collect candidates from every registered source (propose() is pure)
    candidates = []
    for src in sources:
        if moment not in src.moments: continue
        for item in src.propose(target_session, moment):
            if not window_eligible(item, moment, now, proximity(item, seed_emb)): continue   # §1a windows
            if not trust_gate(item, src): continue                                            # §6 gate
            item.score = score(item, now, seed_emb)
            candidates.append(item)

    # 2. Cross-source dedup by citation-anchor / content-hash (global cooldown state)
    candidates = dedup_by_ref(candidates)        # keep highest-score per resolvable ref
    candidates = drop_on_cooldown(candidates, now, ledger)   # per-item cooldown_hours from window

    # 3. Class budget allocation: fixed proportions, then borrow unused
    class_budget = {c: budget_tokens * PROPORTION[c] for c in CLASSES}   # correctness .40 directives .25 recall .25 ambient .10
    included = []; remaining = budget_tokens
    for cls in CLASS_ORDER:                       # correctness → directives → recall → ambient
        pool = sorted([c for c in candidates if c.priority_class==cls], key=-score)
        cap = class_budget[cls]
        for item in pool:
            cost = item.token_cost
            if cost <= cap and cost <= remaining:
                include(item); cap -= cost; remaining -= cost
            elif ref_cost(item) <= remaining:      # DEGRADE full→ref-only
                include_as_ref(item); remaining -= ref_cost(item)
                ledger.record(item, disposition="degraded_ref")
            else:
                ledger.record(item, disposition="dropped", reason="budget")
        borrow_pool += cap                         # unused class budget flows forward (borrowing)
        class_budget[next_class] += cap

    # 4. Deterministic assembly + ledger (INVARIANT: Σ token_cost(included) ≤ budget_tokens)
    assert sum(i.token_cost for i in included) <= budget_tokens
    image = compile_into_context_image(sorted(included, key=(class_order, -score, ref)))  # stable sort ⇒ byte-identical
    for i in included: ledger.record(i, disposition="included", budget_after=...)
    return ScheduledContext(image=image, ledger_rows=..., moment=moment)
```

**Determinism:** `propose()` is pure; all randomness excluded; final sort key is total order `(class_index, -score, item_ref)`. Same inputs ⇒ byte-identical context (property-testable per AC).

---

## (3) Migration

| Tier | Change | Mechanism | Ceremony |
|---|---|---|---|
| **user.db** (durable v4) | `context_policy_json` window vocab; `trust_class` field | **None — JSON only.** Extend `AssertionContextPolicy` + `InjectionWindow` value object; write-time validation in `upsert_assertion` | No numbered migration, no backup manifest. Column is already `TEXT`; old rows (no `windows`) get the back-compat default at read. Passes `devtools lab policy schema-versioning` because durable DDL is unchanged. |
| **ops.db** (disposable v1→v2) | add `context_injection_ledger`, `context_read_access` | Edit canonical `OPS_DDL` + bump `OPS_SCHEMA_VERSION=2`; schema mismatch triggers tier rebuild | `polylogue ops reset --ops && polylogued run`. **No** upgrade helper (rejected by policy lint for derived tiers). |

Rollback: revert JSON accessors (rows harmlessly ignore `windows`); ops.db rebuild is disposable. Zero risk to `user.db` irreplaceable data.

---

## (4) Test strategy

- **Determinism property test** (`tests/property/context/test_scheduler_determinism.py`): Hypothesis-generated source/candidate sets ⇒ `schedule()` twice ⇒ byte-identical `ContextImage`. (AC-mandated.)
- **Budget invariant property test**: over any source combination, `Σ token_cost(included) ≤ budget` AND every included item carries a resolvable ref (parse via `ObjectRef/EvidenceRef.parse`). (AC-mandated.)
- **Ranking unit tests** (`tests/unit/context/test_ranking.py`): staleness half-life (correctness `hl=∞` ⇒ decay=1.0; expired ⇒ 0); proximity (structural ⇒ 1.0, `seed_emb=None` ⇒ 0.5, cosine wiring via a stub vector repo); attention boost monotonic in read-recency, capped 2×.
- **Window semantics** (`tests/unit/context/test_injection_windows.py`): `when` gating (`on_topic` blocked below `min_proximity`, `on_error` only at error moment); `ttl_days` hard-excludes; `max_injections` enforced against ledger count; `cooldown_hours` suppresses within window. Use `frozen_clock` (CLAUDE.md clock-hygiene lint).
- **Trust gate / red-team** (`tests/unit/security/test_context_trust_gate.py`, **protected security dir**): a source without a judgment gate cannot emit `operator`; seeded session containing an injection string ("ignore previous instructions…") never reaches assembled output unfenced (`quoted` items always fenced+attributed); no verbatim tool-output injectable. (AC-mandated red-team fixture.)
- **Cross-source dedup** (`tests/unit/context/test_dedup.py`): advisory item and blackboard item with same ref ⇒ one survives (higher score), other ledgered `dropped/dedup`. (AC-mandated scenario.)
- **Ledger round-trip** (`tests/unit/storage/test_context_ledger.py`): every disposition writes a row; `polylogue context ledger <session>` + MCP tool read it back; content-hash invariant (recording a ledger row never mutates `sessions.content_hash`, mirror `test_feedback.py`).
- **E2E stage integration** into the 37t four-stage test (`tests/unit/context/test_judged_memory_loop.py`): a judged claim flows through the scheduler into a SessionStart preamble as a ref, ledgered.

Inner loop: `devtools test tests/unit/context tests/property/context` (testmon-affected). Never blanket-run.

---

## (5) Bead breakdown (children of polylogue-37t.11)

Sliced to honor the bead's **contract-first split** (slice 1 unblocks mhx.4/rvh/1hj/bfv/gjg in parallel).

1. **`.a` ContextSource protocol + minimal scheduler (slice-1 spine)** — `size:M`
   *AC:* `ContextSource` Protocol (`source_id`, `moments`, `priority_class`, `propose()`, `degrade_order`, `trust_class`) + `schedule()` with fixed-proportion allocation (no borrowing) in `context/scheduler.py`; deterministic assembly property test green; budget-invariant property test green; **trust_class carried from slice 1** (not retrofit). Unblocks parallel source builders.

2. **`.b` Programmable window vocab on `context_policy_json`** — `size:S`
   *AC:* `InjectionWindow` value object + `AssertionContextPolicy.windows()/eligible_at()`; `upsert_assertion` write-time validation rejects unknown `when`; admission read (`list_assertion_claims`, `user_write.py:1585`) honors `ttl/max_injections/cooldown/when`; back-compat default for windowless rows; window-semantics unit tests green. No durable DDL change (verify `schema-versioning` policy passes).

3. **`.c` Injection ledger tier (ops.db v2) + write path** — `size:M`
   *AC:* two ops.db tables via canonical DDL + version bump; scheduler writes included/degraded/dropped rows with scores+budget state; daemon owns the write; content-hash invariant test; `polylogue ops reset --ops` rebuilds cleanly.

4. **`.d` Read-access-log + attention factor** — `size:M`
   *AC:* `context_read_access` populated by `resolve_ref`/`read`/MCP get paths; `attention_boost()` reads it with decay+cap; ranking unit test proves recently-read items resurface; no double-write on repeated reads within a debounce window.

5. **`.e` Migrate 37t.4's two sections as the first ContextSources** — `size:M`
   *AC:* repo-brief + resume-delta sections re-expressed as `ContextSource.propose()` returning ref-style items; SessionStart hook calls the single `schedule_context` entrypoint; preamble honors raw-log restraint (timestamped, expiry metadata, navigable refs, refs-not-dumps); existing 37t.4 preamble test passes through the scheduler.

6. **`.f` Recursive-safety / trust gate + red-team fixtures** — `size:M`
   *AC:* `trust_class` derived from `author_kind` (`user`→operator only if `promotion_required` satisfied; `transform`/`detector`→quoted/system); source without judgment gate cannot emit operator (type-level); assembled output never contains unfenced quoted content (property test); seeded injection-string fixture never reaches preamble unfenced; no verbatim tool-output/web text injectable (refs-only). **Blocking/security — lands with slice 1's protocol.**

7. **`.g` Ledger read surfaces (slice-2)** — `size:S`
   *AC:* `polylogue context ledger <session>` CLI + MCP tool (update `EXPECTED_TOOL_NAMES` + tool contract); webui session Info panel reads the same envelope; uplift instrumentation (cfk) can read arm evidence as ledger rows.

8. **`.h` Borrowing + cross-source dedup + mid-session moments (slice-2)** — `size:M`
   *AC:* unused class budget borrows forward; global cooldown/dedup by ref/content-hash (advisory suppresses same-ref blackboard item — seeded scenario, AC-mandated); mid-session advisory moment (budget = one item) shares the arbiter; s7ae coordination source registers here as a ContextSource, not a separate injection path.

---

## (6) Top-3 risks

1. **Ledger durability vs. write volume (tier placement).** The bead frames the ledger as "queryable fact" and uplift-arm evidence, but I place it in **disposable ops.db** to keep `user.db` (irreplaceable) lean and avoid backup-manifest ceremony on every injection. Risk: an `ops reset` erases the historical memory-map that experiments (cfk) may want long-term. *Mitigation:* ledger is append-only forensics, not correctness state; if long-horizon retention is required, add a periodic export to a durable artifact under `.local/` rather than promoting the hot table into `user.db`. Flag this trade-off to the operator explicitly — it is the one genuinely contestable placement decision.

2. **Trust-class retrofit is an injection hole (must be slice-1, not deferred).** The bead's security constraint is explicit: "retrofitting trust classes is how injection holes ship." The temptation to land `.a`'s scheduler without `.f`'s gate (to unblock parallel sources faster) directly reopens the hole. *Mitigation:* `.f` is marked blocking and its `trust_class` field is required in the `.a` protocol from the first commit; the red-team property test gates the slice-1 merge.

3. **Embedding availability at schedule time (proximity degradation).** `topic_proximity` depends on the seed session already having a vec0 embedding, but embeddings converge asynchronously (`embeddings.db`, daemon catch-up) — at SessionStart the current session is brand-new and unembedded, so `seed_emb=None` ⇒ every item gets neutral 0.5 proximity, collapsing the ranking to staleness×attention×class. *Mitigation:* accept graceful degradation (documented, tested with `seed_emb=None`), and optionally seed proximity from the *repo/cwd* centroid rather than the empty session; do **not** block scheduling on embedding readiness (would stall the hook). Watch that `on_topic` windows don't silently never-fire on fresh sessions — they correctly fall to the `session_start` window.

---

**Key files a builder touches:** `polylogue/context/scheduler.py` (new), `polylogue/context/compiler.py` (assembly primitive, unchanged interface), `polylogue/core/assertions.py` (`InjectionWindow` + policy accessors), `polylogue/storage/sqlite/archive_tiers/ops.py` (ledger DDL + version bump), `polylogue/storage/sqlite/archive_tiers/user_write.py:1520,1585` (window-aware admission), `polylogue/context/preamble.py` (single entrypoint via scheduler). No `user.db` DDL migration; ops.db is a disposable rebuild.

---

## [afea246b237d68799] Wave-5 Spec: Blue-Green Derived-Tier Rebuilds (bead polylogue-b5l)

I have enough grounding. Here is the implementation spec.

---

# Wave-5 Spec: Blue-Green Derived-Tier Rebuilds (bead polylogue-b5l)

**Goal:** kill the 20–40 min degraded window (`ops reset --index && polylogued run` at 12–15 rows/s). Build `index.gN+1.db` beside the live `index.gN.db`, serve the old until the new reaches convergence parity, swap the pointer under a <100 ms write pause, reap the old after a grace window. Scope is **index.db only** (embeddings is money-bounded and already tier-resettable; source/user are never rebuilt).

## Grounding (real files)

- **Path choke point:** `paths/_roots.py:active_index_db_path()` → `index_db_path()` → `archive_root()/"index.db"` (hardcoded basename). `config.py:32` imports it as `default_db_path`; ~40 modules consume it. This is the single seam to make generation-aware.
- **`resolve_active_index_db_path(db_anchor, index_db)`** already exists (`_roots.py:76`) but is a trivial passthrough — the natural home for generation resolution.
- **Connection model** (`storage/sqlite/connection.py:62` `_SIBLING_TIER_ATTACHMENTS`, `_attach_sibling_tiers`): every reader opens **one `index.db` connection** and `ATTACH`es `source`/`user`/`embeddings` as siblings *by path*. A generation swap is therefore purely "which `index.gN.db` file do new connections open" — siblings are unaffected. `archive.py:679-693` (`ArchiveConnection.__init__`) opens `archive_root/"index.db"` ro/rw directly — also needs generation resolution.
- **Regime:** index ∉ `DURABLE_MIGRATION_TIERS` (`migration_runner.py:15`); `bootstrap.py:175` currently raises *"move it aside and rebuild the archive root"* on index version mismatch — the exact degraded path b5l replaces. The lab-policy lint (no upgrade chains) stays **unchanged**: this is still rebuild-from-source, just concurrent.
- **Readiness oracle (4bu):** `daemon/status_snapshot.py` already computes `fts_readiness.messages_ready` + `raw_materialization_readiness`. Parity = these two green on the new generation.
- **`reset.py`:** `_REBUILDABLE_ARCHIVE_DATABASES` = index/embeddings/ops; `reset --index` deletes `index.db`. b5l turns this into "schedule blue-green"; `--offline` keeps old behavior.
- **ops.db** (`ops.py`, `OPS_SCHEMA_VERSION=1`, disposable) — but note `reset --index` currently also wipes ops.db, so the pointer **cannot** live only in ops.db without survival care (see §1).

---

## 1. Schema / DDL / Tier / Regime

The generation pointer must survive `reset --index` and be readable **without** opening any index generation (chicken-and-egg: you resolve the path *before* connecting). Two-layer design — a **durable file pointer** as source of truth, an **ops.db row** as queryable mirror.

**Layer A — pointer file** (source of truth for path resolution), `archive_root/index.pointer.json`:

```jsonc
{
  "format": "polylogue-index-pointer-v1",
  "active_generation": 42,        // reader-facing; the file readers open
  "building_generation": 43,      // null unless a rebuild is in flight
  "index_schema_version": 24,     // must equal INDEX_SCHEMA_VERSION of active gen
  "generations": {
    "42": {"state": "active",    "schema_version": 24, "created_at_ms": ...},
    "43": {"state": "building",  "schema_version": 24, "created_at_ms": ...},
    "41": {"state": "reaping",   "schema_version": 23, "retired_at_ms": ...}
  }
}
```

- Files on disk: `index.g42.db`, `index.g43.db`. Legacy plain `index.db` is treated as generation 0 (bootstrap migration, §3).
- Pointer swap is **atomic**: write `index.pointer.json.tmp`, `fsync`, `os.replace()` (atomic rename on POSIX same-filesystem). No SQLite txn needed for the swap itself.

**Layer B — ops.db mirror** (`OPS_SCHEMA_VERSION` 1 → 2, disposable regime, no migration chain — DDL edit + rebuild). Purely for `status`/webui/metrics; never the resolution authority:

```sql
CREATE TABLE IF NOT EXISTS index_generations (
    generation      INTEGER PRIMARY KEY,
    state           TEXT NOT NULL CHECK(state IN
                       ('building','active','draining','reaping','failed')),
    schema_version  INTEGER NOT NULL,
    rows_target     INTEGER NOT NULL DEFAULT 0,
    rows_done       INTEGER NOT NULL DEFAULT 0,
    rows_per_s      REAL,
    eta_ms          INTEGER,
    created_at_ms   INTEGER NOT NULL,
    activated_at_ms INTEGER,
    retired_at_ms   INTEGER
) STRICT;
```

State machine (single-writer daemon owns all transitions):
`building → active → draining → reaping → (file deleted)`; `building → failed` (aborted rebuild; files reaped, pointer untouched). At most one `active` and at most one `building` at a time.

**Tier / regime placement:** pointer file lives at archive root (durability = the *set* of generations, not any single file). ops mirror is disposable. Because the pointer is durable-file + disposable-mirror, `reset --index` deleting ops.db is harmless — the mirror is rebuilt from the pointer file on next daemon start.

---

## 2. Algorithms (pseudocode)

### 2a. Path resolution (the seam)

```
def active_index_db_path():                       # replaces hardcoded index.db
    ptr = read_pointer(archive_root)              # cached, mtime-invalidated
    if ptr is None:                               # pre-blue-green archive
        return archive_root / "index.db"          # legacy gen-0
    return archive_root / f"index.g{ptr.active_generation}.db"
```

Every reader (CLI cold path, daemon, MCP, API) resolves through this. A long-lived connection **pins** the generation it opened — it keeps serving the old file even after a swap (SQLite keeps the inode alive until close; `os.replace` on the *pointer*, not the db file, so the old db file is never renamed out from under an open handle). New connections pick up the new generation.

### 2b. Rebuild driver (daemon, background)

```
def blue_green_rebuild(reason):
    ptr = read_pointer()
    old = ptr.active_generation
    new = old + 1
    newdb = archive_root / f"index.g{new}.db"
    set_gen_state(new, "building"); pointer.building_generation = new; commit_pointer()

    initialize_archive_tier(connect(newdb), INDEX)          # fresh DDL, current version
    t_start = now_ms()

    # BULK REPLAY from source.db — reuses 20d.15 bulk lane:
    #   parallel parse (process_pool), single-writer batched txns,
    #   FTS bulk-trigger-drop, insights deferred to 2nd pass, idle IO class.
    for batch in source_raw_rows(order_by=rowid):
        materialize_into(newdb, batch)
        update_progress(new, rows_done+=len(batch), rows_per_s, eta)  # feeds status

    # PARITY GATE (4bu oracle): raw_materialization_ready AND fts messages_ready
    if not converged(newdb): 
        set_gen_state(new,"failed"); reap(new); return FAIL

    # DELTA CATCH-UP: replay source rows ingested since t_start
    #   (idempotent-by-content-hash ⇒ replay is safe; no double-write needed)
    replay_delta(newdb, since_ms=t_start)

    swap_pointer(old, new)                                   # §2c, <100ms
    set_gen_state(old,"draining")
    schedule_reap(old, after=grace_window)
```

**Writes-during-rebuild decision (b5l offers two; spec picks replay):** because writes are idempotent by content hash and 20d.15 targets <5 min rebuilds, **delta-replay** beats double-write — the daemon is the sole writer, so it records `t_start` and, right before swap, replays only source rows with `updated_at_ms >= t_start` into the new gen. No dual-write plumbing through the ingest path; the window is bounded by 20d.15 throughput. Double-write is the fallback only if measured windows exceed ~2 min.

### 2c. Pointer swap (atomicity, <100 ms)

```
def swap_pointer(old, new):
    acquire_write_pause()          # daemon stops accepting NEW ingest txns; in-flight drain
    ptr = read_pointer()
    ptr.active_generation = new
    ptr.building_generation = null
    ptr.generations[new].state = "active"
    ptr.generations[old].state = "draining"
    write_atomic(ptr)              # tmp + fsync + os.replace  — the atomic point
    mirror_to_ops(ptr)            # best-effort, non-authoritative
    release_write_pause()
    emit_daemon_event("index_generation_swapped", {old, new})
```

The write pause covers only the pointer `os.replace` + in-flight ingest drain — no data copy, so it is milliseconds. Reads never pause (they route lazily; open handles stay pinned).

### 2d. Reader routing / staleness

- **New connections** always resolve current `active_generation`.
- **Open connections** stay on their pinned generation (correct: they see a consistent snapshot; the old file exists until reaped).
- **Pointer cache invalidation:** cache pointer by `(path, st_mtime_ns)`; re-stat on each `active_index_db_path()` call (cheap) so a swap is picked up by the next new connection within one stat.

### 2e. Retention / reaping

```
def reap_worker():   # daemon periodic
    ptr = read_pointer()
    for gen, meta in ptr.generations where state == "draining":
        if now - meta.retired_at_ms < grace_window: continue      # e.g. 5 min
        if any_open_handle(gen): continue                          # lease/refcount check
        os.remove(index.g{gen}.db); os.remove(-wal, -shm)
        ptr.generations[gen].state = "reaping" → drop entry; write_atomic(ptr)
```

Grace window (default 300 s, configurable) covers long-lived reader connections opened just before the swap. `any_open_handle` uses a lightweight refcount lease (open registers, close deregisters in ops.db, mirroring the blob-GC lease discipline the codebase already uses) plus a `fuser`-style fallback. Never reap `active` or `building`. On daemon crash mid-rebuild: startup sees a `building`/`failed` gen with no `active` promotion → reaps it and leaves the last `active` serving.

---

## 3. Migration / Rebuild Plan

1. **Bootstrap existing archives (gen-0 adoption), zero-downtime:** on first daemon start with the new code, if `index.pointer.json` is absent but `index.db` exists: rename `index.db` → `index.g0.db` (single atomic `os.rename`, done before any reader opens — guarded by daemon startup lock), write pointer `{active_generation:0}`. Readers were already down at daemon start; cold CLIs resolve via pointer thereafter. Fallback: if pointer absent AND no `index.gN.db`, `active_index_db_path` returns legacy `index.db` so nothing breaks pre-adoption.
2. **`initialize_archive_database` change (`bootstrap.py:175`):** the index-tier "move it aside and rebuild the archive root" `RuntimeError` becomes: emit a *schedule-blue-green-rebuild* signal (daemon) instead of hard-failing. Offline/daemonless path keeps the raise behind `reset --index --offline`.
3. **`reset --index` semantics (`reset.py`):** default → schedule blue-green rebuild (builds gen N+1, swaps, reaps N). `--offline` → today's delete-and-rebuild (accepts downtime; for daemonless/recovery). ops.db mirror survives because pointer file is authoritative.
4. **Schema bump flow (index v24 → v25):** operator/daemon detects DDL version delta → triggers `blue_green_rebuild("schema_bump v24→v25")`. New gen initialized at v25, old stays v24 serving reads until parity. `index_schema_version` in pointer flips at swap. No migration chain — lab-policy lint unchanged.
5. **Docs:** `docs/internals.md` schema-versioning section replaces "operator moves the index aside" with the generation flow; operator runbook loses the 20–40 min degraded step.
6. **Topology:** new modules (`storage/index_generation.py`, daemon rebuild driver) → run `devtools render topology-projection && topology-status`, commit updated `docs/plans/topology-target.yaml` + `docs/topology-status.md` (else `render all --check` fails).

---

## 4. Test Strategy

- **Pointer atomicity (unit):** interleave `swap_pointer` with concurrent `read_pointer` under thread stress; assert every read sees either fully-old or fully-new, never torn JSON (guaranteed by `os.replace`, but test the wrapper). Kill-between-tmp-and-replace → old pointer intact.
- **Reader-routing invariant (integration):** open a read connection on gen N, trigger a full rebuild+swap to N+1, assert the held connection still returns gen-N rows and a *new* connection returns gen-N+1 rows; assert no `no such table`/`database is locked` during the swap.
- **Zero-failed-queries harness (the AC):** on `seeded_db`, run a continuous read loop (find/search/get) in one thread while a schema-bump rebuild runs in another; assert **0** query failures, swap pause <100 ms (measure the pointer-write critical section), delta replayed (post-swap row count == source count including rows written during rebuild), old gen reaped after grace. Mirror the `test_live_read_amplification.py` continuous-probe pattern.
- **Reaping safety:** assert a gen with an open lease is *not* deleted; assert reap fires after grace + lease release; assert crash-recovery reaps orphan `building` gen and never the `active` one.
- **Gen-0 adoption:** legacy `index.db`-only archive → daemon start renames to `index.g0.db`, pointer written, reads uninterrupted; idempotent on restart.
- **Throughput SLO (20d.15 gate, seeded corpus):** `devtools bench` asserts bulk replay ≥100 rows/s whole-run and full rebuild <5 min on the seeded corpus; regression-gate in `slo-catalog.yaml` maintenance tier. (Live-archive ≥100 rows/s is operator-machine acceptance, not CI-gated — host-dependent.)
- **Clock hygiene:** all timing tests use `frozen_clock`; grace-window tests advance the frozen clock, never `sleep`.
- **`mypy --strict`** carries the path-resolution refactor across the ~40 consumers of `active_index_db_path`.

---

## 5. Bead Breakdown (children of b5l)

| # | Title | Acceptance |
|---|---|---|
| **b5l.1** | Generation-aware path resolution + pointer file | `active_index_db_path`/`resolve_active_index_db_path` resolve `index.gN.db` via `index.pointer.json`; atomic write helper (tmp+fsync+`os.replace`); gen-0 adoption renames legacy `index.db` idempotently; pre-adoption fallback intact; mypy-strict green across all consumers. |
| **b5l.2** | ops.db generation mirror (v1→v2) + status surface | `index_generations` table added (disposable DDL edit, no migration chain); `status`/webui/`/metrics` show `rebuilding generation N: X% (serving M)` from the pointer/mirror; ops reset rebuilds mirror from pointer file. |
| **b5l.3** | Blue-green rebuild driver (daemon) | Daemon builds `index.gN+1` from source.db in background over the 20d.15 bulk lane (idle IO class) while reads hit gen N; progress (rows/s, ETA) streamed to status; `building→active→draining→reaping` state machine; crash-recovery reaps orphan gens. |
| **b5l.4** | Atomic pointer swap + parity gate + delta replay | Swap gated on 4bu parity (raw-materialization + FTS `messages_ready`); write pause <100 ms (pointer replace + ingest drain only); delta since `t_start` replayed idempotently before swap; post-swap row parity verified. |
| **b5l.5** | Retention / reaping with lease safety | Grace-window reaper (default 300 s, configurable); refcount lease prevents deleting a gen with open readers; never reaps `active`/`building`; WAL/SHM cleaned; crash-safe. |
| **b5l.6** | `reset --index` → schedule blue-green; `--offline` preserves old path | Default `reset --index` schedules blue-green rebuild; `--offline` keeps delete-and-rebuild; `bootstrap.py` index version-mismatch triggers rebuild-schedule instead of hard raise (offline keeps raise). |
| **b5l.7** | Zero-downtime AC harness + SLO gate + docs | Continuous-read harness proves 0 failed queries across a seeded-corpus schema bump; swap <100 ms asserted; ≥100 rows/s + <5 min rebuild gated in `slo-catalog.yaml`; `internals.md` schema-versioning section + operator runbook updated (20–40 min window removed); topology projection regenerated. |

Depends on **20d.15** (bulk throughput makes the delta window trivial) and **4bu** (parity oracle). b5l.1–b5l.2 land first (mechanism), b5l.3–b5l.5 concurrently (driver/swap/reap on shared pointer — serialize edits to the pointer module), b5l.6–b5l.7 last.

---

## 6. Top-3 Risks

1. **Delta-replay race at the swap boundary.** A source row written *after* `t_start` snapshot but *before* the write-pause drain could be missed if the delta cutoff and pause aren't ordered correctly, silently dropping the newest session from the new generation. *Mitigation:* take the delta cutoff **inside** the write pause (pause → snapshot max source `updated_at_ms` → replay tail → swap → release), and assert post-swap row parity against source count in the AC harness. Idempotency makes an over-inclusive replay free, so bias the cutoff early.

2. **Reaping a generation still held by a long-lived reader** (e.g. a stuck MCP session or a cold CLI that opened just before swap) → `no such table`/IO error on a live query, violating the zero-failed-queries AC. *Mitigation:* refcount lease (open registers / close deregisters, mirroring blob-GC lease discipline) **plus** grace window **plus** never-reap-active/building; reaper skips any gen with a live lease and re-checks next cycle rather than force-deleting.

3. **Pointer/mirror divergence or partial adoption corrupting resolution.** If the ops.db mirror is ever treated as authoritative, or gen-0 adoption runs while a stale reader holds `index.db`, path resolution could point at a half-built or deleted file. *Mitigation:* the **pointer file is the sole authority**; ops.db is a best-effort mirror rebuilt from it; gen-0 rename happens under the daemon startup lock before any reader opens; `active_index_db_path` validates the resolved file exists and its `PRAGMA user_version == pointer.index_schema_version`, falling back to legacy `index.db` if the pointer is absent/corrupt.

**Do NOT change** the lab-policy no-upgrade-chains lint — this is still rebuild-from-source, concurrent, not an in-place migration. Index stays out of `DURABLE_MIGRATION_TIERS`.

---

## [a9e91eeefbf0b4c17] Spec: Model-Drift Observatory (Polylogue)

I have enough grounding. Here is the implementation spec.

---

# Spec: Model-Drift Observatory (Polylogue)

**Thesis.** Treat the archive as a longitudinal instrument: hold task *shape* fixed (intent-embedding anchor + `workflow_shape`), vary `(normalized_model, month)`, and measure how cost/turns/error-rate/latency drift — with changepoint detection flagging model-upgrade rate-shifts and honest uncertainty (Wilson/bootstrap) gated by coverage. **Ships entirely as registered measures over the measure-algebra (9l5.7) + temporal series primitive (9l5.8) + cross-provider coverage guard (9l5.2). No bespoke `analyze` mode.** Sits at LAYER 3 of epic `polylogue-9l5`, composing *through* the keystone LAYER 2 registry — this bead is a consumer/validator of `.7`/`.8`, not a parallel path.

## Grounding (real substrate this reuses)

- **Cohort dimensions already materialized.** `SessionInferencePayload` (`polylogue/insights/archive_models.py:105-106`) carries `workflow_shape` + `workflow_shape_confidence`; `SessionEvidencePayload` carries `workflow_shape_features`, `canonical_session_date`, `total_input/output_tokens`, `total_cost_usd`, `cost_provenance`, `latency_percentiles_ms`, `tool_active_duration_ms`, `wall_duration_ms`, `logical_session_id`. The **model axis** is `CostModelBreakdown.normalized_model` (`insights/archive.py:493-494`, rollup key `(source_name, normalized_model_or_model_name)` at `archive_rollups.py:210`) — that `normalized_model` is the cohort key, *not* the raw `model_name`.
- **Intent matching.** `polylogue/daemon/similarity.py` — no session-level vector exists; session similarity is per-message `vec0 MATCH` fanout over `message_embeddings` (Voyage 1024-dim) deduped to sessions, cosine in `[0,1]`, with explicit `disabled/unavailable/not_embedded/ready` states and `embedding_status.needs_reindex` as the embedded-coverage signal. The anchor must be defined in those terms.
- **Coverage matrix.** `storage/usage.py` exact-vs-estimated-per-origin + `cost_provenance` per session is the honesty source the 9l5.2 guard reads.
- **Registry pattern to mirror.** `InsightType` descriptor + `INSIGHT_REGISTRY` (`insights/registry.py`), `project_origin_payload` boundary projection. `MeasureSpec` (9l5.7) is the analogous declare-once contract; `polylogue/analytics/stats.py` (Wilson/bootstrap/Mann-Whitney) is 9l5.7 slice-1 and does not yet exist.
- **Changepoint substrate.** 9l5.8 series primitive + PELT/binary-segmentation + candidate-with-nearby-events honesty rail; `daemon/cursor_lag_baseline.py` is the bespoke rolling-baseline to converge onto, not duplicate.

## 1. Schema / measure surface

**No new physical tables in the hot path.** All observatory outputs are *derived* (index.db tier = rebuildable, no migration chain). Three declaration surfaces:

**(a) Registered `MeasureSpec`s** (in `polylogue/analytics/measures/model_drift.py`, registered into the 9l5.7 `MEASURE_REGISTRY`). Each measure is `⟨reducer × frame × grouping × window × comparison × uncertainty⟩`:

| measure name | construct | reducer | evidence_tier | required_coverage | uncertainty |
| --- | --- | --- | --- | --- | --- |
| `drift.cost_per_logical_session` | $ efficiency on fixed shape | mean/median $ per logical session | provider-reported (priced) | `priced-provenance ≥ τ` per cell | bootstrap CI (heavy tail) |
| `drift.turns_per_task` | verbosity/efficiency | median assistant-turn count | structural | `n ≥ n_min` | bootstrap CI |
| `drift.tool_failure_rate` | reliability | proportion `tool_result_is_error` over actions | structural (index v16 keystone) | `n ≥ n_min` | Wilson interval |
| `drift.session_error_rate` | outcome | proportion sessions with `terminal_state ∈ {failed,abandoned}` | derived (heuristic label) | `n_min` + tier footnote | Wilson interval |
| `drift.latency_p50/p90` | responsiveness | percentile of `latency_percentiles_ms` | structural, confound-flagged (tool-wait mix) | `timing_provenance=structural` share ≥ τ | bootstrap CI on percentile |

Fixed shared **frame** for all five: `SelectionSpec = intent_anchor(anchor_ref, θ) ∧ workflow_shape=S ∧ workflow_shape_confidence≥c` over **LOGICAL_SESSION grain** (`logical_session_id`, one row per lineage family — avoids fork/resume double-count, #2467). Fixed **grouping** = `(normalized_model, month=canonical_session_date→YYYY-MM)`. **Window** = the monthly series. **Comparison** = adjacent-month + cross-model. **Uncertainty** as above.

**(b) Intent-anchor definition** stored durably as a `user.db` assertion — `AssertionKind` is schema-free `TEXT`, so a new kind `cohort_anchor` needs **no user-tier migration**, only enum-embedding regen (`render openapi` + `render cli-output-schemas`). Payload: `{anchor_session_id | centroid_vec_ref, cosine_threshold θ, workflow_shape S, shape_confidence_floor, notes}`. `context_policy_json` default `{"inject":false}`.

**(c) DSL surface (grammar in `archive/query/expression.py`, per fnm doctrine).** New nothing-bespoke — a `cohort` frame modifier + reuse of 9l5.7 `measure` and 9l5.8 `series`/`over` stages:
```
sessions where similar:<anchor> shape:mvc window month
  | cohort by model
  | measure drift.tool_failure_rate
  | over 2025-01..2026-06
  | changepoint annotate model-switch,hook-event
```
renders per `(model,month)`: `rate [lo,hi] n=… (structural)` + candidate changepoints + coverage footnote.

## 2. Cohorting + changepoint (pseudocode)

```python
# --- COHORT BUILD --------------------------------------------------
def build_drift_cohort(anchor, S, θ, c, date_range):
    # 1. intent match — reuse similarity.py, never re-embed
    sim = find_similar_sessions(anchor, limit=MAX)        # cosine in [0,1]
    if sim.status != "ready":
        return CohortResult.absent(sim.status)            # honest absent-state
    matched = {h.session_id for h in sim.hits if h.score >= θ}

    # 2. shape gate + collapse to logical grain
    cells = defaultdict(list)                              # (model, month) -> [logical]
    embedded, total = 0, 0
    for sess in load_profiles(matched, date_range):
        if sess.workflow_shape != S: continue
        if sess.workflow_shape_confidence < c: continue
        total += 1
        if embedding_status(sess).needs_reindex == 0: embedded += 1
        lg = sess.logical_session_id or sess.session_id   # dedup lineage (#2467)
        model = sess.normalized_model or "unknown"        # NOT raw model_name
        month = sess.canonical_session_date[:7]           # YYYY-MM
        cells[(model, month)].append(logical_view(lg, sess))

    # 3. per-cell coverage gate (9l5.2 / 9l5.7 refusal, not silent partial)
    embed_cov = embedded / total if total else 0.0
    for key, rows in cells.items():
        rows = collapse_by_logical(rows)                  # 1 row per lineage
        n = len(rows)
        priced = mean(r.cost_provenance == "priced" for r in rows)
        cells[key] = Cell(rows, n=n, priced_frac=priced)
    return CohortResult(cells, embed_coverage=embed_cov)

# --- MEASURE + UNCERTAINTY (per cell) ------------------------------
def measure_cell(cell, spec):
    if cell.n < spec.n_min:                     return TieredValue.insufficient(cell.n)
    if spec.required_coverage.priced and cell.priced_frac < τ:
        return Refusal(f"{spec.name}: cell priced coverage {cell.priced_frac:.0%} < τ")
    if spec.reducer is PROPORTION:              # Wilson
        k = sum(spec.numerator(r) for r in cell.rows)
        lo, hi = wilson_interval(k, cell.n)     # analytics/stats.py (9l5.7)
        return TieredValue(k/cell.n, (lo,hi), cell.n, spec.evidence_tier)
    else:                                        # mean/median/percentile -> bootstrap
        vals = [spec.value(r) for r in cell.rows]
        pt, lo, hi = bootstrap_ci(vals, spec.reduce, B=2000)
        return TieredValue(pt, (lo,hi), cell.n, spec.evidence_tier)

# --- CHANGEPOINT (per model's monthly series) ----------------------
def detect_drift_changepoints(series, known_events):
    # series: [(month, TieredValue)] for one model, one measure
    xs = [tv.point for _, tv in series if tv.usable]
    if len(xs) < MIN_SEG*2: return []
    cps = pelt(xs, penalty=BIC_penalty(len(xs)))     # ruptures [analytics];
                                                     # fallback: binary_segmentation() ~60 lines
    out = []
    for idx in cps:
        month = series[idx].month
        # HONESTY RAIL: candidate, never causal
        nearby = [e for e in known_events            # model-version transition
                  if abs(months_between(e.month, month)) <= 1]  # + hook/harness events
        delta, sig = compare_segments(series, idx)   # two-sample test on the split
        out.append(Changepoint(month, delta, sig, candidates=nearby, causal=False))
    return out
```

Key correctness points: (i) **`normalized_model` transitions inside a model series must first *split the series*** — a "model upgrade" is itself a cohort re-key, so the drift comparison is *between* adjacent model versions on the same `(shape,intent)` frame; changepoint-within-a-model-series flags *non-upgrade* drift (data-shift, harness change). (ii) coverage gate runs *before* CI, and a failing cell returns `Refusal`/`insufficient`, never a bare zero. (iii) bootstrap (not normal CI) on cost/latency because those distributions are heavy-tailed (matches 9l5.8 MAD rationale).

## 3. Migration

- **index.db (derived, rebuildable):** no numbered migration. The observatory reads existing materialized profiles/cost rollups; if it needs a persisted cohort-membership cache, add it to the **canonical index DDL + rebuild plan**, then `polylogue ops reset --index && polylogued run` — never an upgrade helper (rejected by `devtools lab policy schema-versioning`). Preferred: **compute on read**, cache nothing (cohorts are query-time; the anchor is the only durable state).
- **user.db (durable):** new `AssertionKind = "cohort_anchor"` is `TEXT`, **no SQL migration**; but regenerate `render openapi` + `render cli-output-schemas` (enum is embedded) and add a `user_audit` surface entry (the every-kind invariant, per MEMORY.md #2383 gotcha).
- **embeddings.db:** untouched (read-only consumer).
- **Topology:** any new module under `polylogue/` → run `devtools render topology-projection && devtools render topology-status` or `render all --check` fails.

## 4. Test strategy

- **Property (`tests/property`, Hypothesis):** `wilson_interval` / `bootstrap_ci` nominal coverage on synthetic Bernoulli / log-normal streams (this is 9l5.7 AC; the observatory piggybacks). Cohort key is a total function: every session lands in exactly one `(model,month)` cell or is gated out with a reason.
- **Seeded-corpus behavior (`devtools test`, testmon-affected only):** (a) synthetic **step-series** — inject a cohort where `tool_failure_rate` jumps at a known month across a model boundary; assert changepoint located within ±1 month and rendered as *candidate + nearby model-switch*, never causal (mirrors 9l5.8 AC). (b) **coverage-refusal**: a `(model,month)` cell with unpriced provenance for a cost measure is refused with actionable error, not a silent partial (mirrors 9l5.2 AC #2). (c) **LOGICAL grain**: seed a fork/resume family, assert it contributes n=1 not n=3 to its cell (guards #2467 double-count). (d) **embedded-coverage gate**: anchor over a partially-embedded corpus emits `embed_coverage` footnote and low-coverage flag rather than a confident intent match. (e) **absent-state**: embeddings disabled → observatory returns `status=disabled`, not empty-looks-like-zero.
- **Snapshot:** one end-to-end DSL query on the demo archive rendering CI + tier footnote + coverage line (reuse `demo seed && demo verify`, private-data-free). Use `frozen_clock` for month bucketing.
- **Registry conformance:** `insight_rigor_audit` / registry test asserts each `drift.*` measure declares tier + required_coverage + confounds (the 9l5.7 enforcement extends to these).

## 5. Bead breakdown (children of a new `polylogue-9l5.13`, "Model-drift observatory")

Ordered by dependency; each depends on 9l5.7 (registry) and, where noted, 9l5.8 (changepoint) / 9l5.2 (coverage guard).

1. **`.13.1` Intent-anchor primitive + `cohort_anchor` assertion kind** — define/persist an anchor `(similar-session set | centroid, θ, shape, conf-floor)` in user.db; regen openapi/cli-output-schemas + user_audit entry. *AC:* anchor round-trips; `find_similar_sessions` + shape gate returns a deterministic session set on the seeded corpus; disabled-embeddings path returns honest absent-state. *(dep: none beyond substrate)*
2. **`.13.2` LOGICAL-grain cohort builder `(normalized_model, month)`** — collapse matched sessions to `logical_session_id`, key by `normalized_model`+`canonical_session_date[:7]`, compute per-cell n / priced_frac / embed_coverage. *AC:* fork/resume family counts n=1; every session gated with a machine-readable reason; property test = total function. *(dep: .13.1)*
3. **`.13.3` Register the 5 `drift.*` MeasureSpecs** — cost/turns/tool-failure/session-error/latency, each with tier + required_coverage + confounds, wired to `analytics/stats.py` Wilson/bootstrap. *AC:* registry conformance test passes; each measure emits `point [lo,hi] n (tier)`; census-vs-sample flag correct (no CI on full-population counts). *(dep: 9l5.7 slice-1, .13.2)*
4. **`.13.4` Coverage-gate composition guard for cohort cells** — a cell failing priced/embedded/n_min thresholds refuses as bare number with actionable error; footnote sourced from `storage/usage.py` matrix. *AC:* reproduces 9l5.2 refusal AC on an unpriced cell; passing cells carry per-cell coverage footnote. *(dep: .13.3, 9l5.2 guard)*
5. **`.13.5` Changepoint over the drift series with model-upgrade honesty rail** — split series at `normalized_model` transitions; PELT/binary-seg within-model; annotate candidates with nearby model-switch + hook/harness events; never causal. *AC:* synthetic step located ±1 month, rendered candidate+nearby-events, `causal=false`. *(dep: .13.3, 9l5.8)*
6. **`.13.6` DSL wiring + renderers + demo snapshot** — `cohort by model` frame modifier composing with `measure`/`over`/`changepoint`; terminal sparkline (CLI) / table (`--plain`); one snapshot on demo archive. *AC:* the end-to-end query renders CI + tier + coverage footnote; `query_units`/`completions` discover the `drift.*` measures. *(dep: .13.4, .13.5, fnm.1)*

(Optional `.13.7` — cross-provider drift column: same frame, `origin:claude-code-session vs codex-session`, `$0-lane` counterfactual once local sessions exist; folds directly into 9l5.2 rather than duplicating.)

## 6. Top-3 risks

1. **Intent-anchor validity is the whole construct — and it's the weakest link.** There is no session-level embedding vector; "same task shape" is reconstructed from per-message neighbor fanout (`similarity.py`) which is dormant on most archives (`embedding_status.needs_reindex`). If embed coverage is low or θ is mis-tuned, cohorts silently mix intents and every downstream drift number is confounded. **Mitigation:** mandatory `embed_coverage` gate + footnote; treat the anchor as a declared, inspectable `cohort_anchor` assertion (auditable, not implicit); require `workflow_shape` AND embedding agreement (two independent signals) before a session enters a cell.
2. **Model-drift vs workload-drift is not separable observationally.** A shifted `tool_failure_rate` at a model upgrade may be the model, or a concurrent harness/hook/repo-mix change — the classic confounder the 9l5.8 design explicitly refuses to auto-assert. Over-claiming "the upgrade regressed reliability" is the failure mode. **Mitigation:** changepoints ship strictly as *candidates + nearby known events*, `causal=false`; confounds declared per MeasureSpec; seasonality/workload-mix flagged; render never emits a causal sentence.
3. **Small-n cells + heavy tails make most cohorts statistically empty.** Fixed `(intent ∧ shape)` framing is deliberately narrow; per-`(model,month)` cells will frequently fall below `n_min`, and cost/latency tails make means unstable. Risk: the observatory renders mostly `insufficient`/`Refusal`, looking broken. **Mitigation:** bootstrap (not normal) CIs; Benjamini-Hochberg when a grouping fans >5 comparisons (9l5.7); allow month→quarter window coarsening as a declared frame value; surface `insufficient` honestly rather than lowering `n_min` — an empty-but-honest observatory beats a confident-but-underpowered one.

---

**Convergent-design commitments honored:** measure-algebra (everything is a registered `MeasureSpec`, zero bespoke `analyze` mode), LOGICAL_SESSION grain (dedup lineage, #2467), construct-validity (evidence_tier + required_coverage + confounds per measure, coverage gate at composition, tier footnotes auto-rendered, changepoints candidate-only). **Load-bearing files:** `insights/archive_models.py` (cohort dims), `insights/archive.py`/`archive_rollups.py` (`normalized_model`), `daemon/similarity.py` (intent match), `storage/usage.py` (coverage), `insights/registry.py` (descriptor pattern to mirror), `archive/query/expression.py` (DSL grammar), and the yet-to-exist `analytics/stats.py` + `MEASURE_REGISTRY` from bead `polylogue-9l5.7`. This bead is a downstream consumer of `.7`/`.8`/`.2` — sequence it after the keystone lands.

---

## [a8ec68603920858f3] SPEC — Fix non-injective origin collapse for aggregates (key on `Source`, mark lossy group

I have the full picture now. Writing the spec.

---

# SPEC — Fix non-injective origin collapse for aggregates (key on `Source`, mark lossy groupings)

## 0. Grounded problem statement

The collapse is **not** merely a display-time projection — it is baked into stored identity, then re-collapsed on read.

**Write path** (`storage/sqlite/archive_tiers/archive.py:958,1035`): both `sessions` (index.db) and `raw_sessions` (source.db) store `origin = origin_from_provider(session.source_name)`. `origin_from_provider` maps **both** `Provider.GEMINI` and `Provider.DRIVE` → `Origin.AISTUDIO_DRIVE` (`core/sources.py:220-232`). `sessions.origin` is a PK component (`sessions(origin, native_id)`, `index.py:39-75`) and feeds `session_id = origin || ':' || native_id`.

**Read path** (cost/usage/analytics): rollups read `s.origin`, pass it through `_provider_for_origin(row["origin"]).value` (`archive.py:6565` etc.) — which for `aistudio-drive` returns `gemini` for **both** runtimes — key the aggregate on that `source_name` (`archive_rollups.py:217` keys `(source_name, model)`), then `project_origin_payload`/`_source_name_origin` (`insights/registry.py:114-143`) project `gemini → aistudio-drive` at the output boundary.

Net: every origin-grouped number (cost rollups, `provider_usage`, tool-usage, summaries `origin_breakdown`) **silently sums two distinct runtimes** — AI-Studio web/Takeout export (`drive-takeout`) and AI-Studio direct (`gemini-export`) — with no marker. The `Source` vocabulary that *can* tell them apart (`core/sources.py`: `_SOURCE_GEMINI.family="gemini-export"` vs `_SOURCE_DRIVE.family="drive-takeout"`, both `originating_lab="google"`) is discarded at write.

**Recoverability:** the distinguishing token `session.source_name` (parser output, provider-wire) exists at parse time; `provider_to_source(Provider.from_string(source_name))` yields the full `Source`. It survives in the raw payload and is re-derived on any `reprocess` (parse+materialize+index). It is *not* recoverable from a stored `origin` alone.

Tier facts that gate the migration: `sessions` lives in **index.db (v24, rebuildable derived tier)** → additive column = canonical-DDL edit + rebuild plan, **no numbered migration**. `raw_sessions.origin` is in **source.db (durable)** but need not change — family is recoverable on reparse.

---

## 1. Keying change + `lossy_grouping` marker schema

### 1a. Store `Source.family` alongside `origin` (index.db, derived)

Add one column to `sessions`:

```sql
source_family TEXT NOT NULL DEFAULT 'unknown'
    CHECK (source_family IN ({','.join canonical SourceFamily tokens}))
```

- Populated at write from `provider_to_source(session_provider).family` — the **pre-collapse** token (`gemini-export` / `drive-takeout` / `claude-code-session` / …).
- `origin` stays exactly as-is (still the collapsed public token) → `session_id` identity, all public `--origin` filters, and every existing read surface are byte-unchanged. This is deliberately additive and non-breaking.
- Add index `idx_sessions_source_family ON sessions(source_family, sort_key_ms DESC)` mirroring the existing `idx_sessions_origin_sort`.

`source_family` is the storage proxy for `Source` (family carries `originating_lab` and `runtime_root` deterministically via `source_for_family`). Aggregates key on `source_family`; `Source`'s other fields are looked up, never stored redundantly (consistent with the "identity is computed" invariant).

### 1b. Re-key aggregates on `source_family`, project at boundary

All rollup SQL that currently does `_provider_for_origin(row["origin"]).value AS source_name` changes to select `s.source_family` directly and key the Python grouping on it. Boundary projection replaces the current `_source_name_origin` provider round-trip with a direct `family → origin` map (`origin_from_provider(source_to_provider(source_for_family(fam)))`, memoized).

### 1c. `lossy_grouping` marker

A structured, **presence-signals-lossiness** field attached to any origin-projected aggregate whose displayed origin bucket conflates ≥2 distinct `source_family` values in the underlying data. Absent when the bucket is 1:1 (no noise on honest rows).

```
lossy_grouping: {
    "origin": "aistudio-drive",              # the projected (collapsed) token
    "merged_families": ["drive-takeout", "gemini-export"],  # sorted, the conflated Source families actually present
    "merged_labs": ["google"],               # distinct originating_labs among them
    "reason": "non-injective-origin-collapse"
}
```

Emission rule: attach iff `len(distinct source_family in group) > 1`. In today's `Source` set the only trigger is `aistudio-drive`; the marker is **data-driven, not hard-coded** on that origin, so a future non-injective collapse lights up automatically. Envelope placement: per-row on rollup/breakdown items, and a top-level `lossy_groupings: [...]` list on list-envelopes for surfaces that flatten (MCP `cost_rollups`, `provider_usage`, summaries `origin_breakdown`).

---

## 2. Algorithms (pseudocode)

**Aggregate keyed on Source, project to Origin with marker:**

```
def aggregate_and_project(rows):                 # rows carry source_family
    groups_by_family = groupby(rows, key=r.source_family)   # true grouping unit
    family_totals = {fam: reduce(measure_algebra.plus, g) for fam, g in groups_by_family}

    # project to public origin, tracking which families merge
    origin_acc      = defaultdict(measure_zero)
    origin_families = defaultdict(set)
    for fam, total in family_totals.items():
        origin = origin_of_family(fam)           # family -> provider -> origin (memoized)
        origin_acc[origin]       = origin_acc[origin].plus(total)   # associative/commutative merge
        origin_families[origin].add(fam)

    out = []
    for origin, total in origin_acc.items():
        row = project_origin_payload(total | {"origin": origin})
        fams = origin_families[origin]
        if len(fams) > 1:                        # non-injective collapse actually occurred
            row["lossy_grouping"] = {
                "origin": origin,
                "merged_families": sorted(fams),
                "merged_labs": sorted({lab_of_family(f) for f in fams}),
                "reason": "non-injective-origin-collapse",
            }
        out.append(row)
    return out
```

The measure-algebra assumption: totals form a commutative monoid (`plus`, identity `zero`), so `origin_acc` re-merge is order-independent and the projected origin total equals the sum of its families' totals (conservation — no double count, no drop). This is the construct-validity guarantee: the origin number is *honest as a sum* and *labeled as lossy*.

**Write-time family derivation** (already have the token):

```
source_family = provider_to_source(session_provider).family   # at archive.py:958 write
# origin unchanged: origin = origin_from_provider(session.source_name)
```

**Backfill for existing rows** (no reparse):

```
for session in sessions where source_family = 'unknown':
    provider = provider_from_origin(session.origin)   # canonical inverse
    session.source_family = provider_to_source(provider).family
```

Note the honesty limit: backfill can only restore the **canonical** family of each origin (`aistudio-drive → gemini-export`, since `_ORIGIN_TO_PROVIDER[AISTUDIO_DRIVE]=GEMINI`). It cannot re-split pre-existing drive-takeout rows — that distinction was never stored. Full fidelity for the gemini/drive split requires a `reprocess` rebuild (parser re-derives the true family from raw payload). The spec's migration therefore ships **both**: cheap backfill (correct for all origins except the drive/gemini split, which it approximates) and a documented rebuild for full fidelity.

---

## 3. Migration

**Tier classification:** `sessions` ∈ index.db = rebuildable derived tier → **no numbered migration**. Per CLAUDE.md schema-regime rules and `devtools lab policy schema-versioning`:

1. Edit canonical DDL (`storage/sqlite/archive_tiers/index.py`): add `source_family` column + CHECK + index. Bump index.db `SCHEMA_VERSION` 24 → 25.
2. Add the rebuild plan entry (edit the derived-tier rebuild plan, not an upgrade helper — the policy check rejects upgrade helpers on derived tiers).
3. Deploy path: `polylogue ops reset --index && polylogued run` re-materializes the whole index tree from source.db, and the parser repopulates `source_family` with true families → the gemini/drive split becomes real for the entire corpus on rebuild.
4. Backfill (§2) is an *interim* convenience for environments that can't afford an immediate full reset; it gives correct numbers for every origin except the one lossy pair, which it marks canonically. The `lossy_grouping` marker fires correctly either way because it keys on distinct `source_family` present after rebuild.

**source.db (durable): no change.** `raw_sessions.origin` staying collapsed is fine; the raw payload is the source of truth and the parser recovers family on reparse. Avoids a numbered durable migration + backup-manifest ceremony.

**Regenerate generated surfaces:** `lossy_grouping` enters MCP/CLI/JSON output → `render openapi`, `render cli-output-schemas`, `render all --check` (grep for `out of sync`, don't trust the tail line). No new module = no topology-projection bump, unless the OriginSpec slice (§5) lands a package.

---

## 4. Test strategy

**T1 — non-injectivity round-trip (property, the load-bearing test).** For every `Source` in `ALL_SOURCES`: assert `family → origin → provider_from_origin → provider_to_source.family` returns to the *canonical* family, and that the set `{f : origin_of_family(f) == o}` has size >1 **iff** `o == AISTUDIO_DRIVE`. Directly encodes the non-injective invariant as data, so adding a future collapsing origin fails the test until its marker path is wired.

**T2 — marker fires exactly (golden).** Seed an archive with ≥1 `gemini-export` and ≥1 `drive-takeout` session (distinct `native_id`s), each with cost. Assert: (a) the cost rollup groups them into **two** `source_family` keys internally; (b) the projected origin-level output has a single `aistudio-drive` row whose total = sum of both; (c) that row carries `lossy_grouping` with `merged_families == ["drive-takeout", "gemini-export"]`, `merged_labs == ["google"]`.

**T3 — no false-positive marker.** A single-family origin (e.g. `claude-code-session`) rollup row has **no** `lossy_grouping` key. Guards against noise.

**T4 — conservation.** Property over random measures: `origin_total == Σ family_totals` for each origin (measure-algebra associativity/commutativity), so projection never double-counts or drops.

**T5 — backfill approximation honesty.** After §2 backfill (no reparse) on a fixture that pre-collapsed both into `aistudio-drive`, assert every backfilled row → canonical `gemini-export`, and document (assert) that drive-takeout is NOT recovered without reparse — locks the known limitation so no one later claims backfill is lossless.

**T6 — schema/round-trip.** `sessions.source_family` CHECK rejects non-canonical tokens; a full reset+reingest of the gemini+drive fixture yields the true split (T2 holds post-rebuild). Uses `frozen_clock`; runs via `devtools test <files>`.

---

## 5. Bead breakdown (6, with acceptance)

Suggested parent: contracts/origin-purge program under **polylogue-9e5.8**.

1. **`source_family` column + write-path population.** Add DDL column/CHECK/index (index.py), bump v24→25, populate at both write sites (archive.py:958,1035) from `provider_to_source().family`. *AC:* new archives carry correct `source_family`; `check("source_family", …)` generated from canonical families; mypy-green; rebuild plan entry added.

2. **`family → origin` boundary map + lossy detector.** Add memoized `origin_of_family`/`lab_of_family` to `core/sources.py`; `lossy_grouping` builder. *AC:* T1 passes; `source_for_family` is the single family resolver; no hard-coded `aistudio-drive`.

3. **Re-key cost/usage rollups on `source_family`.** Change rollup SQL + `aggregate_cost_rollup_insights` key (archive_rollups.py:217) + `_provider_for_origin` read sites to select `s.source_family`; project + attach marker at boundary (registry.py). *AC:* T2, T4 pass; `cost_rollups`/`provider_usage`/`session_costs` numbers unchanged for 1:1 origins, split for gemini/drive.

4. **Propagate marker to summaries/tool-usage/analytics envelopes.** `origin_breakdown` (archive_summaries.py:121, archive_rollups.py:123), tool-usage grouping, MCP list envelopes get per-row + top-level `lossy_groupings`. *AC:* T3 passes; MCP + CLI + JSON all surface the marker via the one registry projection.

5. **Migration: backfill command + rebuild-plan doc + regenerated surfaces.** Interim backfill (§2), rebuild-plan text, `render openapi`/`render cli-output-schemas`/`render all --check`. *AC:* T5 passes; `render all --check` clean (grep `out of sync`); `devtools lab policy schema-versioning` green (no upgrade helper on derived tier).

6. **Retirement advance (`OriginSpec` seam, polylogue-2qx).** Fold `family`/`runtime_root`/`originating_lab`/canonical-origin into a per-origin `OriginSpec` so `origin_of_family`, the collapse map, and lossy-pair detection derive from declared specs instead of hand-kept dicts in `sources.py`. *AC:* the GEMINI/DRIVE collapse is expressed once as two specs sharing an origin; adding a future non-injective origin needs only a spec, and T1 auto-covers it. (Depends on 2; unblocks polylogue-2qx / jnj.7 help-leak cleanup.)

Beads 1-5 are the mergeable correctness slice (one coherent PR or a short stack); bead 6 is the structural retirement follow-up.

---

## 6. Top-3 risks

1. **Identity collision hidden beneath the aggregate bug.** `session_id = origin || native_id` means a `gemini-export` and a `drive-takeout` session sharing a `native_id` are *already one row* — undetectable, un-splittable even by reparse. Adding `source_family` fixes aggregation but **not** identity. *Mitigation:* keep this spec scoped to aggregates (as tasked); file a separate bead noting that true per-runtime identity would require `native_id` namespacing or moving `source_family` into the session_id generator — a durable/breaking change out of scope here. State the residual honestly rather than implying identity is fixed.

2. **Backfill masquerading as fidelity.** The cheap backfill restores only canonical families; drive-takeout history stays merged until a full `ops reset --index` rebuild. If operators read post-backfill numbers as the true split, they're misled in the *opposite* direction (all-gemini). *Mitigation:* T5 locks the limitation; the `lossy_grouping` marker + rebuild-plan doc must state that pre-rebuild drive rows appear canonically as gemini. Consider emitting the marker even on single-family backfilled rows when `SCHEMA_VERSION`-vs-materialization provenance indicates a pre-rebuild archive.

3. **Marker/measure drift across the ~130 MCP tools + CLI + JSON.** `lossy_grouping` must ride the *single* `project_origin_payload` boundary, not be re-implemented per surface, or some surfaces will show honest splits while others silently merge — reintroducing the exact inconsistency being fixed. *Mitigation:* enforce one projection chokepoint (registry.py), assert in a contract test that every origin-grouped insight descriptor routes through it (extend the existing `insight_rigor_audit`/every-kind invariant), and regenerate `render openapi`/`cli-output-schemas` so the field is contractually present everywhere at once.

---

Grounding files (absolute): `/realm/project/polylogue/polylogue/core/sources.py` (collapse map lines 220-308), `/realm/project/polylogue/polylogue/insights/registry.py:99-143` (`project_origin_payload`), `/realm/project/polylogue/polylogue/insights/archive_rollups.py:203-274` (rollup keying), `/realm/project/polylogue/polylogue/storage/sqlite/archive_tiers/archive.py:958,1035,6565` (write collapse + read re-collapse), `/realm/project/polylogue/polylogue/storage/sqlite/archive_tiers/index.py:39-90` (sessions DDL). Beads: 9e5.8 (retirement plan / parent), 2qx (OriginSpec seam for bead 6), jnj.7 (help-leak follow-on). No code or beads were edited.

---

## [aae50fd05527b25a5] Wave-5 Spec: Scale vector search — quantized store + coarse prefilter + multi-probe batchi

I now have complete grounding. Here is the implementation spec.

---

# Wave-5 Spec: Scale vector search — quantized store + coarse prefilter + multi-probe batching

## Grounding (real files, verified)

- **Canonical DDL:** `polylogue/storage/sqlite/archive_tiers/embeddings.py` — `EMBEDDINGS_SCHEMA_VERSION = 1`, one `vec0` virtual table `message_embeddings(message_id PK, embedding float[1024], +session_id, +origin)`, plus `message_embeddings_meta` (STRICT) and `embedding_status` (STRICT).
- **Legacy/second DDL:** `sqlite_vec_runtime.py:_ensure_tables` still creates a `+source_name` variant — divergence noted; unify on canonical during this work.
- **Brute-force query paths:** `sqlite_vec_queries.py` — `query` (L116), `query_by_session` (L143, fans out over `_SESSION_SEED_FANOUT=20` seed vectors, one `MATCH k=?` each), `query_by_provider` (L200).
- **Per-seed fan-out (session similarity):** `daemon/similarity.py` — `_PER_MESSAGE_K=20`, loop at L510-512 / L382-383 issues up to 20 separate `MATCH` scans then `_aggregate_hits`.
- **Callers:** `archive/query/archive_execution.py` (L94, L336, L590), `cli/archive_query.py:805`, `storage/search_providers/hybrid.py:159`, `daemon/http.py:3158` (`/api/sessions/{id}/similar`), `mcp/server_insight_tools.py` (`find_similar_sessions`).
- **Rebuild/reset:** `cli/commands/reset.py:39` — `embeddings.db` is in `_REBUILDABLE_ARCHIVE_DATABASES`; `bootstrap.py` blue-green re-inits derived tiers from `PRAGMA user_version` mismatch (no migration chain — confirmed `DURABLE_MIGRATION_TIERS` excludes embeddings).
- **Distance semantics:** L2 over L2-normalized Voyage vectors; `_l2_to_cosine_similarity` = `1 - d²/2` (`similarity.py:217`).
- **Bead:** `mhx.6` owns "quantization, matryoshka, scoped drain"; gated by `mhx.3` (eval lane). This spec implements the retrieval/scale half of `mhx.6` and consumes `mhx.3`'s labeled set.

**Runtime capability probe (sqlite-vec v0.1.9, live):** `int8[N]`, `bit[N]`, `partition key` columns, plain metadata columns (filterable inside KNN `WHERE … MATCH … AND col=?`), multiple vector columns per table, and `vec_quantize_int8(v,'unit')` / `vec_quantize_binary(v)` / `vec_slice` **all confirmed working**. `vec_quantize_binary` requires dim % 8 == 0 (1024 ✓). Verified: `SELECT id,distance FROM v WHERE e MATCH ? AND k=3 AND origin='claude' AND cluster=1` returns partition+metadata-filtered KNN in one statement.

---

## (1) Schema / DDL + tier + regime

**Tier:** `embeddings.db`, `ArchiveTier.EMBEDDINGS`. **Regime: derived/rebuildable** — bump `EMBEDDINGS_SCHEMA_VERSION 1 → 2`, edit canonical DDL, ship a rebuild plan. **No migration helper** (`devtools lab policy schema-versioning` rejects one for a derived tier).

Replace the single float table with a **two-representation store in one vec0 table** (binary coarse + int8 rerank + partition/cluster metadata):

```sql
-- embeddings.db, user_version = 2
CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
    message_id       TEXT PRIMARY KEY,
    origin           TEXT partition key,        -- coarse prefilter #1 (cheap shard)
    cluster_id       INTEGER,                   -- coarse prefilter #2 (centroid bucket; metadata col)
    embedding        int8[1024]                 -- PRIMARY scan representation (4x smaller than f32)
        distance_metric=cosine,
    +embedding_bin   bit[1024],                 -- optional ultra-coarse pass (32x smaller; hamming)
    +embedding_f32   float[1024],               -- full precision, unindexed aux — rerank source
    +session_id      TEXT
);

CREATE TABLE IF NOT EXISTS message_embeddings_meta (
    message_id      TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    dimension       INTEGER NOT NULL CHECK(dimension = 1024),
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32),
    embedded_at_ms  INTEGER,
    quant_version   INTEGER NOT NULL DEFAULT 1, -- which quantizer produced the int8/bit rows
    needs_reindex   INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0,1))
) STRICT;

-- Centroid table: cluster_id -> centroid vector, for assignment + optional centroid pre-pass.
CREATE TABLE IF NOT EXISTS embedding_centroids (
    cluster_id      INTEGER PRIMARY KEY,
    centroid_f32    BLOB NOT NULL,              -- 1024 f32
    member_count    INTEGER NOT NULL DEFAULT 0,
    built_at_ms     INTEGER
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_status ( ... unchanged, STRICT ... );
```

Design notes:
- `int8` (not `bit`) is the **primary indexed** column: measured recall of int8-cosine is near-lossless; `bit` kept as `+aux` only if scale later demands a two-stage hamming→int8 pass. Start with int8-primary + f32-rerank; treat `bit` as a stretch arm gated by the eval.
- `origin` as a real **partition key** physically shards the vec0 index (Voyage-normalized, so cosine metric declared explicitly). `cluster_id` is a metadata column (filterable, not a partition — keeps re-clustering from forcing a full table rewrite).
- `+embedding_f32` stores the exact bytes today's code already serializes (`_serialize_f32`), so rerank is a local blob read, no re-encode.
- `distance_metric=cosine` lets us drop the `_l2_to_cosine_similarity` conversion for the int8 path (keep the L2 helper for the f32 rerank if it stays L2).

---

## (2) ANN / quantization algorithms (pseudocode)

**A. Quantized scan + float rerank** (replaces `query`):
```
query(text, limit):
    q_f32  = normalize(embed(text))                       # unit vector
    q_int8 = vec_quantize_int8(q_f32, 'unit')
    OVERSAMPLE = 4                                         # tune via mhx.3 eval
    # Stage 1: cheap int8 KNN over the whole (or partitioned) index
    candidates = SELECT message_id, embedding_f32, distance
                 FROM message_embeddings
                 WHERE embedding MATCH q_int8 AND k = limit*OVERSAMPLE
                 [AND origin = :origin]        # if caller scoped
                 ORDER BY distance
    # Stage 2: exact rerank on full precision, in Python or via vec_distance_cosine
    for c in candidates:
        c.exact = cosine(q_f32, deserialize(c.embedding_f32))
    return top-`limit` by exact
```
Scan bytes: int8 index is 1/4 of f32; with `bit` prefilter it is 1/32. Rerank touches only `limit*OVERSAMPLE` full vectors.

**B. Coarse prefilter / centroid pre-pass** (for unscoped global search when partition doesn't apply):
```
build_centroids(n_clusters ~= sqrt(N)):          # offline, rebuild-time
    sample = reservoir_sample(all f32 vectors, 100k)
    centroids = kmeans(sample, n_clusters)        # spherical / cosine
    for each stored vector v: cluster_id = argmax_c cosine(v, centroids[c])
    write embedding_centroids + UPDATE cluster_id

query_with_prepass(q_f32, limit, nprobe=8):
    near = top-`nprobe` cluster_ids by cosine(q_f32, centroid)
    candidates = SELECT ... WHERE embedding MATCH q_int8 AND k=limit*OVERSAMPLE
                             AND cluster_id IN (near)      # single statement, IVF-style
    rerank as in A
```
`nprobe` is the recall/latency knob; falls back to A (no cluster filter) when centroids absent.

**C. Multi-probe batching** (kills per-seed fan-out in `query_by_session` / `similarity._PER_MESSAGE_K`):
```
find_similar_sessions(seed_session, limit):
    seed_vecs = SELECT embedding_f32 FROM message_embeddings WHERE session_id = seed AND ... sample K_seed
    # OLD: K_seed separate MATCH statements (20 scans)
    # NEW: one pooled probe
    centroid = normalize(mean(seed_vecs))          # single representative probe
    #   OR pooled-candidate union in ONE pass:
    q_int8 = vec_quantize_int8(centroid,'unit')
    hits = SELECT message_id, session_id, embedding_f32, distance
           FROM message_embeddings
           WHERE embedding MATCH q_int8 AND k = limit*OVERSAMPLE*FANOUT
           AND session_id != seed
    # rerank each hit against the *closest* seed vec (max-sim), aggregate per session
    for h in hits: h.sim = max(cosine(h.f32, s) for s in seed_vecs)   # exact, in-memory
    aggregate_by_session(best sim, matched_message_count)
```
One `MATCH` instead of 20; exact max-sim rerank preserves the "closest seed vector" semantics `_aggregate_hits` currently computes. Keep an optional small multi-probe (2-3 diverse seed centroids via mini-kmeans on seed vecs) if single-centroid recall regresses in the eval.

---

## (3) Rebuild plan (embeddings is rebuildable)

1. Bump `EMBEDDINGS_SCHEMA_VERSION → 2`; edit canonical DDL; unify `sqlite_vec_runtime._ensure_tables` onto it (delete `+source_name` divergence).
2. `bootstrap.py` sees `user_version 1 < 2` on a derived tier → blue-green re-init (existing behavior). Document operator path: `polylogue ops reset --embeddings && polylogued run` (reset.py already lists `embeddings.db`).
3. **Requantize pass** (no re-embedding — vectors already stored as f32 in `raw`/prior tier is *not* the case here, so): the rebuild path re-embeds from `index.db` authored messages via the normal materialization stage. To avoid re-spending Voyage $, add a **one-shot in-place requantize migration script** run *before* the version bump on existing archives: read each `embedding_f32`, compute `vec_quantize_int8` + `cluster_id`, write the v2 table — purely local CPU, $0. This is a rebuild-plan artifact, not a schema migration helper.
4. Centroid build runs as a **new `ConvergenceStage`** (`daemon/convergence_stages.py`) after embedding catch-up, `false_means_pending` when the vector count grew beyond a rebuild threshold (re-cluster deferred to quiet window).
5. Wire cost preflight: model/dim switch stays a tier reset with `ops embed preflight` (per `mhx` doctrine) — quantization change is $0 and does **not** require re-embed, only requantize.

---

## (4) Test strategy incl. recall@k eval

- **Correctness/unit:** `tests/unit/storage/` — quantize round-trip (int8 within tolerance of f32 cosine); DDL v2 bootstrap; partition+metadata KNN returns same top-1 as brute force on a seeded corpus; `query_by_session` single-probe path returns a superset of the old fan-out's top-`limit` sessions on `corpus_seeded_db`.
- **Rebuild:** version-mismatch triggers blue-green re-init; requantize script produces int8 rows whose reranked top-k == brute-force f32 top-k on the seeded corpus.
- **Recall@k eval (the load-bearing gate) — fork/resume-lineage-labeled, free positives:**
  ```
  Positive pairs come for free from session_links (index v12+):
    a child (prefix-sharing fork/resume) and its parent SHARE content →
    a query seeded from the child's divergent tail MUST retrieve the parent
    (and sibling forks) as top neighbors. inheritance + branch_point give
    ground-truth "these sessions are semantically near" WITHOUT hand-labeling.
  eval_recall_at_k(k in {5,10}):
    for each child C with parent P in session_links:
        neighbors = find_similar_sessions(C, limit=k)   # quantized path
        hit = P in neighbors  (or any lineage sibling)
    recall@k = mean(hit); also MRR
  Report: recall@k_int8 vs recall@k_f32 (brute-force baseline);
          gate: delta < TOLERANCE (e.g. recall@10 drop < 0.02) → else fail.
  ```
  This slots into `mhx.3`'s `devtools bench retrieval` labeled set (bead AC #4: "same labeled set importable by mhx.6"). Add lineage-labeled pairs as a CI-runnable seeded subset; live-archive variant (16K sessions, real Voyage vectors) drives the accept decision.
- **Latency/scale regression:** benchmark int8-scan bytes vs f32 (assert ~4×/~32× fewer), and single-probe vs 20-probe wall time for `find_similar_sessions`, via `tests/benchmarks/`.

---

## (5) Bead breakdown (children of `mhx.6`)

1. **`mhx.6.a` — v2 quantized DDL + tier rebuild.** Bump `EMBEDDINGS_SCHEMA_VERSION`, int8-primary + f32-aux + partition/cluster columns, unify the two `_ensure_tables` definitions, rebuild plan doc. **AC:** `ops reset --embeddings && polylogued run` produces a v2 store; bootstrap version-mismatch blue-greens; no migration helper (schema-versioning policy passes).
2. **`mhx.6.b` — quantize+rerank query path.** Rewrite `query`/`query_by_provider` to int8 KNN oversample → f32 rerank; add `$0` requantize script. **AC:** reranked top-k == brute-force f32 top-k on `corpus_seeded_db`; scan-byte benchmark shows ≥4× reduction.
3. **`mhx.6.c` — coarse prefilter (origin partition + centroid IVF).** Partition-key wiring + `embedding_centroids` + centroid `ConvergenceStage` + `nprobe`. **AC:** `origin`-scoped query hits only its partition; centroid pre-pass recall@10 within tolerance at `nprobe=8`.
4. **`mhx.6.d` — multi-probe batching for `find_similar_sessions`.** Collapse `_PER_MESSAGE_K`/`_SESSION_SEED_FANOUT` loops in `similarity.py` + `query_by_session` to one pooled probe with exact max-sim rerank. **AC:** one `MATCH` per call (assert query count); result sessions ⊇ old top-`limit`; wall-time regression benchmark green.
5. **`mhx.6.e` — lineage-labeled recall@k eval.** Build the `session_links`-derived pair set; integrate into `devtools bench retrieval` (consumes `mhx.3`). **AC:** recall@5/@10 + MRR emitted per arm (f32 baseline vs int8 vs int8+centroid); `compare` exits non-zero on a regression beyond stated tolerance.
6. **`mhx.6.f` — cite + default flip.** Update `docs/search.md` and `docs/internals.md` (schema-version history) with eval artifact citations; make quantized+prefilter the default lane only after `mhx.6.e` proves tolerance. **AC:** default resolution changed with cited artifact path + metric deltas.

(Stretch, optional 7th: **`mhx.6.g` — binary `bit` two-stage pass** — only if int8 scale insufficient; gated by `mhx.6.e`.)

---

## (6) Top-3 risks

1. **Quantization silently degrades recall on the real 16K-session/millions-of-vectors archive even when the seeded corpus looks fine** — int8 range/`'unit'` assumes Voyage L2-normalization holds for every stored row; non-unit rows (mixed models, legacy) skew quantization. *Mitigation:* `mhx.6.e` must gate on the **live** lineage-labeled set, not just CI seed; refuse mixed-model vectors (per `mhx` AC) before quantizing.
2. **Centroid/IVF prefilter kills recall for cross-topic queries** — `cluster_id IN (nprobe)` can prune the true neighbor when a query straddles clusters; recall becomes silently `nprobe`-dependent. *Mitigation:* keep centroid pre-pass **optional and off by default** until the eval proves a safe `nprobe`; always fall back to unfiltered int8 scan; partition-by-`origin` (exact, lossless) is the safe prefilter, centroids are the risky one.
3. **Single-centroid pooling changes `find_similar_sessions` semantics** — collapsing 20 seed probes into one mean-vector probe loses multi-topic sessions whose messages span distinct regions (a long session's mean is meaningless). *Mitigation:* mini-kmeans multi-probe (2-3 diverse seed centroids) as the fallback arm, chosen by the eval; preserve exact per-message max-sim rerank so aggregation semantics (`matched_message_count`, best-distance) stay identical to `_aggregate_hits`.

**Two grounding caveats for the implementer:** (a) there are **two** live vec0 DDL definitions (`archive_tiers/embeddings.py` canonical with `+origin`, and `sqlite_vec_runtime.py` legacy with `+source_name`) — unify them in `mhx.6.a` or the requantize pass will hit the wrong shape; (b) `distance_metric=cosine` on the int8 column lets you drop `_l2_to_cosine_similarity` for that path, but the f32 aux rerank and all callers currently assume L2 — audit `archive_execution.py` score handling when you flip it.

---

## [a475ce796faf72a3f] Spec — webui-v2 "evidence cockpit" (polylogue-bby.11 slice)

I have the grounding I need. Here is the spec.

---

# Spec — webui-v2 "evidence cockpit" (polylogue-bby.11 slice)

Grounded in: `core/refs.py`, `daemon/web_shell_lineage.py`, `daemon/web_shell_provenance.py`, `daemon/web_shell_selection.py`, `daemon/web_shell_realtime.py`, `storage/sqlite/archive_tiers/index.py` (blocks/session_links DDL), `storage/sqlite/archive_tiers/user.py` (assertions), `insights/topology.py`, bead `bby` / `bby.11`.

## 0. The load-bearing fact this spec turns on

Today a block citation is an `EvidenceRef` = `session_id::message_id::block_index` where `block_index` is **`blocks.position`** (`block_id = message_id || ':' || position`, `core/refs.py:117-182`). Position is exactly what re-ingest and fork-replay shift. `messages.content_hash` exists (`index.py:123`) but **blocks carry no content hash** — so there is no re-ingest-stable block identity anywhere in the system yet. Every cockpit feature (citations, marginalia, report footnotes) inherits this fragility. The anchor engine is the foundation; graph/report/overlay are consumers.

---

## 1. Schema / API surface + stack choice

### Stack — adopt bby.11's decision (TS + Preact + Vite), do not re-litigate

bby.11 already committed the stack with rationale (Preact = deepest agent training vein at 4KB; TS keeps the mypy-equivalent net; Vite dev-proxy to daemon; dist committed via `devtools render webui`, no node in deploy). I adopt it unchanged. The one thing the cockpit **adds** is resolving the stated tension between a Preact SPA and the progressive-enhancement AC ("usable on phone / `curl|pandoc`"):

**Two-tier delivery.** The daemon server-renders the reader-transcript and the report to **semantic HTML + a Markdown twin** (the `RenderSpec` static-export path already named in bby.11 notes: `SelectionSpec × ProjectionSpec × RenderSpec`). That layer is the degradation-safe core — works no-JS, phone, `curl … | pandoc`. Preact **hydrates** it for interactivity (graph, command palette, live SSE). Enhancement-only surfaces (force graph, palette) degrade to their existing server forms: the graph falls back to the BFS text list already in `web_shell_lineage.py`. Rule: **nothing that is evidence is JS-only**; only interaction is.

### New durable state — user.db assertions, ZERO migration

`kind` is free `TEXT` with no CHECK (`user.py:11-30`), so the entire cockpit vocabulary lands with no user-tier schema bump — the sharpest durability win. Three new `AssertionKind` values (`core/enums.py:399`):

| kind | target_ref | value_json payload | body_text |
|---|---|---|---|
| `citation_anchor` | the captured `EvidenceRef` | `{block_hash, message_hash, session_content_hash, captured_position, text_quote, char_range}` | — |
| `evidence_report` | `report:<uuid>` | `{blocks:[{type:"prose"|"cite", …}], title, theme}` (ordered document) | rendered Markdown cache |
| `report_basket` | `basket:<uuid>` | `{anchor_ids:[…]}` (ordered working set) | — |

`evidence_refs_json`, `supersedes_json`, `confidence`, `status`, `context_policy_json` are reused as-is. Baskets/reports are just assertions → they inherit `list_assertion_claims`, superseding, and the existing `/api/user/annotations` plumbing.

### New derived state — index.db, rebuildable (no migration chain)

Add `blocks.block_content_hash BLOB` (plain column, Python-populated by the materializer — SQLite has no built-in SHA so it cannot be a generated column), reusing the block-payload NFC normalization already in `core/hashing.py`. Plus `CREATE INDEX idx_blocks_content_hash ON blocks(session_id, block_content_hash)` for reverse lookup. Because index.db is rebuildable, this is edit-canonical-DDL + `polylogue ops reset --index && polylogued run` to backfill — no upgrade helper (`devtools lab policy schema-versioning` forbids one).

### API surface (extends the OpenAPI render → typed client is generated, per bby.11)

```
POST   /api/anchors                 body EvidenceRef+captured context → CitationAnchor (computes block_hash)
POST   /api/anchors/resolve         body [anchor_id|inline] → [{status, resolved_ref, position_drift, confidence}]
GET    /api/reports/{id}            → report doc + live resolution state
PATCH  /api/reports/{id}            → mutate ordered blocks (prose/cite)
GET    /api/reports/{id}/export     ?format=md|html → static artifact + integrity manifest
GET    /api/reports/{id}/integrity  → {total, exact, drifted, deleted, quarantined}
GET    /api/sessions/{id}/topology  (EXISTS — graph reuses it verbatim; no new lineage API)
```

`resolution.status ∈ {exact, drifted, deleted, quarantined}` mirrors the provenance panel's existing chips (`web_shell_provenance.py:52-54`).

---

## 2. Algorithms (pseudocode)

### A. Block content-hash (write-side, in materializer)

```
def block_content_hash(block):
    # mirror core/hashing.py session-hash block subpayload
    payload = nfc_normalize(join_sentinel(
        block.block_type, block.text, block.tool_name,
        canonical_json(block.tool_input), block.tool_id, block.media_type))
    return sha256(payload)            # stored in blocks.block_content_hash
```

### B. Anchor resolution (the core — survives re-ingest + fork-position shift)

```
def resolve(anchor) -> Resolution:
    sess = load_session(anchor.session_id)
    if sess is None:                      return DELETED
    if sess.quarantined:                  return QUARANTINED   # from session_links.status

    # Tier 1 — fast exact path: message unchanged ⇒ position still valid
    msg = messages[anchor.message_id]
    if msg and msg.content_hash == anchor.message_hash:
        return EXACT(ref = block_at(msg, anchor.captured_position))

    # Tier 2 — hash reverse-lookup across the COMPOSED transcript.
    # Fork/resume replays the parent prefix, so compose parent-up-to-
    # branch_point + child-tail (message_query_reads.compose path) and
    # search that block set — position is irrelevant, hash is stable.
    composed = compose_lineage_blocks(anchor.session_id)   # spans session_links
    hits = [b for b in composed if b.block_content_hash == anchor.block_hash]
    if len(hits) == 1:
        drift = hits[0].position != anchor.captured_position
        return EXACT_or_DRIFTED(ref=hits[0].ref, position_drift=drift)
    if len(hits) > 1:                     # duplicated block (acompact) — disambiguate
        return nearest_by_position(hits, anchor.captured_position)  # DRIFTED, conf<1

    # Tier 3 — content edited: fuzzy recover by quote
    cand = trigram_best_match(anchor.text_quote, composed)
    if cand.score >= THRESHOLD:           return DRIFTED(ref=cand.ref, confidence=cand.score)
    return DELETED   # block gone from the composed transcript
```

Position-keyed refs break at Tier 2 (fork shift); hash-keyed refs resolve because the composed transcript is searched by content, not offset. That is the whole design.

### C. Report render + provenance footnotes

```
def render_report(report) -> Markdown:
    out, foot = [], []
    for blk in report.blocks:
        if blk.type == "prose":  out += blk.markdown
        else:                                             # citation block
            r = resolve(blk.anchor)
            out += blockquote(r.rendered_text) + f"[^{blk.n}]"
            if r.status != EXACT: out += drift_badge(r.status, r.confidence)  # live flag
            foot[blk.n] = provenance_footnote(r)          # title · origin · role ·
                                                          # canonical_url · captured_at ·
                                                          # ref · resolution status
    return join(out) + "\n\n" + join(sorted(foot))        # standard MD footnotes → pandoc-clean
```

Live: SSE `message.appended`/`insight.updated` (`web_shell_realtime.py`) → re-resolve only cited anchors → update drift badges in place; operator prose is never rewritten.

### D. Integrity verifier (export gate)

```
def export(report, fmt):
    res = [resolve(b.anchor) for b in report.cite_blocks]
    manifest = tally(res)                 # {exact, drifted, deleted, quarantined}
    if manifest.deleted or manifest.quarantined:
        require operator_override         # else refuse; watermark if forced
    return RenderSpec(fmt).freeze(report, res), manifest
```

### E. Force graph (consumer of existing `SessionTopology`)

No API change. Vendored dependency-free Verlet/force sim in `lib/graph.ts` (no-CDN posture). `TopologyEdgeKind` → `tokens.css` colors (`fork`/`subagent`/`sidechain`/`resume`/`unresolved_native`); `session_links.status='quarantined'` cycle-break edges dashed-red; `branch_point_message_id` node ringed. No-JS/phone → the existing BFS list from `web_shell_lineage.py` renders server-side unchanged.

---

## 3. Migration

- **user.db (durable):** none. New kinds are TEXT values; `USER_SCHEMA_VERSION` stays 4. This is deliberate — the irreplaceable tier takes zero migration risk.
- **index.db (rebuildable):** add `blocks.block_content_hash` + index to canonical DDL; bump index schema version; rebuild plan is `polylogue ops reset --index && polylogued run` (backfills hashes via materializer). No upgrade helper. Batch this bump with any other pending index-tier bumps before triggering a live rebuild (per CLAUDE.md schema discipline).
- **Existing position-refs:** remain valid. Anchors are additive; a legacy `EvidenceRef` with only a position is a Tier-1-only anchor (no hash) that upgrades on first re-capture.

---

## 4. Test strategy

- **Anchor resolution (property + unit):** given a `SessionBuilder` session, re-ingest with (a) unchanged content → EXACT; (b) prepended fork prefix → EXACT with `position_drift=True`; (c) edited cited block → DRIFTED; (d) deleted block → DELETED; (e) `session_links.status='quarantined'` → QUARANTINED. Hypothesis strategy over block payloads asserting `resolve(anchor(b)) == b` under position permutation.
- **Content-hash stability:** `block_content_hash` invariant to NFC/whitespace-sentinel normalization; equal to the block subpayload used by `core/hashing.py` session hash.
- **Report/integrity:** golden Markdown snapshot (footnotes pandoc-parse cleanly); integrity manifest tallies match seeded drift; export refuses on `deleted`/`quarantined` without override.
- **Progressive enhancement:** `curl /api/reports/{id}/export?format=md | pandoc` produces valid HTML; reader route renders no-JS (server HTML present before hydration).
- **Graph:** vitest component snapshot of edge-kind → color mapping and quarantine styling; CDP smoke lane (the bby.7 ref-walk, run against v2) drives select→cite→report→export.
- **Contract:** `EXPECTED_TOOL_NAMES`/OpenAPI regen for new routes; `render all --check` (grep `out of sync`); topology-projection regen for any new module.

---

## 5. Bead breakdown (children of bby.11 / bby)

1. **`block-content-hash` (P1, blocker):** index.db `blocks.block_content_hash` column + index + materializer population + rebuild plan. *AC:* every block hash matches `core/hashing.py` subpayload; index-schema bump + rebuild backfills; property test for normalization stability; `render all --check` clean.
2. **`citation-anchor-engine` (P1, dep:1):** `citation_anchor` AssertionKind + `POST /api/anchors` + resolution engine (Tiers 1-3, lineage-composed). *AC:* 5-case resolution matrix (exact/drift/edited/deleted/quarantined) green; fork-prefix re-ingest returns EXACT+drift; zero user.db migration.
3. **`report-model-and-render` (P1, dep:2):** `evidence_report`+`report_basket` kinds, PATCH/GET routes, Markdown render with provenance footnotes, live SSE re-resolve. *AC:* golden footnote snapshot pandoc-parses; drift badge updates on SSE without prose rewrite.
4. **`integrity-verifier-export` (P2, dep:3):** `/integrity` + `/export` gate + RenderSpec static freeze. *AC:* manifest tally correct; export refuses deleted/quarantined absent override; `curl|pandoc` PE test passes.
5. **`force-graph-view` (P2, dep: none — uses existing topology API):** Preact force component, edge-kind coloring, quarantine/branch-point styling, BFS list fallback. *AC:* vitest color-map snapshot; no-JS falls back to `web_shell_lineage` list; no new API.
6. **`marginalia-overlay` (P2, dep:2):** assertions rendered as anchor-resolved margin notes over the reader transcript, reusing the resolution engine so marginalia survives re-ingest. *AC:* margin note re-anchors after position shift; `context_policy` display-vs-inject separation preserved.
7. **`basket-capture-ux` (P3, dep:2,3):** extend the existing selection toolbar (`web_shell_selection.py`) + command palette to add block/session to a live basket → seed a report. *AC:* select→cite→basket→report flow drives the CDP smoke lane.
8. **`agent-buildability-proof` (P3):** the bby.11 judge-queue view built purely against scaffold docs, reusing the anchor client. *AC:* a coding agent adds the view with no core edits (the bby.11 buildability criterion).

---

## 6. Top-3 risks

1. **Duplicated-block ambiguity (Tier 2 multi-hit).** Auto-compaction physically replays blocks, so one `block_content_hash` legitimately appears N× in a composed transcript. Nearest-by-position disambiguation is heuristic and can mis-anchor a citation onto the wrong replay copy. *Mitigation:* store `captured_position` + `session_content_hash` on the anchor; prefer the copy in the same lineage segment; downgrade confidence and surface it rather than guessing silently.
2. **Rebuild cost of the hash column on a 38 GB archive.** `ops reset --index` reprocesses the whole tree to backfill `block_content_hash`; that is a heavy live rebuild. *Mitigation:* batch this bump with other pending index-tier schema work (CLAUDE.md rule) so the archive rebuilds once, not per-feature; verify materializer hash cost on the seeded corpus before touching the live archive.
3. **Progressive-enhancement scope creep vs SPA gravity.** The Preact SPA is the path of least resistance; the Markdown/SSR twin is extra work that is easy to skip, which would silently break the `curl|pandoc`/phone AC. *Mitigation:* make the Markdown export the **primary** report render (SPA consumes the same server output), and put the `curl|pandoc` check in the CDP/PE lane so regressions fail CI, not review.

**Convergence check:** citation-anchors, findings/reports-as-objects, and the episode unit all land as **user.db assertion kinds** (schema-free), and the graph reuses the **existing `SessionTopology` envelope** — so this spec adds one rebuildable derived column and zero durable-tier migrations. No code or beads were modified.

---

## [a20ebb94646a0189e] Polylogue positioning plan (Wave-5)

I have complete grounding. Delivering the positioning plan directly.

---

# Polylogue positioning plan (Wave-5)

Grounded in `polylogue-3tl` (legibility epic), `polylogue-212` (construct-valid demo portfolio), `polylogue-cfk`/`jxe` (uplift status), README, architecture-spine, `docs/proof-artifacts.md`, and raw-log (Karpathy contact intent 2026-05-01; "not legible even to fellow nerds" 2025-06-30; explicit "explicit, persistent, legible" doctrine).

## 1. Category claim + pitch

The team already committed to **"the system of record for AI work"** (3tl diagnosis: name the category, don't borrow chat-viewer/observability/memory buckets). Keep it as the *formal* category — but it's bureaucratic and won't grab the AI-engineering sphere on its own. Lead the pitch with a vivid subordinate hook, then land on the category.

**One-sentence claim:**
> Polylogue is the flight recorder for AI work — one local, cross-vendor system of record where every claim an agent made resolves to the structural evidence for what actually happened.

**3-sentence pitch:**
> Your coding agents produce the running notebook, the debug log, and the only proof of what they did before claiming success — then it disappears into per-vendor silos. Polylogue ingests Claude, ChatGPT, Codex, Gemini, and coding-agent sessions into one local archive and answers what ordinary transcript folders can't: what the agent did, what failed, what it cost, what should resume. Every derived number resolves, on `--explain`, to a structural fact — an exit code, a tool-result `is_error`, a usage event, raw bytes — never a regex over the assistant saying "done."

"Flight recorder" is the grab; "system of record" is the category anchor; "resolves to structure not prose" is the wedge. All three in one breath.

## 2. README first-screen structure

The current first screen is close but front-loads the 7-minute skim ladder and prose before the reader has a reason to care. Reorder so the **hero finding is visible above the fold** — a stranger should hit a number that makes them lean in before they hit the architecture.

```
[H1] Polylogue  ·  badges
[1 line]  Flight recorder for AI work — the system of record where every
          claim resolves to structural evidence.
[HERO STRIP]  "On a 42,033-failure archive, agents silently proceeded past a
              structural failure at least 24.1% of the time. Reproduce it in
              one command ↓"   ← the grab, with a link to §Try it
[5 QUESTIONS]  What did it do / what failed / what did it cost / what should
               resume / where's the raw evidence  (keep — this is the best
               part of the current README)
[TRY IT]  polylogue demo tour   (uvx one-liner, 30s to first result)
[WHY TRUST IT]  the 5 evidence bullets (structure not prose, FTS invariant,
                rebuildable-vs-durable, deletes features that guess)
[skim ladder / architecture / links]  ← demote below the fold
```

Rule for the first screen: **a number, a command, a trust claim — in that order.** Everything else (four verbs, five tiers, DSL) is depth the cold reader opts into.

## 3. The single hero finding

Ranking the four candidates against: has-a-number-now, reproducible, construct-valid, lands-in-Karpathy-discourse, emotionally grabby.

| Candidate | Number today | Reproduce cmd | Construct-valid | Discourse fit | Verdict |
|---|---|---|---|---|---|
| **Claim-vs-evidence (silent-proceed rate)** | **24.1% silent lower bound / 42,033 failures; precision 100%, recall 84.2%** | ✅ published | ✅ `is_error`/`exit_code` | ✅ "do agents lie about done?" | **HERO** |
| Replay-duplication % | ~32% dup msgs, 16K vs 8.8K sessions | partial | ✅ | medium (internal-baseball) | supporting |
| Subscription-vs-API cost gap | qualitative (cache-free on Max) | ✅ cost demo | ✅ | medium | supporting |
| Abandonment mortality | no headline % yet | needs build | ✅ | high but not ready | future |

**Winner: the claim-vs-evidence silent-proceed finding.** It is the only candidate that already ships with published numbers *and* a reproduce path, it *is* the category-defining wedge made measurable (structure over prose), and it lands exactly where the AI-engineering/Karpathy discourse already is ("agents claim success; did they?"). The other three become the second-tier "and also" evidence.

**Hero headline:** *"How often does a coding agent sail past its own failure? On a 42,033-failure archive: at least 24.1% of the time, silently."*

**Reproduce command (private-data-free, from `docs/proof-artifacts.md`):**
```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/poly-cve-demo
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
devtools workspace claim-vs-evidence \
  --archive-root "$POLYLOGUE_ARCHIVE_ROOT" --limit 5000 \
  --out-dir /tmp/poly-cve-repro --json
```

One caveat to publish alongside (do not hide): the 24.1% headline is the operator's single private archive. The credibility upgrade is `3tl.3` — the **multi-model, incl. open-models** leaderboard variant. Ship the single-archive finding now with the leaderboard flagged as the next artifact; don't wait for it, but don't imply it exists.

## 4. Flight-recorder vs public-honesty-benchmark: the framing decision

**Decision: they are not competitors — they sit at different layers, and conflating them is the trap.**

- **Flight recorder = the product's identity** (what it *is*, for a stranger, private-utility-first). It's honest about scope: a local recorder you own. Low over-claim risk. This is the README/category framing.
- **Public honesty benchmark = the launch *campaign artifact*, not the product's identity.** The claim-vs-evidence leaderboard is the *proof that the recorder reads truth*, and it's the thing that lands in discourse and reaches Karpathy. But making "honesty benchmark" the product's name would (a) over-promise a benchmark org Polylogue isn't staffed to be, (b) invite endless "your failure predicate is wrong" methodology fights as the *whole product's* credibility, and (c) abandon the durable, defensible "local system of record" moat for a crowded leaderboard genre.

**So: recorder is the noun, honesty-benchmark is the launch verb.** The launch post is "I built a flight recorder for AI work; here's the first thing it measured." The recorder framing owns the repo/README; the benchmark finding owns the launch narrative and the Karpathy-sphere hook. This also matches the operator's own doctrine ("every finished artifact proves the substrate is honest") — the benchmark *proves the recorder*, it doesn't *replace* it.

## 5. 90-second demo script

Obeys the `polylogue-212` compositionality rule (product primitives; shell is glue only) and the ground rule (every number → `--explain` structural evidence).

```
[0:00–0:15]  "AI agents produce the only record of what they actually did,
              then it vanishes into vendor silos. Here's one local archive
              across every provider — no upload, no API key."
              $ polylogue demo tour            # 30s, throwaway det. archive

[0:15–0:40]  ASK, DON'T SCROLL.
              $ polylogue analyze --facets      # what's in here
              $ polylogue find "pytest" then read --view messages
              "Grep gives you lines. This gives you sessions, tool calls,
               costs, outcomes — one query surface across all vendors."

[0:40–1:10]  THE WEDGE — structure over prose.
              $ devtools workspace claim-vs-evidence --limit 5000 --json
              "The agent said 'done.' Did it? The failure predicate isn't
               the word 'error' in prose — it's is_error=1 and non-zero
               exit codes. On the real archive: 24.1% silent-proceed."
              $ polylogue ... --explain          # number → exit_code row → raw bytes

[1:10–1:25]  HONESTY SLIDE (construct validity):
              "Here's a question we CANNOT answer from structure, and we
               say so instead of guessing." (the 212.1 'cannot answer X' slide)

[1:25–1:30]  "Local. Cross-vendor. Every number resolves to evidence.
              One command: polylogue demo tour."
```

The `--explain` drilldown at 0:40–1:10 is the whole show — it's the visible proof that the number isn't prose-mined. Do not cut it for time.

## 6. Top-3 risks

1. **The memory/uplift over-claim trap.** The "remember" verb's benefit is *unproven* — the only run (`jxe`) closed **negative** (raw-ref 8/10 beat handoff-pack 5/10), and the re-test (`cfk`, n≥3 then n=12–20) hasn't reported. If the launch narrative claims memory *uplift*, a critic finds the negative pilot and the whole credibility (built on "we don't guess") collapses. Mitigation: keep every memory claim capability-phrased ("compiles context bundles") until `cfk` reports; the launch post is the *benchmark*, not the memory loop.

2. **Category grab vs. surface breadth.** "System of record" reads enterprise/bureaucratic and won't stop the Karpathy-sphere scroll on its own; meanwhile the real surface (130 MCP tools, 5 tiers, a Lark DSL) is a wall a cold reader bounces off. The hero *finding* must carry the emotional grab that the category name doesn't — which is exactly why #3 (hero above the fold) and the flight-recorder hook exist. Risk if the README leads with architecture: stranger never reaches the number.

3. **Hero-finding construct-validity counterattack + n=1.** The 24.1% invites "silent-proceed ≠ dishonesty; your failure predicate over-counts" and "it's one operator's archive." Pre-empt both: (a) publish the precision/recall calibration (100%/84.2%) and the explicit "cannot answer X" slide *in the launch itself*, not as a footnote; (b) frame the single-archive number as a lower bound with the multi-model leaderboard (`3tl.3`) named as the next artifact. Do not let the benchmark's methodology fight become the *product's* credibility fight — that's the layer-separation in #4.

---

**Net recommendation:** Keep "system of record for AI work" as the category anchor; lead every surface with the **flight-recorder** hook and the **24.1% silent-proceed** hero number (published, reproducible, construct-valid, discourse-native); treat the honesty benchmark as the launch campaign that *proves* the recorder rather than the product's identity; and hold all memory claims capability-phrased until `cfk` reports. The single highest-leverage README change is moving a resolvable number above the fold.

---

## [a809d19336db88d7c] - `episode` as a first-class derived unit in index.db (rebuildable): a `episodes` table + 

- `episode` as a first-class derived unit in index.db (rebuildable): a `episodes` table + `episode_members(session_id, role, confidence, evidence_json)` join, keyed by a content-hash of its ordered member set so re-stitch is idempotent like the ingest hash. Episodes are NOT sessions and NOT session_links rows — they are a higher tier over topology, so within-provider lineage (swarm2) stays the leaf and cross-tool stitching composes above it. — Gives the stitch a durable, queryable home without polluting the session tree or the durable tiers. — NEW

- Stitch algorithm as an explicit 4-signal scorer, never a single similarity: `repo/cwd match` (hard prior) × `time-window adjacency` (decaying kernel) × `embedding cosine over session summaries` × `shared-artifact overlap` (same file paths, same tool_id/commit sha, same error string). Store each signal's contribution in `evidence_json` so a stitch is auditable, not a black-box merge. — Construct-validity-as-substrate: the edge must carry its own evidence or it isn't real. — NEW

- Shared-artifact join is the strongest edge and should dominate: extract file paths from tool_use/tool_result blocks (edits, reads, bash cwd), git SHAs mentioned in prose, PR/issue numbers, and error fingerprints; two sessions touching `src/foo.rs:tests` within a window are the same episode with high confidence even at low embedding similarity. — Thematic drift makes embeddings weak across a Cursor→ChatGPT handoff; the artifact is invariant. — NEW

- Confidence tiers with a hard false-merge floor: `linked` (topology-proven, e.g. session_links prefix-share) > `corroborated` (≥2 independent signals incl. one hard artifact/repo signal) > `candidate` (embedding+time only). Only `linked`+`corroborated` render as one episode by default; `candidate` edges surface as "possibly related" suggestions requiring one confirming click. — The lane's non-negotiable: never a false-merge; asymmetric cost between missing a stitch and fabricating one. — NEW

- Episode reconstruction read-view (`read --view episode <id>`) that renders the interleaved, deduplicated cross-tool transcript in wall-clock order: Cursor exploration → ChatGPT rubber-duck → Claude Code implementation → the commit, with per-segment origin badges and the stitch-evidence inline as citation anchors. — Deepens wave-1 citation-anchors: every cross-origin seam is a load-bearing claim and must be clickable back to its evidence. — NEW

- Terminal artifact-anchor via lynchpin as the missing glue between AI sessions and non-AI work: correlate episode members to ActivityWatch window focus, Atuin shell history, and git commit timestamps in the same repo+window. A `git commit` with no AI session in-window is still an episode member (the "quiet" step); an episode can END at a commit even though no transcript describes it. — "How I actually solved X" includes the parts that never touched an LLM; AI-only stitching under-counts the real work. — NEW/beads (new)

- Episode→commit/PR attribution as a derived edge with direction and lag: match episode terminal window to the produced commit (author time, changed paths ⊆ episode-touched paths), and forward-link the PR number if the commit landed. Store `produced_ref` on the episode. This turns "which chats produced this merged PR" into a lookup, and "how many sessions/$ did this PR cost" into one measure. — Analytics-one-measure-away: cost per merged artifact is the operator's real ROI question. — NEW

- Episode-level cost & effort rollup that correctly spans origins and the honest-authorship axis: sum authored-user words, assistant tokens, tool-call counts, wall-clock span, and idle gaps across all members, using `material_origin` to exclude protocol rows and NOT double-counting the replayed prefixes already deduped at the lineage tier. — Cross-origin cost is currently un-answerable; the episode is the only unit where "what did solving X actually cost" is meaningful. — NEW

- Time-window kernel must be repo-scoped and gap-aware, not fixed: within one repo a 6h gap is same-episode; across repos even 5 minutes is probably a context-switch, not a stitch. Model inter-member gaps and cut an episode boundary where the idle exceeds a repo-conditioned threshold, so a week-long recurring bug becomes ONE episode with sparse members rather than seven. — Thematic/temporal edges need adaptive boundaries or you either shatter or over-merge. — NEW

- Negative-evidence / anti-stitch signals to actively prevent false-merge: different repo root, disjoint file sets, contradicting goals (embedding of stated objective points opposite), or an explicit topology `quarantined`/cycle-break marker suppress a candidate edge even if time+embedding agree. Anti-signals are recorded in `evidence_json` too. — A merge you refuse is as important to justify as one you make; the recursive-safety gate applies to episode assembly. — NEW

- Episode as a stitch-hypothesis object (findings-as-objects): each proposed episode is a queryable, mark-able, correctable assertion (`AssertionKind` metadata/annotation), so the operator can `confirm`/`split`/`reject` a stitch and that correction feeds back as ground-truth. Rejections train the scorer's thresholds; confirmations become `linked`-tier permanently. — Wave-1 queries/findings-as-objects: the stitch is a claim the user can adjudicate and the system learns from. — NEW

- Cold-start / unresolved-parent bridge: reuse the session_links pattern of persisting edges to not-yet-ingested targets — persist candidate episode edges even when one side (e.g. a Cursor export) isn't imported yet, keyed by artifact fingerprint, and resolve them on later ingest via a converger stage. — Cross-tool exports arrive asynchronously; the stitch must survive the gap like topology_edges do. — NEW

- Episode MCP surface as the continuity primitive (not CLI): `get_episode`, `find_episode_for_ref` (given a commit/PR/file → the episode that produced it), `episode_resume_brief` (reconstruct end-to-end context to continue X in a NEW tool). This is the natural home for "how did I solve X last time" agent recall. — MCP is the continuity surface (per project doctrine); cross-tool reconstruction is exactly agent-facing recall. — NEW

- Corpus-compaction at episode granularity: a completed, committed episode compacts to an episode-summary object (goal, approach, key decisions, produced_ref, dead-ends) retaining citation anchors back to member sessions, so the raw cross-tool sprawl can be archived/cold while the reconstructable narrative stays hot. — Deepens wave-1 corpus-compaction: the episode is the right unit to compress because it's a complete arc, not an arbitrary session. — NEW

- Stitch-quality construct-validity audit lane (`devtools lab`): sample stitched episodes, hold out one signal, measure whether the remaining signals still agree (leave-one-signal-out), and report precision proxies (how many episodes rest on a single soft signal, how many artifact-corroborated). Publishes a rigor score per episode. — Construct-validity-as-substrate: the stitcher itself must be measured, not trusted; a stitch resting only on embeddings is flagged. — NEW

- Cross-tool handoff detection as a named sub-pattern: identify the specific seam where the operator moved tools mid-problem (ChatGPT dead-end → switched to Claude Code) via a member-ordering with an origin change + a restated goal + shared artifacts. Surface these as "tool-handoff" episodes for meta-analysis of WHICH tool the operator escalates to and when. — Turns episode data into self-knowledge: the handoff is where the operator's real workflow lives. — NEW/beads (new)

---

GPT-pro prompt stubs:

[A] "Design the confidence model for merging heterogeneous work sessions (from Cursor, Claude Code, ChatGPT, and a git commit log) into one 'episode' where edges are thematic/temporal, not identity-shared. Given four signals — repo/cwd equality, time-window adjacency, embedding cosine over session summaries, and shared-artifact overlap (file paths, git SHAs, error fingerprints) — specify a scoring function with an explicit hard floor that makes false-merges asymmetrically costly vs. missed-stitches. Define confidence tiers, the anti-stitch (negative evidence) signals, and exactly what evidence must be persisted per edge for a human to audit or reject it."

[DR] "Survey how existing systems reconstruct a single logical task that spans multiple tools and time gaps: developer-activity timeline reconstruction, session/episode segmentation in behavioral logs, record-linkage / entity-resolution confidence models, and event-correlation in observability (trace stitching without a shared trace-id). Extract concrete algorithms for (a) adaptive temporal boundary detection, (b) multi-signal linkage scoring with calibrated confidence, and (c) preventing false links. Map each finding to the four available signals (repo, time, embedding, shared-artifact) and note which techniques transfer to a low-volume, high-stakes personal archive."

[DR] "Research correlating AI-assistant transcripts with the concrete artifacts they produce (git commits, PRs, edited files) and with non-AI activity telemetry (window focus, shell history). What signals reliably attribute a commit to the reasoning that produced it, how is authorship-vs-machine noise separated, and how do prior systems compute cost/effort per unit of shipped work when the effort is spread across multiple tools and includes steps that left no transcript? Identify the failure modes of naive attribution and the minimal evidence needed to defend an episode→PR attribution."

---

## [a75b983604cec8338] META-QUALITY R&D — proving Polylogue's own correctness at scale (agent 14/14)

Grounded. Returning the list.

## META-QUALITY R&D — proving Polylogue's own correctness at scale (agent 14/14)

- **Identity round-trip metamorphic property** — Hypothesis-generate arbitrary (origin, native_id, position, variant_index) tuples, write via SessionBuilder, then assert `session_id`/`message_id`/`block_id` generated columns recompose to exactly the inputs AND that no two distinct logical rows collide. The `run_ref` global-PK collision (#2464) was exactly a "global PK derived from non-unique local coordinates" bug — this is the general guard for that whole class. — NEW (child of polylogue-1xc)

- **Content-hash idempotency metamorphic suite** — for a generated session, assert re-ingest with byte-identical payload is a no-op (0 rows changed), a user-metadata mutation (tag/annotate/correct) does NOT change the hash (excluded by construction), and any content mutation DOES. Include NFC-normalization adversaries (combining chars, None/empty/missing sentinels) so the hash boundary is pinned. — NEW

- **Lineage composition ⊕ identity law** — property test: parent full-content ≡ compose(parent-prefix-up-to-branch, child-tail) for prefix-sharing children; assert `branch_point_message_id` survives a parent full-replace DELETE→re-INSERT cycle (the deliberately-not-a-FK invariant). Regression-guard the exact `ON DELETE SET NULL` foot-gun the CLAUDE.md warns about. — NEW

- **Lineage no-double-count invariant at scale** — differential test: sum of physically-stored divergent-tail messages + composed-parent messages must equal the logical transcript length, never 2×. This is the 16K-physical-vs-8.8K-logical / ~32% dup measurement (#2467) turned into an executable law over a synthetic fork/resume/acompact corpus. — NEW (relates polylogue-1xc)

- **FTS5 trigger-coherence metamorphic test** — after any sequence of block insert/update/delete, assert the contentless FTS index and `blocks.search_text` agree (every block matchable by its own tokens, no orphan/ghost rows). Contentless + `contentless_delete=1` + 3 triggers is fragile under bulk rebuild; fuzz the operation ordering, not just single ops. — NEW

- **Construct-validity gate as a test, not just an MCP tool** — promote `insight_rigor_audit` / `readiness_check` into a self-verify pytest lane: for every `INSIGHT_REGISTRY` entry, assert that any insight claiming a numeric measure has non-empty backing rows (an insight that reports a number over zero rows must FAIL). This is wave-1's "insight claiming a number whose rows are empty must fail" made a CI gate. — NEW

- **Empty-vs-fabricated differential over the recovery digest** — regression-test the fixed "PR #123 merged" fabrication (regex `_events_from_text`, no authorship gating): feed prose that mentions a PR merge with `material_origin` proving it's assistant-authored speculation, assert it surfaces as an *unverified candidate*, never a claimed event. Construct-validity for the text-mining path specifically. — NEW

- **Daemon-vs-direct differential lane** — ingest the same synthetic corpus through (a) `polylogued run` full convergence and (b) the direct `reprocess` path, then assert byte-identical index.db logical state (sessions, messages, blocks, session_links, insights) modulo ops-tier cursors. The daemon owning all writes means these must converge to the same fixpoint; today nothing proves it. — NEW

- **Convergence idempotency + `false_means_pending` fixpoint property** — property: running any `ConvergenceStage` twice is a no-op the second time; a stage that returns `False` must have pushed exactly the un-done remainder into `convergence_debt` (no work silently dropped, no work double-scheduled). Chaos-order the stage execution to catch ordering-dependent debt leaks. — NEW (guards docs/retro/2026-05-24-1498-cascade)

- **Fault-injection / chaos harness on the single writer** — SIGKILL the daemon mid-transaction at each ingest stage boundary (acquire/parse/materialize/index) against a synthetic corpus, restart, assert the archive converges to the same fixpoint with no partial-session visibility and no orphaned blobs. Directly exercises the blob-GC two-invariant (lease + snapshot ref) acquire→commit window. — NEW

- **Blob-GC race property** — model the acquire-blob→commit-row gap as an interleaving and assert no reachable blob is ever collected and no leaked blob survives two GC generations. Extend `test_blob_store_props.py` with a concurrent lease/GC-generation state machine (Hypothesis `RuleBasedStateMachine`). — NEW (extends property/test_blob_store_props.py)

- **Adversarial parser fuzz on real export shapes** — grow `fuzz_json_parsers.py` with structure-aware generators per provider: grouped-JSONL split by `sessionId`, drive-like nesting, the 384MB single-artifact Codex raw row, duplicate native ids, and detector-tightness collisions (a payload two detectors could both claim). Assert: no crash, no cross-provider misroute, and streaming path ≡ non-streaming path for multi-GiB Claude Code JSONL. — NEW (extends fuzz/)

- **Scale-shaped synthetic corpus generator as first-class infra** — the one thing polylogue-1xc.1 needed but hand-rolled: a parameterized generator (N sessions, M messages/session, hash-collision density, duplicate-native-id rate, giant-artifact injection) behind a `scale` pytest marker, so every scale bug (WAL blowup, single-transaction rebuild, PK collision) gets a cheap reproduction tier instead of a live 38GB copy. This is the missing substrate the whole epic's AC ("a lane that would have caught each shipped bug class") depends on. — NEW (polylogue-1xc infra)

- **WAL/transaction-boundary assertion helper** — a reusable test fixture that spies `conn.commit()` and measures peak WAL growth, so any "must chunk into bounded transactions" claim (rebuild, migration, bulk reingest) is asserted by *commit-boundary count + bounded intermediate WAL*, not by hope. Generalizes polylogue-1xc.1's one-off spy into the property the whole tier-1 class needs. — NEW

- **Schema-regime conformance test** — assert the durability axis is real: derived tiers (index/embeddings) have NO migration files and rebuild cleanly from source; durable tiers (source/user) have a gapless additive migration chain each bumping `user_version` by one behind a backup manifest. Machine-check what `devtools lab policy schema-versioning` asserts, plus a round-trip "reset --index && reingest ≡ prior index state" fixpoint. — NEW

- **CHECK-constraint ↔ Python-type drift property** — since CHECK constraints are generated from `typing.Literal`/`get_args`, add a test that fuzzes each such column with values just outside `get_args(...)` and asserts SQLite rejects exactly the set Python would reject. Catches the day someone edits the enum but the embedded SQL constraint silently diverges. — NEW

- **Encoding/normalization boundary at the hash + FTS seam** — extend `test_encoding_boundary_matrix.py` to prove NFC normalization in `core/hashing.py` and the FTS `unicode61` tokenizer agree: two payloads that are NFC-equal must share a content hash AND be mutually FTS-matchable; NFC-distinct must not collide. Pins the "no porter stemmer, don't change tokenizer" invariant against accidental change. — NEW (extends property/test_encoding_boundary_matrix.py)

---

### GPT-pro prompt stubs

**[A]** "Design a metamorphic/property test taxonomy for a content-addressed, single-writer SQLite archive whose row identities are *generated columns* (id = concat of parent id + local coordinates) and whose fork/resume lineage is stored as divergent-tail-plus-branch-point (parent prefix physically shared, recomposed on read). Enumerate the algebraic laws that must hold (identity round-trip & injectivity, hash idempotency excluding user metadata, lineage compose ⊕ no-double-count, FTS trigger coherence) and for each give the metamorphic relation, the adversarial input generator, and the oracle. Rank by which real bug class each would have caught."

**[DR]** "Survey the state of the art (2023–2026) in *scale-only* and *fault-injection* testing for local-first / embedded-database systems: WAL-blowup and single-transaction regressions, crash-consistency fuzzing at transaction boundaries, deterministic simulation testing (FoundationDB/TigerBeetle/Antithesis style), and property-based state machines for GC/lease races. What techniques are cheap enough to run as an optional CI lane over a *synthetic* scale corpus rather than a multi-GB live copy, and what do they provably miss?"

**[DR]** "Research 'construct validity as a queryable/executable substrate' for derived analytics: how do systems assert that a reported metric's backing rows are non-empty, that a text-mined 'event' is not fabricated from speculative prose, and that an insight over a lineage-deduplicated corpus doesn't double-count? Compare data-observability (Great Expectations, dbt tests, Monte Carlo), semantic-diff, and authorship/provenance-gating approaches, and propose a minimal gate a self-verifying archive could run against its own insight registry."

---

## [af76765e69cf16717] Adversarial construct-validity / honesty audit — findings:

Adversarial construct-validity / honesty audit — findings:

- **Rigor audit silently covers only 5 of ~11+ insight products** — `insights audit`/`insight_rigor_audit` presents as a whole-archive rigor profile but `_RIGOR_MATRIX` omits the highest-stakes number-bearing products (session_costs, cost_rollups, usage_timeline, archive_coverage, tool_usage, archive_debt); `audit.py:188` iterates only contracts and "products without contracts are skipped" — an unaudited cost number looks equally blessed. — Fix: add contracts for the cost/coverage/tool/debt products, or make the report enumerate `uncovered` products instead of silently dropping them.

- **`classify_aggregate_hwm_source` launders temporal provenance** — `temporal_source.py:97` unconditionally returns `provider_ts` for any non-empty input, docstring asserting "Each input is itself a provider_ts," but `classify_profile_hwm_source` can return `fallback_date`/`materialization_ts`; a day-summary/tag-rollup HWM is stamped provider-grade even when contributing sessions had synthetic/None timestamps. — Fix: propagate the weakest source across inputs (min-reliability), not blanket provider_ts.

- **"No regex over prose" holds only for the outcome axis** — `insights/transforms.py` mines prose with `_TEST_PASS_RE`/`_TEST_FAIL_RE`/`_CHECK_PASS_RE` (test pass/fail counts + `_test_evidence` lines), `_COMMIT_SHA_RE`, `_DECISION_RE`, `_STATUS_HEADING_RE`/`_RUNSTATE_SECTION_RE`, `_INSTRUCTION_DUMP_MARKER_RE`; these feed forensic claims and resume/handoff bundles while the keystone comment loudly says "not regex-guessed from prose (#2482)." Only is_error/exit_code is structural. — Fix: label these digest fields `text_derived`/unverified in the payload contract, or subject them to the same authorship gating the outcome axis got.

- **`handler_kind="test"` is partly prose-derived** — `_looks_like_test_output` (transforms.py:2231) classifies via `\d+ passed/failed` in output_text; pathology `wasted_loop` filters on handler_kind∈{shell,git,github,test}, so a structural-looking pathology finding's scope silently depends on keyword/regex classification of tool output. — Fix: derive handler_kind only from command/tool_name tokens, not output prose.

- **Cost enrichment keeps a stale stored estimate silently** — `cost_enrichment.py:54` retains the older stored estimate whenever re-derivation returns non-confident, so displayed `cost_usd` can reflect a superseded pricing basis with no marker that a fresher-but-unavailable derivation was suppressed. — Fix: surface a `stale_basis`/`derivation_suppressed` flag when that fallback fires.

- **Aggregate cost rollups/outlook sum USD across mixed coverage tiers** — exact (codex/claude-code) + estimate-only (chatgpt/claude-ai) + partial (gemini) collapse into one "cycle cost"; the per-origin coverage taxonomy exists only in `diagnostics usage`, not on the headline `CostRollup`/`CycleOutlook` number (docs/cost-model.md). — Fix: attach the exact/estimate/partial USD split to the rollup/outlook payload itself.

- **Provider→origin projection is non-injective on public aggregates** — GEMINI and DRIVE both map to `AISTUDIO_DRIVE` yet payloads project provider→origin at the boundary; any per-origin aggregate over aistudio-drive merges two distinct populations, and `_provider_for_origin` (readiness.py) reverse-picks one arbitrarily. — Fix: keep a sub-origin discriminator or document the merge on the payload (polylogue-2qx OriginSpec).

- **"Every number resolves to bytes" is only block-granularity** — `EvidenceRefKind` is session/message/block only (core/refs.py); a cost/token/`occurrence_count`/phase-tool-count figure's ref points at a message, never at the specific usage counter/field that produced it. — Fix: state the granularity honestly, or add a value-level ref (block + json-path/field).

- **Positional block ids + lineage recomposition make refs silently stale** — `block_id = message_id:position` with `position.variant_index` fallback; prefix-sharing forks recompose parent prefix on read, so an EvidenceRef captured against a composed position can point at a different block after a parent full-replace re-inserts with shifted positions. — Fix: `resolve_ref` should validate target existence/identity and return a `stale_ref` marker rather than a confidently-wrong block.

- **Attachment referenced-bytes reported as if retrievable** (#2468) — 8.4GB referenced, ~0 blobs present, 56% zero-byte; attachment/coverage surfaces count attachments and byte totals that never resolve to stored bytes, breaking "resolves to bytes" for a whole object class. — Fix: split `referenced_bytes` from `stored_bytes` everywhere; never sum referenced bytes as retrievable.

- **`_caveats` substring mining inverts polarity** — transforms.py:2183 matches "caveat"/"blocker"/"not included" over prose to populate handoff caveats, so "no blockers" registers as a blocker and raw model prose surfaces as structured caveat evidence. — Fix: drop, or tag as unverified text snippet.

- **Two unreconciled confidence cutoff systems overstate "high"** — audit `_bucket_confidence` calls ≥0.67 "high" while `confidence.py` `_STRONG_CONFIDENCE_FLOOR=0.78`; the rigor-audit high-confidence count includes rows the confidence vocabulary would call MODERATE. — Fix: unify the audit histogram on `ConfidenceBand` thresholds.

- **`stale_version_count` counts missing versions as fresh** — `audit.py:99-118` returns not-stale whenever the version field is absent/non-int, so a materializer that dropped its version stamp reports 0 stale and looks current. — Fix: distinguish `unknown_version` from `current`.

- **Rebuild-sensitive headline figures presented as facts** — MEMORY: 376.6B Codex tokens were stale rows (clear on re-ingest), credit-rate 5× output bug, per-model partitioning #2472; any doc/demo citing raw archive token/cost totals without a "rebuild-sensitive, not reconciled to provider bill, codex input includes cache" caveat overclaims. — Fix: gate demo/analysis numeric output behind a freshness caveat and per-origin token-semantics normalization before display.

- **Readiness/coverage denominator can exclude convergence-debt sessions** — `false_means_pending` pushes deferred insight backlog into `convergence_debt`; a readiness/coverage query run while debt is nonzero can report sessions as materialized/ready whose insights are actually deferred. — Fix: readiness surfaces must count debt-pending sessions as `partial`/`deferred`, not omit from the denominator.

- **`stale_context` pathology fires on by-design lossy inheritance** — pathology.py counts resume + summary/prefix inheritance as a pathology, but lossy inheritance after auto-compaction is normal; the pathology count inflates with expected behavior (no corroborating post-resume failure required). — Fix: require a corroborating redo/failure signal before counting, or downgrade to informational.

GPT-pro prompt stubs:

- **[A]** "Given Polylogue's insight registry (11+ products) and its `_RIGOR_MATRIX` (5 products) plus the `insight_rigor_audit` runner that silently skips uncovered products: design an honesty-preserving audit contract where every number-emitting insight either declares evidence/inference/fallback provenance or is explicitly reported as `uncovered`. Specify the schema, the enforcement lint, and the failure mode if a new insight ships without a contract."

- **[DR]** "Survey how mature data-provenance / lineage systems (e.g. OpenLineage, W3C PROV, dbt exposures, Great Expectations, dataframe column-lineage tools) distinguish source-clock vs processing-clock timestamps and injective-vs-lossy dimension collapses. Contrast with Polylogue's `TemporalSource` taxonomy that launders aggregate HWMs to `provider_ts` and its non-injective provider→origin (GEMINI+DRIVE→AISTUDIO_DRIVE) collapse. What patterns prevent provenance-laundering on aggregates?"

- **[DR]** "Research the honesty pitfalls of text-mined vs structurally-derived claims in transcript/log analysis tools (regex-over-prose pass/fail counts, commit-SHA scraping, decision/caveat substring detection). How do incident-analysis and observability tools mark heuristic-extracted fields as unverified vs authoritative, and what UX/contract patterns keep a text-derived count from masquerading as a structured fact? Apply to Polylogue's `transforms.py` digest, whose keystone outcome axis is structural but whose counts/decisions/caveats/commit-refs remain regex-over-prose."

---

## [a3f25be5cb29d0bf4] Grounded in polylogue-3tl (legibility epic), polylogue-212 (demo portfolio), polylogue-9e5

Grounded in polylogue-3tl (legibility epic), polylogue-212 (demo portfolio), polylogue-9e5 (audit lane), plus operator raw-log wants (retroactive mining, "extract powerful parametrized prompts / move one layer higher", Karpathy-sphere, dogfood-to-prove-utility). Ideas below deliberately avoid the already-designed set (README skim-ladder, one-command demo, "system of record" name, interchange schema publish, Datasette, install matrix, codebase atlas, D1–D8).

- Hero-finding discipline: designate ONE robust+surprising+clickable number (e.g. material_origin "you author only X% of your Claude Code 'user' turns" or the ~32% replay-duplication) as the single spearhead — README hero, launch post, and lead demo all point at it — instead of shipping a finding *catalog* nobody enters. — A catalog has no front door; one number does. — NEW
- `finding` as a first-class AssertionKind: a published finding becomes a substrate object bundling claim + producing-query + provenance refs + publication URL, listable via `find kind:finding`. — Makes the "citable finding" a queryable artifact, not a loose doc; dogfoods user.db. — NEW (one materializer away, extends AssertionKind vocab)
- Findings-as-tests provenance manifest: each published finding ships a CI-re-runnable manifest (query + corpus version + expected value ± tolerance) so a stale finding fails the devloop, not the reader. — Honesty made mechanical; findings can't silently rot the way the fabricated "PR #123" report did. — NEW (findings-scoped sibling of 3tl.9)
- Head-to-head "lie detector" demo: pose the same question to a naive grep/log-tailer (which fabricates from prose) and to polylogue (which answers from structure or refuses), side by side. — Weaponizes the moat directly; seeded by the real regex-over-prose fabrication incident. — NEW (distinct from 212.1's tracer-can't-answer framing)
- Claims-ledger dogfood: every self-marketing claim polylogue makes is itself a `finding` with an evidence status (proven / capability / aspirational); the project holds its own copy to its own evidence bar, published. — Radical-honesty as brand; operationalizes the "capability-phrased until uplift re-run" discipline as a public artifact. — NEW
- Interactive "click-the-number" Artifact: a single self-contained web page where a stranger clicks a headline stat and watches it decompose number → query → structural field → raw bytes. — The moat as a shareable link; the strongest possible cold-reader proof in 10 seconds. — NEW (Artifact, no code)
- "No regex over prose" creed: a one-page, quotable manifesto that defines the category by its *method* (every number resolves to bytes) rather than its feature list. — Categories anchored on a method are un-miscategorizable; this is the line others will quote. — NEW
- Category-defense "what polylogue is NOT" bake-off: explicit contrast vs LangSmith/Langfuse/Helicone (live tracing, single-provider, online) and Rewind/QS (surveillance framing) — offline, cross-provider, reconstructed-from-exports-you-already-have. — Pre-empts the exact mis-buckets 3tl names as the core problem. — NEW
- `polylogue tour` self-legibility command: the CLI narrates its own category and runs the hero demo for a cold reader from inside the tool. — "A stranger can understand" satisfied without leaving the terminal; the tool explains itself. — NEW (one command away)
- Named media type + paste-your-export validator: register `application/vnd.polylogue.session+json` for the normalized model and ship a self-contained "paste any provider export → normalized JSON" page. — Turns 3tl.6 from "published schema" into an *adoptable standard* + a lead-gen utility that proves parser breadth. — NEW
- Ingest Simon Willison's `llm` SQLite logs as a source adapter: doubles as an outreach hook into the Datasette/`llm` community that already cites SQLite-native LLM tooling. — A feature that manufactures its own citation path to the most on-brand influencer. — NEW
- Self-hosting proof badge: polylogue's own repo is the reference deployment — every dev decision/finding links back to the session that produced it, with a live "N of our own decisions are citable" counter. — Dogfood is the most credible demo; makes the repo page prove the pitch. — NEW
- "Numbers that resolve" as the signature brand gesture: standardize click-a-number → see-bytes as the first-interaction identity across CLI `--explain`, MCP, and docs. — One repeated gesture becomes the brand; consistency across surfaces reads as one system. — NEW (positioning)
- Findings campaign identity ("Ground Truth" field-notes series): a named, serialized publication so findings accrue as a *body someone follows*, not disconnected one-off posts. — Outreach cadence and citability compound; a series earns subscribers a tool cannot. — NEW
- Provenance-trace hero poster: one annotated static image (number → query → structural field → raw byte) as the single visual that sells the moat in social/README OG-card contexts. — The Artifact widget needs a click; the poster works in a scroll feed. — NEW
- [RADICAL] Flight-recorder / black-box positioning: reframe from viewer/memory/QS to the *forensic recorder you consult after a crash* — post-hoc, structural, "what did the agent actually do." — A borrowed-from-aviation frame that no competitor owns and that anchors the forensic-Q&A demos. — NEW (naming)
- [RADICAL] Public honesty benchmark: reframe claim-vs-evidence into a NAMED public eval — "does this agent do what it claims it did?" — with a leaderboard others run their own models against. — Flips personal archive → public science instrument; the leaderboard becomes the thing the world cites, tool is the substrate. — NEW (distinct from 3tl.3's private multi-model variant)
- [RADICAL] Prompt/meta-workflow distillery: reframe polylogue as the *input to a self-improvement loop* that mines your history into better parametrized prompts — "your history is training data for how you should work." — The operator's own stated dream ("move one layer of abstraction higher"); repositions from record-keeping to leverage. — NEW

---

Strongest 3 as GPT-pro prompt stubs:

- [A] **Hero-finding selection** — "Over this bundle of candidate findings computed from the live 38GB archive (authored-user-word share by material_origin, replay-duplication %, subscription-vs-API cost gap, abandoned-session mortality, Codex reasoning-token inflation), rank each on {robustness to corpus changes, surprise to an outsider, one-sentence clickability, resolves-to-bytes cleanliness}. Output the single spearhead finding, its headline sentence, the exact reproduce command, and the 2 backups — with the disqualifying weakness of each rejected one."
- [A] **Prompt distillery over own history** — "Given a bundle of my highest-value past coding-agent sessions (tool-call shapes, corrections I issued, what succeeded vs stalled), induce 5–8 general *parametrized* meta-prompts (parameters: repo, task-type, risk-tier) that would have produced better outcomes than what I actually typed. For each: the pattern it generalizes, the sessions it's distilled from, and a falsifiable A/B test to validate it against a fresh target."
- [DR] **Honesty-benchmark landscape** — "Research the current landscape of evals/benchmarks measuring whether an AI agent's self-reported actions match its actual actions (agent honesty, claim-vs-trace fidelity, tool-call faithfulness). Cover named benchmarks, papers, and tools (2024–2026), who runs them, and what's missing. Assess whether a public 'did the agent do what it claimed, verified from structured session records' leaderboard occupies open space, and what naming/positioning would land in the AI-engineering/Latent-Space/Karpathy-sphere discourse."

---

## [ab6e0ef0d388cbec8] Missing data-model units / relations (ontology gaps)

Grounded in existing units (sessions/messages/blocks, `actions` VIEW, session_links, assertions, files, runs, observed-events, context-snapshots) and the query grammar's unit sources. Here is the lane deliverable.

## Missing data-model units / relations (ontology gaps)

- **episode / investigation** — a logical task spanning sessions/tools/time; enables cost/outcome/abandonment accounting at the unit an operator actually thinks in. Derived: transitive closure over `session_links` lineage ∪ temporal-proximity ∪ shared-repo ∪ embedding-cohesion clustering. index.db; **table** (recursive-CTE + embedding join too heavy for a live VIEW; materialize with a rebuild plan). Extends wave-1 `delegation`/#2467 — **NEW**.

- **turn-pair (exchange)** — human↔assistant adjacency pair as one row; enables per-turn latency, cost, correction-rate, and answer-anchoring (the natural grain for "was this answer good"). Derived: VIEW over `messages` pairing a `human_authored` run to its next `assistant_authored` run within a session by position. index.db; **VIEW** (like `actions`) — **NEW**.

- **entity-mention / reference** — files, `#N` issues, PR/URL, symbols mentioned anywhere; enables backlinks ("every session touching #2467"), the citation-anchor substrate as queryable rows. Derived: parser over `blocks.search_text` + tool args → (entity, block_id, offset) mentions. index.db; **table** (mention edges) + entity dim. Builds on wave-1 citation-anchors — **NEW**.

- **artifact (produced/consumed file)** — a file a session created/edited/read, distinct from `files` (attachments); enables artifact provenance ("which sessions wrote src/x", artifact→artifact lineage). Derived: `actions` where tool∈Write/Edit/Read, normalize path args, join `files`/`observed-events`. index.db; **VIEW** first, promote to **table** if path-normalization needs precompute — **NEW**.

- **world-effect edge (cause→effect)** — links an `observed-event` (commit, file write, test run) to the action/turn that caused it; enables "which turn produced commit abc", transcript↔reality binding. Derived: temporal + path/hash join between `observed-events` and `actions`. index.db; **VIEW** — **NEW**.

- **verification-run outcome** — a check/pytest/`devtools verify`/CI invocation with pass/fail extracted from structure; enables "did this session's change actually verify", failure-loop detection. Derived: `actions` where tool is a runner, read `tool_result_is_error`/`exit_code` (v16 keystone) + stdout tail. index.db; **VIEW/table** — extends `runs` — **NEW**.

- **tool-outcome + retry-chain** — elevate `actions` into an outcome-typed relation carrying error/exit_code and self-linked retry runs; enables failure taxonomy, retry-storm and thrash detection. Derived: `actions` VIEW + self-join on (tool, normalized-args, subsequent position). index.db; **VIEW** — **extend `actions`**.

- **correction edge (anchored)** — a `CORRECTION` assertion bound to the exact block/turn it corrects; enables error-rate-per-tool/per-model and "what the assistant got wrong". Derived: VIEW joining `assertions` (kind=correction) to `blocks` via citation-anchor `block_id`. user.db read-join → index.db surface; **VIEW** — **extend** (wave-1 citation-anchor).

- **project (beyond repo string)** — durable project identity aggregating repo ∪ worktrees ∪ cwd-prefixes ∪ ChatGPT `g-p-` gizmo_id; enables cross-repo/cross-origin project rollups and cost-by-project. Derived: `session_profiles.repo`/cwd + `raw_provider_payload.gizmo_id` mapping. Identity is user-meaningful → **user.db** dim table + index.db membership VIEW — **NEW** (memory: gizmo_id==project).

- **topic / theme cluster** — semantic cluster over session/message embeddings with a label; enables topic timelines and "what have I been working on lately". Derived: HDBSCAN/k-means over `embeddings.db` vec0 → assignments + centroid label. index.db (rebuildable from embeddings); **table** — **NEW**.

- **cross-origin thread** — a logical conversation continued across providers where NO replay `session_link` exists (claude→chatgpt same topic); enables true continuity beyond same-provider replay. Derived: embedding-sim ∪ shared-entity ∪ temporal across differing `origin`, excluding lineage edges. index.db; **table** — formalizes the `threads` MCP tool — **extend**.

- **semantic handoff edge** — session A's intent resumed by session B (possibly cross-machine), distinct from physical replay; enables materialized resume graph. Derived: materialize `find_resume_candidates`/`get_resume_brief` logic as persisted edges. index.db; **table** — **extend** (continuity surface).

- **phase segment** — planning/implementing/debugging/verifying spans as first-class rows (today computed-on-read by `session_phases`); enables phase-duration analytics and workflow-shape queries as unit sources. Derived: sequence classification over turn-pairs/actions. index.db; **table** (or VIEW if classifier is cheap) — **extend** (`workflow_shape_distribution`).

- **goal / intent** — the stated objective of a session/episode, construct-gated so it's a *candidate* until confirmed; enables goal→outcome and abandonment accounting. Derived: first `human_authored` message + `operator_command` mining; unverified until asserted. index.db candidate rows + **user.db** `AssertionKind.goal` when confirmed; **table** — **NEW** (pairs with `find_abandoned_sessions`).

- **decision-object (derived + supersession)** — a decision reached in-transcript plus decision→superseding-decision edges (today `AssertionKind.decision` is only user-authored). Enables decision provenance and "what changed my mind". Derived: mine assistant decision markers as candidates (recursive-safety-gated), link to prior decisions. index.db candidates + user.db assertion + **relation table** for supersession — **extend AssertionKind**.

- **spend-episode (cost attribution)** — cost attributed to episode/project/topic, not just session; the "analytics one measure away" rollup. Derived: VIEW rolling `session_costs` over episode/project/topic membership. index.db; **VIEW** (depends on episode + project units) — **NEW/derived**.

## GPT-pro prompt stubs

- **[A]** "Given Polylogue's contentless-FTS + generated-column identity model and `actions`-as-VIEW precedent, design the boundary rule for when a derived ontology unit (episode, topic-cluster, entity-mention) should be a materialized index.db table vs a live VIEW. Produce a decision matrix keyed on: recursion depth, embedding-join cost, rebuild-plan cost, and query-unit exposure — and apply it to the 16 proposed units."

- **[DR]** "Survey how comparable systems (knowledge graphs, Roam/Obsidian backlinks, Datomic/EAV, git/Gerrit change-object models, OpenTelemetry span↔resource attribution) model (a) mention/reference edges, (b) cause→effect provenance edges, and (c) logical 'episode' clustering over event streams. Extract concrete schema patterns Polylogue can borrow for entity-mention, world-effect, and episode units without storing content N× (lineage-dedup constraint)."

- **[A]** "Design a construct-validity gate for *derived intent* units (goal, decision-object, phase segment) so mined candidates never masquerade as asserted facts — specify the candidate→verified state machine, the citation-anchor each candidate must carry, and how it plugs into user.db `assertions` + `context_policy_json` injection gating. Include the failure mode from the 2026-06-29 recovery-digest fabrication incident as a test case."

---

## [a458d6320d1588e12] Grounded in flake.nix, pyproject.toml, packaging/Containerfile, systemd/polylogued.service

Grounded in flake.nix, pyproject.toml, packaging/Containerfile, systemd/polylogued.service, docs/{getting-started,installation,cloud-agents}.md, the nix/{module,hm-module}.nix pair, and .github/workflows/{release,container,homebrew-bump,flakehub}.yml. Key finding: the full distribution stack is **already built** (PyPI Trusted Publishing via OIDC + Sigstore keyless + CycloneDX SBOM + cross-Python/OS installed-wheel smoke matrix; ghcr slim+distroless × amd64+arm64 with provenance; homebrew-bump; FlakeHub; NixOS + HM modules; three console scripts) — but **no release tag has ever fired**, version is frozen at 0.1.0, and both getting-started.md and installation.md hard-code "not documented as current until artifacts exist and smoke-tested." The dominant lane theme is therefore *release is a decision, not more code*, plus a genuinely absent cross-machine story.

- Cut the first real `vX.Y.Z` tag — release.yml (PyPI trusted-publish + sigstore + SBOM + smoke matrix), container.yml, flakehub.yml, homebrew-bump.yml are all wired and untriggered; the "no packaged install path" claim is now a stale doc, not a missing capability — polylogue-3tl.7
- Lead getting-started with `uvx polylogue` once PyPI publishes — pure-Python + prebuilt wheels makes ephemeral uvx the true one-command stranger install; the 30s demo already proved the uvx path — polylogue-3tl.2 / polylogue-3tl.7
- Auto-reconcile getting-started.md + installation.md at first publish — both files literally say "clone + nix develop only"; leaving that after PyPI/ghcr go live is a self-inflicted legibility regression; gate it in the release checklist — polylogue-3tl.13 / polylogue-3tl.7
- Document a cross-machine model: ingest locally on each host, sync only the durable tiers (source.db + user.db), rebuild index/embeddings on the peer — the archive is single-writer local yet laptop+desktop both emit Claude/Codex sessions; derived tiers are throwaway so they must not be synced — NEW
- Ship a merge/conflict boundary for naive rsync sync — bidirectional dir sync corrupts index/ops WAL and races user.db; codify that assertions are append-keyed (mergeable), derived tiers are regenerable (never copy), source.db is content-hash-idempotent (safe union) — NEW
- Add `polylogued install-service [--user]` mirroring the `hooks install` pattern — systemd/polylogued.service assumes a `~/.local/bin/polylogued` (pipx/uv-tool) layout, so uvx/PyPI users get no unit; the daemon is the product and must self-install its unit — NEW (reuse polylogue-d1y idempotent-merge design)
- Document the pipx / `uv tool install polylogue` persistent lane — the "I have Python, not Nix" non-NixOS Linux path for a durable CLI+daemon (vs ephemeral uvx); this is where the installed systemd unit lives — polylogue-3tl.7
- Make the container's session-dir mount explicit — Containerfile is daemon-first with archive/config volumes, but `polylogued run` auto-discovers `~/.claude/projects` / `~/.codex/sessions` that aren't in the image; without a documented read-only host-path mount the image only ingests pre-mounted exports — NEW
- Consolidate the `_build_info.py` generator — Nix injects it via `postPatch`, hatch_build.py injects it via a build hook: two codepaths for one file with subtly different dirty-flag semantics; unify or add a parity test so Nix vs wheel builds report identical provenance — NEW
- Strengthen installed-smoke beyond `--help` — release.yml's smoke matrix and Nix `checks.default` only run `--help`; a wheel can import+help yet break on first real parse through the Rust-backed deps; add a synthetic-fixture import + `find` + `read` to the smoke — NEW
- Pin a supported-wheel matrix and fail release on sdist fallback — sqlite-vec, watchfiles, nh3, orjson are the load-bearing platform wheels; "pure Python, prebuilt wheels" silently breaks on any target lacking a wheel (musl/alpine, older arm-mac pyvers); assert no source compile in the smoke — NEW
- Publish the normalized session model as a versioned interchange schema — cross-machine moves and third-party readers need a stable export contract decoupled from the internal 5-tier DDL; makes "take my archive elsewhere" first-class — polylogue-3tl.6
- Datasette (or published read-only exhibit) as the zero-install read path — lowest-friction "run it" for a stranger who installs nothing, and doubles as the cross-machine read-only viewer of a synced durable snapshot — polylogue-45i
- Foreground `nix run github:Sinity/polylogue` as the ready-today Nix-stranger install — app outputs + flakehub publish make this work now, yet docs bury it under clone+dev-shell; for NixOS/darwin users this is the real one-liner — polylogue-3tl.7
- Turn release into one green button: run `devtools release verify-distribution` against the tag as the sole gate — the command already exists (release.yml uses it); wire it + the launch-kit artifacts so "release is a decision" is a checklist, not a manual dance — polylogue-3tl.10 / polylogue-3tl.7
- Verify homebrew-bump points at a live tap before advertising macOS — the bump workflow exists but a formula only resolves post first GitHub-release/PyPI; don't list the Homebrew channel until the tap round-trips — polylogue-3tl.7

GPT-pro prompt stubs:

- [DR] "Best practices for distributing a pure-Python CLI+daemon application in 2026: compare uvx/`uv tool install`, pipx, PyPI wheels, Nix flake apps, Homebrew, and OCI images for a single-writer local-data tool. Cover PyPI Trusted Publishing (OIDC) vs API tokens, Sigstore keyless signing + attestations, CycloneDX/SPDX SBOM expectations, platform-wheel coverage for Rust-backed deps (sqlite-vec/watchfiles/orjson/nh3) including musl, and what a modern installed-smoke matrix should assert beyond `--help`. Give a recommended channel priority for a solo maintainer."
- [DR] "Design patterns for cross-machine use of a local-first, single-writer SQLite application where the store is split into durable vs rebuildable tiers. Survey how tools (Datasette, litestream, dqlite, syncthing-over-SQLite, Obsidian-style vaults) handle multi-host divergence; what's safe to rsync vs what needs an append-mergeable log; and how to expose a versioned interchange schema so a second machine can rebuild derived data. Give a concrete sync-boundary recommendation."
- [A] "Given a Python project whose PyPI/Sigstore/SBOM/container/Homebrew/FlakeHub release automation is fully implemented but has never been triggered (version frozen at 0.1.0, docs say 'no packaged install path'), draft the minimal release-cutover checklist and the doc-reconciliation diff so the first tag also flips getting-started/installation to the shipped channels in one atomic decision."