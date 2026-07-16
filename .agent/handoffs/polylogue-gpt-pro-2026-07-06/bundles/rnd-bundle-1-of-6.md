# Polylogue R&D bundle — file 1 of 6

> Preamble below is the MASTER SYNTHESIS, identical across all 6 files.
> The unique payload of THIS file is the 14 subagent outputs after the PAYLOAD divider.
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
  PART 2 - PAYLOAD: 14 subagent outputs (unique to file 1 of 6)
=====================================================================


---

## [brainstorm] rnd-brainstorm-2026-07-05.md (~170-idea corpus + 66 GPT-pro prompts)

---
created: "2026-07-05"
purpose: "Parallel R&D brainstorm (Opus ultrathink + 8 grounded zone-agents) — ideas + GPT-pro prompt fan-out for the polylogue program. Scope deliberately orthogonal to the swarm2 daemon/composer/DSL/cost/lineage set."
status: "active"
project: "polylogue"
---

# Polylogue R&D brainstorm — 2026-07-05

Inputs: my ultrathink pass (68 ideas, 8 zones) + 8 background zone-agents grounded in
live beads/code/raw-log. Deduped below. Scope avoids what swarm2 already owns
(hot-daemon/UDS/thin-client/lifecycle, CLI-thinning/PolylogueService contract,
live-preview composer, query-DSL grammar/set-algebra/projection, cost-token math,
lineage dedup, attachment bytes, scale/WAL, bead-set meta-audit).

Reintegration note (operator-corrected): GPT-pro webui chats are **auto-captured into
the archive by the browser extension** (extension → receiver :8765 → spool → daemon →
archive). There is **no manual export step**. The flywheel is: lynchpin bundle → fan
out chats → they self-capture → distill into beads → coding agents execute. The only
manual work is the *analytical* half (link captured R&D sessions to lanes; distill).

---

## Convergent themes (hit independently by ≥2 agents — highest signal)

1. **Queries / findings as first-class stored objects.** self-ref: `query:<id>` ObjectRef,
   content-hashed result-set snapshots, query→query dependency edges, standing-queries as
   change-detectors. product: `finding` as an `AssertionKind` (claim + producing-query +
   provenance + URL, `find kind:finding`), findings-as-tests manifest. UX: content-hash
   citation anchors as the atom everything cites. → **the convergent frontier.**
2. **Construct-validity as a queryable substrate, not a markdown header.** trust:
   `finding_provenance` table + sample-frame drift detector (re-run each finding's COUNT on
   every converger pass, auto-stamp "population 412→1,088, re-run before citing"). product:
   findings-as-tests (CI re-runs the number ± tolerance). content: summary must carry
   citation edges (coverage>0) or be rejected at write. Brand line: **"every number resolves
   to bytes / no regex over prose."**
3. **Content-hash citation anchors that survive re-ingest.** UX + ecosystem (CIF) both landed
   on stable `session:message:block` anchors (not `position`) as the load-bearing atom — the
   honest fix for the fork-position-shift + recovery-digest-fabrication class.
4. **The "so what" analytics is mostly one materializer away.** The keystone substrate already
   exists (`tool_duration_ms`, `tool_result_is_error`/`exit_code` v16, `actions` view,
   `workflow_shape`, `terminal_state`, `session_links`). Gap-taxonomy, retry-storm,
   cost-per-outcome, delegation-yield are new *measures*, not new plumbing.
5. **Recursive-safety gate.** self-ref + trust: agent-authored assertions default to
   CANDIDATE + `inject:false` until operator-promoted. Load-bearing now that the archive
   auto-captures and feeds itself — prevents the archive laundering its own hallucinations
   back into context (the recovery-report fabrication class, structurally).
6. **Corpus-compaction pack (`find <q> | compact`).** analytics: token-budgeted, decision-dense
   digest (drop tool-spam via `material_origin`, keep human turns + decisions + error→fix +
   outcomes, dedup replayed prefixes, fit a context window, emit a drop-manifest). This is the
   flywheel enabler — it packages the archive's own sessions for the next GPT-pro lane.

---

## Ideas by zone (deduped; NEW = no bead yet; else epic/bead anchor)

### A. Product identity / positioning / legibility (3tl, 212, 9e5)
- Hero-finding discipline: ONE robust+surprising+clickable number as the spearhead (README hero
  + launch post + lead demo all point at it), not a finding *catalog* with no front door. — NEW
- `finding` as first-class AssertionKind; findings-as-tests provenance manifest (stale finding
  fails the devloop, not the reader). — NEW / sibling of 3tl.9
- Head-to-head "lie detector" demo: same question to naive grep/log-tailer (fabricates from
  prose) vs polylogue (answers from structure or refuses). — 212
- Interactive "click-the-number" Artifact: click a headline stat → decompose number→query→
  structural field→raw bytes. The moat as a shareable link, 10-second cold-reader proof. — NEW
- Claims-ledger dogfood: every self-marketing claim is itself a `finding` with evidence status
  (proven/capability/aspirational), published; radical-honesty as brand. — NEW
- Category-defense bake-off vs LangSmith/Langfuse/Helicone (live/single-provider/online) and
  Rewind/QS (surveillance) — offline, cross-provider, reconstructed-from-exports. — 3tl
- [RADICAL] Flight-recorder / black-box positioning: the forensic recorder you consult after a
  crash (post-hoc, structural, "what did the agent actually do"). No competitor owns it. — NEW
- [RADICAL] Public honesty benchmark: claim-vs-evidence as a NAMED public eval + leaderboard
  others run their models against; personal archive → public science instrument. — NEW
- [RADICAL] Prompt/meta-workflow distillery: "your history is training data for how you should
  work" — mine history into better parametrized prompts (operator's stated dream). — NEW
- `polylogue tour` / `explain-project`: the CLI narrates its own category + runs the hero demo
  for a cold reader. — 3tl
- Self-hosting proof badge: polylogue's own repo as reference deployment, live "N of our
  decisions are citable" counter. — NEW
- Weekly auto "state of my AI work" report — recurring proof artifact (dogfood + marketing). — 9l5/212
- Ingest Simon Willison's `llm` SQLite logs as a source — manufactures its own citation path
  into the Datasette/`llm` community. — l4kf
- Honesty demo: showcase what polylogue *refuses* to claim (unfetched attachments, stale rows). — 9e5/83u

### B. Deep analytics / "so what" (9l5) — most are one measure away
- Gap-taxonomy materializer: partition each session's wallclock into typed spans (edit/compile/
  test/rebuild/model-thinking/human-idle) from inter-event gaps + tool identity + command regex.
  The literal raw-log-1841 ask. — 9l5.8
- Long-call time-sink ranking: tool calls whose wall-duration > N min, labeled by command class
  (full-test/nix-switch/rebuild), per session, rolled up per repo. — NEW
- Retry-storm / guess-and-check detector: edit→same-target-test→edit cycles where
  `tool_result_is_error` stays true and error signature doesn't change = churn. — 9l5.9
- Agent-efficiency score with peer baseline: `tool_active_minutes/wallclock` z-scored against a
  per-repo, per-`workflow_shape` cohort (judge like-work, not global). — 9l5.2/.7
- Tool-error recovery latency: first-error → next-success on same tool/target, per model. — NEW
- Context-gluttony index: whole-file `Read` (no offset/limit) + bytes ingested vs targeted
  `Grep`/`Glob` (raw-log-2001 "skillful grep"). — NEW
- Cost-per-outcome join: `terminal_state` × `session_costs` → "$ per *successful* session",
  "spend burned on abandoned+stuck". The sharpest so-what. — 9l5.1
- Delegation yield (subagent ROI): child cost/tokens/wallclock vs parent terminal success. — NEW
- Rework half-life / churn survival: time until a touched file is re-edited later; short = thrash. — 9l5.9
- Measure→prose narration (local-LLM): `analyze --narrate` passes only the numeric result to
  gemma/deepseek → grounded prose, never invents facts. — 9l5/mhx.5/1944
- Pathology epidemiology over time: prevalence per ISO-week × model + changepoint on upgrades. — 9l5.8
- Wasted-test-run detector: long test span that produced no caught error = wasted minutes
  (quantifies the blanket-run anti-pattern). — NEW
- Abandonment autopsy clustering: cluster stuck/abandoned by (last-tool, last-error-signature). — 9l5.1
- Context-switch / interruption tax: abandon-A→spawn-B→return-A patterns; re-orientation cost. — NEW
- Thinking-vs-doing drift: `thinking_duration_ms` vs `tool_duration_ms` trended per model — early
  "this model got worse for my work" signal. — 9l5.8
- Model head-to-head from real corpus: same task-shape across models — cost/turns/errors/success. — 9l5
- Per-session "so what" scorecard `reader_panel`: efficiency + gap-sparkline + retry count +
  cost-per-outcome + delegation-yield in one slot. — NEW

### C. Self-referential substrate (s7ae, 37t, rii) — the frontier
- Promote saved queries to `query:<id>` ObjectRef so any assertion can target/cite a query
  (a query accrues annotations/judgments/supersession). — 37t / NEW
- Materialized result-set snapshots in ops.db `(query_id, frontier_ids, result_hash, ran_at)` →
  `read query:abc --diff` ("3 in / 1 out since last run"); assertions attach to a frozen answer. — NEW
- Query→query dependency edges + invalidation (storage layer, not grammar). — NEW
- Standing queries as change-detectors: `context_policy_json {watch:true}` re-runs on quiet
  window; a delta writes a CANDIDATE assertion (archive's own alerting). — feeds rii
- Self-mine BLOCKER/LESSON/DECISION assertions → discovered beads (cluster same-target,
  `bd create` with evidence_refs). — 37t
- Recursive-safety gate: agent-authored assertions default CANDIDATE + `inject:false`. — 7aw (load-bearing)
- Ingest polylogue's own dev sessions into a `self` workspace (cwd-prefix == POLYLOGUE_ROOT). — NEW
- Read-access log as a queryable unit source: most re-read sessions, saved queries returning
  nothing, recall packs never opened (dead-memory detection); the signal the scheduler needs. — 37t
- Context scheduler: budget-aware inject selection ranked by staleness × attention ×
  topic-proximity (embedding of current session); decay by `staleness_json`. — 37t
- Programmable `context_policy_json` windows: `{inject, when:[start|on_topic|on_error], ttl_days,
  max_injections, cooldown}` — assertion as scheduled memory primitive. — NEW
- Annotation recipes: parameterized assertion templates (a postmortem recipe auto-attaches
  decision/blocker/lesson skeletons pre-filled from `session_phases`/`get_pathologies`). — NEW
- Belief-history read over `supersedes_json`: `belief_timeline(target_ref)` shows reversal history
  so an agent stops re-litigating settled reversals. — NEW
- Live recall packs bound to a `query:<id>` (recompute membership live, keep human narrative). — NEW
- Semantic dedup of saved queries via embeddings (avoid a 40th near-duplicate view). — NEW
- Reverse "annotation density" insight (session → assertions-about-it): most-annotated,
  unresolved-blockers, zero-human-touch. — rii
- Blackboard posts carry `query:<id>` refs → coordinator posts a query, workers run-and-report
  (notepad → task bus). — s7ae
- Synthetic "meta-session" of the archive's own runtime (ingest/convergence/read events as blocks)
  → every read-model applies to polylogue operating on itself, zero new insight code. — NEW

### D. Ecosystem / interop / breadth (l4kf, 7aw, fs1)
- Publish **CIF (Conversation Interchange Format)**: content-addressed transcripts with portable
  `session:message:block` anchors + per-source fidelity declaration; the neutral hub spec every
  export/import profile targets. — l4kf
- `polylogue-export` as a first-class re-ingestable Origin: round-trip reconstructs identical
  ids → free correctness invariant + federation primitive. — NEW
- `.well-known/ai-sessions` federation manifest + selective content-hash sync (local-first peer
  sharing, dedup via idempotency). — NEW
- Emit `git notes` (refs/notes/polylogue) attaching a session citation to the commit it produced
  (`git log --notes=polylogue`). — NEW (outbound counterpart to 7xv)
- `polylogue cite` → gh PR/issue provenance footer + reverse `session_link` (PR ↔ session). — NEW
- Export mined pathologies as **SARIF** → GitHub code-scanning surfaces agent-behavior defects. — NEW
- Signed session attestation `(content_hash, anchor, sig)` → tamper-evident citations. — NEW (pairs ale)
- MCP sessions/messages/blocks as **resources** (`resource://` + subscriptions), not only tools —
  the embeddable, live-updating continuity primitive. — NEW
- Recall-packs / saved-views as MCP **prompts** (slash-command surface) — zero-friction injection. — NEW
- `llms.txt`/`AGENTS.md` continuity artifact for MCP-less harnesses (Aider/Cursor/Zed read a file). — NEW
- Attach Sinex/ActivityWatch ground-truth (window-focus, Atuin, file-activity) to a session as
  observed-events by time window — transcript says what the agent did; this says what the human did. — NEW
- Cross-origin "same investigation" stitching (repo + time-window + embedding), edge=thematic not
  prefix-sharing: Cursor→Claude Code→ChatGPT as one logical unit. — NEW
- **Cursor parser** (chats in `state.vscdb` SQLite, binary keyed blobs — a SQLite-shaped acquire
  path, unlike every current file detector). Largest missing corpus. — NEW
- Continue.dev dev-data JSONL (purpose-built for export); Zed via ACP (source + live lane);
  OpenCode/Warp stores; Aider `.aider.chat.history.md` (first prose-shaped parser). — l4kf
- Datasette read-only publish of the archive (already SQLite; QS/Willison ecosystem for free). — NEW
- IDE Timeline/SCM provider: "which sessions touched this file" in the editor. — NEW
- Eval-corpus export: promptfoo test-cases + SWE-agent `.traj` downstream of CIF. — 7k7/fs1.5
- Wire grok-export (reserved origin, no parser). — 611/l4kf

### E. Content transform / multimodal / embeddings (4smp, mhx)
- `mechanical` vs `generative` variant provenance axis (OCR/transliteration trusted 100%; LLM
  translation/summary lossy). The trust axis is the point. — 4smp
- Variant staleness keyed to `source_content_hash` (mark stale/orphaned on source hash change,
  never auto-repaint). — 4smp
- Summary variants must carry citation edges (coverage>0) or be rejected at write. — 4smp
- Coverage-gap "dark matter" rendering: the source spans a summary did NOT cover ("what got
  dropped"). — 4smp
- Faithfulness score via embedding drift (cosine between summary node and its cited source span). — mhx+4smp
- Per-block language detection (not per-session) driving translation defaults; exclude code/tool
  blocks by construction. — 0v9p
- Glossary-pinned translation: deterministic pre/post substitution of domain terms, span-preserving. — 4smp
- Image blocks → caption variant (searchable) + separate OCR variant with bbox alignment
  (mechanical) — two variant kinds over one image, never conflated. — 4smp+mhx
- Audio transcript as a first-class origin (per-block audio offsets, "jump to audio at this line")
  — the operator's oldest itch. — NEW
- Embedding eval harness mining ground truth from lineage: resume/fork pairs = semantically-near,
  few shared tokens → **free labeled positives** for recall@k (FTS vs vector vs hybrid). — mhx
- Multi-provider embedding shootout on a frozen probe set (Voyage vs local nomic/bge/gemma);
  second vec0 space per provider with hard provenance. — mhx
- Reranker as a measured post-retrieval stage (bge-reranker top-50→top-10), promoted only if it
  lifts precision@10 on the probe set. — mhx
- Tool-result-aware chunking: embed a distilled head+tail+error variant, keep raw block as evidence. — mhx+4smp
- Retrieval unit = `action` (embed call-intent + outcome jointly via the actions view). — mhx
- Semantic diff across a fork's divergent tail as topic-drift vector ("did the fork abandon the
  goal?"). — NEW
- Hierarchical coarse→fine retrieval (session-summary embeddings → action/message embeddings). — mhx
- `variant`/`coverage` as queryable DSL units (`sessions where variant:translation and coverage<0.8`). — 4smp

### F. Trust / verifiability / privacy (cpf, kwsb, 9e5)
- `finding_provenance` as a queryable `index.db` table (cursor-id/position, measure+version, git
  SHA, sample-frame predicate, run-date) + sample-frame drift auto-flag. — cpf/3tl.4
- `trust_class ∈ {OPERATOR, SYSTEM, QUOTED}` on every context-injectable row; the compiler cannot
  concatenate QUOTED (attacker-controllable) into an OPERATOR block without a visible fence + ref. — cpf/37t.11
- Injection-tripwire fixture: known injection strings must round-trip through `compile_context`/
  `compose_context_preamble` as fenced QUOTED or the test fails. — cpf
- Confidence + evidence-ref REQUIRED (NOT NULL) at assertion write. — cpf/9e5.1
- Secret-scan at the READ boundary (fold into read, per operator): emit spans as `secret_candidate`
  overlays with a reveal affordance; keep bytes durable, never re-surface a pasted key by accident. — kwsb/27m
- Secret candidates → judgment queue as assertions (never auto-redact); acceptance → excision,
  rejection → suppress pattern → labeled corpus tunes the detector. — 27m
- Verifiable excision with content-hash tombstone (records old content_hash, recomputes new;
  idempotency can't resurrect the removed span; auditor can prove exactly-this-hash-removed). — 27m
- One shared mutation-audit contract across excise/reset/delete/MCP-admin (actor, ref,
  dry-run-diff hash, reason, --yes provenance); fixes the reset-before-preview bug class. — kwsb/jnj.5
- Forget-audit reconciliation: converger invariant re-scans index/FTS/embeddings/blob for the
  removed hash and fails loudly on reappearance. — kwsb/27m
- Degradation-rung in every read payload envelope (full/partial-index/embeddings-stale/
  evidence-only); EVIDENCE-ONLY floor is sacred (raw reads always answer). — cpf
- Claim-vs-evidence as a standing property test over insight OUTPUT: no rendered insight string
  carries a quantitative claim whose backing rows are empty/NULL. — 9e5
- "Unverified candidate" as a first-class render mode distinct from "fact" (codifies the text-mined
  recovery-report fix into the type system). — 9e5
- Semantic-version-bump invalidation cascade for findings/assertions (old-version findings → stale
  + diff banner; append-only supersession). — cpf/3tl.4
- Adversarial self-audit shipped as `polylogue audit --adversarial <insight>` (generates the
  strongest counter-frame as `caveat` assertions before you cite). — 9e5
- Injected-context provenance receipt: `build_context_image` attaches a manifest of what it
  contains + why (source refs, trust classes, freshness per component). — cpf/37t.11
- `lossy_grouping` marker on the origin projection (GEMINI+DRIVE→AISTUDIO_DRIVE is non-injective). — 9e5

### G. UX beyond the composer (bby, jnj)
- Citable-report builder ("evidence basket"): accrete selected blocks into a live Markdown doc
  with provenance footnotes, exportable as the deliverable — the missing "report" end of bby. — NEW
- Content-hash citation anchors (`cite this block`, keyed on hash not position) + citation-integrity
  verifier (re-resolve on export, flag drifted/deleted/quarantined). — NEW
- Real force-directed lineage/topology graph (replace BFS text list): edge-kind coloring, quarantined
  cycle-break nodes flagged, branch-point highlighted, click-to-refocus. — complements bby.5
- Assertion/evidence overlay graph + marginalia layer in the reader (marks/corrections as sticky
  margin notes) — the durable user.db tier is currently invisible at read time. — NEW
- Year-heatmap calendar (contribution-graph) → click into day-page (bby.13); the navigable temporal
  index. — bby
- Unified temporal scrubber: one zoom spine over year-heatmap → firehose (bby.10) → replay (bby.12). — meta over bby
- `material_origin` texture gutter (minimap strip colored by authoredness) — reveals a 4k-msg
  session's human/assistant/tool texture. — extends bby.5
- Delegation call-tree render layout (nested collapsible cards, not flat inline splice; after #2545). — NEW
- Specimen-gallery browse mode (grid of session-glyph cards for visual triage of 200 sessions). — NEW
- Side-by-side fork/parent diff pane (surface `compare_sessions`, highlight only divergent tail). — NEW
- Ambient theming from pywal/`prefers-color-scheme` on the web reader (extend `il` palette to web). — jnj
- Progressive-enhancement / server-rendered read routes (readable on phone / `curl | pandoc`);
  bake into arch-v2 (bby.11) as a constraint. — feeds bby.11
- Obsidian-vault read-view profile (wikilink-cross-referenced Markdown, one file per session,
  lineage as `[[…]]` backlinks; reuses canonical-URL projection). — NEW/jnj
- Cmd-K command palette (kill the last `window.prompt()`, bby.6). — extends bby.6
- Voice recall loop (push-to-talk → local Whisper → DSL; TTS read-back of resume-briefs). — NEW
- "Now playing" ambient strip (reader header + optional Waybar module live-tailing daemon SSE). — rides bby.4

### H. Radical reframes / moonshots
- Polylogue as a local RL/eval environment: sessions = labeled trajectories (task, tool-calls,
  outcome, cost, corrections in user.db); `polylogue eval export --format atropos`. — fs1
- Verifiable-reward RL tasks auto-mined from CI-passing sessions (`tool_result_exit_code=0` → the
  verify command IS the reward). — eval-env
- Reward model from your corrections (CORRECTION assertions = human-preference dataset over agent
  behavior; train a tiny local scorer "would Sinity flag this turn"). — eval-env
- Session replay/reproduction: re-execute recorded tool-calls against a fresh checkout at the
  session's git SHA → determinism, "worked-then-breaks-now" regressions, RL rewards. First step: a
  dry replayer printing the executable plan (no execution). — replay
- Counterfactual re-run ("what if model X"): freeze human turns + initial repo state, re-drive with
  a different model, diff cost/turns/outcome. — replay
- Spec-cards per session (initial SHA, human intent, acceptance signal, final diff) → any session
  becomes a portable benchmark item without transcript leakage. — eval-env
- Trajectory quality index (latency + correction density + abandonment + tool-error + phase thrash
  → one 0–1 score) as RL reward-shaping AND personal dashboard. — eval-env/observatory
- Model-drift observatory: same task-shape across (model, month) → cost/turns/error drift. — observatory
- Pathology-driven auto-guardrails: recurring failures → injected `caveat` assertions ("last 3× you
  edited convergence_stages.py you broke X"). — resident (links #1498 cascade)
- Standing "resident intellect" agent mining the archive nightly (context = polylogue MCP +
  assertions; drops ≤5 findings/day via save_annotation). — resident
- Self-improving CLAUDE.md/skill synthesis from repeated corrections (weekly digest drafts a
  candidate memory snippet; operator reviews, never auto-commits). — resident
- Cross-project agent memory: `recall(task_hint)` MCP tool returning most-similar prior sessions +
  their corrections/lessons across ALL repos. — cross-project-memory
- Personal-model distillation substrate ("golden turns" subset as SFT/preference data). — personal-model
- Privacy-preserving cross-archive federation: exchange only derived statistics/embeddings/
  pathology-rates, never content. — federation
- Analysis-recipe library (parameterized DSL pipelines you name/share/run: `run-recipe <name>`). — recipes
- The umbrella: polylogue as substrate for a personal AI that knows your whole collaboration
  history (curated "resident" MCP profile bundling recall + assertions + lynchpin). — resident

---

## GPT-pro prompt fan-out (deduped, final)

Prepend the **standing depth contract** to every prompt:

> You have the forked context: the polylogue bundle + our swarm2 daemon/composer/thin-client
> designs + our discussion of queries/findings-as-objects, delegation units, annotation recipes,
> and browser-auto-capture. Ground every claim in the bundle (cite file:line / bead id); tag
> [evidence] vs [proposal]. Be opinionated — one defended recommendation + the runner-up you
> rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate
> change, (3) the full direction. End with **Open questions for the operator** and **What's
> missing** (a claim unverified, a source unread, a modality not run). Snapshot is dated; live code
> may have moved — flag assumptions to verify locally. Don't re-derive swarm2.

### Design [A] (analysis over the bundle)
A1. **"So what" analytics + gap-taxonomy materializer.** Design `session_gap_profile` partitioning
    each session's wallclock into typed spans (edit/compile/test/rebuild/model-thinking/human-idle)
    from inter-event gaps + tool identity + command classification; the gap decision table, the
    STRICT schema, registration as a measure (grouping×window×uncertainty) with construct-validity
    metadata, and a compact sparkline. Then the measure family it seeds (retry-storm, recovery
    latency, gluttony, cost-per-outcome, delegation-yield).
A2. **Queries+findings as first-class objects (the convergent frontier).** Storage+ref model:
    `query:<id>` and `finding` ObjectRef/AssertionKinds; content-hashed result-set snapshots (which
    tier?); query→query dependency edges + invalidation; standing-query change-detection emitting
    CANDIDATE assertions; findings-as-tests manifest (CI re-runs the number ± tolerance). DDL
    placement across the 5 tiers, invariants, read/write API — without touching the DSL grammar.
    Enumerate the recursive-loop failure modes.
A3. **Context scheduler + memory loop + recursive-safety gate.** Given a token budget, a topic
    embedding, a read-access log, and assertions with `context_policy_json`/`staleness_json`/
    `confidence`/`author_kind`: the ranking function, the programmable-window schema (when/ttl/
    max_injections/cooldown), and the gate that keeps agent-authored assertions out of context until
    operator promotion. Prove it prevents self-hallucination feedback.
A4. **Delegation graph as substrate.** The `delegation` unit + extraction from Claude-Code/Codex
    subagent structure + artifact/scratchpad linking + delegation-card projection + analytics DSL
    (`delegations where … | group by subagent_model_family | count`). Generalize the Fable demo.
A5. **Session replay/reproduction critique.** Re-run recorded tool-calls against the repo at the
    recorded SHA for determinism / RL rewards / model A-B. Enumerate the correctness/safety/
    nondeterminism problems (side effects, network, time, FS state), which session types are safely
    replayable, and the smallest trustworthy first slice (dry replayer).
A6. **Content-variant model + honesty invariants.** Typed variants over refs (translation/
    transliteration/simplification/summary/caption/OCR) with alignment edges (incl. partial),
    a `mechanical` vs `generative` provenance axis, `source_content_hash` staleness, coverage>0
    write invariant, and query surfaces that guarantee variant text is never returned as evidence.
A7. **Construct-validity as a live surface.** `finding_provenance` table + sample-frame drift
    detector + `trust_class` (OPERATOR/SYSTEM/QUOTED) on injected context + the injection-tripwire
    fixture. Make honesty a query + a regression gate, not a doc convention.
A8. **Redaction/forget done right.** Fold secret-scan into `read` (spans as `secret_candidate`
    overlays → judgment queue), verifiable excision with hash tombstone, one shared mutation-audit
    contract, and a converger reconciliation that proves no derived tier retained the removed hash.
A9. **Webui evidence cockpit: anchors → report.** Citation-anchor scheme (content-hash, survives
    re-ingest + fork-position shift), evidence-basket→citable-report data flow, and citation-
    integrity verifier (drift/quarantine handling). Anchor format, resolution algorithm, minimal API.
A10. **Unified temporal-navigation instrument.** Year-density heatmap + scrubbable firehose +
    per-day narrative + per-session replay under one zoom model; the zoom state machine, aggregation
    tiers (year→week→day→session→message), data contracts per level, and forks as timeline branching.
A11. **Corpus-compaction pack (`find <q> | compact`).** Token-budgeted, decision-dense digest for
    feeding an external LLM: block-selection scoring (drop tool-spam via material_origin, keep human
    turns + decisions + error→fix + outcomes), replayed-prefix dedup, per-session boundary markers,
    and a fidelity drop-manifest. This is the R&D-flywheel enabler.
A12. **Own-export as a first-class source + CIF round-trip + federation.** The case for/against making
    the archive's own export re-ingestable (round-trip id identity as a correctness invariant +
    federation primitive); where non-injective origin collapse or lossy fidelity declarations break
    it; manifest-pull vs feed-push federation.
A13. **Product spearhead + positioning.** Rank candidate hero-findings (authored-user-word share,
    replay-duplication %, subscription-vs-API cost gap, abandonment mortality, Codex reasoning
    inflation) on {robustness, surprise, clickability, resolves-to-bytes}; output the single
    spearhead + headline + reproduce command + 2 backups, plus the flight-recorder/honesty-benchmark
    framing vs named competitors.
A14. **Prompt/meta-workflow distillery.** From my highest-value past sessions, induce 5–8 general
    *parametrized* meta-prompts (params: repo, task-type, risk-tier) that would have beaten what I
    actually typed; for each, the pattern, the sessions it's distilled from, and a falsifiable A/B test.
A15. **The R&D flywheel (corrected).** Design the fan-out + reintegration loop given that browser-auto-
    capture already ingests GPT-pro chats: lane clustering, making each lane's captured sessions a
    retrievable cohort (title stub / tag), the recursive-safety gate on distilled assertions, and the
    distill→bead step. No manual ingest.

### Deepresearch [DR] (web)
D1. **Competitive/landscape scan + positioning.** Tools that archive/analyze AI-agent sessions
    (exporters, LLM-observability: LangSmith/Langfuse/Helicone, agent-trace, eval harnesses,
    Rewind/QS). White space for "system of record for AI work"; assess the flight-recorder + public
    honesty-benchmark framings and what naming lands in the Latent-Space/Karpathy discourse.
D2. **Local-LLM + embedding + reranker stack on a consumer box (13700K + ~10GB GPU).** Models for
    classify/summarize/narrate a 40GB personal corpus (gemma/qwen/deepseek/flash-lite) and embeddings/
    rerankers (nomic/bge-m3/gte vs Voyage baseline): decision matrix over quality (MTEB + code),
    footprint, throughput, quantization, licensing. + a self-labeled recall@k eval from fork/resume pairs.
D3. **Embedding & retrieval for long tool-heavy transcripts.** SOTA chunking, dense/hybrid/late-
    interaction, and eval for very long agent sessions (not chat-sized). Does vector beat BM25 here,
    and how do you prove it locally (mhx.3)? How methods degrade on code + tool-output vs prose.
D4. **AI-session harness + interchange-format landscape.** Where Cursor/Zed/Continue/Aider/OpenCode/
    Cline/Windsurf/Amp/Warp store sessions + schema shape + open-vs-reverse-engineered + stability;
    and the format standards (OTel GenAI, Zed ACP, llms.txt/AGENTS.md, promptfoo/inspect/SWE `.traj`).
    Which ONE interchange standard should a neutral local archive champion vs merely adapt to.
D5. **Portable content-addressed citation-anchor standard.** Prior art (W3C Web Annotation selectors,
    git notes, DOI/PID, IPFS/Nostr content addressing, Gwern link archival); what makes an anchor
    durable, tamper-evident, re-resolvable across independent archives; failure modes (re-parse drift,
    schema change, redaction).
D6. **Cost/pricing & subscription-credit models.** Current API vs subscription-credit accounting
    across Anthropic/OpenAI/Google; honest dual-view presentation; pricing-catalog sourcing (f2qv).
D7. **RL/eval-environment landscape + trajectory formats.** Atropos/verifiers/NeMo ATOF-ATIF/OpenAI
    evals/terminal-bench/SWE-bench: input schema, reward model, whether human-corrections are
    first-class, and how a local ~10k-session corpus (tool-calls, exit codes, corrections, SHAs)
    lowers in with minimal fabrication. The single most leverageable export target + exact JSONL shape.
D8. **Personal-model training/distillation from own history.** Corpus size/quality for useful SFT vs
    preference-tuning (DPO/reward-model) of a local model; data-quality gating; federation techniques
    for comparing behavioral statistics without sharing transcripts. A minimal pipeline over 40GB.
D9. **Agent memory & context-engineering prior art.** How leading agent frameworks manage long-term
    memory, context assembly, provenance — what the context scheduler (37t) should steal.
D10. **Construct-validity as a product feature.** dbt tests / Great Expectations / metric-store
    semantic layers / postmortem tooling: how best-in-class systems expose sample-frame, denominator
    provenance, and staleness so a number can't be silently miscited. Concrete design for auto-flagging
    findings whose population predicate drifted.
D11. **Verifiable deletion in append-only content-addressed stores.** Tombstoning, hash-of-removed,
    crypto-shredding, Merkle-consistency proofs; interaction with content-hash idempotency + derived-
    index rebuilds; what an auditor can prove without retaining bytes; resurrection failure modes.
D12. **Summary-faithfulness/attribution scoring.** Embedding-drift cosine vs NLI entailment vs
    citation-coverage; SOTA for detecting hallucinated/unsupported summary sentences cheaply + locally;
    degradation on code + tool-output.
D13. **Persisted queries as first-class objects — prior art.** Datalog/materialized-view engines,
    dbt lineage, Datasette/Steampipe saved queries, graph views, event-sourced read models: patterns
    for query→query invalidation, result-set snapshotting, standing-query alerting on single-writer SQLite.
D14. **Big-graph in-browser + progressive-enhancement + ambient theming.** Force-directed vs
    hierarchical node-link rendering at ~16k nodes; SPA-plus-hydration usable on mobile/curl; pywal/
    prefers-color-scheme piped into a locally-served app. Techniques, ceilings, failure modes.

Run split: fork the current GPT-pro session for continuation lanes (A2, A3, A4, A5, A11, A15, D9);
fresh chats + bundle for the rest. 14 [A] + 14 [DR] = 28 chats — well inside the quota budget.

---
---

# WAVE 2 — 14 agents (3 deepen the convergent frontier, 11 open new dimensions)

## New convergent themes (wave-2 — hit across lanes)

7. **The measure algebra is now explicit.** Analytics = a 5-tuple `⟨reducer(column) over unit-frame × grouping × window × comparison × uncertainty⟩` — the catalog is a *cartesian product*, not 16 hand-written `analyze` modes. Highest-leverage first: cache-amplification ratio, latency three-lane share, engaged-vs-wall efficiency, tool-mix entropy, credit-vs-API divergence. Every "cross-model" / "longitudinal" variant is a grouping/window swap on a base measure, never new code.
8. **Queries+findings-as-objects got a full spec.** `query:<hash>` ObjectRef (hash over the planned AST post-macro-expansion, mirroring content-hash idempotency); `AssertionKind.FINDING` reusing the pathology candidate→judge lifecycle verbatim; `result_sets` snapshots in the *derived* tier (droppable by `reset --index`); `query_edges` DAG persisted from the set-algebra EXPLAIN nodes; a `StandingQueryStage` in the converger; **findings-as-tests** (a promoted finding with `value.expected` becomes a re-runnable invariant). Multiple other lanes (search self-testing saved-queries, proactive standing-queries, attention frontier-as-query) independently assume this.
9. **Recursive-safety is a coherent subsystem, not a flag.** Across self-capture / proactive / attention / queries-deep: agent-authored assertions default CANDIDATE+inject:false; a **closed-loop laundering gate** (a claim whose evidence resolves *only* to other agent-authored sessions is quarantined until an external-grounded citation or human judgment breaks the cycle); `author_kind` differential trust in the inject gate; provenance-cycle quarantine reusing `TopologyEdgeStatus`; never alert/surface on `generated_context_pack` material. This is what makes auto-capture-of-own-R&D safe.
10. **"Wire what already exists" beats "build new."** proactive found a full 5-backend notification fan-out (only carrying ops alerts); search found `score_components` already carries per-lane RRF decomposition; incident found postmortem-bundle machinery; economics found lease/GC + retention already present. Most wave-2 wins are *routing an existing pipe to a new signal*.
11. **The episode is the missing keystone unit.** episode + missing-units + self-capture converge: a logical task spanning sessions/tools/time, derived above `session_links`, stitched by a 4-signal scorer with a hard false-merge floor, the right grain for cost-per-outcome and "how I actually solved X."
12. **Construct-validity became executable everywhere.** measure-catalog's coverage-precondition registry; meta-quality's "insight claiming a number over zero rows must FAIL" CI gate; ingestion's byte-fidelity ratio; temporal's `time_confidence` field. The brand ("every number resolves to bytes") is now a test lane, a column, and a gate.

## Ideas by lane (curated strongest ~5 each)

### W2-Q — Queries/findings-as-objects (deep)
- `query:<hash>` keyed on planned AST post-macro-expansion (dedup, cache-key, "have I run this"). — NEW
- `result_sets` snapshots in index.db (derived), Merkle root over sorted keys for O(1) change-detect; only a promoted finding referencing one is durable. — NEW
- `AssertionKind.FINDING` reuses pathology candidate→judge lifecycle + inject-gating verbatim (zero new lifecycle code). — NEW
- Findings-as-tests: promoted finding with `value.expected` → a `ConvergenceStage` re-runs it, emits `finding-drift` candidate on divergence. — NEW
- Content-anchor evidence as `session:X@<content_hash>` so a re-materialize mismatch flags possibly-stale instead of lying; provenance-cycle quarantine via `TopologyEdgeStatus`. — NEW

### W2-M — Measure catalog (deep)
- The 5-tuple algebra as the registry contract; catalog = product not list. — 9l5.7
- 16 measures spec'd (cache-amplification, thinking-tax, latency three-lane, interaction-latency asymmetry, engaged-vs-wall, stuck-tool density, compaction pressure, substantive density, outcome-conditioned cost, tool-mix entropy, workflow-shape transition matrix, session-redundancy zstd ratio, thread-cost Gini, subagent fan-out, pathology epidemiology, credit-vs-API divergence). — 9l5.x
- Uncertainty layer: Wilson for proportions, bootstrap CI for median/pXX, coverage-tier gate; one-line honest footnote (`n=…, coverage=evidence, timing_provenance=structural`) from one registry declaration. — 9l5.7
- Denominator/unit-frame is the #1 construct-validity trap — the registry must enforce it. — 9l5.7

### W2-S — Self-capturing recursive loop (deep)
- Synthetic runtime meta-session (convergence/ingest/debt events as a session tree) → every insight applies to polylogue on itself. — NEW
- R&D cohort auto-minted per agent wave (session-tree ∪ repo ∪ time-window), judged/diffed/cited as a unit. — NEW
- Closed-loop laundering gate (THE invariant); hallucination-provenance cohort down-weights sessions that historically fabricated. — NEW
- Design-chat→bead→PR provenance chain (resolve_ref on a bead walks back to the originating GPT-pro design chat). — NEW
- Loop-latency as a native measure (design-chat ingest → bead → PR merged → postmortem); cohort drift/novelty score vs prior waves. — NEW

### W2-Act — Activation & adoption
- `polylogue install` one-command idempotent hook wiring (all 16 CC + 6 Codex events) + `polylogue doctor` hook-liveness heartbeat. — d1y
- Adoption measured FROM the archive: count `mcp__polylogue__*` tool_use per session/repo → adoption-rate insight; "why isn't this used" diagnosis for configured-but-zero-usage repos. — NEW
- `what_now(intent)` router + primary-tool tiering (progressive disclosure over 130 tools). — pj8
- Assertions-as-injected-continuity (assertions > CLAUDE.md), token-budgeted ranked preamble; adoption canary as a dogfood CI gate. — 3gd
- A/B activation experiment rail (config versioned + adoption measured → empirical, not faith). — 3gd/7aw

### W2-Inc — Incident/compaction/resilience
- `CompactionEvent` as a first-class archived object; loss-forensics as a construct measure; always-on JSONL-boundary fallback snapshot (degrade honestly when the hook is absent). — gjg
- `compaction_forgot(session_id)` MCP tool; content-addressed compaction snapshots via blob dedup (near-free). — gjg
- `daemon_lifecycle` table + heartbeat-backed liveness (not pid-file truth); crash-recovery convergence-debt reconciliation. — peo
- Postmortem bundle ingests the incident (daemon deaths, debt strandings, compaction events overlapping the window). — 8jg9
- The voluntary-handoff / involuntary-compaction / cross-session-resumption triad = one OS-memory story under 37t. — gjg/37t

### W2-T — Temporal doctrine
- Name the four times as a typed enum (source/ingest/derived-order/event); cross-kind comparison → mypy error. — cpf.1
- `sort_key_provenance` (explicit/inherited/fell-back-to-ingest/synthesized-0) so "ordered by real time" is queryable; the epoch-0 vanish bug. — NEW
- LIVE BUG: `RELATIVE_BASE = datetime.now()` at import → `since:7d` anchors to process-start, unpatchable by frozen_clock. — NEW
- Skew-tolerance band (co-temporal within band); total stable tiebreak `(sort_key_ms, session_id, position, block_position)` — a citation-anchor prerequisite. — NEW
- `time_confidence` on read payloads; tz-unknown-by-default (`source_tz`/`tz_provenance`); backfill must never move `sort_key_ms`; `docs/doctrine/time.md`. — cpf

### W2-Econ — Archive economics & longevity
- Storage-growth ledger insight (bytes/session/day by tier×origin + trend + 1/3/10yr projection). — NEW
- Promote/re-parent zstd blob compression (36GB→~4-7GB, address unchanged) OUT of the attachment epic; per-origin trained dictionaries. — 83u.5
- Access-temperature tier orthogonal to durability (cold >18mo → level-19 recompress + `cold/` shard); yearly epoch: repack fully-cold years into frozen read-only segments. — NEW
- Embeddings are the second storage bomb: int8/binary quantize vec0 (4×), embed only human+assistant blocks, drop cold; measure the derived-rebuild cost (the "rebuildable ≠ free" liability). — NEW
- `reacquirability` class on blobs (drop re-derivable-from-`~/.claude` first) + citable drop-tombstones + per-origin bytes/session SLO circuit-breaker. — 83u.4

### W2-Ing — Ingestion robustness & coverage
- Byte-fidelity ratio per raw_session (parsed_bytes/blob_size) — silent-drop becomes queryable. — NEW
- Round-trip reconstruction check vs blob_hash; unparsed-key census (ranked ignored provider keys). — NEW
- Detection-ambiguity score + misclassification tripwire (run other detectors post-parse); streaming-window blind-spot guard (islice 32). — NEW
- `parser_fingerprint` on raw_sessions → force reprocess on parser improvement (fidelity gains backfill). — NEW
- Zero-message parse-success anomaly detector; `unknown/grok-export` volume funnel; decode-failure taxonomy (encoding/framing/schema). — NEW

### W2-Ep — Cross-tool episode reconstruction
- `episode`/`episode_members` table in index.db, content-hashed member set; a tier ABOVE topology (lineage stays leaf). — NEW
- 4-signal scorer (repo × time-kernel × embedding × shared-artifact) with contributions in `evidence_json`; shared-artifact (file/SHA/error-fingerprint) dominates. — NEW
- Confidence tiers with a hard false-merge floor (linked > corroborated > candidate); anti-stitch negative evidence. — NEW
- Episode→commit/PR attribution (`produced_ref`); episode-level cost/effort rollup honoring material_origin + lineage-dedup. — NEW
- Episode as a stitch-hypothesis assertion (confirm/split/reject feeds the scorer); lynchpin terminal-artifact glue (a commit with no in-window session is still a member). — NEW

### W2-U — Missing data-model units
- episode; turn-pair (VIEW); entity-mention/reference (table); artifact produced/consumed (VIEW→table); world-effect cause→effect edge (VIEW). — NEW
- verification-run outcome (exit_code keystone); tool-outcome + retry-chain (extend `actions`); correction edge anchored to the block it corrects. — extend/NEW
- project (repo ∪ worktrees ∪ cwd ∪ `g-p-` gizmo_id) as durable user.db dim; topic/theme cluster (HDBSCAN over vec0). — NEW
- cross-origin thread (formalize `threads`); semantic handoff edge (materialize resume graph); phase segment as rows (extend session_phases). — extend
- goal/intent + decision-object as construct-gated candidates (recursive-safety); spend-episode cost VIEW. — NEW

### W2-Search — Relevance & explainability
- Lineage-aware result dedup/collapse (one thread ≠ N near-identical hits) with `variant_count`. — NEW
- `explain search`/`--why` narrating existing `score_components`; `AssertionKind.RELEVANCE` judgments from `mark` → the LtR corpus. — NEW
- `devtools lab search-eval` (nDCG@10/MRR/recall per lane, gate weight/k changes); weighted RRF; query-intent classifier before lane routing. — NEW
- Did-you-mean / query relaxation in miss-diagnostics (each with the count it would yield); field-boosted bm25 (title/opening > tool-log); recency×relevance knob. — NEW
- MMR diversity re-rank; score-calibration for cross-query comparability (never show raw bm25 as %); `weak_evidence` band on single-lane deep-rank hits. — NEW

### W2-Prox — Proactive/ambient surfacing
- Semantic alert tier on the EXISTING `HealthAlert` fan-out (add a `Notice` severity + `content` family); zero new channel. — NEW
- Standing queries as `daemon_events` producers (`notify_on: appeared|disappeared|count_crossed`); `polylogue brief --since 24h` = a query over the event ledger (the oracle habit, deterministic). — NEW
- "You're repeating a past mistake" real-time nudge (embed live tail → find_similar vs pathology/lesson sessions that ended badly, cite the prior failure). — NEW (highest-value ambient signal)
- Three-cadence policy table (on-event/daily/weekly) + per-family token bucket + `SUPPRESSION`-assertion snooze; recursive-safety (no alert on generated context); now-quiet deferral reusing hot-file logic. — NEW
- MCP-injected ambient preamble (brief the next agent before it starts); ambient-event replay audit ("what did you NOT tell me"). — NEW

### W2-Att — Attention/triage frontier
- A context-free `frontier` view ranking all 16.7k (invert cwd-coupled find_resume_candidates) + one `worth_reviewing_score` per logical session. — NEW
- `AssertionKind.triaged` (resumed/wont_resume/archived/snoozed:<until>) → the frontier is a true inbox that empties; snooze-with-wake. — NEW
- Durable-vs-disposable classifier as a first-class axis; "started and never finished" = terminal-state + no lineage/file follow-up. — NEW
- Cluster loose ends by thread/repo into cards (triage topics not rows); surface the actual open-question text with a citation-anchor. — NEW
- Blocker-centric lane ("N sessions blocked on the same thing" via embedding-cluster of blocker prose); inverted-U attention-urgency curve; frontier honesty (render low-confidence as a guess). — NEW

### W2-MQ — Meta-quality (proving polylogue's own correctness)
- Identity round-trip + injectivity metamorphic property (the `run_ref` global-PK collision #2464 class); content-hash idempotency suite (NFC adversaries, user-metadata excluded). — NEW
- Lineage compose⊕identity law + no-double-count invariant at scale (the ~32% dup #2467 as an executable law). — NEW
- FTS5 trigger-coherence metamorphic test; construct-validity CI gate (insight over zero rows must FAIL); recovery-digest fabrication regression. — NEW
- Daemon-vs-direct differential lane (same corpus → byte-identical index state); convergence idempotency + `false_means_pending` fixpoint; SIGKILL fault-injection at each stage boundary. — NEW
- Scale-shaped synthetic corpus generator as first-class infra (the missing 1xc.1 substrate); WAL/commit-boundary assertion helper; CHECK↔Literal drift property. — 1xc infra

## Prompt additions (wave-2) — titles + tag + focus (ask me to expand any to paste-ready)

Design [A]:
- A16 [A] Measure catalog + construct-validity registry — fill the 5-tuple registry row (construct, formula, evidence-tier, denominator, top-2 confounds, coverage precondition) for the 16 measures; flag provenance-mixing denominators. (sharper A1)
- A17 [A] Query→finding object-model migration — canonical AST hashing, result_sets tier placement + `reset --index` interaction, supersession-versioning vs content-addressed ids; DDL + the dropped-snapshot failure mode. (sharper A2)
- A18 [A] Recursive-safety invariant for a self-ingesting archive — the closed-loop predicate, quarantine state machine, author-kind trust, release conditions; SQL-expressible check + too-loose/too-tight failure modes.
- A19 [A] Runtime meta-session schema — project daemon/convergence/debt events into a queryable synthetic session tree; identity/hashing, tier, new self-analytics unlocked.
- A20 [A] `CompactionEvent` schema + loss-forensics diff — structural retained/lost/transformed classification, rank-by-later-reference, honest degradation on JSONL-boundary snapshots.
- A21 [A] Four-times stable ordering — total, skew-tolerant comparator with per-key provenance; identical corpora → identical order; backfill sorts by history; invariant tests.
- A22 [A] Tiering + retention policy engine — durability × access-temperature classification driving recompress/drop/repack; tombstone schema, space-pressure trigger, lease/GC-safe invariants.
- A23 [A] Adoption-measurement layer — signals distinguishing "used" vs "invisible", adoption-rate controlling for irrelevant repos, the zero-usage failure taxonomy.
- A24 [A] Progressive-disclosure/intent-routing over 130 MCP tools — primary-tagging vs `what_now()` router vs prompts vs skill; selection-accuracy/token tradeoffs.
- A25 [A] Ingest-fidelity metric — byte round-trip loss vs intended omission, cheap on multi-GB JSONL, one coverage number that never hides truncation/depth-cap drops.
- A26 [A] Provider misclassification detector + ambiguity score — winner-vs-runner-up margin, bounded cost on sampled JSONL, late-signal-past-window failure mode.
- A27 [A] Episode confidence/merge model — 4-signal scorer with a hard false-merge floor, confidence tiers, anti-stitch signals, per-edge auditable evidence.
- A28 [A] Table-vs-VIEW boundary for derived units — decision matrix (recursion depth, embedding-join cost, rebuild cost, query-unit exposure) applied to the 16 proposed units.
- A29 [A] Derived-intent construct-validity gate — candidate→verified state machine for goal/decision/phase units; citation-anchor requirement; the recovery-digest incident as a test case.
- A30 [A] Explainable ranking UX — "why ranked here" template over `score_components`, calibrated cross-query confidence (no raw bm25 %), `weak_evidence` threshold.
- A31 [A] Semantic notification policy layer — operational-vs-content taxonomy, per-family cadence + token-bucket + durable snooze, "worth interrupting" scoring, no-self-alert constraint.
- A32 [A] `worth_reviewing_score` + triage inbox state model — decomposable score, inverted-U urgency curve, force-to-zero conditions; triaged/snooze-with-wake state transitions + the "today's frontier" query.
- A33 [A] Metamorphic/property-test taxonomy — the algebraic laws (identity round-trip, hash idempotency, lineage compose⊕no-double-count, FTS coherence); per-law relation + generator + oracle, ranked by bug-class caught.

Deepresearch [DR]:
- D15 [DR] Metric-as-composable-object prior art — dbt-metrics/MetricFlow, PostHog, Cube.js, Honeycomb: dimension/window/comparison/uncertainty algebra + the traps (Simpson's, ratio-of-ratios, censored means).
- D16 [DR] Self-ingesting loops without error amplification — model-collapse mitigation, agent memory poisoning, provenance/taint tracking, retrieval down-weighting of self-generated content.
- D17 [DR] Compaction handling across agent frameworks — does any framework snapshot full pre-compaction context, measure loss, or re-ground from an external store? Prior art for "compaction as a first-class observable event."
- D18 [DR] Daemon-death forensics & heartbeat liveness — systemd Restart, faulthandler stack dumps, atexit clean-vs-vanish, heartbeat-age vs pid-file; reconciling half-finished derived work after unclean crash.
- D19 [DR] Event-time vs ingest-time vs processing-time — Datomic/OTel/Git/IMAP/journald conventions for clock skew, unknown tz, backfilled records, temporal-uncertainty display.
- D20 [DR] Relative-date & calendar-bucketing failure modes — import-time-frozen bases, DST/NTP-corrupted durations, UTC-midnight buckets misassigning local evening activity; hermetic time-injection patterns.
- D21 [DR] Compression/retention/dedup for personal append-only content-addressed stores — zstd trained dicts vs generic, FastCDC economics, restic/borg/Kopia incremental over immutable chunks, embedding int8/binary recall, time-decay sampling.
- D22 [DR] The "rebuildable tier" fallacy — when delete-and-rebuild beats retaining (re-embed $ + CPU-hours + rebuild risk); how search engines/feature-stores/vector DBs budget reindex; tiered rebuild-on-demand.
- D23 [DR] Silent field-drop & source-drift in ETL/archiving — dovecot/notmuch, ES ingest pipelines, Singer/Airbyte, OTel collectors: residual capture, dead-letter queues, coverage/lag, reprocess-on-parser-improvement.
- D24 [DR] Cross-tool episode/record-linkage — activity-timeline reconstruction, session segmentation, entity-resolution confidence, trace stitching without a shared id; adaptive temporal boundaries + false-link prevention.
- D25 [DR] Commit↔reasoning attribution — signals attributing a commit to the session that produced it; separating human vs machine authorship; cost/effort per shipped unit spread across tools + untranscribed steps.
- D26 [DR] Knowledge-graph/backlink/EAV schema prior art — Roam/Obsidian, Datomic, Gerrit change-objects, OTel span↔resource: mention edges, cause→effect provenance, episode clustering without storing content N×.
- D27 [DR] RRF tuning + result diversification — choosing k / per-lane weights / score normalization; offline relevance judgment sets; MMR vs cluster-then-rank vs canonical-collapse over forked document trees.
- D28 [DR] Notification fatigue & proactive-surfacing — interruptibility (Horvitz), digest-vs-realtime, peripheral-awareness UI, "repeating a past mistake" nudging; thresholds + empirical fatigue-onset.
- D29 [DR] Saved-query-as-subscription — incremental view maintenance / Materialize / Debezium / changedetection: appeared/disappeared/count-crossed, dedup by stable identity, no re-notification storms after backfill.
- D30 [DR] Human triage of large heuristic queues — email imbox, issue triage, SOC alert de-dup, read-later: scoring, snooze, clustering, zero-inbox pressure, staleness decay, dismissal audit; what fails when items are immutable.
- D31 [DR] Persisted-queries-as-objects — dbt exposures/snapshots, Malloy, Dagster asset checks, Great Expectations, Datasette canned queries, LookML, Metabase: query identity, result-set versioning, staleness triggers, test-bound-to-query.
- D32 [DR] Scale-only + fault-injection testing for embedded DBs — WAL-blowup regressions, crash-consistency fuzzing at txn boundaries, deterministic simulation (FoundationDB/TigerBeetle/Antithesis), property state-machines for GC/lease races; what's cheap on a synthetic corpus and what it misses.
- D33 [DR] Construct-validity as an executable substrate — Great Expectations / dbt tests / Monte Carlo / semantic-diff / authorship-gating: assert non-empty backing rows, non-fabricated text-mined events, no lineage double-count; a minimal self-verifying-archive gate.

Total fan-out now: 33 [A] + 33 [DR] = 66 prompts. Recommended first batch (highest leverage, run these ~20 first): A16, A17, A18, A2/A17-merge, A1/A16-merge, A11, A20, A32, A27, A13, A14, D1, D2, D7, D15, D16, D21, D24, D30, D33.


---

## [ab1286f914a4692b9] Spec — Honesty-audit completeness (Wave-5)

I have the grounding I need. Here is the implementation spec.

---

# Spec — Honesty-audit completeness (Wave-5)

## 0. Ground truth established

- **Registry has 11 number-bearing `InsightType` products** (`registry.py`): `session_profiles`, `session_work_events`, `session_phases`, `threads`, `session_tag_rollups`, `archive_coverage`, `tool_usage`, `session_costs`, `cost_rollups`, `usage_timeline`, `archive_debt` — plus `session_latency_profile` (served in `archive.py`, not a top-level registry name) and Day/Week aggregate summaries.
- **`_RIGOR_MATRIX` (`rigor.py`) covers exactly 5**: profiles, work_events, phases, threads, tag_rollups. **Uncovered and silently skipped: `archive_coverage`, `tool_usage`, `session_costs`, `cost_rollups`, `usage_timeline`, `archive_debt`** — precisely the cost/coverage/tool/debt cluster the red-team flagged.
- **The skip is structural**: `build_insight_rigor_audit_report` (`audit.py:188`) iterates `list_rigor_contracts()` only. A registry product with no contract produces **no entry at all** — not an `uncovered` row. There is no reconciliation against the registry.
- **`_audit_one` is also silent for aggregate-only products**: `session_tag_rollups` declares empty `evidence_payload`/`inference_payload`, so every counter stays 0 and `has_*_payload=False` — the audit "passes" a product it never actually inspected. There is no check that a product emitting numbers has *any* provenance-bearing field, and no check that a number is emitted over non-empty/non-NULL rows.
- **`classify_aggregate_hwm_source(source_updates: list[str])` (`temporal_source.py:97`) launders provenance**: it returns `provider_ts` whenever the list is non-empty, on the *documented assumption* that "each input is itself a `provider_ts`". That assumption is false — `classify_profile_hwm_source` can return `materialization_ts`/`fallback_date`, and the aggregate only receives timestamp *strings*, having already discarded each input's `TemporalSource`. Callers: `archive_summaries.py:85`, `archive_rollups.py:99`.
- **`transforms.py` honesty is exit-code-only**: `#2482` correctly moved test pass/fail *outcome events* to `SessionDigestEvent` read from keystone `exit_code`/`is_error`. But `_COMMIT_SHA_RE`, `_DECISION_RE`, `_STATUS_HEADING_RE`, `_TEST_PASS_RE`/`_TEST_FAIL_RE` (used for `test_evidence` preview), `_ISSUE_RE`/`_PR_RE` still regex-mine prose into `ToolSummary.commit_refs/test_evidence/pr_refs/issue_refs`, `DecisionCandidate.text`, `RunStateSummary`. These feed `ForensicIndex` and `SuccessorContextEntry`. The `EvidenceSupport` marker (`raw_evidence`/`inference`/`assertion`/`caveat`) exists only on the **bundle entry**, and only accepted decisions get `inference`; the **payload models themselves carry no `text_derived`/unverified flag**, so a forensic consumer reading `ToolSummary.commit_refs` cannot tell a structured value from a prose guess.
- **Attachments** (`MEMORY #2468`): `blob_hash` is fabricated, bytes never fetched, ~8.4 GB referenced / 0 stored. No model splits *referenced* (declared in export manifest) from *stored* (blob actually present) bytes.
- **CI gate pattern exists to mirror**: `devtools lab policy schema-versioning` (`command_catalog.py:714`, `verify_schema_upgrade_lane.py`) is the template for a structural policy lane; `devtools/coverage_gate.py` / `claim_vs_evidence.py` for a data-exercising lane.

---

## 1. Audit-contract schema — what changes

The contract moves from **product-level payload paths** to **field-level provenance declarations**, and gains a **completeness dimension**.

### 1a. New provenance vocabulary (`rigor.py`)
```
ProvenanceClass = Literal[
    "evidence",        # direct from archive structure (counts, keystone exit_code, provider ts)
    "inference",       # probabilistic, model/heuristic derived, confidence-scored
    "fallback",        # deterministic placeholder when source-anchored value absent
    "text_derived",    # regex/NLP-mined from message prose — UNVERIFIED
    "aggregate",       # min/sum/max over child rows; provenance = weakest contributor
]
```

### 1b. `RigorFieldContract` (new, replaces the coarse `evidence_payload`/`inference_payload` tuples)
```
class RigorFieldContract(ArchiveInsightModel):
    field_path: tuple[str, ...]       # dotted path to the number/collection
    provenance: ProvenanceClass
    nullable_when_ungrounded: bool    # True => must emit None, not 0, when no evidence row
    unverified_tag_path: tuple[str,...] = ()   # required iff provenance == "text_derived"
    aggregates_over: tuple[str, ...] = ()      # child field paths, iff provenance=="aggregate"
```
`RigorContract` keeps `insight_name`/`display_name`/`readiness_semantics`/`version_fields` and **gains `number_fields: tuple[RigorFieldContract, ...]`**. Every field that renders a number on any surface must appear here. Legacy `evidence_payload`/`inference_payload`/`confidence_field` become derived views over `number_fields` (back-compat for the existing rollup).

### 1c. Completeness contract (`rigor.py`)
```
def registry_number_bearing_names() -> frozenset[str]   # from INSIGHT_REGISTRY, minus explicitly-exempt
RIGOR_EXEMPT: frozenset[str] = frozenset()  # products declared to emit zero numbers; must be justified inline
def uncovered_insight_names() -> frozenset[str]:
    return registry_number_bearing_names() - set(rigor_contract_names()) - RIGOR_EXEMPT
```

### 1d. `InsightRigorAuditEntry` gains (`audit.py`)
```
coverage_status: Literal["covered", "uncovered", "exempt"]
number_field_reports: tuple[FieldReport, ...]   # per number_field: provenance, nonzero_rows, evidence_rows, ungrounded_number_emitted (bool)
temporal_source_observed: dict[str, int]         # histogram of input_high_water_mark_source values in the sample
laundering_suspected: bool                        # aggregate row claims provider_ts while a weaker input existed
```
`build_insight_rigor_audit_report` now iterates **`registry_number_bearing_names()`**, not `list_rigor_contracts()`: uncovered products get an entry with `coverage_status="uncovered"` and no silent omission.

### 1e. TemporalSource (`temporal_source.py`)
```
_TEMPORAL_STRENGTH = {  # ascending
  "fallback_date":0, "materialization_ts":1, "file_mtime":2, "sort_key":3,
  "hook_event_ts":4, "provider_ts":5,
}
def weakest_source(sources: Sequence[TemporalSource]) -> TemporalSource

def classify_aggregate_hwm_source(inputs: Sequence[TemporalSource]) -> TemporalSource:
    if not inputs: return "fallback_date"
    return weakest_source(inputs)
```
Signature change: callers must pass the **per-input `TemporalSource` classifications**, not bare timestamp strings. Where a caller only has timestamps today (`archive_summaries`, `archive_rollups`), the contributing per-session rows already store `input_high_water_mark_source`; thread that column through instead of the raw `updated_at` strings.

### 1f. Attachment bytes (models in `insights/archive.py` + coverage/debt)
Split every attachment byte total into two fields wherever one exists or is added:
```
referenced_bytes: int   # declared in export manifest; provenance = "fallback" (unverified)
stored_bytes: int       # sum of blob sizes actually present; provenance = "evidence"
```
Any surface that today prints a single "attachment bytes" number reports both, and never labels `referenced_bytes` as retrievable/available.

### 1g. `transforms.py` payload contract
Add an explicit unverified marker on the prose-mined models so honesty is carried in the data, not only the bundle renderer:
```
ToolSummary:        text_derived_fields: tuple[str,...] = ()   # e.g. ("commit_refs","test_evidence")
DecisionCandidate:  evidence_class: EvidenceSupport = "inference"   # never "raw_evidence"; set at extraction
RunStateSummary:    text_derived: bool = True
```
`ForensicIndexEntry` gains `verified: bool` (True only for keystone-structured claim kinds: `event`, `run_state`-from-structure). `SuccessorContextEntry.support` for these must resolve from the model flag, not be hard-coded per branch.

---

## 2. Enforcement algorithms (pseudocode)

### 2a. Completeness (structural, no data)
```
def check_rigor_completeness():
    uncovered = uncovered_insight_names()
    if uncovered: FAIL(f"number-bearing products without a rigor contract: {sorted(uncovered)}")
    for c in list_rigor_contracts():
        # every declared number_field must resolve on the product's Pydantic model
        model = insight_model_for(c.insight_name)
        for f in c.number_fields:
            assert path_exists_on_model(model, f.field_path)
            if f.provenance == "text_derived": assert f.unverified_tag_path
            if f.provenance == "aggregate":     assert f.aggregates_over
```

### 2b. Number-over-empty / NULL gate (data, over seeded demo corpus)
```
def check_no_ungrounded_numbers(rows, contract):
    for f in contract.number_fields:
        if f.provenance in ("inference","fallback","text_derived"): continue  # allowed to be soft
        nonzero = [r for r in rows if numeric(resolve_payload(r, f.field_path))]
        for r in nonzero:
            ev = evidence_backing(r, f)          # e.g. contributing non-NULL source column / keystone field
            if ev is None or ev == 0:
                if f.nullable_when_ungrounded and resolve_payload(r,f.field_path) is None:
                    continue                     # correctly emitted None
                FAIL(f"{contract.insight_name}.{'.'.join(f.field_path)} = "
                     f"{resolve_payload(r,f.field_path)} over row with no backing evidence")
```
For aggregate products (`archive_coverage`, `cost_rollups`), `evidence_backing` = count of contributing sessions whose underlying column is non-NULL. A `total_cost_usd > 0` over a bucket where every session has NULL cost fails.

### 2c. Temporal laundering gate
```
def check_no_temporal_laundering(rows, contract):
    for r in rows:
        src = resolve_payload(r, ("provenance","input_high_water_mark_source"))
        if src == "provider_ts" and is_aggregate(contract):
            inputs = contributing_input_sources(r)     # from the materializer's recorded child sources
            if any(strength(i) < strength("provider_ts") for i in inputs):
                FAIL(f"{contract.insight_name} row claims provider_ts but had weaker input {min-by-strength}")
```
(Enforced structurally at build time by 2e; this gate catches regressions in already-materialized rows.)

### 2d. text_derived tagging gate
```
def check_text_derived_tagged(digest):
    for tool in digest.tool_summaries:
        if tool.commit_refs or tool.test_evidence:
            assert set(tool.text_derived_fields) >= {name for name in ("commit_refs","test_evidence") if getattr(tool,name)}
    for d in digest.decision_candidates:
        assert d.evidence_class != "raw_evidence"
    for entry in digest.successor_context().entries:
        if entry.section in ("decisions","tools") and derived(entry):
            assert entry.support in ("inference","caveat")
```

### 2e. Weakest-source propagation (build-time, in the materializers)
Replace `classify_aggregate_hwm_source(source_updated_at_strings)` with `classify_aggregate_hwm_source([row.input_high_water_mark_source for row in contributing_rows])`. `weakest_source` then guarantees no upgrade.

---

## 3. Migration

All touched tiers are **derived** (`index.db` v24 rows, transforms are storage-free read models) or **pure contract** (`rigor.py`, `temporal_source.py`). **No numbered durable migration; `user.db`/`source.db` untouched.**

1. Land the pure-module changes (`rigor.py`, `temporal_source.py`, `audit.py`) + call-site threading in `archive_summaries.py`, `archive_rollups.py`, session materializers, and `transforms.py` model fields.
2. Bump `SESSION_INSIGHT_MATERIALIZER_VERSION` (and add `referenced_bytes`/`stored_bytes`, `input_high_water_mark_source` provenance now weakest-correct) → schema mismatch triggers derived rebuild: **`polylogue ops reset --index && polylogued run`**. No upgrade helper (rejected by `lab policy schema-versioning`).
3. Regenerate every embedded surface: `devtools render openapi`, `render cli-output-schemas` (new `ProvenanceClass` enum + coverage_status), `render insights-rigor-matrix` (`docs/insights-rigor-matrix.md` now shows all 11 products), and — if a new module is added for the gate — `devtools render topology-projection && render topology-status`. Verify with `render all --check` (grep for `out of sync`).
4. New MCP field on `insight_rigor_audit` output → no new tool, so `EXPECTED_TOOL_NAMES` unchanged, but tool contract snapshot regenerates.

---

## 4. Test strategy

- **Contract-completeness unit test** (`tests/unit/insights/test_rigor_completeness.py`): assert `uncovered_insight_names() == frozenset()`; parametrized over `INSIGHT_REGISTRY`, assert each product has a contract and each `number_field.field_path` resolves on its model. This is the regression net that fails the moment someone adds a 12th insight without a contract.
- **Audit-emits-uncovered test**: monkeypatch a registry product out of `_RIGOR_MATRIX`, assert the report still contains an entry with `coverage_status="uncovered"` (never a silent omission).
- **Number-over-empty property test** (Hypothesis, extend `test_properties.py` style): generate coverage/cost rows with all-NULL underlying columns → assert emitted total is `None`, never `0.0`, for `nullable_when_ungrounded` fields.
- **Temporal weakest-source unit tests** (`temporal_source`): `weakest_source` table-driven over all 6 tokens; `classify_aggregate_hwm_source([provider_ts, fallback_date]) == "fallback_date"`; a materializer test that an aggregate over one fallback-dated session never yields `provider_ts`.
- **transforms text_derived tests**: a digest built from a session whose prose contains a SHA and "decided: X" → assert `ToolSummary.text_derived_fields` includes `commit_refs`, `DecisionCandidate.evidence_class != "raw_evidence"`, and the successor bundle marks them `inference`/`caveat`.
- **Attachment split test**: a session referencing bytes with zero stored blobs → `stored_bytes == 0`, `referenced_bytes > 0`, and no surface string calls referenced bytes retrievable.
- **CI gate integration**: `devtools lab policy insight-honesty` run against `polylogue demo seed` corpus in the `test` lane; a golden `.local/` report. Inner loop: `devtools test tests/unit/insights/test_rigor_completeness.py` + `devtools verify` (testmon-affected). Do **not** blanket-run.

---

## 5. Bead breakdown (children of a new execution epic; 9e5's audit half is read-only, these are the split-out execution beads per 9e5's AC)

| Bead | Title | Acceptance |
|---|---|---|
| **B1** | Field-level `RigorFieldContract` + `ProvenanceClass`; declare `number_fields` for all 5 existing products | `rigor.py` carries `number_fields`; legacy `evidence_payload`/`confidence_field` derived from it; `render insights-rigor-matrix` clean; no behavior change to the 5 rollups |
| **B2** | Author contracts for the 6 uncovered products (coverage, tool_usage, session_costs, cost_rollups, usage_timeline, archive_debt) + `session_latency_profile` | `uncovered_insight_names() == ∅`; completeness unit test green; matrix doc lists all 11 |
| **B3** | Audit iterates registry, emits `coverage_status`/`uncovered`; `number_field_reports`; kill silent skip | `build_insight_rigor_audit_report` produces one entry per registry product; monkeypatch-out test proves an `uncovered` row is emitted; MCP/CLI output schemas regenerated |
| **B4** | TemporalSource weakest-source: `weakest_source`, change `classify_aggregate_hwm_source` to `Sequence[TemporalSource]`, thread `input_high_water_mark_source` through `archive_summaries`/`archive_rollups` + session materializers | No aggregate row emits `provider_ts` when any contributor is weaker; unit + materializer tests green; index rebuilt |
| **B5** | `transforms.py` payload honesty: `text_derived_fields`/`evidence_class`/`ForensicIndexEntry.verified`; support resolves from model flag | prose-mined SHA/decision/runstate fields carry unverified tags in the model, not only the bundle; forensic-bundle tests assert markers; exit-code axis unchanged |
| **B6** | Split `referenced_bytes` vs `stored_bytes` across attachment-byte surfaces (coverage/debt insight models + renderers) | every attachment-byte number reports both; referenced never labeled retrievable; test with 0 stored blobs |
| **B7** | `devtools lab policy insight-honesty` CI gate (completeness + number-over-empty + laundering + text_derived), wired into `test` lane over demo corpus | gate fails a synthetic ungrounded-number/laundered-ts/untagged-prose fixture; passes clean archive; `command_catalog.py` + `render devtools-reference` updated |

Dependency order: B1→B2→B3; B4, B5, B6 independent; B7 depends on B2/B3/B4/B5/B6.

---

## 6. Top-3 risks

1. **`nullable_when_ungrounded` breaks byte-compat consumers.** Flipping `total_cost_usd`/attachment bytes from `0.0`→`None` when ungrounded is the *correct* honesty fix but changes JSON shape for MCP/API/CLI-schema consumers and existing snapshot/envelope-contract tests. Mitigation: gate the flip behind the materializer-version bump + derived rebuild, and treat every failing snapshot as a deliberate `fix(test):`/schema regen, not a revert. Distinguish "honest None" from a real regression in review.

2. **Weakest-source propagation exposes how weak the archive actually is.** Once aggregates stop laundering to `provider_ts`, `usage_timeline`/`cost_rollups`/day-week summaries may broadly report `sort_key`/`file_mtime`/`fallback_date`, and downstream code or dashboards that assume `provider_ts` (e.g. recency sorting, staleness gates) could change behavior. This is truth surfacing, but validate `classify_profile_hwm_source` first — it *also* launders (`updated_at not None ⇒ provider_ts`) even when `updated_at` came from mtime; if the leaf classifier is itself dishonest, weakest-source at the aggregate is only half the fix. Scope note: this spec fixes the aggregate; a follow-up bead should audit the leaf.

3. **Completeness gate becomes a maintenance tax / false-confidence surface.** Forcing a contract for every registry product means a contract can be written *wrong* (mislabel a `text_derived` field as `evidence`) and still pass structural completeness — the gate proves coverage, not correctness. The number-over-empty and text_derived data gates (B7) are the real teeth; without them B2 gives green checks over unverified declarations. Ensure B7 lands with B2/B3 and never merges completeness-only. Also define `RIGOR_EXEMPT` with inline justification so genuinely number-free products don't force hollow contracts (which would themselves be dishonest).

---

Files grounding this spec (all absolute): `/realm/project/polylogue/polylogue/insights/rigor.py`, `/realm/project/polylogue/polylogue/insights/audit.py`, `/realm/project/polylogue/polylogue/insights/temporal_source.py`, `/realm/project/polylogue/polylogue/insights/transforms.py`, `/realm/project/polylogue/polylogue/insights/registry.py`, `/realm/project/polylogue/polylogue/insights/archive.py`, `/realm/project/polylogue/polylogue/insights/archive_summaries.py`, `/realm/project/polylogue/polylogue/insights/archive_rollups.py`, `/realm/project/polylogue/polylogue/mcp/server_insight_tools.py`, `/realm/project/polylogue/polylogue/cli/commands/insights.py`, `/realm/project/polylogue/devtools/command_catalog.py`, `/realm/project/polylogue/devtools/verify_schema_upgrade_lane.py`. Bead: `polylogue-9e5` (audit epic; children are read-only, so these execution beads must be split out per its AC).

---

## [a505b4db8666244ef] Recursive-Safety Subsystem — Implementation Spec

I have the full grounding I need. Here is the implementation-grade spec.

---

# Recursive-Safety Subsystem — Implementation Spec

**Scope:** the safety substrate that makes a *self-ingesting* archive (browser-extension auto-captures the operator's own agent R&D chats → same `assertions`/session store those agents later read) non-laundering. Five mechanisms: closed-loop laundering gate, `author_kind` differential trust in the inject gate, provenance-cycle quarantine (reusing `TopologyEdgeStatus`), evidence-content-hash drift auto-downgrade, and a hard `material_origin=generated_context_pack` no-inject/no-alert/no-mine rule.

## Grounding (verified, file:line)

- **`assertions`** (user.db, `USER_SCHEMA_VERSION=4`, durable/irreplaceable tier) — `polylogue/storage/sqlite/archive_tiers/user.py:12-31`. STRICT table already carrying `author_kind TEXT DEFAULT 'user'`, `evidence_refs_json TEXT DEFAULT '[]'`, `status TEXT DEFAULT 'active'`, `context_policy_json TEXT DEFAULT '{"inject":false}'`, `supersedes_json`, `confidence`, `staleness_json`. No CHECK on `status`/`author_kind` → both vocabularies are schema-free (grow like `AssertionKind`).
- **`AssertionStatus`** closed enum `polylogue/core/enums.py:437-447`: active/candidate/accepted/rejected/deferred/superseded/deleted/inactive. **No quarantine state exists.**
- **`TopologyEdgeStatus`** `enums.py:314-320`: unresolved/resolved/repaired/**quarantined** — the reuse target.
- **`material_origin`** closed enum `enums.py:184-192`: human_authored / assistant_authored / operator_command / runtime_protocol / runtime_context / tool_result / **generated_context_pack** / generated_analysis_pack / unknown. Lives on `messages` in **index.db** (rebuildable tier).
- **Candidate pipeline** (`user_write.py`): `upsert_transform_candidate_assertions` (`:1002`, `author_kind='transform'`, `status=CANDIDATE`, `context_policy={"inject":False,"promotion_required":True}`) and `upsert_pathology_findings_as_assertions` (`:1060`, `author_kind='detector'`). Judgment: `judge_assertion_candidate` (`:1245`) → `_promote_candidate_assertion` (`:1338`) which **stamps `author_kind='user'` and `author_ref=actor_ref` on the promoted active row** (`:1360-1361`). This is load-bearing: *judged ⇒ author_kind flips to `user`*.
- **Inject gate** = `list_assertion_claims(..., statuses=('active',), context_inject=True)` — `context/preamble.py:88-95` and `list_assertion_claims` SQL at `user_write.py:1531-1586`; the `context_inject` filter is currently **app-level Python** (`:1584-1585`), not SQL, and applies **no `author_kind` and no `material_origin` predicate**.
- **Cycle/quarantine to reuse**: `session_links.py` — `_would_create_cycle` (`:90`), `_quarantine_link` (`:125`, `SET status='quarantined'` + `reason='cycle_rejected'` + `cycle_path`), `count_quarantined_session_links` (`:170`), upsert clamp that never un-quarantines (`:60-62`).
- **Cross-tier joins are available**: user.db is `ATTACH`ed as **`user_tier`** onto the main index connection — `archive.py:3639-3644`. So `user_tier.assertions ⋈ main.messages(material_origin) ⋈ main.sessions(content_hash, origin)` is expressible in one SQL statement on the archive connection.
- **Content anchors exist**: `sessions.content_hash BLOB(32)` and `messages.content_hash BLOB(32)` (`index.py:70,123,168`). Evidence ref kinds (`core/refs.py:40`) are session/message/block; `ObjectRef` supports qualifiers (`:98`, `:179`) → `session:X@<hex>` is representable today with no ref-grammar change.
- **The incident class**: 2026-06-29 recovery digest fabricated "PR #123 merged" via `_events_from_text` regex over prose with **no authorship gating** (MEMORY.md). Structural fix = candidates may only cite content with a known non-generated `material_origin`.

---

## 1. Schema / DDL changes + tier

**Tier: user.db (durable) — one additive numbered migration; index-tier reads only.** Two orthogonal lifecycles: judgment `status` (operator intent, unchanged) and a new **provenance/safety** axis (auto-computed, reversible). Keeping them orthogonal is deliberate: an operator-accepted `active` assertion can later be auto-quarantined by evidence drift without destroying the judgment record.

Add to `assertions`:

```sql
provenance_state TEXT DEFAULT 'resolved'   -- TopologyEdgeStatus vocabulary: resolved|quarantined|repaired|unresolved
safety_json      TEXT                       -- nullable verdict detail (see below)
```

`safety_json` shape (written by the convergence stage):
```json
{
  "verdict": "ok | closed_loop_laundering | provenance_cycle | evidence_drift | generated_pack",
  "checked_at_ms": 0,
  "evidence_classification": {"grounding": 0, "recursive": 0, "unresolved": 0},
  "cycle_path": ["assertion:...", "assertion:..."],   // provenance_cycle only
  "drifted_refs": ["session:X@<expected>"],            // evidence_drift only
  "evidence_snapshot": {"session:X": "<hex>"}          // captured first pass for legacy unanchored refs
}
```

`provenance_state` **reuses `TopologyEdgeStatus` literally** as its value enum (no new enum): `resolved` = passed, `quarantined` = held (laundering or cycle), `repaired` = re-grounded/re-judged after a prior quarantine, `unresolved` = drift/pending. Index for the gate:

```sql
CREATE INDEX IF NOT EXISTS idx_assertions_inject_gate
  ON assertions(status, author_kind, provenance_state, kind);
```

`author_kind` gets a canonical (still schema-free TEXT) vocabulary — introduce `AuthorKind` StrEnum in `core/enums.py` for producers only: `user / agent / transform / detector / assistant / system`. No CHECK; consumers treat "anything != `user`" as machine-authored.

## 2. Exact SQL-expressible predicates (pseudocode)

All run on the archive connection with `user_tier` attached.

**(a) Inject gate — the single hardened predicate** (replaces app-level `context_inject` filter and adds differential trust + generated-pack + quarantine exclusion):

```sql
-- an assertion is INJECTABLE iff:
SELECT a.assertion_id
FROM user_tier.assertions a
WHERE a.status = 'active'
  AND json_extract(a.context_policy_json, '$.inject') = 1
  AND a.author_kind = 'user'                 -- (§differential trust) machine rows never inject
                                             --   unless judged, which stamps author_kind='user'
  AND a.provenance_state != 'quarantined'    -- (§cycle + §laundering) held rows excluded
  AND NOT EXISTS (                           -- (§generated_context_pack) never inject own recall packs
        SELECT 1 FROM main.messages m
        WHERE m.material_origin = 'generated_context_pack'
          AND ('message:' || m.message_id = a.target_ref
               OR 'session:' || m.session_id = a.target_ref)
  );
```
The `author_kind='user'` clause is the injection tripwire: an agent that writes `context_policy={"inject":true}` directly cannot self-promote because its row carries `author_kind='agent'` (stamped by the write surface, §5/bead 37t.14), and only `judge_assertion_candidate`→`_promote_candidate_assertion` flips it to `user`.

**(b) Closed-loop laundering** (machine-authored claim grounded only in machine content):

```
LAUNDERED(A) :=
  A.author_kind != 'user'
  AND len(evidence_refs(A)) > 0
  AND grounding_leaf_count(closure(A)) == 0

grounding_leaf(ref) :=
  resolve(ref) is a message/block with material_origin IN ('human_authored','operator_command')
  OR ref is a raw source-bytes / external-URL anchor

closure(A) := DFS over evidence edges; ref 'assertion:B' descends into evidence_refs(B)
              (bounded by MAX_DEPTH, cycle-guarded)
-- verdict → provenance_state='quarantined', safety_json.verdict='closed_loop_laundering'
```
SQL kernel per assertion (recursive CTE over the evidence graph):
```sql
WITH RECURSIVE reach(ref, depth) AS (
    SELECT value, 0 FROM json_each((SELECT evidence_refs_json FROM user_tier.assertions WHERE assertion_id=:A))
  UNION
    SELECT je.value, r.depth+1
    FROM reach r
    JOIN user_tier.assertions b
      ON r.ref = 'assertion:' || b.assertion_id AND r.depth < :MAX_DEPTH
    JOIN json_each(b.evidence_refs_json) je
)
SELECT SUM(is_grounding(ref)) AS grounding, COUNT(*) AS total FROM reach;
-- is_grounding(ref): join to main.messages on message_id/session_id, test material_origin membership
-- grounding=0 AND total>0 → laundered
```

**(c) Provenance cycle** (reuse `_would_create_cycle` DFS pattern, `session_links.py:90`, over `assertion→assertion` evidence edges): if `A ∈ closure(A)` → quarantine every node on `cycle_path`, `safety_json.verdict='provenance_cycle'`. Same clamp rule as `session_links` upsert: a quarantined row is never auto-cleared by re-computation, only by judgment/repair.

**(d) Evidence-content-hash drift → auto-downgrade:**
```
for anchored ref session:X@<expected_hex> (or snapshot fallback in safety_json.evidence_snapshot):
    actual := hex(SELECT content_hash FROM main.sessions WHERE session_id='X')  -- or messages
    if actual != expected_hex:
        drifted_refs += ref
if drifted_refs and A.status == 'active':
    status  active → candidate           -- re-enter judgment queue (37t.12)
    context_policy.inject := false
    provenance_state := 'unresolved'; safety_json.verdict='evidence_drift'
```
Downgrade is reversible: re-judging (accept) re-promotes and sets `provenance_state='repaired'`.

**(e) generated_context_pack hard rule** — three enforcement points:
1. inject gate excludes (predicate (a));
2. candidate miner refuses to emit candidates whose source/evidence `material_origin='generated_context_pack'` (§5);
3. any alerting/surfacing query carries `AND NOT EXISTS (... material_origin='generated_context_pack')`.

## 3. Migration

Durable tier ⇒ **additive numbered migration behind a verified backup manifest**, one `PRAGMA user_version` step (`CLAUDE.md` schema regime; validated by `devtools lab policy schema-versioning`).

`polylogue/storage/sqlite/migrations/user/005_assertion_provenance_state.sql`:
```sql
ALTER TABLE assertions ADD COLUMN provenance_state TEXT DEFAULT 'resolved';
ALTER TABLE assertions ADD COLUMN safety_json TEXT;
CREATE INDEX IF NOT EXISTS idx_assertions_inject_gate
  ON assertions(status, author_kind, provenance_state, kind);
PRAGMA user_version = 5;
```
Bump `USER_SCHEMA_VERSION 4→5` in `user.py:5` and add the two columns to `USER_DDL` (for fresh archives) + `_ASSERTION_COLUMNS` (`user_write.py:1407`) and `_assertion_row_to_envelope` (`:1381`). Pure additive with safe defaults → no copy-forward, no consent gate beyond the standard backup manifest. Index-tier reads need no migration (rebuildable). Existing rows default to `provenance_state='resolved'`; the first convergence pass computes real verdicts and populates `evidence_snapshot` for legacy unanchored refs (drift baseline).

## 4. Test strategy + injection/laundering tripwire fixture

**Placement:** `tests/unit/storage/test_recursive_safety.py` (+ property law in `tests/property/`), using `SessionBuilder`/`make_message`/`workspace_env` and the `frozen_clock` fixture (clock-hygiene lint). The daemon owns writes → drive the real convergence stage, not a mock.

**Tripwire fixture `laundering_corpus`** (the load-bearing one):
- Session `S_agent`: all messages `material_origin=assistant_authored` (a self-captured agent R&D chat).
- Session `S_human`: at least one `material_origin=human_authored` / `operator_command` message.
- `A_laundered`: `author_kind='agent'`, `evidence_refs=['session:S_agent']`, `context_policy={"inject":true}` (the attack: agent wrote inject:true directly).
- `A_grounded`: `author_kind='agent'`, `evidence_refs=['session:S_human']`.
- `A_cycle_1/A_cycle_2`: mutual `assertion:` evidence.
- `A_pack`: targets a `generated_context_pack` message.

Assertions:
1. **Injection tripwire** — `A_laundered` never appears in the inject gate (blocked by `author_kind != 'user'` *and* by quarantine); assert both independently so removing either clause fails.
2. **Laundering** — after convergence, `A_laundered.provenance_state='quarantined'`, `verdict='closed_loop_laundering'`; `A_grounded` stays `resolved` (grounding leaf present).
3. **Cycle** — both cycle nodes quarantined with `cycle_path` populated; re-running convergence does not un-quarantine (clamp invariant, mirrors `session_links`).
4. **Drift** — mutate `S_human.content_hash`; assert an anchored active assertion downgrades `active→candidate`, `inject=false`, re-enters the judgment queue; re-judge → `provenance_state='repaired'`.
5. **generated_context_pack** — `A_pack` never injectable; the candidate miner emits **zero** candidates for the pack session.
6. **Release** — operator `judge_assertion_candidate(accept)` on a quarantined-laundered claim clears quarantine (human judgment breaks the loop).

**Regression test** (tie to the incident): a "recovery-digest" candidate mined from an agent-only session with no human-authored grounding must be quarantined, not injected — the exact structural failure of the 2026-06-29 `_events_from_text` fabrication.

**Lint / executable policy** (`devtools lab policy`, modeled on `verify-test-clock-hygiene`): a static check `no-unauthored-inject` that greps for any inject-gate read path missing the `author_kind='user'` + `provenance_state != 'quarantined'` + generated-pack clauses — prevents a future surface from re-opening the laundering hole. Add a construct-validity gate: an inject/alert query returning rows sourced only from machine content **fails**.

## 5. Bead breakdown (children of 37t; acceptance)

- **37t.13 — Provenance-state schema + safety verdict column.** Migration 005 (additive, backup manifest), `USER_SCHEMA_VERSION 4→5`, `provenance_state` reuses `TopologyEdgeStatus` values, `safety_json`, gate index, `AuthorKind` enum, envelope/columns wiring. *AC:* `PRAGMA user_version=5` on migrated + fresh archives; `devtools lab policy schema-versioning` passes; round-trip read/write of both columns; existing rows default `resolved`.
- **37t.14 — Inject-gate hardening (differential trust + generated-pack + quarantine).** Push the inject filter into SQL (predicate (a)); require `author_kind='user'`; exclude quarantined and `generated_context_pack`. Force the MCP/agent write surface (`mcp/server_mutation_tools.py`) to stamp `author_kind` from caller **role**, never caller input. *AC:* an agent-written `inject:true` assertion is not injected; a judged (promoted) claim is; unit test asserting each clause independently; `no-unauthored-inject` lint green.
- **37t.15 — Recursive-safety convergence stage.** `make_recursive_safety_stage` in `convergence_stages.py` (daemon-owned write, `false_means_pending`, `check_sessions`/`execute_sessions` for debt retry), implementing laundering (b), cycle (c) reusing `_would_create_cycle`, and writing `provenance_state`/`safety_json`; clamp so re-run never un-quarantines. Registered in `make_default_convergence_stages`. *AC:* laundering + cycle fixtures quarantined; idempotent re-run; topology-projection/render regenerated.
- **37t.16 — Content-anchored evidence + drift downgrade.** Candidate writers emit `session:X@<hex>`/`message:X@<hex>` (ObjectRef qualifier); resolver + drift check; legacy-ref snapshot fallback; auto-downgrade `active→candidate` on mismatch feeding 37t.12's queue. *AC:* drift fixture downgrades + re-queues; re-judge → `repaired`; anchors resolve via `resolve_ref`.
- **37t.17 — Candidate-miner authorship gating (incident fix).** `upsert_transform_candidate_assertions`/`upsert_pathology_findings_as_assertions` and any text-mining path refuse `material_origin ∈ {generated_context_pack, unknown}` and require ≥1 non-generated cited origin. *AC:* zero candidates mined from a generated-pack session; recovery-digest regression test passes; no fabricated-event candidate from an agent-only source.
- **37t.18 — Tripwire fixtures + laundering/injection lint + construct-validity gate.** `laundering_corpus` fixture, the six assertions above, regression test, `no-unauthored-inject` lint, construct-validity CI gate. *AC:* all six assertions pass; lint fails on a deliberately-broken gate; gate fails on a machine-only inject/alert query.

Suggested order: 37t.13 → (37t.14 ∥ 37t.16) → 37t.15 → 37t.17 → 37t.18. 37t.14's differential-trust clause and 37t.15's quarantine writer share the gate index from 37t.13; 37t.16 is independent until 37t.15 consumes drift verdicts.

## 6. Top-3 risks (too-loose vs too-tight)

1. **Laundering predicate too-tight → the archive can never remember its own valid work.** The operator's *legitimate* agent R&D — genuinely useful conclusions reached in an assistant-authored session — is exactly the self-capture target, and by construction has no `human_authored` leaf. Quarantining all of it makes the memory loop (37t) inert. *Mitigation:* `operator_command` counts as grounding (the operator steering an agent *is* human grounding), and quarantine is a **hold for judgment**, not a delete — 37t.12's queue is the release valve. Tune: laundering quarantines only *auto-inject* candidates, never blocks the claim from being *listed/judged*.
2. **Differential-trust too-loose → the tripwire is bypassable at the write boundary.** The whole gate rests on `author_kind='user'` meaning "operator-judged." If any surface lets an agent set `author_kind` from input (or a code path forgets to stamp role), a self-captured agent launders directly into inject. *Mitigation:* 37t.14 forces role-stamping server-side + the `no-unauthored-inject` lint; user.db being durable/irreplaceable makes a bad write costly, so the boundary check is the real security surface, not the schema.
3. **Drift auto-downgrade too-tight → convergence thrash / mass downgrade.** Content-hash anchoring means any legitimate re-materialize (a reprocess that changes normalization, or an FTS/derived rebuild touching content) mass-drifts every anchored assertion and floods the judgment queue — punishing the operator for routine `ops reset --index`. *Mitigation:* anchor on the **durable** `sessions.content_hash` (which is content-idempotent by construction, `pipeline/ids.py` — excludes user metadata), not on any derived-tier value; treat drift as `unresolved`+downgrade (reversible) rather than `quarantined`; batch under the existing hot-file quiet-deferral so a live rebuild doesn't storm the queue.

**Not edited:** no code or beads changed (read-only spec). All file:line references above are from the live tree at `/realm/project/polylogue`.

---

## [abf033d819a8461dc] Spec: `polylogue-export` origin + Conversation Interchange Format (CIF)

I have all the grounding I need. Here is the spec.

---

# Spec: `polylogue-export` origin + Conversation Interchange Format (CIF)

**Grounding (real files, read this session):**
- `polylogue/core/enums.py:42` — `Origin` enum, 10 members incl. reserved `GROK_EXPORT = "grok-export"` (the template for a reserved-token-first origin).
- `polylogue/core/sources.py:220,285` — bidirectional `_PROVIDER_TO_ORIGIN` / `_ORIGIN_TO_PROVIDER` + `_ORIGIN_TO_LAB` (line 238). `origin_from_provider`/`provider_from_origin` are the round-trip bridge.
- `polylogue/pipeline/ids.py:40-66` — `session_id = f"{origin.value}:{provider_session_id}"`, `message_id = f"{session_id}:{provider_message_id}"`; `session_content_hash` (126) hashes NFC-normalized title+timestamps+messages+blocks+attachments+events, **excludes user metadata**.
- `polylogue/sources/dispatch.py:124` — `_detect_provider_from_record` is the tightness-ordered detector chain; `LoweredPayloadSpec` (49) is what a payload lowers to.
- `polylogue/sources/parsers/base_models.py:51,95,248` — `ParsedContentBlock` / `ParsedMessage` / `ParsedSession` field sets (the normalized target the parser must emit; captured below).
- `polylogue/sources/provider_completeness.py:35` — `_PackageModeSpec` (`origin`, `capture_mode`, `maturity`, 11 `_REQUIRED_ITEM_NAMES`); `devtools lab provider completeness` reads `PACKAGE_MODE_SPECS`.
- Bead **polylogue-l4kf** (epic "Ecosystem interop") — parent of `2qx` (OriginSpec), `611` (Grok), `0cg` (OTel), `4g5` (HPI/Promnesia), and export-lane children `wmj/7k7/r47/4g5`. This spec is the natural sibling of those export children.

---

## The load-bearing design decision (state it first)

Identity is `session_id = origin:native_id` and archive writes are **idempotent by content hash**. Therefore the *correct* round-trip is not "re-import produces equal-but-new rows" — it is **re-import is a no-op** because the content hash matches. This gives a free, strong invariant:

> `import(export(archive)) ⇒ zero new/changed sessions`

For that to hold, a CIF record must re-emit the session under its **original origin**, not under `polylogue-export`. So:

- **`Origin.POLYLOGUE_EXPORT` is the envelope/container identity** — what `detect_provider` recognizes. It is transparent: reconstructed sessions preserve their **embedded** origin token (`claude-code-session`, etc.), reproducing identical `session_id/message_id/block_id`.
- `POLYLOGUE_EXPORT` appears as a *session's own* origin only for CIF-native content with no upstream origin (hand-authored / federation-minted), mirroring how `GROK_EXPORT` is reserved.

This is the pivot: the same mechanism is a **correctness invariant** (self round-trip) and a **federation primitive** (a CIF file carries N sessions from N origins, each citably anchored).

---

## (1) CIF schema + `polylogue-export` Origin/parser

### CIF envelope (content-addressed, portable anchors, per-source fidelity)

```jsonc
{
  "cif_version": 1,                      // magic key — strictest detector signal
  "generator": "polylogue/<version>",
  "exported_at": "2026-07-05T00:00:00Z",
  "content_hash_algo": "sha256-nfc-json",// pins ids.py/hashing.py algorithm identity
  "fidelity": {                          // per-source declaration (l4kf: "fidelity declaration")
    "claude-code-session": {
      "capture_mode": "export-jsonl",
      "level": "lossless",               // lossless | lossy-derived | metadata-only
      "excludes": ["raw_bytes", "blob_content"],
      "notes": "attachments metadata-only (Ref #2468)"
    }
  },
  "sessions": [
    {
      "origin": "claude-code-session",   // ORIGINAL origin — reproduces session_id
      "native_id": "<provider_session_id>",
      "session_id": "claude-code-session:<native_id>",   // asserted anchor (verify on import)
      "content_hash": "<sha256>",        // asserted; recompute + compare on import
      "title": "...", "created_at": "...", "updated_at": "...",
      "lineage": {                       // preserves session_links topology
        "parent_native_id": "...", "parent_origin": "...",
        "branch_point_message_id": "...", "inheritance": "prefix-sharing"
      },
      "messages": [
        {
          "native_id": "<provider_message_id>",          // or null → position-derived id
          "message_id": "<session_id>:<native_id>",       // asserted anchor
          "role": "assistant",
          "material_origin": "assistant_authored",        // authoredness axis preserved
          "position": 12, "variant_index": 0,
          "timestamp": "...", "model_name": "...",
          "input_tokens": ..., "output_tokens": ...,      // cost columns preserved
          "blocks": [
            { "position": 0, "block_id": "<message_id>:0",
              "type": "tool_use", "text": "...",
              "tool_name": "...", "tool_id": "...", "tool_input": {...},
              "is_error": null, "exit_code": null,        // keystone structural outcomes
              "media_type": null }
          ]
        }
      ],
      "attachments": [ /* metadata only, per fidelity.excludes */ ],
      "session_events": [ /* event_type, timestamp, payload */ ]
    }
  ]
}
```

**Design notes:**
- Anchors (`session_id`, `message_id`, `block_id`, `content_hash`) are **asserted, then re-derived and verified** on import — never trusted. The asserted values are the round-trip contract; recomputation is the check. A mismatch is a parser/format-drift signal, not silently accepted.
- `native_id` may be `null`; the parser then supplies the same positional fallback the ids use (`message_id = session_id:position.variant_index`) so the derived id still matches.
- Field set is a faithful projection of `base_models.py`: `ParsedMessage` (`material_origin`, `variant_index`, `position`, token columns, `model_name`) and `ParsedContentBlock` (`is_error`/`exit_code` keystone, `tool_id` for the `actions` view join) — nothing that feeds the content hash may be dropped, or self round-trip breaks.
- `fidelity` is CIF's honesty surface: attachments are metadata-only by construction (Ref #2468), so a lossless CIF must declare `excludes: ["blob_content"]` rather than pretend.

### Origin + wiring

- Add `Origin.POLYLOGUE_EXPORT = "polylogue-export"` (`core/enums.py`).
- Add `Provider.POLYLOGUE` (wire token), and entries in **all three** maps in `core/sources.py`: `_PROVIDER_TO_ORIGIN`, `_ORIGIN_TO_PROVIDER`, `_ORIGIN_TO_LAB` (lab = `"polylogue"` / self). This mapping is **injective** (unlike GEMINI+DRIVE→AISTUDIO_DRIVE), so no `project_origin_payload` hazard.
- New package `sources/parsers/polylogue_cif/` with `looks_like` (checks `cif_version` int + `sessions` list) and `parse`. Register the detector **first** in `_detect_provider_from_record` (dispatch.py:124) — tightest tier: a required integer magic key no other format carries.
- Parser lowers the envelope into **one `ParsedSession` per `sessions[]` entry**, each with `source_name = <embedded origin's provider>` (not `POLYLOGUE`), so `session_content_hash`/`session_id` reproduce the original. `LoweredPayloadSpec` gets a bundle-style multi-spec path (dispatch already has grouped/bundle lowering, e.g. `_claude_code_grouped_record_specs:354`).

### `.well-known/ai-sessions` manifest (federation, thin)

```jsonc
// GET https://host/.well-known/ai-sessions
{ "cif_version": 1, "endpoint": "/cif/sessions",
  "sessions": [ { "session_id": "...", "content_hash": "...", "updated_at": "...",
                  "origin": "...", "bytes": 12345 } ] }
```
Selective sync = fetch manifest, diff `content_hash` against local, request only missing/changed sessions as CIF records. Content-hash idempotency means an unchanged remote session is skipped even if fetched — the manifest just avoids the fetch. (Federation-transport is the last, optional bead; the format + parser stand alone.)

---

## (2) Round-trip algorithms (pseudocode)

**Export:**
```
export(filter) -> CIF:
  envelope = {cif_version:1, generator, exported_at, content_hash_algo:"sha256-nfc-json"}
  for session in repository.iter(filter):              # SessionRepository read mixins
    rec = project(session)                             # emit base_models fields verbatim
    rec.origin        = session.origin.value           # ORIGINAL origin
    rec.native_id     = session.native_id
    rec.session_id    = session.session_id             # assert current id
    rec.content_hash  = session.content_hash           # from index.db
    rec.lineage       = session_links_for(session)     # parent + branch_point + inheritance
    envelope.fidelity[session.origin] |= declare_fidelity(session)  # capture_mode+excludes
    envelope.sessions.append(rec)
  return envelope
```

**Import (via normal dispatch → daemon writer):**
```
parse_cif(envelope) -> [ParsedSession]:
  assert envelope.content_hash_algo == "sha256-nfc-json"   # algorithm identity guard
  out = []
  for rec in envelope.sessions:
    origin   = Origin.from_string(rec.origin)
    provider = provider_from_origin(origin)                # sources.py bridge
    ps = ParsedSession(source_name=provider,
                       provider_session_id=rec.native_id,
                       title=rec.title, created_at=rec.created_at, updated_at=rec.updated_at,
                       messages=[to_parsed_message(m) for m in rec.messages],
                       attachments=..., session_events=...,
                       parent_session_provider_id=rec.lineage.parent_native_id, ...)
    # VERIFY asserted anchors reproduce (fail loud on drift):
    assert session_id(provider, ps.provider_session_id) == rec.session_id
    assert str(session_content_hash(ps)) == rec.content_hash
    out.append(ps)
  return out
  # pipeline then hashes; matching content_hash => idempotent skip (no-op)
```

**Round-trip identity invariant:**
```
import(export(A)) is a no-op:
  export(A) -> CIF
  for rec in CIF.sessions:
    reparsed = parse_cif(single(rec))
    assert session_content_hash(reparsed) == A[rec.session_id].content_hash  # => ingest skip
```

---

## (3) Migration

Classify per CLAUDE.md schema regimes:

- **`Origin`/`Provider` enum + `sources.py` maps** — pure Python; no SQL migration.
- **index.db (derived)** — if `origin`/`provider` appears in a generated `literal_check` CHECK, adding a member is an **additive-derived** change: edit canonical DDL, bump `index.db` schema version, rebuild via `polylogue ops reset --index && polylogued run`. **No** upgrade helper (`devtools lab policy schema-versioning` rejects one). Verify with: `grep -rn "literal_check.*[Oo]rigin\|get_args(Origin" polylogue/storage/sqlite/` before assuming a CHECK bump is needed.
- **source.db (durable)** — `raw_sessions.provider` is `TEXT`, not a CHECK on the Origin literal, so a new token needs **no numbered durable migration**. Confirm there is no `literal_check` over provider in `migrations/source/`; if there were, it would require a numbered additive migration + backup manifest.
- **Generated surfaces (mandatory, or `render all --check` fails):**
  - `devtools render topology-projection && devtools render topology-status` (new `sources/parsers/polylogue_cif/` module).
  - `render openapi` + `render cli-output-schemas` (Origin enum embedded).
  - Add a `_PackageModeSpec` to `PACKAGE_MODE_SPECS` (`provider_completeness.py`) so `devtools lab provider completeness` is green for the new origin.
- **Import direction only** in phase 1 (l4kf children `wmj/r47/7k7` own the emit lane / OTel-GenAI export separately); this spec ships **format + parser + reserved origin** so a `polylogue export` CLI verb can land against a stable format.

---

## (4) Test strategy — round-trip identity as the primary invariant

1. **Property (Hypothesis) — the free correctness invariant.** Over schema-driven `SessionBuilder` sessions across every real origin: `content_hash(parse_cif(export(s))) == content_hash(s)` and `session_id`/`message_id`/`block_id` reproduce exactly. This is the spec's whole justification — it must be a property, not an example. Lives beside `tests/unit/sources/test_parsers_props.py` (protected).
2. **Idempotent-skip integration.** Seed archive → `export` → re-`import` through the real pipeline → assert 0 new/0 changed sessions (content-hash skip path). Demo-seeded, private-data-free (`polylogue demo seed`).
3. **Dispatch tightness regression.** Feed a CIF fixture through the full `detect_provider` chain; assert it is claimed by the CIF detector and that a `claude-code`/`codex` fixture is *not* mis-claimed by CIF (the l4kf-2qx footgun: a loose detector stealing records). Also assert every *other* origin's fixture is unchanged after inserting the CIF detector at the head.
4. **Anchor-drift guard.** Corrupt an asserted `content_hash`/`session_id` in a CIF fixture → parser raises (format-drift is loud, never silently re-ingested under a new id).
5. **Federation / non-injectivity.** A CIF carrying both `gemini-cli-session` and `aistudio-drive` sessions round-trips without the GEMINI/DRIVE→AISTUDIO_DRIVE collapse corrupting either (guards the `project_origin_payload` seam).
6. **Fidelity honesty.** Attachment-bearing session exports with `fidelity.excludes: ["blob_content"]` and metadata-only round-trips cleanly (Ref #2468) — no fabricated blob presence.
7. **Completeness lane.** `devtools lab provider completeness` green for `polylogue-export` (all 11 required items present).

Use `frozen_clock` for any timestamp-sensitive case (clock-hygiene lint).

---

## (5) Bead breakdown (children of **polylogue-l4kf**; 6 beads)

| # | Title | Acceptance |
|---|---|---|
| **B1** | `feat(sources): CIF envelope schema + fidelity declaration` | CIF v1 documented (`docs/providers/polylogue-cif.md` + schema package); fidelity block has `capture_mode`/`level`/`excludes`; `content_hash_algo` pins the ids/hashing algorithm identity. `render all --check` clean. |
| **B2** | `feat(core): Origin.POLYLOGUE_EXPORT + Provider.POLYLOGUE bridge` | Enum member added; all three `sources.py` maps + lab entry present and injective; `render openapi`+`render cli-output-schemas` regenerated; existing origin tests green. |
| **B3** | `feat(sources): CIF detector + parser (import lane)` | `looks_like` gates on `cif_version` int; registered first in dispatch tightness order; parser lowers to N `ParsedSession` under **embedded** origins; anchors re-derived and verified. Dispatch-tightness regression green. |
| **B4** | `test(sources): CIF round-trip identity property` | Hypothesis property `content_hash(parse_cif(export(s)))==content_hash(s)` + id reproduction across all origins; idempotent-skip integration (0 new/0 changed on re-import); anchor-drift guard. |
| **B5** | `feat(cli): polylogue export -> CIF` | `export` verb (or `read --view cif`) emits CIF for a query; folds into `read` per operator "export shouldn't be a separate flow"; `_PackageModeSpec` added → `devtools lab provider completeness` green for `polylogue-export`. |
| **B6** | `feat(daemon): .well-known/ai-sessions manifest + selective content-hash sync` | Manifest endpoint lists `{session_id, content_hash, origin}`; selective sync fetches only content-hash-divergent sessions; re-fetch of unchanged session is a content-hash no-op. (Depends B1–B5; can defer.) |

Ordering: B1→B2→B3→B4 is the correctness core (federation-independent); B5 adds the emit surface; B6 is optional transport. B3 must land before B5 so export is validated against a working import.

---

## (6) Top-3 risks

1. **Content-hash algorithm drift silently breaks federation.** `session_content_hash` is NFC-normalized `json.dumps(sort_keys, separators=(",",":"))` (deliberately stdlib, *not* orjson, per `hashing.py:38`). Any future change to hash inputs/serialization invalidates every previously-emitted CIF `content_hash` → cross-instance sync degrades to full-refetch and self round-trip stops being a no-op. **Mitigation:** `content_hash_algo` is a required, version-pinned envelope field; a hashing change bumps it and the importer refuses/renormalizes rather than trusting stale anchors. Add a golden-hash regression test that fails loudly if the algorithm changes.

2. **Detector tightness footgun (l4kf-2qx).** CIF is a generic JSON container; a too-loose `looks_like` (e.g. only checking `sessions` is a list) could steal `chatgpt-export` or `codex` records, or vice-versa. Hand-ordered dispatch makes this a standing hazard. **Mitigation:** gate strictly on `cif_version` being an `int` **plus** structural shape; add the tightness regression (test #3) that runs every origin's fixture through the full chain. Consider landing atop the `OriginSpec` refactor (2qx) so tier ordering becomes type-level rather than vigilance-maintained.

3. **Federation identity collision vs. transparency.** Preserving embedded origin gives clean self round-trip, but importing *another instance's* export can collide `session_id`s (same `origin:native_id`) with genuinely different content, or import a foreign session as if locally-authored. **Mitigation:** phase 1 is self-round-trip only (content-hash mismatch on collision surfaces as a normal update, not corruption); federation (B6) must record import provenance (a `session_link`/import-source assertion) and define a collision policy (content-hash equal ⇒ dedup; unequal same-id ⇒ quarantine, reusing the `TopologyEdgeStatus.quarantined` pattern) **before** two-way sync is enabled. Non-injective origins (GEMINI+DRIVE→AISTUDIO_DRIVE) are a narrower instance of the same seam — covered by test #5.

**No code or beads were edited.** All file/line citations above are from live source read this session.

---

## [a37ebb1642d581fc0] Spec: RL/eval export — sessions as labeled trajectories

I have complete ground truth. Delivering the spec.

---

# Spec: RL/eval export — sessions as labeled trajectories

**Parent bead exists:** `polylogue-fs1.5` — *"Export: Atropos/eval JSONL downstream of the canonical archive"* (open, under epic `fs1`). This spec expands it into implementable children. Sibling `fs1.4` (forensics report) shares the read substrate; `fs1.6` (sovereign loop demo) is the downstream consumer. The design mandate on fs1.5 is explicit and load-bearing: **downstream projection only — canonical archive → eval JSONL, not a bespoke snapshot→export parser**, implemented as a *render/export profile over the read substrate*, with **selection = a normal query** so any archive slice becomes an eval set.

## Ground-truth primitives (verified in source)

- **`actions` VIEW** (`storage/sqlite/archive_tiers/index.py`, index.db) — the outcome lane. Columns: `session_id, message_id, tool_use_block_id, tool_name, semantic_type, tool_command, tool_path, tool_input, output_text, is_error, exit_code, tool_result_block_id`. `exit_code`/`is_error` are provider-reported (`NULL` = unknown, never prose-guessed). **This is the structural reward source.**
- **`session_commits`** (index.db) — `session_id, commit_sha, repo_id, detection_type ∈ {time_window,file_overlap,explicit_ref,origin_reported}, method, confidence, evidence_json, created_at_ms`. **The commit SHA is the checkout-able tree state for a verifiable task.**
- **`assertions`** (user.db, durable) — unified table; `AssertionKind.CORRECTION` rows carry `target_ref` (an `ObjectRef`), `evidence_refs_json`, `confidence`, `body_text`. **Corrections are session-scoped** (`record_correction(session_id, kind, payload)` keys by canonical session id, targeting a derived insight) — per-turn anchoring only exists when the payload/evidence_refs carry a `message:`/`block:` ref. This is the **human-preference label**, and it is coarse by default. `AssertionKind.PROMPT_EVAL` and `TRANSFORM_CANDIDATE` **already exist** in the enum — the mined-task/candidate store needs no new kind.
- **`MaterialOrigin`** (`core/enums.py`): `human_authored, assistant_authored, operator_command, runtime_protocol, tool_result, …` — the authoredness axis that segments prompt-prefix vs. scored assistant turn vs. protocol noise.

Non-negotiable per fs1.5 design: **VERIFY the current `NousResearch/atropos` trajectory schema before freezing field names, and round-trip output through their `jsonl2html.py` as the acceptance check.** Treat field names below as a proposal gated on that verification.

---

## (1) Export schema + `polylogue eval export` surface

### Surface — two entry points, both thin adapters over the read substrate

The CLI is query-first with a strict command floor (#1842). Reconcile the two shapes the task names:

- **Primary: a query action verb `export`.** `polylogue find "<query>" then export --format atropos|sft|jsonl [--out PATH]`. Fits the floor (signalled intent via `find`), lets any slice (`repo:x outcome:pass since:30d`) become an eval set exactly as fs1.5 demands. Register as a query action alongside `read`/`analyze`/`mark` (`cli/query_verbs.py` / `query_actions.py`); new Click params go **last** (positional-shift gotcha).
- **Convenience: a top-level `eval` group** (modeled on `commands/demo.py`'s `@click.group("demo")`), for the RL operations that are *not* per-slice lowering: `polylogue eval mine …` (verifiable-task mining), `polylogue eval reward-model train|score …`. `polylogue eval export …` becomes a thin alias that dispatches the same lowering as the query verb. Register the group in `click_app.py` and add `CommandSpec` entries; run `devtools render devtools-reference` + `render all --check`.

Implementation home: an **export profile** parallel to the read-view/read-package registry (`cli/read_views/`, `read_view_registry.py`) — descriptor-driven, not a script silo. The Atropos lowering is a `RenderProfile` that consumes composed logical sessions (`get_logical_session`, so forks/resumes are recomposed, not duplicated — critical given the lineage-duplication finding #2467).

### Canonical eval record (provider-neutral, one line = one trajectory)

Emit an internal `EvalTrajectory` first, then lower to each `--format`:

```
EvalTrajectory {
  trajectory_id: session_id (logical/composed id)
  origin, repo, model, created_at
  turns: [ Turn { role, material_origin, text_or_blocks,
                  actions: [ {tool_name, tool_command, exit_code, is_error} ],
                  turn_reward: float|null } ]
  outcome: { terminal_exit_code|null, any_error: bool,
             session_commits: [{sha, confidence, detection_type}],
             ci_pass: bool|null }               # see mining §2
  reward: float                                  # session-level scalar
  labels: { has_correction: bool, correction_refs:[...],
            pathologies:[...], judgment: str|null }
  provenance: { source_refs:[...], fidelity: "messages exact; spans …" }  # reuse fs1.3
}
```

### Atropos lowering (`--format atropos`, gated on schema verification)

```
{ "messages": [ {"role":"user"|"assistant"|"tool", "content": ...} ],   # from turns, material_origin→role
  "scores":   [ per-assistant-turn reward ]  OR  "reward": <scalar>,     # match live atropos field
  "metadata": { trajectory_id, origin, repo, commit_sha, ci_pass,
                has_correction, pathologies, source_refs } }
```
`runtime_protocol`/`generated_context_pack` turns are dropped from `messages` (protocol noise), preserved in `metadata` counts. `--format sft` emits only accepted, correction-free, `ci_pass` trajectories as prompt→completion pairs; `--format jsonl` is the raw canonical record for debugging.

**No new DB tier for export itself** — it is a pure read projection over `actions` ⋈ `session_commits` ⋈ `assertions`.

---

## (2) Reward / mining algorithms (pseudocode)

### A. Session/turn reward from `exit_code` + correction presence

```
def session_reward(session):
    acts = actions_for(session.id)                      # actions VIEW rows
    terminal = last_action_with_exit_code(acts)         # by block position
    r_struct = ( +1.0 if terminal and terminal.exit_code == 0
                 -1.0 if terminal and (terminal.exit_code != 0 or terminal.is_error)
                  0.0 otherwise )                        # NULL exit_code = abstain, not penalty
    err_frac = count(a.is_error) / max(1, count(a with outcome))
    r_struct -= 0.5 * err_frac                          # dampen thrash even on eventual success

    corr = corrections_targeting(session.id)            # AssertionKind.CORRECTION (user.db)
    r_human = -1.0 * sum(c.confidence or 1.0 for c in corr)   # any operator flag => penalty
    patho   = -0.5 * count(pathology_assertions(session.id))

    return clip(w1*r_struct + w2*r_human + w3*patho, -1, +1)

def turn_reward(turn):                                  # only when granularity available
    # per-turn only if a CORRECTION/PATHOLOGY assertion carries a message:/block: evidence_ref
    if any(ref anchors turn.message_id): return -1.0
    if turn has verify-action with exit_code==0:        return +1.0
    return None                                          # unlabeled => excluded from turn scoring
```

### B. Verifiable-reward task mining (the command IS the reward)

```
VERIFY_PATTERNS = [pytest, "cargo test", "devtools verify", "npm test",
                   "make check", "go test", ...]        # curated, extensible

def mine_verifiable_tasks(session):
    for a in actions_for(session.id):
        if is_shell(a) and matches(a.tool_command, VERIFY_PATTERNS) and a.exit_code == 0:
            commits = session_commits(session.id)        # tree state
            prompt  = nearest_preceding_human_turn(a)    # material_origin=human_authored
            yield PromptEvalTask {
              repo, verify_cmd = a.tool_command,
              start_sha  = parent_of(best(commits)),     # pre-change tree
              answer_sha = best(commits).sha,            # solution tree
              prompt, expected_exit = 0,
              confidence = best(commits).confidence }     # low if time_window-only
```

- Persist mined tasks as **`AssertionKind.PROMPT_EVAL`** assertions (user.db, TEXT kind → **no schema bump**), `value_json` = task, `evidence_refs_json` = the action block + commit refs, `context_policy` default `{inject:false}`.
- Honesty gate: a task is only *replayable* (not just *mined*) when `session_commits.detection_type ∈ {explicit_ref, file_overlap}` with confidence above threshold — `time_window`-only commits mine as **candidate** tasks, flagged unreproducible.

### C. Reward-model-from-corrections (tiny local scorer)

```
def features(session):                                  # all from actions + material_origin + insights
    return [ n_tool_calls, n_errors, err_frac, max_exit_code,
             n_shell, n_edit, distinct_tools, n_turns,
             frac_assistant_authored, frac_runtime_protocol,
             session_duration_s, mean_tool_latency, n_pathologies ]

def label(session):  return 1 if corrections_targeting(session.id) else 0   # "operator would flag"

def train_reward_model(sessions):
    X,y = zip(*[(features(s), label(s)) for s in sessions])
    groups = [s.logical_root for s in sessions]         # group by lineage to prevent leakage
    Xtr,Xte,ytr,yte = grouped_split(X,y,groups, holdout=0.25)
    model = LogisticRegression()                        # pure numpy; no native dep
    model.fit(standardize(Xtr), ytr)
    auc = roc_auc(yte, model.predict_proba(Xte))
    return model, Report{ auc, base_rate=mean(y), n_pos=sum(y), n=len(y) }
```
Report **AUC alongside base rate and positive count** — corrections are rare and session-scoped, so a high-AUC claim on a dozen positives is noise. Persist weights as an artifact under `.local/eval/reward_model.json` (untracked); the scorer is `polylogue eval reward-model score <query>`.

---

## (3) Migration

- **Export lane: zero migration.** Pure read over the existing `actions` view, `session_commits`, and `assertions` — all already present.
- **Mined tasks / candidates: zero user.db schema bump.** Land as `AssertionKind.PROMPT_EVAL` / `TRANSFORM_CANDIDATE` rows; `assertions.kind` is `TEXT` and both enum values already exist. (If a *new* kind were added it would need `render openapi` + `render cli-output-schemas` regen — avoided here.)
- **Reward-model weights:** artifact file under `.local/eval/` (disposable/untracked), not a tier.
- **Optional (deferred) fast-path:** a materialized `trajectory_rewards` table for large-corpus mining is a **derived-tier (index.db) additive DDL** change — edit canonical DDL + bump `index.db` schema (currently 24), **no migration chain**, rebuild via `polylogue ops reset --index && polylogued run`, and **never** a `_upgrade` helper (`devtools lab policy schema-versioning` rejects it). Prefer view-time computation first; only materialize if profiling shows the mining query is hot.

---

## (4) Test strategy

- **Golden lowering (unit):** `SessionBuilder` fixtures with known exit codes / corrections → assert exact `EvalTrajectory` and Atropos JSONL bytes. Use `frozen_clock`; DBs under `/realm/tmp/polylogue-pytest`.
- **Atropos round-trip (acceptance, fs1.5 mandate):** export → feed through Nous `jsonl2html.py` → assert it renders without error. Gate behind schema-verification note recorded on the bead. Vendor a pinned copy or fixture-mock if network-free CI requires it (cloud lane forbids real corpora).
- **Reward invariants (property, Hypothesis):** `NULL exit_code ⇒ abstain (never penalty)`; `any correction ⇒ session_reward < correction-free counterpart`; reward monotonic in `err_frac`; `clip` bounds hold.
- **Mining precision:** synthetic sessions with `pytest … exit 0` action + `explicit_ref` commit ⇒ exactly one replayable task; `time_window`-only commit ⇒ candidate (unreproducible) flag. Assert no task mined when `exit_code ≠ 0`.
- **Reward-model:** deterministic-seed train on a fixture corpus; assert AUC computed, grouped split prevents lineage leakage (a fork of a labeled session cannot straddle train/test), report includes base rate.
- **Composition/dedup:** export of a session with a fork asserts the **composed** trajectory (no replayed-prefix duplication — guards #2467).
- **Surface:** CLI help snapshot (`__snapshots__`), `EXPECTED_TOOL_NAMES`/tool-contract update if an MCP `eval_export` tool is added, `devtools render all --check` (grep `out of sync`, don't trust tail).
- Run via `devtools test <files>` / `-k`, never blanket directories. `render topology-projection` + `topology-status` after adding modules.

---

## (5) Bead breakdown (children of `fs1.5`, with acceptance)

1. **fs1.5.1 — `EvalTrajectory` model + Atropos lowering (read profile).** AC: canonical record composes from logical session + `actions` + `session_commits` + `assertions`; `--format atropos|sft|jsonl` lower deterministically; forks composed not duplicated; golden-bytes tests pass.
2. **fs1.5.2 — Atropos schema verification + `jsonl2html.py` round-trip.** AC: current `NousResearch/atropos` trajectory schema confirmed and the verification note recorded on the bead; field names frozen against it; round-trip renders. (Blocks freezing 5.1's Atropos field names.)
3. **fs1.5.3 — `export` query verb + `eval` group wiring.** AC: `find "<q>" then export --format atropos` and `polylogue eval export` both produce identical output; params added last; help snapshots + `render all --check` green; devtools-reference regenerated.
4. **fs1.5.4 — Reward function (exit_code + correction + pathology).** AC: `session_reward`/`turn_reward` implemented per pseudocode; `NULL` exit_code abstains; property tests for monotonicity/bounds/correction-penalty pass; rewards attach to `EvalTrajectory`.
5. **fs1.5.5 — Verifiable-reward task mining → `PROMPT_EVAL` assertions.** AC: mines `{repo, verify_cmd, start_sha, answer_sha, prompt}` only from `exit_code==0` verify actions; replayable vs candidate split keyed by `session_commits.detection_type`/confidence; persisted as `PROMPT_EVAL` (no schema bump); `polylogue eval mine` lists them.
6. **fs1.5.6 — Reward-model-from-corrections (train/score + AUC report).** AC: features from `actions`+`material_origin`+insights; grouped-by-lineage split; pure-Python logistic model; report emits AUC, base rate, positive count, n; weights persisted under `.local/eval/`; `eval reward-model score` runs.
7. **fs1.5.7 (optional) — SFT fine-tuning corpus profile.** AC: `--format sft` emits only accepted + correction-free + `ci_pass` trajectories as prompt→completion; judgment/outcome gate documented; synthetic-corpus reproducibility so the set is shareable without private data.
8. **fs1.5.8 (deferred) — materialized `trajectory_rewards` (index.db derived).** AC: only if mining profiles hot; additive DDL + schema bump + rebuild plan, no upgrade helper; `schema-versioning` policy green.

Sequence: 5.2 gates 5.1's field-freeze; 5.1→5.3→5.4→5.5/5.6 parallel; 5.7/5.8 opportunistic.

---

## (6) Top-3 risks

1. **Correction labels are session-scoped and rare** — `record_correction` keys by `session_id`, so per-turn credit assignment is unavailable unless the payload carries a `message:`/`block:` ref, and positives are sparse. The reward-model degenerates to session-level with severe class imbalance; **mitigate** by reporting AUC *with* base rate + positive count, treating it as a weak-signal prototype, and defining turn-level labels only where an evidence_ref anchors a turn. Overclaiming here would be the classic "claim the code doesn't support."
2. **Verifiable-task reproducibility hinges on commit-attribution confidence** — `session_commits` includes `time_window` (0.1 confidence) and `file_overlap` heuristic edges; a mined "the verify command IS the reward" task is only re-runnable if `start_sha` truly is the pre-change tree. **Mitigate** by gating replayable tasks to `explicit_ref`/high-confidence `file_overlap`, emitting the rest as flagged candidates, never asserting reproducibility a `time_window` edge can't support.
3. **Atropos schema drift** — fs1.5 warns Hermes/Nous ship ~1,700 commits per minor; a frozen field guess rots. **Mitigate** by making 5.2 (live schema verification + `jsonl2html.py` round-trip) a hard gate before 5.1 freezes Atropos field names, and by lowering from a stable internal `EvalTrajectory` so only the thin Atropos adapter changes when upstream moves.

**Note:** no code or beads were modified. `fs1.5` already scopes this; the eight children above are proposals for you to create, not created.

---

## [a599387e70b53b827] SPEC — The Attention/Triage Frontier

I have the grounding I need. Delivering the spec.

---

# SPEC — The Attention/Triage Frontier

Turns the three detectors (`find_abandoned`/`find_stuck`/`find_resume_candidates`) into a single context-free inbox that *empties*. Grounded in `polylogue/insights/resume.py`, `mcp/server_insight_tools.py`, `core/enums.py`, `storage/sqlite/archive_tiers/{index,user}.py`. Bead alignment: 9l5 (register as a measure), 9e5.1 (assertion-adoption — this is the surface that *creates* judged assertions).

## Core architectural decision (read first)

`find_resume_candidates` (resume.py:633) is cwd/repo-coupled: it *requires* `repo_path`, groups profiles by `logical_session_id`, and scores `0.35·recency + 0.25·file_overlap + 0.15·cwd + 0.15·terminal + 0.10·workflow`. The frontier is its **inversion**: drop file/cwd terms, keep the intrinsic-worth terms, add blocker/question/durable-prior, and rank *all* logical sessions with no operator context.

**Split time-invariant vs time-dependent.** Materialize only the time-invariant worth into `session_profiles` (index.db is rebuildable but not re-run daily). Apply `staleness_urgency(age)` at **read time** in the frontier query. This is the keystone — a materialized inverted-U over age goes stale the moment it's written. `worth_reviewing_score` column = invariant composite; frontier surface = `score × staleness_urgency(now − last_message_at)`.

---

## (1) Schema + tier

### 1a. `session_profiles` (index.db, currently v24 → **v25**; rebuildable derived tier — no numbered migration)

Add to the canonical DDL at `storage/sqlite/archive_tiers/index.py:799`:

```sql
worth_reviewing_score        REAL NOT NULL DEFAULT 0
    CHECK(worth_reviewing_score BETWEEN 0 AND 1),
worth_reviewing_breakdown_json TEXT NOT NULL DEFAULT '{{}}',
worth_reviewing_gate         TEXT NOT NULL DEFAULT ''   -- '', 'disposable', 'superseded', 'in_flight'
```
```sql
CREATE INDEX IF NOT EXISTS idx_session_profiles_worth
ON session_profiles(worth_reviewing_score DESC);
```

- `worth_reviewing_score` = **time-invariant** composite (pre-staleness), computed in the insights convergence stage (`daemon/convergence_stages.py:make_insights_stage`, alongside existing profile materialization).
- `worth_reviewing_breakdown_json` = decomposed sub-scores `{terminal, blocker, question, durable_prior}` + gate reason, so every surfaced number is auditable (matches resume.py's existing `score_breakdown` discipline).
- `worth_reviewing_gate` non-empty ⇒ demoted (see risk #2 — *demote, not hard-zero*, except `disposable`).
- Bump `INDEX_SCHEMA_VERSION = 24 → 25`; edit the rebuild plan; **rerun topology projection** (new scorer module → `devtools render topology-projection topology-status`).

### 1b. `AssertionKind.TRIAGED` (user.db, v4 — **no schema bump**)

Add to `core/enums.py:399`:
```python
TRIAGED = "triaged"
```
The `assertions.kind` column is `TEXT` with no CHECK — the vocabulary grows without a user-tier migration (per CLAUDE.md). A triage row:

| column | value |
|---|---|
| `kind` | `triaged` |
| `target_ref` | `session:<logical_session_id>` (registered `ObjectRef` kind `session`, core/refs.py:43) |
| `status` | `active` (re-triage → old row `superseded`, via `supersedes_json`) |
| `value_json` | `{"verdict": "resumed" \| "wont_resume" \| "archived" \| "snoozed", "wake_at_ms": <int\|null>, "reason": <str\|null>}` |
| `author_kind` | `user` |
| `evidence_refs_json` | `[<EvidenceRef of the open-question anchor>]` when snoozing a question |

Verdict lives in `value_json` (not `AssertionStatus`) because the four triage verdicts are frontier-domain, orthogonal to the generic lifecycle enum. `snoozed` + `wake_at_ms` = the re-surface timer.

**Gotchas triggered (from memory):** new `AssertionKind` is embedded in `render openapi` + `render cli-output-schemas` → regenerate both; `user_audit` has an every-kind invariant → add a surface entry; `render all --check` prints `sync OK` yet exits 1 → grep `out of sync`.

---

## (2) Scoring function + "today's frontier" query

### Time-invariant worth (materialized, per logical session)

```
# All inputs from SessionProfileInsight (collapsed across lineage members,
# same representative-pick as resume.py:677: max last_message_at).
inference  = profile.inference        # terminal_state, *_confidence, workflow_shape
enrichment = profile.enrichment       # blockers, is_goal_session, goal_outcome, support_level
evidence   = profile.evidence         # message_count, word_count, is_continuation, logical_session_id

# --- gates (multiplicative 0/1, but record WHICH gate fired) ---
disposable = (workflow_shape == "chat" and terminal_state == "clean_finish")
in_flight  = last_message_at within QUIET_WINDOW (e.g. 30m) OR hot-file still appending
superseded = a descendant in lineage continued past this session's branch_point
             (session_links: this logical id is a proper prefix of a longer live child)

if disposable:           return 0.0, gate="disposable"        # hard zero
if in_flight or superseded:
    gate = "in_flight"|"superseded"                            # DEMOTE, don't zero (risk #2)
    demotion = 0.15
else:
    gate, demotion = "", 1.0

# --- decomposable value factors, each in [0,1], weighted by their own confidence ---
terminal      = _terminal_weight(terminal_state) * terminal_state_confidence      # reuse resume.py:252
blocker       = min(1.0, 0.5 * len(enrichment.blockers)) * band(enrichment.support_level)
question      = (1.0 if terminal_state == "question_left"
                 else 0.6 if last_authored_message.role == "user" and unanswered
                 else 0.0) * terminal_state_confidence
durable_prior = 0.6 * _workflow_weight(workflow_shape)                             # reuse resume.py:263
              + 0.2 * saturate(word_count / 4000)
              + 0.2 * (1.0 if enrichment.is_goal_session and goal_outcome != "achieved" else 0.0)

value = 0.40*terminal + 0.25*blocker + 0.20*question + 0.15*durable_prior
worth_reviewing_score = round(demotion * value, 6)   # STORED; staleness applied at read
breakdown = {terminal, blocker, question, durable_prior, gate}
```

`_terminal_weight` and `_workflow_weight` already exist in resume.py — **refactor both into a shared `insights/frontier.py` scorer** consumed by *both* the frontier and `find_resume_candidates` (bead 6).

### Read-time staleness (inverted-U) + the frontier query

```
def staleness_urgency(age_hours):
    # inverted-U: fresh work is still in-flight (defer); ancient work is likely dead.
    # peak around ~3–10 days; log-gaussian over age.
    PEAK_H, WIDTH = 120.0, 1.1
    if age_hours <= 0: return 0.0
    return exp(-((log(age_hours) - log(PEAK_H))**2) / (2*WIDTH**2))
```

**"Today's frontier"** (cross-tier: index.db `session_profiles` ⋈ attached user.db `assertions`; the repository already attaches tiers for DSL reads):

```sql
SELECT sp.logical_session_id, sp.worth_reviewing_score, sp.worth_reviewing_breakdown_json,
       sp.worth_reviewing_gate, sp.terminal_state, sp.terminal_state_confidence,
       sp.last_message_at, sp.title
FROM session_profiles sp
WHERE sp.worth_reviewing_score > 0
  -- context-free: NO repo/cwd/file filter (the inversion of find_resume_candidates)
  AND NOT EXISTS (
    SELECT 1 FROM user.assertions a
    WHERE a.kind = 'triaged'
      AND a.status = 'active'
      AND a.target_ref = 'session:' || sp.logical_session_id
      AND ( json_extract(a.value_json,'$.verdict') IN ('resumed','wont_resume','archived')
         OR ( json_extract(a.value_json,'$.verdict') = 'snoozed'
              AND json_extract(a.value_json,'$.wake_at_ms') > :now_ms ) )
  )
-- collapse lineage to representative row per logical id upstream (or GROUP BY logical_session_id)
ORDER BY sp.worth_reviewing_score DESC
LIMIT :prefetch;   -- e.g. 3×limit
```

Then in Python, apply read-time staleness and re-sort (keeps the inverted-U out of stored state):

```
final = worth_reviewing_score * staleness_urgency((now - last_message_at).hours)
frontier = sorted(rows, key=lambda r: -r.final)[:limit]
```

`WHERE NOT EXISTS ... triaged` is what makes the inbox **empty**: `resumed/wont_resume/archived` remove permanently; `snoozed` removes until `wake_at_ms`, then the row re-enters automatically.

### Clustering into cards (blocker-centric)

```
# candidates = frontier rows with non-empty blockers OR question text
for each candidate: anchor = EvidenceRef(session_id, message_id_of_question_or_last_user_turn)
if embeddings enabled (poly.config.embedding_enabled):
    # greedy leader clustering over cosine of session embeddings (embeddings.db vec0),
    # reuse poly.neighbor_candidates(session_id) — the same path find_similar_sessions uses.
    cluster candidates s.t. cosine(a,b) > 0.82 → connected components
else:
    # fallback: group by (thread_id) then (dominant_repo) — no embeddings needed
card = { blocker_text, member_logical_ids[], count, open_question_text, anchor_ref,
         min_confidence, "N sessions blocked on the same thing" if count>1 }
```
Each card surfaces the **open-question text** verbatim with a citation-anchor (`EvidenceRef` → `session_id::message_id`), never opaque prose.

### Frontier honesty

Every row/card carries `confidence = terminal_state_confidence × band(enrichment.support_level)`; render `confidence < 0.4` explicitly as **"guess"** (the `ConfidenceBand.WEAK` band already exists in the payloads). Low-confidence rows are shown but visually demoted — the frontier admits when it's guessing rather than silently ranking noise.

---

## (3) Migration

Derived tier → **no migration chain** (per CLAUDE.md schema-regime rule; `devtools lab policy schema-versioning` rejects upgrade helpers for index.db):

1. **index.db**: edit canonical DDL (columns + index), bump `INDEX_SCHEMA_VERSION 24→25`, update the rebuild plan. Deploy = `polylogue ops reset --index && polylogued run` (full re-materialize computes the score). Incrementally, bumping `SESSION_INSIGHT_MATERIALIZER_VERSION` re-materializes stale profiles.
2. **user.db**: enum addition only — **additive, no migration, no version bump** (TEXT column). Regenerate `render openapi` + `render cli-output-schemas`; add the `user_audit` surface entry.
3. **Writes**: score written by the daemon insights convergence stage (sole SQLite writer). Triage assertions written through the user-tier write path (`user_write.py`) — surfaced as an MCP write-role tool + CLI `mark`-family verb.

---

## (4) Test strategy

- **Unit (pure scorer, `insights/frontier.py`)** — table-driven synthetic profiles: monotonicity (higher `_terminal_weight` ⇒ higher score); gate semantics (`disposable`⇒0, `in_flight`/`superseded`⇒0.15× demotion, not 0); `breakdown` sub-scores reconstruct the composite; each factor weighted by its confidence.
- **Property (Hypothesis)** — `score ∈ [0,1]`; `disposable ⟺ score==0`; `staleness_urgency` unimodal (fresh < peak, ancient < peak); breakdown keys stable across inputs.
- **Frontier query** — seeded corpus: add a `triaged`/`resumed` assertion ⇒ session leaves frontier; `snoozed` with future `wake_at_ms` ⇒ absent; past `wake_at_ms` ⇒ re-surfaces. Drive `wake_at_ms`/age with the **`frozen_clock`** fixture (the `verify-test-clock-hygiene` lint blocks raw `datetime.now`).
- **Clustering** — synthetic near-duplicate blocker texts cluster (count>1, "N blocked on same thing"); distinct blockers stay separate; embeddings-disabled path falls back to thread/repo grouping.
- **Honesty snapshot** — `confidence<0.4` row carries the guess flag; anchor `EvidenceRef` resolves to a real message.
- **MCP contract** — new `frontier` tool ⇒ update `EXPECTED_TOOL_NAMES` + tool contract, or `test_tool_discovery` fails.
- **Integration** — daemon materialize → frontier read → triage write → frontier read (session gone), end-to-end.
- Inner loop: `devtools test <file>` / `-k frontier` (never blanket-run directories).

---

## (5) Bead breakdown (6, each with AC)

1. **`feat: materialize worth_reviewing_score in session_profiles` (index.db v25)** — add 3 columns + index; extract `_terminal_weight`/`_workflow_weight` into `insights/frontier.py` scorer; compute invariant score in the insights convergence stage. **AC:** seeded profiles carry `score`+`breakdown`+`gate`; `disposable⇒0`, `in_flight`/`superseded` demoted (not zeroed); INDEX_SCHEMA_VERSION=25 + rebuild plan updated; topology projection regenerated; `find_resume_candidates` numerics unchanged on the cwd path (shared scorer).
2. **`feat: AssertionKind.triaged with verdict + wake_at`** — enum add; `value_json` verdict/wake_at shape; write path + re-triage-supersedes. **AC:** triage writes an `active` assertion `target_ref=session:<id>`; re-triage supersedes prior; `render openapi`+`cli-output-schemas` regenerated; `user_audit` surface entry present; **no** user-tier version bump.
3. **`feat: context-free frontier query (NOT EXISTS triaged, snooze-aware)` + surfaces** — frontier read (cross-tier join), read-time staleness, CLI verb + MCP `frontier` tool. **AC:** ranked non-triaged rows; snooze respected/expiring under `frozen_clock`; `EXPECTED_TOOL_NAMES`+contract updated; inverts (never requires) repo/cwd.
4. **`feat: cluster frontier loose ends into blocker/thread cards`** — embedding-optional clustering; open-question text + `EvidenceRef` anchor. **AC:** near-dup blockers cluster with "N blocked on same thing"; card surfaces verbatim question + resolvable anchor; embeddings-off falls back to thread/repo.
5. **`feat: frontier honesty rendering (confidence bands)`** — per-row confidence, guess flag for weak support. **AC:** `confidence<0.4` flagged as guess; snapshot pins render; gate reason shown (`superseded`/`in_flight` visible, not silently dropped).
6. **`refactor: unify resume + frontier scoring core`** — single `insights/frontier.py` consumed by both `find_resume_candidates` and frontier. **AC:** one scorer module; no duplicated `_terminal_weight`/`_workflow_weight`; resume candidates byte-identical on the cwd-coupled path (regression snapshot).

Register the score as a **measure under 9l5.7** (construct note: *blocker-presence operationalizes "unresolved work", NOT "importance"; terminal_state from provider structure, not prose*).

---

## (6) Top-3 risks

1. **Staleness materialization staleness (design keystone).** A materialized inverted-U over age is wrong the day after it's written; index.db isn't rebuilt daily. **Mitigation (baked into this spec):** store only time-invariant worth; apply `staleness_urgency(now − last_message_at)` at query time. If a future author folds staleness into the stored column, the frontier silently mis-ranks — enforce with a test that the stored score is age-independent.
2. **Gates hide instead of resolve (inbox-empties-by-cheating).** `superseded`/`in_flight` rest on heuristic lineage + hot-file signals; over-aggressive hard-zeroing drops real loose ends and the inbox "empties" by concealment. **Mitigation:** hard-zero only `disposable` (chat+clean_finish); `superseded`/`in_flight` **demote** (0.15×) into a visible gated bucket with the reason rendered — honest emptiness requires triage verdicts, not gate suppression.
3. **Enrichment noise dominates ranking.** `blockers`/`open_question`/`intent` come from heuristic enrichment carrying `ConfidenceBand.WEAK` over much of the 16.7k corpus; a blocker-weighted frontier amplifies that noise. **Mitigation:** weight every factor by its own `*_confidence`/`support_level` (already in the payloads), render `<0.4` as guess, and register under 9l5.7 with explicit construct metadata so partial-provenance can't masquerade as a ranked finding.

**Not re-derived** (convergent across the 38 prior agents, taken as given): three-level content tree, `material_origin` axis, contentless FTS5, lineage-tail composition, and that the daemon is the sole writer.

---

## [a717d18b4210fe8a7] Standing-Query Subscription Substrate — Implementation Spec

I have full grounding. Delivering the spec.

---

# Standing-Query Subscription Substrate — Implementation Spec

Ties query-objects `StandingQueryStage` → notification-policy → `daemon_events`. Grounded in `daemon/events.py`, `daemon/convergence.py` (`ConvergenceStage`, `converge_sessions`), `daemon/convergence_stages.py` (`false_means_pending`, hot-file quiet deferral, `make_default_convergence_stages`), `storage/sqlite/archive_tiers/ops.py` (ops tier DDL), `storage/sqlite/archive_tiers/user_write.py` (`upsert_saved_view` / `AssertionKind.SAVED_QUERY`), `docs/plans/query-pipeline-substrate.md`.

## Core design decision (grounds everything below)

Dedup is by **stable finding-identity**, not by event. Identity in this archive is computed, never stored (`sessions.session_id`, `messages.message_id`, `blocks.block_id` are generated columns; writes are content-hash-idempotent). A re-ingest / full-replace of an already-seen session yields the *same* finding_id. Therefore comparing finding-identity *sets* — not event streams — makes re-notification storms after backfill structurally impossible: a re-ingested old row is already in membership, so it is not an `appeared` delta regardless of how many `session.updated` events fire. Everything else (baseline-then-notify, quiet window, epoch scoping) is defense-in-depth on top of this primitive.

**Subscription = a SAVED_QUERY assertion whose payload carries a `notify_on` block.** No new `AssertionKind` (avoids the `render openapi` / `render cli-output-schemas` enum-embedding fallout — see CLAUDE.md gotcha). `notify_on` is additive JSON in the existing `value` TEXT column of `upsert_saved_view`; no user-tier schema change, no migration.

---

## (1) Schema / DDL + tier

**Durable (user.db, no schema change).** Extend the `SAVED_QUERY` assertion `value` payload:
```jsonc
{ "query": "sessions where origin:codex-session and cost_usd>5",   // existing
  "notify_on": {                                                     // NEW, additive
    "mode": "appeared" | "disappeared" | "count_crossed",
    "threshold": 10,          // count_crossed only
    "direction": "above"|"below"|"either",  // count_crossed only
    "enabled": true } }
```
A saved query is a *standing* query iff `notify_on.enabled` is truthy. Durable because it is operator intent.

**Disposable (ops.db).** All evaluation state is rebuildable — bump `OPS_SCHEMA_VERSION` 1→2, append to `OPS_DDL` in `archive_tiers/ops.py` (both `CREATE TABLE IF NOT EXISTS`, self-healing on every daemon open via `initialize_archive_database`):

```sql
CREATE TABLE IF NOT EXISTS standing_query_cursor (
    query_id            TEXT PRIMARY KEY,          -- saved_view id
    query_fingerprint   TEXT NOT NULL,             -- sha256(normalized query text + notify_on)
    dep_signature       TEXT NOT NULL,             -- JSON: affected-by tags (origins/units) from lowered AST
    last_event_id       INTEGER NOT NULL DEFAULT 0,-- daemon_events watermark consumed
    last_count          INTEGER NOT NULL DEFAULT 0,
    last_count_side     TEXT,                       -- 'above'|'below' hysteresis for count_crossed
    baseline_established INTEGER NOT NULL DEFAULT 0 CHECK(baseline_established IN (0,1)),
    last_evaluated_ms   INTEGER,
    updated_at_ms       INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS standing_query_membership (
    query_id      TEXT NOT NULL,
    unit          TEXT NOT NULL,          -- 'sessions'|'messages'|'actions'|... terminal unit
    finding_id    TEXT NOT NULL,          -- stable identity of the result row
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms  INTEGER NOT NULL,
    PRIMARY KEY (query_id, finding_id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_standing_query_membership_q ON standing_query_membership(query_id);
```

**Notices ride the existing `daemon_events` ledger** (ops.db, unchanged DDL): `kind = "notice.appeared" | "notice.disappeared" | "notice.count_crossed"`, `operation_id = query_id`, payload = `{query_id, name, unit, finding_ids:[...], count, threshold, transition}`. New granular kind constants added to `events.py` alongside `GRANULAR_EVENT_KINDS`; the SSE reader subscribes via `?kinds=notice.appeared,...`.

**Self-trigger firewall (three independent layers):** (a) the change-detector reads events only via a `kinds` whitelist `{session.appended, session.updated, message.appended, insight.updated}` — `WHERE kind IN (...)` structurally excludes `notice*`; (b) defensive `AND kind NOT LIKE 'notice%'` on that scan; (c) standing queries lower over archive units (sessions/messages/…), never over `daemon_events`, so an emitted notice can never enter any result set.

---

## (2) Change-detector algorithm (pseudocode)

Invoked once per converger quiet window (see §integration), after session-scoped stages drain.

```
run_quiet_window(budget_n):                     # StandingQueryStage entry
  subs      = load_saved_views_with(notify_on.enabled)          # user.db
  new_events = query_events_since(min(sub.last_event_id for subs),
                 kinds=[SESSION_APPENDED, SESSION_UPDATED,
                        MESSAGE_APPENDED, INSIGHT_UPDATED])       # NEVER notice*
  touched   = derive_change_signature(new_events)   # {origins, session_ids, units}
  max_event = get_latest_event_id()

  affected = [s for s in subs
              if fingerprint_changed(s)                          # re-baseline needed
              or dep_signature(s) ∩ touched ≠ ∅]                 # scoped epoch: skip untouched

  evaluated = 0
  for s in stable_order(affected):
      if evaluated >= budget_n:
          record_convergence_debt(stage='standing_query',
              target_type='standing_query', target_id=s.id, status='deferred')
          continue                                               # false_means_pending backlog
      if depends_on_hot_source(s, touched):                      # quiet-window gate
          record_convergence_debt(..., status='deferred'); continue   # flap guard
      evaluate_one(s, max_event)
      evaluated += 1
  return evaluated == len(affected)   # False ⇒ remaining retried as debt (no failure)

evaluate_one(s, max_event):
  cur = execute_query(s.query)                     # lower→SQL, terminal unit
  cur_ids = { stable_finding_id(row, s.unit) for row in cur }
  prev_ids = membership_ids(s.id)                  # ops.db

  # BASELINE-THEN-NOTIFY: cold cursor or changed predicate ⇒ silent
  if s.baseline_established == 0 or fingerprint_changed(s):
      replace_membership(s.id, cur_ids)
      set_cursor(s.id, last_event_id=max_event, last_count=len(cur_ids),
                 baseline_established=1, fingerprint=fp(s)); return

  appeared    = cur_ids - prev_ids                 # identity-set delta (storm-proof)
  disappeared = prev_ids - cur_ids

  if s.mode == 'appeared'    and appeared:    emit_notice('notice.appeared',    s, appeared)
  if s.mode == 'disappeared' and disappeared: emit_notice('notice.disappeared', s, disappeared)
  if s.mode == 'count_crossed':
      side = 'above' if len(cur_ids) >= s.threshold else 'below'
      if side != s.last_count_side and matches_direction(side, s.direction):
          emit_notice('notice.count_crossed', s, count=len(cur_ids))
      set_side(s.id, side)

  replace_membership(s.id, cur_ids)                # touches first/last_seen_ms
  set_cursor(s.id, last_event_id=max_event, last_count=len(cur_ids))
```

`emit_notice` is idempotent per `(query_id, finding_id, transition)` within a window — a re-emit for an already-noticed appear is suppressed by the membership row already existing.

**Scoped predicate-fingerprint epoch** = the `dep_signature ∩ touched` gate. `dep_signature` is derived once (cached in cursor) from the lowered AST: the set of origins named/implied, the terminal unit, and whether it is FTS/semantic/structural. `touched` comes from decoding the events delta. A query whose dependency signature does not intersect the batch's touched signature is skipped entirely — no re-evaluation, no thrash — while its `last_event_id` is still advanced to `max_event` (so its cursor never lags and never re-scans that delta).

---

## (3) Migration

- **ops.db (disposable/derived tier — no numbered migration chain per Schema-regimes):** bump `OPS_SCHEMA_VERSION` 1→2; append the two `CREATE TABLE IF NOT EXISTS` blocks + index to `OPS_DDL`. Purely additive and idempotent; `initialize_archive_database` runs the DDL on every daemon open, so existing archives self-heal on next `polylogued run`. No backup manifest (that gate is durable-tier only). No `polylogue ops reset` required.
- **user.db (durable):** **no migration.** `notify_on` is additive JSON inside the existing `assertions.value` TEXT column; the closed-vocabulary risk (new `AssertionKind`) is avoided by reusing `SAVED_QUERY`. Old saved views without `notify_on` are inert (not standing).
- **Generated surfaces:** new `notice.*` kinds + any new CLI/MCP verb for standing-query CRUD require `devtools render all` (openapi, cli-output-schemas, topology projection for the new module — else `render all --check` fails; grep for `out of sync`, don't trust the tail line).

---

## (4) Test strategy

Protecting behavior/invariants, not diffs (per Operating Contract).

1. **no-storm-after-backfill (the load-bearing test).** Create standing query `mode=appeared` over origin X; ingest N sessions; drain quiet window → baseline established, **0 notices**. Full-replace re-ingest all N (same content-hash / same session_ids) → **0 `notice.appeared`** (identity already in membership). Then ingest 1 genuinely-new session matching → exactly **1** notice with that finding_id. Assert notice count, not just "no error".
2. **dedup-by-identity.** Two evaluation passes with an identical result set emit the delta once; a disappeared-then-reappeared finding emits `disappeared` then `appeared`, never a duplicate `appeared` while continuously present.
3. **baseline-then-notify.** First-ever evaluation of a new subscription is silent; changing `notify_on.mode` or query text bumps `query_fingerprint` → silent re-baseline (no notices), then transitions notify.
4. **scoped-epoch / no-thrash.** Subscription depends only on origin A; ingest into origin B → subscription is *not* re-evaluated (assert via a spy/counter on `execute_query`), yet its `last_event_id` advances (no cursor lag).
5. **self-trigger exclusion.** After emitting notices, run another quiet window → the `notice.*` events do not appear in the change delta and produce zero further notices. Assert the change-scan SQL never selects `kind LIKE 'notice%'`.
6. **count_crossed hysteresis.** Oscillating counts around the threshold emit exactly one notice per genuine crossing (side change), not per evaluation.
7. **false_means_pending backlog.** With budget=1 and 3 affected queries, one evaluates and two land in `convergence_debt` (status `deferred`, stage `standing_query`), retried next window — no `FAILED`, no error_count bump. Mirror the existing insights-deferral assertions.
8. **quiet-window gate.** A query whose dependent source is still hot-appending is deferred (debt), not evaluated mid-append. Use `frozen_clock` + synthetic mtimes (clock-hygiene lint).

Use `SessionBuilder` / `corpus_seeded_db`, `workspace_env`, `frozen_clock`. Run via `devtools test tests/unit/daemon/test_standing_query.py`.

---

## (5) Bead breakdown (6 issues)

Parent epic ties them; each has acceptance criteria (AC).

1. **`notify_on` on saved-query payload + subscription read model.** AC: `upsert_saved_view` accepts optional `notify_on`; a `load_standing_subscriptions()` reader returns only `enabled` ones; round-trips through user.db; old views inert. Tests: read/write round-trip, inert-old-view.
2. **ops.db v2 tables + cursor/membership store.** AC: `OPS_SCHEMA_VERSION=2`; two tables added idempotently; a `StandingQueryStore` with `replace_membership`, `membership_ids`, cursor get/set; self-heals on open. Tests: fresh + upgrade-in-place, membership CRUD.
3. **`notice.*` event kinds + emitters.** AC: three kinds + constants added to `events.py`; `emit_notice` idempotent per `(query_id, finding_id, transition)`; `query_events_since` whitelist excludes them; `render all` regenerated. Tests: emit + SSE `?kinds=` filtering, self-exclusion.
4. **Change-detector + `evaluate_one` (dedup, baseline-then-notify, count_crossed).** AC: identity-set delta; baseline silent; count hysteresis; fingerprint-change re-baseline. Tests: dedup-by-identity, baseline, count_crossed (T2/T3/T6).
5. **Scoped predicate-fingerprint epoch (dep_signature ∩ touched).** AC: `dep_signature` derived from lowered AST + cached; untouched queries skipped but cursor advanced; storm-free after backfill. Tests: no-storm-after-backfill, no-thrash (T1/T4/T5).
6. **`StandingQueryStage` quiet-window integration + `false_means_pending` debt.** AC: evaluated once per quiet window from the daemon loop after `converge_sessions`; budgeted; overflow + hot-dep queries → `convergence_debt` (deferred), retried; wired without breaking `make_default_convergence_stages`. Tests: backlog-deferral, quiet-window gate (T7/T8).

Ordering: 1→2→3 parallel-ish; 4 depends on 2+3; 5 depends on 4; 6 depends on 4+5.

---

## (6) Top-3 risks

1. **Backfill storm if the identity primitive is bypassed.** If anyone re-derives deltas from the event stream instead of the membership identity-set (e.g. "session.updated ⇒ appeared"), a single re-ingest of the ~8.8k logical / 16k physical corpus emits thousands of notices. Mitigation: identity-set comparison is the *only* delta source; baseline-then-notify on cold cursor; test T1 is a required gate. Reset of ops.db (disposable) must re-baseline silently, never replay.
2. **`dep_signature` under-approximation → missed notices (false-quiet).** If the AST-derived dependency tags miss a way a query can be affected (e.g. lineage/semantic predicates whose inputs changed in an unmodeled origin), the scoped-epoch gate skips a query that *should* fire. This fails *closed on silence*, which is worse than a spurious notice. Mitigation: derive `dep_signature` conservatively (unknown/complex predicate shapes ⇒ `depends_on=*`, always-evaluate); add an explicit `*` fallback for FTS/semantic/lineage units until per-unit signatures are proven; log skips at debug for auditability.
3. **Evaluation cost / SQLite-writer contention in the quiet window.** Standing queries execute full archive queries on the sole writer process; a heavy `near:`/semantic query per window can stall ingest convergence. Mitigation: strict per-window budget (`false_means_pending` overflow → debt, mirroring insights paging); mark evaluation `cpu_bound`-friendly read-only and keep writes to the tiny cursor/membership tables; reuse `_HOT_INSIGHT_QUIET_SECONDS` gating so evaluation only runs when sources are quiet. Cap membership set size per query (large result sets are a config error, not a subscription).

---

**Not built/edited** (spec only, per instructions): no code, no beads, no schema changes applied. All file paths above are absolute-resolvable under `/realm/project/polylogue/`.

---

## [a3cf0207524854172] Spec: FTS coherence & drift — metamorphic invariant, O(1) drift gauge, bounded repair

I have complete grounding. Here is the tight implementation spec.

---

# Spec: FTS coherence & drift — metamorphic invariant, O(1) drift gauge, bounded repair

## Grounding (real files)

| Concern | File | Load-bearing detail |
|---|---|---|
| Contentless FTS DDL + 3 triggers + ledger | `polylogue/storage/sqlite/archive_tiers/index.py:282–322` | `messages_fts` (`content=''`, `contentless_delete=1`, `tokenize='unicode61'`); triggers `messages_fts_ai/ad/au` keyed on `blocks.rowid`; `fts_freshness_state` table already has `source_rows/indexed_rows/missing_rows/excess_rows/duplicate_rows/detail` |
| Exact invariant model | `polylogue/storage/fts/fts_lifecycle.py:867–943` | `_trigger_invariant_sync` + `fts_invariant_snapshot_sync` — the canonical coherence algebra already exists |
| Coherence SQL | `polylogue/storage/fts/sql.py` | `FTS_INDEXABLE_MESSAGE_COUNT_SQL`, `FTS_INDEX_DOC_COUNT_SQL`, `insert_missing_message_rows_range_sql`, `excess_message_rows_sql` |
| Startup probe | `polylogue/daemon/fts_startup.py` | Trusts ledger first, else does `COUNT(*)` scans + bounded rebuild (`≤10_000` drift rows) else `convergence_debt` |
| Readiness projection | `polylogue/daemon/fts_status.py` | `fts_readiness_info(exact=False)` = ledger-backed; `exact=True` = full COUNT/LEFT-JOIN scan |
| Metrics | `polylogue/daemon/metrics.py:423–452, 1125–1147` | Emits only booleans: `polylogue_fts_trigger_present{trigger}`, `_triggers_all_present`, `polylogue_fts_freshness_ready{surface}` |
| Bounded repair | `polylogue/storage/fts/dangling_repair.py`, `fts_lifecycle.py:414–487` | `repair_stale_fts_rows`, `insert_missing_message_rows_batched_sync`, `delete_excess_message_rows_batched_sync` |

**The rowid identity is the coherence keystone:** `messages_fts.rowid == blocks.rowid`, and `messages_fts_docsize.id` is the FTS5 shadow of that rowid. Every coherence check is a rowid LEFT JOIN between `blocks WHERE search_text != ''` and `messages_fts_docsize`.

---

## 1. Schema / gauge surface

**Two tiers, durability-keyed (this is the correct convergent placement):**

- **Current-state ledger — reuse `fts_freshness_state` (index.db, derived).** It *already* carries the drift magnitudes; nothing to add. This is the O(1) single-row-per-surface read that the drift gauge and readiness probe both project from.
- **Continuous drift history — new `fts_drift_samples` in `ops.db` (disposable).** Time-series belongs beside the existing `cursor-lag samples` / `convergence_debt` in ops.db, *not* index.db. One row per scrape/quiet-tick:

```sql
CREATE TABLE IF NOT EXISTS fts_drift_samples (
    sampled_at_ms  INTEGER NOT NULL,
    surface        TEXT    NOT NULL,
    source_rows    INTEGER NOT NULL,
    indexed_rows   INTEGER NOT NULL,
    missing_rows   INTEGER NOT NULL,   -- indexable blocks with no FTS shadow (orphan source)
    excess_rows    INTEGER NOT NULL,   -- FTS shadow rows with no indexable block (ghost)
    drift_rows     INTEGER NOT NULL,   -- missing + excess (total divergence magnitude)
    exact          INTEGER NOT NULL CHECK (exact IN (0,1)),
    PRIMARY KEY (sampled_at_ms, surface)
) STRICT;
```
Retention: trim by age like other ops samples (disposable tier; a SIGKILL loses only history, never truth).

**New Prometheus gauges** (add to `metrics.py` alongside the boolean family — magnitudes the ledger already holds):

| Metric | Type | Labels | Source |
|---|---|---|---|
| `polylogue_fts_drift_rows` | gauge | `surface`, `kind` ∈ {missing,excess,total} | `fts_freshness_state` row (O(1)) |
| `polylogue_fts_source_rows` / `_indexed_rows` | gauge | `surface` | ledger row |
| `polylogue_fts_drift_exact` | gauge 0/1 | `surface` | 1 iff ledger counts came from an exact pass, 0 if bounded/trusted |

Keeps the existing boolean `_freshness_ready` — the gauge is the *magnitude* companion, so a surface can be `ready=0` **and** you can see whether it is off-by-3 or off-by-3-million (drives repair-vs-debt decision).

---

## 2. Coherence-check algorithms (pseudocode)

All keyed on the rowid identity. `missing` = orphan source blocks; `excess` = ghost FTS rows.

```
# O(1) ledger read — the probe path (no scan)
fn drift_from_ledger(conn, surface) -> {source, indexed, missing, excess, drift, trusted}:
    row = SELECT source_rows, indexed_rows, missing_rows, excess_rows, state
          FROM fts_freshness_state WHERE surface = ?
    if row is None: return UNKNOWN
    return {..row, drift: row.missing + row.excess,
            trusted: freshness_ready_record_trusted(row)}   # reuse existing predicate

# Exact coherence pass — used only on demand / periodic, wraps a read txn (BEGIN..COMMIT)
# so source and shadow describe one instant (already done in fts_invariant_snapshot_sync)
fn exact_drift(conn, surface="messages_fts"):
    source  = COUNT(blocks WHERE search_text != '')
    indexed = COUNT(messages_fts_docsize)
    missing = COUNT(blocks b LEFT JOIN messages_fts_docsize d ON d.id=b.rowid
                    WHERE b.search_text!='' AND d.id IS NULL)     # orphan source
    excess  = COUNT(messages_fts_docsize d LEFT JOIN blocks b
                    ON b.rowid=d.id AND b.search_text!='' WHERE b.rowid IS NULL)  # ghost
    return {source, indexed, missing, excess, drift: missing+excess}

# Coherence predicate (matches FtsSurfaceInvariant.ready)
fn coherent(inv): return inv.triggers_present and inv.source==inv.indexed
                          and inv.missing==0 and inv.excess==0 and inv.duplicate==0

# Continuous sampler (daemon convergence tick / metrics scrape)
fn sample_drift(surface):
    d = drift_from_ledger(conn, surface)          # O(1), preferred
    write fts_drift_samples(now_ms, surface, d..., exact=d.trusted?)
    emit gauges from d
    if d.drift > 0:
        alert_debt(surface, d)                     # §4

# Debt-style alert (reuses convergence_debt, mirrors _record_message_fts_surface_debt)
fn alert_debt(surface, d):
    if d.drift > STARTUP_REBUILD_MAX_DRIFT_ROWS (10_000):   # archive-scale → don't surprise-rebuild
        CursorStore.record_convergence_debt(stage="fts", subject_type="fts_surface",
            subject_id=surface, error="fts drift {d.drift} exceeds bounded repair")
    elif drift persisted across >= N quiet ticks:            # small but stuck
        record_convergence_debt(... "fts drift not self-healing")
```

---

## 3. Migration

**index.db and ops.db are both derived/disposable tiers → NO numbered migration chain** (`devtools lab policy schema-versioning` rejects upgrade helpers on these tiers).

- `fts_drift_samples` (ops.db, v1→bump): add to the **canonical ops.db DDL** + bump `ops.db` schema version; recovery = `polylogue ops reset` (ops is disposable, so a mismatch simply drops+recreates — no data to preserve).
- `fts_freshness_state`: **unchanged** — already carries every magnitude column. Zero schema delta on index.db.
- Metrics gauges are runtime-only — no schema; but regenerate `render openapi` + `render cli-output-schemas` if the status payload gains drift fields, and `render all --check` (grep `out of sync`).
- Adding the `metrics.py` module surface requires no topology change (existing module); if any new `polylogue/` file is added, run `devtools render topology-projection && topology-status`.

---

## 4. Metamorphic test strategy

**Core metamorphic law:** for *any* sequence of `blocks` insert/update/delete applied through the ordinary write path (triggers live), the exact invariant must hold — this is a relation the triggers are *supposed* to maintain, so it's a legitimate contract test (not diff-memorialization).

```
@given(ops = lists of BlockOp(insert|update_text|update_to_empty|delete), over a small block pool)
def test_fts_coherent_after_any_sequence(seeded_index_db, frozen_clock, ops):
    apply each op via real INSERT/UPDATE/DELETE on blocks   # triggers fire
    inv = fts_invariant_snapshot_sync(conn).messages
    # 1. count agreement
    assert inv.missing == 0 and inv.excess == 0 and inv.source == inv.indexed
    # 2. every indexable block matchable by its OWN tokens (no orphan)
    for b in blocks where search_text != '':
        hit = SELECT rowid FROM messages_fts WHERE messages_fts MATCH quote(first_token(b.search_text))
        assert b.rowid in hit_rowids
    # 3. no ghost row resolves to a dead/empty block (no excess)
    for rid in SELECT id FROM messages_fts_docsize:
        assert exists(blocks WHERE rowid=rid AND search_text != '')
    # 4. transition edge: update text '' -> nonempty inserts; nonempty -> '' deletes (au trigger)
```

Metamorphic *relations* to assert as separate properties:
- **Idempotence:** re-applying the identical op set leaves invariant unchanged (content-hash idempotency parallel).
- **Commutativity of disjoint blocks:** ops on different `message_id`s in any order → identical final FTS state.
- **Repair convergence:** inject artificial drift (raw `DELETE FROM messages_fts WHERE rowid=…` bypassing triggers, and orphan `INSERT`), then `insert_missing_message_rows_batched_sync` + `delete_excess_message_rows_batched_sync` ⇒ `coherent(inv)` and `drift==0`.
- **Ledger-vs-exact agreement:** after a materialize pass, `drift_from_ledger` == `exact_drift` (the O(1) probe never lies when trusted).

Use `frozen_clock` (clock-hygiene lint), build on `corpus_seeded_db`/`SessionBuilder`, Hypothesis profile bounded. Place under `tests/property/` (metamorphic) + `tests/unit/storage/` for the repair-convergence case. Do **not** touch protected `test_crud.py`.

---

## 5. Bead breakdown (6 beads, acceptance criteria)

1. **`fts-drift-1` — Emit drift-magnitude gauges from the ledger (O(1)).**
   AC: `polylogue_fts_drift_rows{surface,kind}`, `_source_rows`, `_indexed_rows`, `_drift_exact` exported by `metrics.py` reading `fts_freshness_state` only (no COUNT scan on scrape); metric help + registry list updated; `render` metrics-doc surface regenerated; unit test asserts gauge values equal ledger columns.

2. **`fts-drift-2` — `fts_drift_samples` continuous-sample table (ops.db).**
   AC: canonical ops.db DDL adds `fts_drift_samples`; ops.db version bumped with rebuild-plan note (no numbered migration); sampler writes one row/tick from the ledger; age-based retention trim; `reset` recreates cleanly.

3. **`fts-drift-3` — Metamorphic coherence property test.**
   AC: property test over random block op-sequences asserts `missing==excess==0`, own-token matchability, no ghost rows; commutativity + idempotence relations; green under `devtools test`; clock-hygiene lint passes.

4. **`fts-drift-4` — Repair-convergence test + wire bounded repair to drift.**
   AC: injected orphan/ghost drift is driven to `drift==0` by `insert_missing`/`delete_excess` batched primitives; test asserts convergence and that `drift > 10_000` records `convergence_debt` instead of rebuilding.

5. **`fts-drift-5` — True O(1) startup probe.**
   AC: `ensure_fts_startup_readiness_sync` fast-path returns from the trusted ledger without any `COUNT(*)` when `freshness_ready_record_trusted` holds; COUNT path taken only on ledger-miss/untrusted; test proves no `blocks`/`docsize` COUNT executes on the healthy path (spy/counter on `conn.execute`).

6. **`fts-drift-6` — Debt-style drift alert with quiet-window hysteresis.**
   AC: sampler records `convergence_debt(stage="fts")` when drift persists ≥N quiet ticks or exceeds bounded threshold; does not thrash-alert on transient mid-ingest drift (hot-file deferral respected); test covers transient-vs-stuck classification.

Dependencies: 1←(2 optional), 3 & 4 independent, 5 independent, 6 depends on 2. Ship 1/3/5 first (pure additive + tests).

---

## 6. Top-3 risks

1. **rowid reuse / non-stability breaks the identity.** All coherence rests on `messages_fts.rowid == blocks.rowid == docsize.id`. `blocks` PK is `(message_id, position)`; SQLite may recycle a freed rowid on a later INSERT, silently rebinding a stale ghost FTS row to a *different* block — an invisible ghost that passes count checks. Mitigation: coherence check must LEFT JOIN on rowid **and** confirm `block_id` equality (the `messages_fts.block_id UNINDEXED` column), not rowid alone; add a property case that deletes+reinserts to force rowid churn.

2. **Ledger-vs-truth divergence under the exact failure mode the module exists for.** Startup trusts `fts_freshness_state`; the `au` trigger fires on *every* UPDATE (even non-text) and the SIGKILL-between-trigger-drop-and-restore path (#1242) bypasses FTS without updating the ledger. An O(1) probe that trusts a stale-but-"ready" ledger serves silent-empty search. Mitigation: keep the existing `_message_fts_freshness_row_ready_sync` distrust rules; make `drift_exact=0` visible so operators know the gauge is ledger-trusted not scanned; periodic (non-startup) exact reconcile writes truth back.

3. **Sampler cost / lock contention against the single writer.** The daemon is the sole SQLite writer; an exact drift scan holds a read txn (`BEGIN..COMMIT` in `fts_invariant_snapshot_sync`) that can stall ingest on a 38GB archive, and a per-scrape exact pass would be pathological. Mitigation: sampler is **ledger-only O(1)** by default; exact passes are quiet-window-gated and bounded (reuse convergence hot-file deferral); `fts_drift_samples` writes go to ops.db (separate file) so history sampling never contends with index.db writes.

---

**Anti-goals honored:** no numbered migration on derived tiers; drift *history* placed in disposable ops.db, drift *truth* stays in index.db `fts_freshness_state` (already sufficient); metamorphic test asserts a maintained trigger contract, not a diff. No code or beads were modified.

---

## [a02acbbcadc07a13f] LANE: the full object/reference model for queries-and-findings-as-first-class-objects

Grounded. Here is the tight deliverable.

---

**LANE: the full object/reference model for queries-and-findings-as-first-class-objects**

Grounding: `saved_view`/`recall_pack`/`insight`/`transform`/`assertion`/`context-snapshot` are already `ObjectRef` kinds (`core/refs.py`), but **there is no `query`, `result_set`, or `finding` kind**; a saved query is just a `SAVED_QUERY` assertion whose `value` is an opaque query dict keyed by mutable name; the candidate→judgment→promote lifecycle + `supersedes` list + `inject:false`/`promotion_required` gating already exist (`user_write.py`); set-algebra (`docs/design/query-set-algebra.md`, `fnm.13`) is designed but persists **no** result-sets and mints **no** query identity.

- Add a `query:<hash>` ObjectRef kind as the keystone — an arbitrary query (saved or not) has no addressable identity today; only `@macro` names and `saved_view` names are handles, and both are mutable-by-name. A stable id is the prerequisite for set-op operands, evidence anchors, and query→query edges. — NEW (extends `fnm.13`/`4p1`)
- Define the query hash over the **planned AST post-macro-expansion**, not the surface string — `auth and test` ≡ `test and auth` ≡ `@arm_pack`-expanded must collapse to one `query:` id, mirroring the NFC content-hash idempotency doctrine used for sessions (`pipeline/ids.py`). Gives dedup, cache-keying, and "have I run this before" for free. — NEW
- Add `AssertionKind.FINDING` in user.db — `kind=FINDING`, `target_ref=query:<hash>` (or `session:`/`insight:`), `body_text`=claim, `value`={measure,n,statistic}, `evidence_refs`=content-anchored. It reuses the entire pathology/transform machinery: machine findings default `CANDIDATE`+`inject:false`+`promotion_required`, operator `judge_assertion_candidate` promotes. Zero new lifecycle code. — NEW (build on #2383 pattern)
- Put result-set snapshots in a **derived** tier (index.db), not user.db — `result_sets(result_set_id PK, query_id, grain, member_keys_json, member_count, corpus_epoch, ranking_json, computed_at_ms)`, `result_set_id = hash(query_id || corpus_epoch || sorted(member_keys))`. A result-set is rebuildable-from-source, so the durability axis puts it derived; only the *decision to keep* one (a promoted finding referencing it) is durable in user.db. Lets `ops reset --index` legitimately drop it. — NEW
- Make a scoped `corpus_epoch` the change-detector primitive — daemon bumps an epoch on ingest/materialize commit; a standing query is stale iff `current_epoch > snapshot.epoch` **and** re-materialized members differ. The daemon already owns the sole write path, so the bump is cheap. — NEW
- Scope the fingerprint to the query's pushdown predicate, not a global counter — a global epoch makes every standing query look stale on every ingest. `SessionQueryPlan` already separates SQL-pushdown from post-filters, so the pushdown predicate IS the scope key: fingerprint = hash of session content-hashes whose (origin,repo,date-window) intersect it. Only genuinely-affected queries re-evaluate. — NEW
- Persist the query DAG as `query_edges(src_query_id, dst_query_id, edge_kind, created_at_ms)`, edge_kind ∈ {operand-of, refines, supersedes, derived-from} — the set-algebra parser already builds this tree (`§6` EXPLAIN shows two sub-plans joined by a `set_op` node); `A | intersect (B)` naturally emits `operand-of` edges. Just persist the node so "what did this finding descend from?" is queryable. — NEW (extends `fnm.13` EXPLAIN node)
- Version queries via the existing `supersedes` list, not a new subsystem — editing a saved_view writes a fresh `SAVED_QUERY` assertion with `supersedes=[old_ref]`, old → `SUPERSEDED`, both retained. Content-addressing gives immutable versions (`query:<hash>` changes); the human `key` is the mutable pointer that repoints. Exactly the git model, reusing assertion supersession verbatim. — NEW
- Content-anchor a finding's evidence, not id-anchor — a finding citing `session:X` is silently wrong when X re-ingests with a differing hash. Store `session:X@<content_hash>` (ObjectRef already carries a trailing qualifier for block index — extend it) so a promoted finding records the corpus state it was true over; a re-materialize mismatch flags *possibly-stale* instead of lying. — NEW (construct-validity + citation-anchor themes)
- Findings-as-tests: a promoted FINDING with `value.expected` (`count==12`, `measure<thr`) becomes a re-runnable invariant — a `ConvergenceStage` re-executes its `query_id` against the current corpus and emits a candidate `finding-drift` (inject:false) on divergence. The check/execute + `false_means_pending` backlog shape is already the executor; findings become monitored claims, not frozen prose. — NEW (convergence stage)
- Add a `StandingQueryStage` to `DaemonConverger` — for each saved_view with `context_policy.standing=true`, if the scoped epoch advanced, re-materialize, diff against the last `result_sets` snapshot, write a candidate `query-delta` FINDING (inject:false) + refresh the snapshot. Use the session-scoped `check_sessions`/`execute_sessions` variant so it retries through `convergence_debt` under hot-file quiet deferral. — bead: extends #1498-cascade converger; NEW stage
- Redefine recall packs as **compositions over the object graph**, not frozen blobs — today `recall_pack.value` is opaque payload; make it an ordered list of `query:`/`finding:`/`session:` refs + a render projection, resolved live at recall time by `build_context_image`/`compose_context_preamble`. This inherits inject-gating + staleness and re-derives from source, avoiding the exact "frozen fabricated PR#123" failure the recovery digest had. — NEW (extends `recall_pack` semantics)
- Single inject gate for the context scheduler: only `status=ACTIVE ∧ inject=true ∧ not-stale` enters a preamble — a FINDING whose evidence content-hashes drifted is auto-downgraded to `inject:false` by the StandingQueryStage *before* it can be injected. This closes the "agent cites its own outdated finding as ground truth" loop the moment the corpus basis moves. — NEW (recursive-safety gate)
- Provenance-cycle guard on query→finding→query — if query B references finding F whose evidence transitively references F, mark the edge `quarantined` (reuse `TopologyEdgeStatus` from `session_links`); the converger refuses to auto-promote a finding whose query closure contains a quarantined edge. Mirrors the session-lineage cycle-break design; prevents an agent bootstrapping a self-justifying finding loop. — NEW (borrow `TopologyEdgeStatus`)
- Wire `author_kind` into the inject gate for differential trust — every `query:` run and finding already can carry `author_ref`/`author_kind` (`agent:<id>` vs `user:local`); make the scheduler treat `author_kind=agent` findings as promotion-required and `author_kind=user` as inject-eligible. Recursive-safety must distinguish operator from agent claims at the substrate, not by convention — the columns exist, just gate on them. — NEW
- Make the new kinds resolvable: `resolve_ref` (MCP) + CLI `read query:<hash>`/`read finding:<id>`, and an `explain_query_expression --as query:` mode returning the canonical hash — closes the audit loop: a finding cites `query:<hash>`, an auditor dereferences it to the exact re-runnable query + its `result_sets` snapshot. Requires an `EXPECTED_TOOL_NAMES` update. — NEW (extends `resolve_ref`)
- Store large result-sets as a Merkle root over sorted keys, not the full member list — the set-algebra already caps operands; a root gives O(1) "did this set change" for the change-detector and cheap intersect/except membership without persisting millions of keys in a derived tier. — NEW (scales the change-detector)

---

**GPT-pro prompt stubs**

- **[A]** "Design the migration path to introduce `query:<hash>` and `AssertionKind.FINDING` into a split-tier SQLite archive where saved queries currently live only as opaque `SAVED_QUERY` assertion values keyed by mutable name. Specify: (1) canonical query-AST hashing post-macro-expansion so equivalent queries collapse; (2) whether `result_sets` snapshots go in the rebuildable derived tier vs durable user tier, and the exact `ops reset --index` interaction; (3) how supersession-based query versioning coexists with content-addressed immutable ids and a mutable human name. Produce DDL, tier placement, and the failure mode when a promoted finding references a result-set that a derived-tier rebuild has dropped."
- **[A]** "Given a context scheduler that injects promoted assertions into future agent preambles, and agents that can author both queries and findings, enumerate the recursive-loop failure modes of a query→finding→query provenance graph and design the guard rails: content-hash-anchored evidence with auto-downgrade on drift, a quarantine status on provenance cycles, author-kind differential trust, and a standing-query convergence stage that re-tests findings-as-invariants. Show the exact predicate that decides whether a finding may self-inject."
- **[DR]** "Survey how mature systems model 'saved query + its result set + a claim derived from it' as versioned, addressable, content-hashed objects with change-detection — dbt exposures/snapshots, Malloy, Dagster asset checks, Great Expectations checkpoints, Datasette canned queries, LookML, Metabase questions, and notebook-provenance systems (Vizier/nbgather). For each: query identity model, result-set materialization/versioning, staleness/change-detection trigger, and how a downstream assertion (test/expectation) is bound to the query and re-evaluated. Extract the design patterns transferable to a single-writer local SQLite archive with a durable/derived tier split."

---

## [a490f010447e1bbce] The standard Polylogue should champion:

Grounded in the three epics, the enums, and `dispatch.py`. The tracked cluster already covers OTel in/out (0cg/wmj), OriginSpec (2qx), HPI/Promnesia (4g5), Grok (611), inspect-ai/Docent (7k7), inbound git (7xv), link-rot (ale), Obsidian export (r47), rest-of-Claude-Code (t0p), origin-enum + generic openai-chat detector (uiw), agent-config-as-source (7aw/h6r), the Hermes/Codex/Sinex bridges (fs1.x, ox0). Below deliberately routes around those and the excluded zones.

---

**The standard Polylogue should champion:**

- Publish **CIF — a Conversation Interchange Format**: content-addressed transcripts with *deterministic, portable anchors* (`session_id:message_id:block_id` already computed as generated columns) that survive export and re-resolve back to the exact block. The pitch is the one thing no vendor export has: stable citation identity + per-source fidelity declaration in one envelope. Distinct from r47 (a render *profile*); CIF is the anchor+provenance *spec* every export/import profile targets. — provider exports are all one-way, lossy, and un-citable; owning the interchange makes Polylogue the neutral hub instead of one more silo — NEW (parent l4kf)

- Make **`polylogue-export` a first-class Origin with a parser** (re-ingest of Polylogue's own CIF export). A round-trip that reconstructs identical `session_id/message_id/block_id` becomes a free correctness invariant *and* enables federation (import a teammate's exported archive; hash-idempotency dedups). — export fidelity is currently unverifiable and un-testable; closing the loop turns "did export lose anything?" into a passing provider-completeness lane — NEW

- **`.well-known/ai-sessions` federation manifest + selective content-hash sync.** Two archives advertise a manifest of `(content_hash, origin, anchor)`; a pull fetches only unknown hashes and dedups through existing idempotency. Local-first sharing without a server. — the QS/HPI audience (4g5) wants peer sync, not cloud; the content-hash substrate already makes this nearly free — NEW

**Outbound enrichment (Polylogue writes back into the dev ecosystem):**

- **Emit `git notes` (refs/notes/polylogue) attaching a session citation to the commit it produced.** Non-destructive, so `git log --notes=polylogue` shows "authored in session X" without rewriting history. This is the *outbound* counterpart to 7xv's inbound git awareness. — provenance currently dies in the archive; git notes put it where every future `blame`/bisect reader already looks — NEW

- **`polylogue cite` → gh PR/issue provenance footer.** Given a PR, inject a citable evidence block (the actual transcript spans that motivated the change) and register the reverse `session_link`. Two-way: PR ↔ session, both resolvable. — matches the repo's own "Verification/Problem = evidence not 'user asked'" discipline; makes that evidence a link, not a retype — NEW

- **Export mined pathologies as SARIF.** `AssertionKind.PATHOLOGY` rows already exist; emitting SARIF surfaces them in GitHub code-scanning and any SARIF consumer with zero new UI. — reuse a standard instead of inventing a report; agent-behavior defects become first-class findings alongside linters — NEW

- **Signed session attestation.** Sign `(content_hash, anchor)` so a session cited in a paper/PR/postmortem is tamper-evident: "this transcript, hash X, sig Y produced this result." Pairs with ale (link-rot) to give reproducible, non-rotting provenance. — single-writer + content-addressing already gives the integrity substrate; a signature makes it externally verifiable — NEW

**MCP as the continuity surface (deepen beyond ~130 tools):**

- **Expose sessions/messages/blocks as MCP *resources* with `resource://` URIs + subscriptions**, not only tools. Resources are the embeddable, citable, *live-updating* primitive any MCP client can mount — a tool call is a one-shot; a resource is durable continuity. — MCP is declared the continuity surface, but tools can't be embedded or subscribed to; resources are how other agents *hold* a session, not just query it — NEW

- **Ship recall-packs / saved-views as MCP *prompts*** (the slash-command surface). Any MCP client then gets Polylogue continuity as first-class prompts without knowing the tool catalog. — the recall-pack/saved-query assertions already exist; the prompts surface is the missing zero-friction injection path — NEW

- **`llms.txt` / `AGENTS.md` continuity artifact for MCP-less harnesses.** Render `compose_context_preamble`/`build_context_image` output into the emerging file conventions so Aider/Cursor/Zed (no MCP client) still consume Polylogue continuity via a file the harness already reads. — MCP is a hard dependency many harnesses lack; a file bridges continuity to the whole non-MCP fleet — NEW

**Inbound from the /realm data lake (not just git):**

- **Attach Sinex/ActivityWatch ground-truth to a session as observed-events**: window-focus timeline, shell commands (Atuin), and file-activity *during* the session wall-clock window. A transcript says what the agent did; this says what the *human* actually did around it. — the lake already has `window_focus`/`file_activity`/`command_frequency`; correlating by time window makes a session a full episode, not a half-record — NEW (aligns fs1.9's Polylogue→Sinex with the reverse)

- **Cross-origin "same investigation" stitching** (repo + time-window + embedding neighborhood), explicitly *not* within-provider replay lineage: a human debugging in Cursor → continuing in Claude Code → asking ChatGPT is one logical unit across three origins. Edge type = thematic/temporal, not prefix-sharing. — real work spans harnesses; origin breadth is pointless if the archive can't see the seam between them — NEW

**Concrete new-source parsers worth wiring (beyond uiw's aider/Cline/OpenHands enumeration):**

- **Cursor** — chats live in `state.vscdb` SQLite under `workspaceStorage` (binary, keyed blobs), not JSON files. The non-obvious wiring: a SQLite-shaped detector/acquire path, unlike every current file-payload detector. Cursor is the single largest missing corpus. — NEW
- **Continue.dev "dev data"** — Continue emits purpose-built JSONL specifically for export/analysis (the friendliest possible source; it *wants* to be ingested) plus `.continue/sessions`. — NEW
- **Zed via ACP (Agent Client Protocol)** — Zed champions an open editor↔agent protocol; wiring ACP gives both an ingest source *and* a live lane, and positions Polylogue inside the one editor ecosystem with a real standard. — NEW
- **OpenCode (sst/opencode)** local session store and **Warp** AI blocks — additional harness JSON stores; mechanical once OriginSpec (2qx) lands. — NEW
- **Aider `.aider.chat.history.md`** — a *Markdown* transcript, not JSON; forces the first prose-shaped parser and a fixture that stresses the block model. (In uiw's list but the markdown-parse detail is the non-obvious risk.) — NEW

**Ecosystem-reuse surfaces:**

- **Datasette read-only publish of the archive.** It's already SQLite; a `datasette`-compatible publish exposes the whole archive to the Simon-Willison/QS ecosystem's tooling for free — same audience as 4g5/r47, near-zero effort. — NEW

- **IDE Timeline / SCM provider: "which sessions touched this file."** A tiny VS Code/Zed extension backed by the file-touch data surfaces sessions in the editor's timeline for the open file — provenance where the developer already is. — NEW

- **Eval-corpus export extension: promptfoo test-cases + SWE-agent `.traj`** downstream of CIF (extends 7k7/fs1.5 to the two most-used OSS eval formats, so archived sessions become redteam/benchmark datasets). — NEW (extends 7k7)

---

**GPT-pro prompt stubs:**

- **[DR]** "Survey the 2025–2026 AI coding-agent harness landscape (Cursor, Zed, Continue.dev, Aider, OpenCode, Cline/Roo, Windsurf/Cascade, Amp, Warp) and the emerging session/trace *format* standards (OTel GenAI semantic conventions, Zed's Agent Client Protocol, `llms.txt`/`AGENTS.md`, promptfoo/inspect-ai/SWE-agent trajectory formats). For each: where sessions are physically stored, the schema shape, whether it's an open spec or reverse-engineered, and format stability. Conclude with which *one* interchange standard a neutral local archive should champion vs. merely adapt to, and why."

- **[DR]** "Research the design space for a portable, content-addressed *citation anchor* standard for AI transcripts that survives export and round-trips back to an exact message/block. Compare prior art (W3C Web Annotation selectors, `git notes`, DOI/PID practice, Nostr/IPFS content addressing, Gwern-style link archival). What properties make an anchor durable, tamper-evident, and re-resolvable across independent archives, and what are the concrete failure modes (re-parse drift, provider schema change, redaction)?"

- **[A]** "I have a single-writer, content-addressed SQLite archive of AI coding sessions across many harnesses. Argue the strongest case for and against making the archive's *own export a first-class re-ingestable source* (round-trip identity as a correctness invariant + a federation primitive). Cover: what invariants it buys, where non-injective provider→origin collapse or lossy fidelity declarations break the round-trip, and whether federation should be manifest-pull or feed-push."

---

## [a8c5e7a43a3e65027] Grounded on polylogue-4smp (content-variant substrate: translation/transliteration/simplif

Grounded on polylogue-4smp (content-variant substrate: translation/transliteration/simplification/summary as aligned variants over refs, never confused with evidence), polylogue-mhx (embedding substrate), polylogue-0v9p (language detection), and raw-log itches (reranker, audio+transcription, screencap OCR). Ideas below.

- **Transliteration as a mechanical variant kind, distinct from generative ones** — romanization (Cyrillic/CJK/Polish) is rule-based, so it can be an always-on variant with 100% coverage and exact char-level alignment; tagging variant provenance `mechanical` vs `generative` lets the reader trust transliteration/OCR fully while flagging LLM translations as lossy — the trust axis is the point of the substrate — polylogue-4smp
- **Variant staleness keyed to source content_hash** — variants target refs whose source can re-ingest with a differing hash (the update path); without an invalidation edge a translation silently re-aligns to changed text and becomes evidence-corrupting. Store `source_content_hash` on each variant node; on hash change mark `stale`/`orphaned`, never auto-repaint — polylogue-4smp
- **Summary variants must carry citation edges or be rejected at write** — generalizes the "recovery report fabricated PR #123" incident: a summary node with zero alignment edges back to the source spans it compresses is an unverifiable claim masquerading as a read model. Make coverage>0 a write invariant; a summary is only honest if you can click each sentence back to its source blocks — polylogue-4smp
- **Coverage-gap ("dark matter") rendering for lossy variants** — for a summary/simplification, compute the source spans NOT covered by any alignment edge and surface them; the operator's real question about a summary is "what did it drop," which no summary can self-report — polylogue-4smp
- **Faithfulness score as a variant-quality assertion via embedding drift** — cosine between a summary node's embedding and its claimed aligned source span; a summary whose vector drifts far from what it cites is a hallucination signal. Turns the embedding substrate into a cheap honesty auditor for the variant substrate — polylogue-mhx + polylogue-4smp
- **Per-block language detection driving variant defaults, not per-session** — real sessions are mixed (Polish prose, English code comments, English tool output); a session-level language guess is wrong for most blocks. Store detected lang as block metadata so translation variants only offer where `source_lang ≠ operator_lang`, and code/tool_result blocks are excluded from translation by construction — polylogue-0v9p
- **Glossary-pinned translation with span-preserving substitution** — domain terms (identifiers, project names, `polylogued`) must survive translation verbatim; run deterministic pre/post substitution around the LLM keyed to an operator-maintained term dictionary (an assertion kind), and keep original-token alignment spans so the pinned terms stay clickable to source — polylogue-4smp
- **Image blocks → caption/description variants for search, never as evidence** — Claude/ChatGPT exports already carry inline image blocks; a vision-model caption is a `variant(kind=caption)` over the image ref that makes the image FTS/vector-searchable without pretending the caption is what the model actually saw. `find "diagram of the pipeline"` should match an image via its variant text — polylogue-4smp + polylogue-mhx
- **Screenshot OCR as a mechanical variant with bbox alignment, separate from semantic caption** — agents paste screenshots; the literal on-screen text (OCR, mechanical, high-trust) and a semantic "what this shows" caption (generative) are two different variant kinds over the same image block, and conflating them loses the trust distinction that makes OCR usable as evidence-adjacent — NEW (depends on polylogue-4smp)
- **Audio transcript as a first-class origin where timestamps become block metadata** — the operator's oldest itch (2025-04 audio recording + transcription); ingest diarized transcript as source content with per-block audio offsets, enabling "jump to audio at this line" and making the whole capture lake retrievable through the same search algebra — NEW
- **Embedding eval harness that mines ground truth from lineage** — resume/fork pairs are semantically-near sessions that share few literal tokens: free positive pairs. Build a labeled probe set from `session_links`, then measure recall@k for FTS vs vector vs hybrid to actually *prove* vector earns its cost instead of asserting it — polylogue-mhx
- **Multi-provider embedding shootout on a frozen probe set (Voyage vs local)** — embeddings.db is rebuildable, so run Voyage vs local (nomic/bge/gemma via Ollama) on the *same* eval reporting recall + $/1k + latency; "should I run local embeddings" becomes an evidenced decision. Second vec0 space per provider with hard provenance so an ANN query never mixes spaces — polylogue-mhx
- **Reranker as a measured post-retrieval stage, not a reflex** — hybrid fetches top-50, a local cross-encoder (bge-reranker) reorders to top-10 (raw-log: "Reranker seems potentially very useful"); wire it behind the same eval harness and only promote into MCP search if it lifts precision@10 on the probe set — polylogue-mhx
- **Tool-result-aware chunking: embed a distilled variant, keep raw as evidence** — embedding a 40KB stack-trace block whole is noise that pollutes ANN neighborhoods; chunk by the keystone structure (`tool_result_is_error`/`exit_code`) and embed a head+tail+error distillation (itself a variant) while the raw block stays untouched as evidence — polylogue-mhx + polylogue-4smp
- **Retrieval unit = "action", not raw block** — the `actions` view already joins tool_use↔tool_result by tool_id; embedding call-intent + outcome *jointly* lets `find "the grep that found the conn leak"` retrieve one coherent action rather than two disjoint blocks the query splits across — polylogue-mhx
- **Semantic diff across a fork's divergent tail as a topic-drift vector** — go beyond the text diff of the divergent tail: embed each side's divergent messages and report which concepts entered/left, answering "did the fork abandon the original goal?" cheaply. Optionally diff two *summary variants* of the forks to answer at outcome granularity instead of token granularity — NEW (uses polylogue-mhx + polylogue-4smp)
- **Hierarchical coarse→fine retrieval for very long transcripts** — session-summary-variant embeddings for coarse recall, then action/message embeddings for the fine pass; two vec0 spaces so a 500-message tool-heavy transcript doesn't drown the fine index and coarse recall stays cheap — polylogue-mhx
- **`variant` and `coverage` as queryable DSL units** — `sessions where variant:translation and coverage < 0.8` to find partial/abandoned translations, or `messages where lacks variant:caption and has image` to find un-captioned image debt; makes the variant substrate observable through the existing query algebra rather than a side inspection tool — polylogue-4smp

---

GPT-pro prompt stubs:

- **[DR]** "Survey the 2025–2026 landscape of *local* text-embedding and cross-encoder-reranker models runnable via Ollama/llama.cpp on a single 13700K+consumer-GPU box (nomic-embed, bge-m3, gte, gemma-embedding, Voyage-3 as the cloud baseline). For a personal AI-session archive (~40GB, code-heavy multilingual transcripts), give a decision matrix over retrieval quality (MTEB + code retrieval), memory footprint, throughput, quantization tradeoffs, and licensing — and specify how to build a self-labeled recall@k eval from fork/resume pairs to validate any choice locally."
- **[DR]** "Compare architectures for *faithfulness/attribution scoring of summaries* (summary-as-variant that must cite source spans): embedding-drift cosine, NLI-based entailment scoring, and citation-alignment coverage metrics. What's the state of the art for detecting hallucinated/unsupported summary sentences cheaply and locally, and how do these methods degrade on code + tool-output content vs prose?"
- **[A]** "Design a typed content-variant model over addressable refs (session/message/block/assertion) supporting translation, transliteration, simplification, summary, caption, and OCR, with alignment edges (incl. many-to-one and partial), a `mechanical` vs `generative` provenance axis, and a `source_content_hash` staleness invariant. Specify the closed relation/status/coverage vocabularies, the write-time honesty invariants, and how query surfaces guarantee variant text is never returned as original evidence."

---

## [a9d14b82e1ef1ef35] Grounded: read the `bby` epic ("evidence cockpit") and its 16 children (timeline/firehose 

Grounded: read the `bby` epic ("evidence cockpit") and its 16 children (timeline/firehose bby.10, day-page bby.13, pinboard bby.14, replay bby.12, minimap bby.5, live-tail bby.4, aggregates bby.3, completions bby.2, arch-v2 bby.11, interaction-debt bby.6, virtualized-list bby.8, WebUI posting b1n/ptx, session-diff yrx), plus `jnj` surface-algebra. Inspected the 2.4k-line embedded-JS three-pane SPA (`web_shell.py` + shards), the lineage tab (currently a **BFS text list**, not a graph), realtime SSE, the Textual TUI, and the `il`/pywal-shaped theming gap. Ideas below deliberately step past the existing children.

- Citable-report builder ("evidence basket") — the `bby` arc promises result-list→evidence-graph→**citable-report** but no child delivers the report end; add a persistent basket that accretes selected blocks/messages into a live Markdown doc with provenance footnotes, exportable as the actual deliverable of an investigation — NEW under bby
- Content-hash citation anchors (`cite this block`) — every block/message gets a copyable permalink keyed on content-hash, not `position`, so a citation survives re-ingest when a fork's divergent-tail shifts positions; this is the atom the whole cockpit stands on and the honest fix for the recovery-digest fabrication class — NEW under bby
- Citation-integrity verifier — on report export, re-resolve every anchor against the live archive and flag drifted/deleted/quarantined citations (same honesty doctrine as text-mined→unverified-candidates); a report the operator can trust is the difference between "cockpit" and "screenshot" — NEW
- Real force-directed lineage/topology graph — replace the rooted-tree BFS list with an actual node-link SVG: edge-kind coloring (prefix-sharing vs spawned-fresh), quarantined cycle-break nodes visibly flagged, branch-point highlighted, click-to-refocus; the list structurally hides fork *shape* which is the sharpest part of the data model — NEW under bby (complements bby.5)
- Assertion/evidence overlay graph — extend the lineage graph so nodes include the marks/tags/corrections/assertions/recall-packs that *reference* sessions; the durable `user.db` tier is currently invisible at read time, so "which of my judgments cite which sessions" can't be seen — NEW
- Marginalia layer in the reader — render marks/corrections/annotations as sticky margin notes anchored to their blocks, editable in place; assertions are the irreplaceable tier yet the reader shows raw transcript with zero overlay of the operator's own accumulated meaning — NEW
- Year-heatmap calendar (contribution-graph shape) — a density grid over tokens/sessions/tool-calls that click-throughs into the day-page (bby.13); bby.13 is *narrative*, this is the *navigable temporal index* that gets you there, and it makes "what was a busy week" answerable at a glance — NEW under bby
- Unified temporal scrubber — one draggable time axis with zoom semantics that goes year-heatmap → firehose (bby.10) → session-replay (bby.12); today those beads are three unrelated widgets, a shared zoom spine turns them into one instrument — NEW (meta over bby.10/12/13)
- material_origin texture gutter — a vertical minimap strip colored by `material_origin` (human/assistant/tool_result/runtime_protocol) so a 4k-message Claude Code session reveals its authoredness texture and you click to jump; extends bby.5's phase minimap with the load-bearing axis Role can't express — extends bby.5
- Delegation call-tree render layout — after #2545 shipped subagent-exchange rendering, present Task/subagent delegations as nested collapsible cards forming a call-tree, not a flat inline splice; the specimen the operator most wants to inspect ("what did my subagents actually do") is buried — NEW
- Specimen gallery browse mode — a grid-of-cards view (vs the list) where each card is a session-glyph: origin badge, tool-timing sparkline, phase ribbon, cost chip; built for *visual triage of 200 sessions* which a virtualized list (bby.8) can't do — NEW under bby
- Side-by-side fork/parent diff pane — surface the existing `compare_sessions` MCP as a two-pane aligned reader view that highlights only the divergent tail past `branch_point_message_id`; lineage dedup is invisible unless you can *see* what a fork changed — NEW (distinct from yrx's edit-changelog)
- Ambient theming from pywal / prefers-color-scheme — the web reader reads `~/.cache/wal/colors.json` (or `POLYLOGUE_THEME`) and honors `prefers-color-scheme`, extending the existing `il` palette (already used for terminal syntax/diff) to the web surface so the cockpit matches the operator's desktop instead of a fixed dark theme — NEW (ties `ui/il` → web)
- Progressive-enhancement / server-rendered read routes — every reader route also returns clean semantic HTML (not only the JS SPA), so a phone browser, `curl | pandoc`, or a reverse-proxy render is readable; "I'm blind without webui" is also true on mobile, and arch-v2 (bby.11) should bake this in as a constraint not a retrofit — feeds bby.11
- Obsidian-vault read-view profile — a read-view that emits wikilink-cross-referenced Markdown (one file per logical session, `[[…]]` backlinks mirroring lineage edges, canonical-URL frontmatter) droppable into the knowledgebase vault for mobile/Obsidian read; reuses the Origin canonical-URL projection already built — NEW (read-view profile; touches jnj surface algebra)
- Cmd-K command palette — a keyboard-first fuzzy palette to jump to session/saved-view/verb and enter DSL, killing the last `window.prompt()` (bby.6) with a real primitive; navigability for an archive means never reaching for the mouse — extends bby.6
- Voice recall loop (push-to-talk) — a reader hotkey → local Whisper → DSL query, and TTS read-back of resume-briefs/day-pages; the raw-log's hands-free/ambient thread implies the archive should answer "what was I doing on X" spoken, not typed — NEW
- "Now playing" ambient strip — a tiny always-visible widget (reader header + optional Waybar module) live-tailing daemon SSE to show what agents are doing *right now*, turning the archive from a thing you open into peripheral awareness — NEW (rides bby.4 realtime)

---

[A] "I'm designing an 'evidence cockpit' web reader over a local SQLite AI-session archive whose read arc is result-list → evidence-graph → citable-report. Given content-hash block identity, a lineage model that stores only divergent tails plus branch-point, and a durable assertions tier (marks/corrections/annotations), design the citation-anchor scheme, the evidence-basket→report data flow, and the citation-integrity verifier so exported reports survive re-ingest and fork-position shifts. Specify the anchor format, resolution algorithm, drift/quarantine handling, and the minimal API surface."

[A] "Propose a unified temporal-navigation instrument for an AI-work archive that composes a year-density heatmap, a scrubbable event firehose, a per-day narrative, and per-session replay under one shared time-axis zoom model. Define the zoom-level state machine, the aggregation tiers (year→week→day→session→message), the data contracts each level fetches, and how session-lineage forks are represented as branching on the timeline rather than duplicated tracks."

[DR] "Survey the strongest patterns and open-source implementations for: (1) large-graph in-browser rendering of node-link topologies (force-directed vs hierarchical), (2) progressive-enhancement / server-rendered-plus-hydration architectures that keep a single-page reader usable on mobile and via curl/pandoc, and (3) desktop-ambient theming (pywal/prefers-color-scheme) piped into a locally-served web app. For each, extract the concrete techniques, performance ceilings, and failure modes relevant to a single-writer local archive of ~16k sessions."

---

## [a604d42fbdaea4ffd] Measure Catalog + Composition Algebra (lane #4, wave-2 agent 2)

## Measure Catalog + Composition Algebra (lane #4, wave-2 agent 2)

Grounding read: `insights/registry.py` (10 registered types), `insights/archive_models.py` (evidence/inference/latency/workevent/phase/thread/cost payloads), `insights/tool_usage.py`, `insights/run_projection.py`, bead 9l5 tower map + children .1–.12. Everything below is **one reducer away** from a column that already materializes — the catalog is *generated*, not authored.

### The composition algebra (crisp form of 9l5's 5-tuple)

A **measure** is a 5-tuple `⟨reducer(column_expr) over unit-frame, grouping, window, comparison, uncertainty⟩`. Each axis is an independent, closed operator over the DSL query algebra (fnm/4p1), so the catalog = cartesian product, not a list of bespoke `analyze` modes:

- **reducer × unit-frame** — `count / sum / ratio / mean / median / pXX / entropy / Gini / zstd-ratio` over a unit frame `∈ {sessions, actions, messages, observed-events, work-events, phases, threads, thread-members}`. The frame decides the denominator (the #1 construct-validity trap).
- **grouping** — `origin | normalized_model | dominant_repo | workflow_shape | terminal_state | action_kind | heuristic_label | depth | day | iso_week | thread | timing_provenance-tier`.
- **window** — `all | trailing-N | rolling-N | changepoint-segment | since-event`.
- **comparison** — `absolute | vs-baseline(trailing) | vs-group(A/B) | paired-arms | delta-over-window`.
- **uncertainty** — Wilson for proportions; bootstrap CI + n for median/pXX; two-sample effect size; **coverage tier** (evidence/inference) + `timing_provenance`/`cost_provenance` footnote gate — a partial-provenance group cannot masquerade as a finding.

The registry (9l5.7) stores per-measure: construct operationalized, formula, evidence tier, sample frame, declared confounds, coverage precondition. This lane's job is to fill that registry.

### ~5 highest-leverage (build first)

1. **Cache-amplification ratio** — `total_cache_read_tokens / total_input_tokens` — a finding-grade number (216× in forensics) that recolors *every* throughput/cost claim; disjoint-lane honest.
2. **Latency three-lane decomposition** — `thinking_duration_ms : tool_duration_ms : output_duration_ms` as shares of `wall_duration_ms` — the single most reusable "where does time actually go" primitive, groups by everything.
3. **Engaged-vs-wall efficiency** — `engaged_minutes / (wall_duration_ms/60000)` — separates real attention from idle span; the honest denominator that makes duration comparisons non-lying.
4. **Tool-mix entropy** — Shannon over `tool_call_count_by_category` / `tool_counts` — one scalar for "workflow diversity," cross-model comparable, construct-transparent (9l5.12).
5. **Credit-vs-API divergence** — `subscription_equivalent_usd / api_equivalent_usd` per model — the "what you'd pay vs what the number says" ratio; nothing surfaces it yet despite five-axis basis existing.

### 12–16 concrete measures (`measure — answer + column — bead/NEW`)

- **Cache-amplification ratio** — fraction of apparent token throughput that is cache-read, per origin/model over time — `total_cache_read_tokens / total_input_tokens` — 9l5.4
- **Reasoning "thinking tax"** — share of output that is invisible reasoning tokens, by model — `reasoning_output_tokens / total_output_tokens` — 9l5.4/NEW
- **Latency three-lane share** — where wall time goes (think/tool/output), by shape and model — `thinking_duration_ms, tool_duration_ms, output_duration_ms / wall_duration_ms` — NEW
- **Interaction latency asymmetry** — is the human or the agent the bottleneck, by model — `median_agent_response_ms vs median_user_response_ms` — NEW (latency payload already carries both)
- **Engaged-vs-wall efficiency** — real-attention fraction of session span — `engaged_minutes ÷ wall_duration_ms` — NEW
- **Stuck-tool density** — rate of pathologically-slow tool calls, by tool/model — `stuck_tool_count / tool_use_count` (+ `max_tool_call_ms` tail) — 9l5.1/NEW
- **Compaction pressure** — context-thrash intensity, longitudinal — `compaction_count` per session and per 1k `message_count` — 9l5.4/9l5.8
- **Substantive density** — signal-to-noise of a session, drift over time — `substantive_count / message_count` — NEW
- **Outcome-conditioned cost/duration/retries** — cost & wall by structural `terminal_state` × exit_code/is_error, with coverage tier — evidence cost/duration cols × actions-view outcome — 9l5.1
- **Tool-mix entropy (diversity)** — full-affordance vs shell+edit-only, per model/repo — Shannon over `tool_call_count_by_category` — 9l5.12
- **Workflow-shape transition matrix + distance** — do models differ in edit→test→commit discipline — first-order Markov over phase/action sequence, matrix L1 distance between models — 9l5.10
- **Session redundancy (zstd ratio)** — repetitiveness / low-new-information thrash, per session/thread — compression ratio of session `search_text`/prose — 9l5.12
- **Thread cost concentration (Gini)** — is thread spend dominated by one runaway child — Gini over member `total_cost_usd` within `ThreadPayload` — NEW
- **Subagent fan-out / dispatch-return latency by depth** — fleet-shape fingerprint — `depth`, `branch_count`, member start/end deltas over topology — 9l5.12
- **Pathology epidemiology rate** — is thrash-looping getting better since a harness change — per-session detector hits ÷ eligible sessions, grouped by `iso_week`/model, Wilson CI — 9l5.3
- **Credit-vs-API divergence** — subscription true-cost vs list-equivalent, by model — `subscription_equivalent_usd / api_equivalent_usd` (cost basis) — 9l5.2/NEW

Cross-cutting note: every row above is the *same* reducer applied across the grouping axis — the registry should enforce that a "cross-model" or "longitudinal" variant is just a `grouping`/`window` swap on a registered base measure, never a new code path. That is the deepest form of theme #4: the catalog is the composition table, not 16 functions.

### 3 GPT-pro prompt stubs

- **[A]** "Given Polylogue's column inventory (`archive_models.py` payload fields listed) and the 5-tuple `⟨reducer/unit-frame, grouping, window, comparison, uncertainty⟩`, produce the full construct-validity registry row for each of the 16 measures above: construct operationalized, exact formula, evidence tier, correct denominator/unit-frame, the top-2 confounds, and the coverage precondition that must hold or the measure is suppressed. Flag any measure whose naive denominator would silently mix provenance tiers."
- **[DR]** "Survey how observability/product-analytics systems (Honeycomb, PostHog, dbt-metrics/MetricFlow, Cube.js semantic layer) formalize a *metric as a composable object* — dimension/grouping algebra, windowing, comparison-to-baseline, and uncertainty surfacing. Extract the 5 design patterns most transferable to a single-writer SQLite archive whose measures must carry evidence-tier + coverage-precondition metadata, and name the traps (Simpson's paradox on grouping, ratio-of-ratios, censored-duration means) each system solves or ignores."
- **[A]** "For the latency three-lane decomposition and cache-amplification measures, design the uncertainty layer: which need Wilson (proportions) vs bootstrap-CI (median/pXX of skewed durations), the minimum-n and coverage-tier gate before a group is reportable, and how to render a one-line honest footnote (`n=…, coverage=evidence, timing_provenance=structural`) in plaintext/JSON/MCP from one registry declaration."

---

## [a22febd55fe44ae89] INGESTION ROBUSTNESS, FIDELITY & COVERAGE — 15 ideas

Grounded in `sources/dispatch.py`, `raw_sessions` DDL, `ArtifactSupportStatus`/`ValidationStatus` enums, and the fact that today's `archive_coverage` only counts sessions-per-provider (no byte-fidelity notion). Here's the lane.

## INGESTION ROBUSTNESS, FIDELITY & COVERAGE — 15 ideas

- **Byte-fidelity ratio per raw_session** (parsed_bytes / blob_size, stored on the row) — `archive_coverage` today only counts sessions-per-origin; there is no measure of "what fraction of the acquired blob became structured content." Make fidelity a first-class column so silent-drop is queryable, not a vibe — NEW.
- **Round-trip reconstruction check against blob_hash** — every raw payload's bytes are already durably stored (`blob_refs ref_type='raw_payload'`, 32-byte hash). Add a `devtools lab` reconstructor that re-serializes the parsed tree back to the source shape and diffs against the original blob; report per-provider round-trip loss as a rigor number — NEW.
- **Unparsed-key census (residual-field mining)** — walk each raw JSON document, subtract the keys/paths the parser actually consumed, and persist the leftover key-set. Turns "we might be dropping fields" into a ranked list of concretely-ignored provider keys per origin, the input to closing fidelity gaps — NEW.
- **Promote `recognized_unparsed` into a queryable backlog surface** — `ArtifactSupportStatus.RECOGNIZED_UNPARSED` and `PARTIAL_DECODE` already exist as durable states but there's no `find`/MCP lane that lists "artifacts we saw, recognized, and could not fully parse." Wire it to `archive_debt` so parser gaps are tracked debt, not enum values no one reads — NEW.
- **Detection-ambiguity score + second-best provider** — `detect_provider` returns first match in tightness order with no record of *how close* the runner-up was. Capture a margin (matched-signal count of winner vs next) into `detection_warnings_json`; low-margin sessions are the misclassification candidates worth auditing — NEW.
- **Misclassification tripwire via post-parse cross-validation** — after a session parses under provider X, run the *other* providers' cheap shape-detectors against the same payload; if a tighter detector would also have claimed it, flag as ambiguous. Catches the "loose dict-key ChatGPT/Claude-web/Gemini collision" class the CLAUDE.md warns about — NEW.
- **Streaming-detection window blind-spot guard** — `_detect_provider_from_raw_bytes` samples only `islice(stream, 32)` records for JSONL detection. A multi-GB Codex/Claude-Code file whose distinguishing record appears after row 32 mis-detects to the fallback. Record when detection *used* the fallback vs a positive match, and alarm when fallback rate rises — NEW.
- **Truncated/partial-JSONL tail accounting** — the streaming path has `truncated_tail_ok` / `_trim_jsonl_detection_prefix` handling but silently discards a malformed trailing record. Count trimmed/undecodable trailing bytes per file and surface as `PARTIAL_DECODE` with a byte count, so truncated exports don't look like clean ingests — NEW.
- **`_MAX_PARSE_DEPTH=10` overflow is a first-class drop event** — deeply-nested drive/bundle payloads that exceed depth-10 lowering are silently not lowered. Emit a structured `detection_warnings_json` entry ("depth cap hit at N") and count these as a coverage-loss category rather than a no-op — NEW.
- **Validation-drift as a fidelity signal, not just a counter** — `validation_drift_count` is persisted but unaggregated. Roll it up per origin/schema-version so a provider export-format change shows up as rising drift *before* it becomes parse failure; ties into the construct-validity substrate wave-1 theme — NEW.
- **Re-ingest-on-parser-improvement safety gate** — content-hash idempotency skips re-import when bytes match, so a *better parser* never re-processes old blobs. Add a `parser_fingerprint` (version/hash of the parser code path) to `raw_sessions`; when it changes, force reprocess (parse+materialize+index, no re-acquire) of affected origins so fidelity gains actually backfill — NEW.
- **Coverage regression harness over the fixture corpus** — snapshot per-provider fidelity ratio + unparsed-key census as a golden file; a PR that lowers fidelity for any provider fails a `devtools lab` gate. Makes "did this parser change lose anything?" an automated, reviewable diff — NEW.
- **Zero-message / zero-block parse-success anomaly detector** — a session that parses "successfully" (`parse_error IS NULL`) but yields 0 messages or all-empty blocks is a silent-drop masquerading as success. Add an invariant check post-materialize that flags structurally-empty successful parses per origin — NEW.
- **`unknown-export` / `grok-export` funnel report** — `grok-export` is a reserved origin with no wired parser and `unknown-export` is the catch-all. Report the volume + byte-weight of everything landing in these buckets so the size of the "provider we can't parse yet" backlog is a visible number, feeding the ecosystem agent's new-parser prioritization — NEW.
- **Decode-failure taxonomy (encoding vs framing vs schema)** — `DECODE_FAILED` collapses several distinct failures (bad UTF-8, malformed JSON framing, schema-mismatch). Split the failure reason into a small closed vocabulary on the row so malformed-export triage can route: encoding bug → decoder fix, framing → streaming fix, schema → parser fix — NEW.

## GPT-pro prompt stubs

- **[A]** "Design a per-origin *ingest fidelity metric* for a content-addressed archive where raw export bytes are durably stored (SHA-256 blob) and parsed into a sessions→messages→blocks tree. Define the metric so it (a) distinguishes byte-level round-trip loss from semantically-intended field omission, (b) is cheap enough to compute on every ingest of multi-GB JSONL, and (c) composes into a single queryable coverage number without hiding truncation or depth-cap drops. Specify the stored columns, the reconstruction-diff algorithm, and how to avoid false 'loss' on deliberately-discarded provider noise."

- **[DR]** "Survey how mature data-ingestion/ETL and email/chat-archiving systems (e.g. dovecot/notmuch, Elasticsearch ingest pipelines, Singer/Airbyte taps, OpenTelemetry collectors) detect and report *silent field drop* and *source-format drift* — schema-on-read residual capture, dead-letter/quarantine queues, coverage/lag metrics, and re-processing when a parser improves. Extract the reusable patterns for a single-writer local SQLite archive of heterogeneous LLM chat exports."

- **[A]** "Given shape-based provider detection over JSON payloads evaluated in strict 'tightness order' (structural detectors → Pydantic-validated → loose dict-key checks), design a *misclassification detector and ambiguity score*: how to quantify the margin between the winning provider and the runner-up, how to bound the cost on streaming JSONL where only the first N records are sampled, and how to turn low-margin/late-signal cases into an auditable backlog rather than silent mis-routes. Include the failure mode where the distinguishing record appears past the detection sample window."