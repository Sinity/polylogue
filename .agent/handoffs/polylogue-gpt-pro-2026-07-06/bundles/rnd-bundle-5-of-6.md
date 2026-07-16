# Polylogue R&D bundle — file 5 of 6

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
  PART 2 - PAYLOAD: 16 subagent outputs (unique to file 5 of 6)
=====================================================================


---

## [a4063f6013d7b4519] Wave-5 Implementation Spec: `phase-segment`, `goal`, `decision-object`

I have complete grounding. Returning the spec.

---

# Wave-5 Implementation Spec: `phase-segment`, `goal`, `decision-object`

## Grounding correction (read first — the task framing collides with two deliberate prior decisions)

Two units in the task are framed as "missing" but the substrate already contains most of the machinery, and one framing directly contradicts a construct-validity decision that must be honored:

1. **`session_phases` are NOT intent-classified and must not become so.** `polylogue/archive/phase/extraction.py:27-46` — `SessionPhase` is time-gap-segmented (5-min `_PHASE_GAP`); its docstring records that the `kind` field *was removed* because "phases represent when activity happened, not what kind of activity it was." The intent-classified equivalent already exists as **`session_work_events`** (`archive/session/extraction.py:89-103`, `WorkEventHeuristicLabel` = planning/implementation/debugging/review/testing/research/configuration/documentation/refactoring/data_analysis/session), materialized as durable index rows (`storage/sqlite/archive_tiers/index.py:689-709`, table `session_work_events` with `work_event_type`, `confidence`, `evidence_json`, FTS). So `phase-segment` is **not a new table** — it is a *DSL unit source + macro-phase projection* over existing `session_work_events` rows. Re-adding `kind` to `session_phases` would revert a construct decision.

2. **The candidate→verified state machine already exists — reuse it, don't reinvent.** `core/enums.py:437-447` `AssertionStatus` = ACTIVE/CANDIDATE/ACCEPTED/REJECTED/DEFERRED/**SUPERSEDED**/DELETED/INACTIVE. `storage/sqlite/archive_tiers/user_write.py:1002-1119` shows the exact pattern for both `TRANSFORM_CANDIDATE` and `PATHOLOGY`: mined findings become **`AssertionStatus.CANDIDATE`, `visibility=PRIVATE`, `context_policy={"inject": False, "promotion_required": True}`**, keyed by a deterministic idempotent id, with the never-downgrade guard `if existing is not None and existing.status != CANDIDATE: keep existing`. `insights/transforms.py:280-292` already mines `DecisionCandidate` (`_DECISION_RE` line 85, instruction-dump rejection gate `_instruction_dump_rejection_reason` line 1978, `candidate_ref` sha256 line 1988) and `upsert_transform_candidate_assertions` (user_write.py:1002) already mirrors them into user.db. This is the recovery-digest fix (#2482) generalized: mined intent lives as non-injected candidates carrying `raw_refs`, never as fact.

Net: all three units are **thin extensions of existing derivation + the existing candidate state machine**, plus new DSL unit-source registration. This keeps them cheap and construct-honest.

---

## Unit A — `phase-segment` (queryable intent-span unit + macro-phase projection)

### (1) Schema / DDL / tier / table-vs-VIEW
- **No new physical table.** Reuse `session_work_events` (index.db, tier v24, rebuildable). It already carries every field a phase-segment needs.
- **Add a derived read projection `phase_kind`** collapsing the 11 `WorkEventHeuristicLabel` values into 4 canonical macro-phases via a *pure deterministic map* (planning←{planning, research}; implementing←{implementation, refactoring, configuration, documentation}; debugging←{debugging}; verifying←{review, testing}; `session`/unknown→`unlabeled`). Implement as a Python mapping applied at read + a generated SQL `CASE` expression, **not** a stored column (derived-tier rule: no migration for a projection).
- **New DSL unit source `phase-segment`** in `archive/query/` (spec.py / unit_results.py / fields.py), joining the existing set `sessions/actions/messages/observed-events`. Fields: `phase_kind`, `work_event_type`, `confidence`, `duration_ms`, `session_id`, `started_at_ms`, `file_paths`, `tools_used`. Enables `sessions where repo:x | group by phase_kind | sum duration_ms` and `phase-segment where phase_kind:debugging with duration`.
- **`workflow_shape` as a second unit-source aggregation** (or a pipeline over `phase-segment`): the session-level `workflow_shape` (`insights/archive_models.py:105`) becomes derivable as the dominant-duration `phase_kind` per session, letting `workflow_shape_distribution` (MCP tool `server_insight_tools.py`) be re-expressed as a DSL pipeline rather than bespoke code.

### (2) Derivation + construct-gate algorithm
- Derivation is unchanged (`extract_work_events`); this unit only *exposes* rows. The construct gate is in the **projection**: a work_event only projects to a confident `phase_kind` when `confidence >= 0.5` AND `support_level != WEAK`; otherwise `phase_kind = "unlabeled"` (never silently promote a `weak_signal`/`no_tools` row — see `_classify_range` fallbacks at extraction.py:355-359 which emit 0.4 confidence). Macro-phase confidence = the underlying work_event confidence; the map never *invents* confidence.
- Aggregations (duration-by-phase, workflow-shape) must expose an `unlabeled_fraction`; if `unlabeled_fraction > 0.5` the session-level `workflow_shape` stays `unknown` (mirrors existing `workflow_shape_confidence` semantics).

### (3) Migration
- Derived tier → **no numbered migration**. Add the unit-source + projection to canonical query modules; if any new stored column were added it would require a `index.db` schema bump (v24→v25) + rebuild-plan doc, but the projection approach avoids that entirely. Ship as index-only/additive-derived. Regenerate topology projection (`devtools render topology-projection`) if new modules are added, plus `render cli-output-schemas`/`render openapi` if the unit surfaces on CLI/MCP.

### (4) Test strategy
- Property test (`tests/property`): every `WorkEventHeuristicLabel` maps to exactly one `phase_kind`; the map is total and stable.
- Construct-gate unit test: a `SessionBuilder` session yielding a `weak_signal` (0.4) work_event projects to `unlabeled`, not `implementing`.
- DSL law test (extend `tests/unit/cli/test_query_exec_laws.py`): `phase-segment` unit obeys projection/group-by/count laws; `sum duration_ms | group by phase_kind` totals equal the session wall-active time within tolerance.
- Aggregation honesty: a session that is 60% `unlabeled` yields `workflow_shape=unknown`.

### (5) Bead — acceptance criteria
`feat(query): phase-segment unit source + macro-phase projection over session_work_events`
- AC1: `phase-segment` registered as DSL unit source alongside sessions/actions/messages/observed-events; `explain_query_expression` lists it.
- AC2: deterministic 11→4 `phase_kind` map with `unlabeled` fallback; confidence-gated (≥0.5 AND non-WEAK).
- AC3: `workflow_shape_distribution` re-expressible as a `phase-segment` pipeline; existing MCP tool output unchanged (regression snapshot).
- AC4: `session_phases` (time-gap) left untouched — no `kind`/intent column added (guard test asserting the table has no intent column).

### (6) Top-3 risks
1. **Re-conflating the deliberate phase/work-event split.** Mitigation: AC4 guard test + doc note; the unit reads `session_work_events`, never `session_phases`.
2. **`workflow_shape` double-source drift** — session-level `workflow_shape` (`_workflow_shape` in `archive/session/runtime.py`) and the new phase-derived one could disagree. Mitigation: pick one as canonical (keep existing session-level as source of truth; the pipeline form is a *view*, verified equal by test) rather than shipping two definitions.
3. **DSL positional/param drift** — new unit + fields must not shift Click positional args (CLAUDE.md gotcha: new params go last); verify via `test_verb_cardinality.py`.

---

## Unit B — `goal` (construct-gated session objective; pairs with abandonment)

### (1) Schema / DDL / tier / table-vs-VIEW
- **Two-tier by durability:**
  - **Candidates → index.db (rebuildable).** Reuse existing `SessionEnrichmentPayload.is_goal_session/goal_text/goal_outcome` (`insights/archive_models.py:200-207`, from #1687) as the materialized candidate surface — no new table; extend the enrichment payload with `goal_source` (`operator_command`|`first_human_message`|`run_state_heading`), `goal_confidence`, `goal_status` (candidate/promoted). Expose a `goal` DSL unit/field for querying.
  - **Confirmed → user.db (durable).** New `AssertionKind.GOAL = "goal"` in `core/enums.py:399`. user.db assertion column is TEXT (no CHECK) so no user-tier schema bump — but the enum **is embedded in `render openapi` + `render cli-output-schemas`** (CLAUDE.md gotcha), so regenerate both. Confirmed goals are `AssertionStatus.ACTIVE`, operator-authored (`author_kind="operator"`), optionally `context_policy={"inject": true}` when the operator wants it in recall.
- **Table-vs-VIEW:** candidate is a materialized enrichment field (row); the "unmet goal" abandonment pairing is a **read-time VIEW** joining goal candidate/assertion ↔ `session_profiles.terminal_state`.

### (2) Derivation + construct-gate algorithm
Mine goal from three ordered sources, highest-trust first:
1. `operator_command` (material_origin, `core/enums.py`) `/goal ...` in first user message — this is #1687's path, highest confidence (~0.9), `goal_source=operator_command`.
2. `RunStateSummary.goal` (`transforms.py:1812` `_extract_run_state`, `_RUNSTATE_SECTION_RE` "goal:" heading) — mid confidence (~0.6).
3. First **`material_origin=human_authored`** message (excludes Claude Code protocol `role=user` rows — the honest-authorship axis from CLAUDE.md) — lowest confidence (~0.4), gated hard.
- **Construct gate (the load-bearing part):** a mined goal is **always `AssertionStatus.CANDIDATE`, `inject:false`, `promotion_required:true`** unless it came from an explicit `/goal` operator_command (which is authored intent, not inference → may go ACTIVE directly). Apply the **`_instruction_dump_rejection_reason` gate** (transforms.py:1978): a first message that is an instruction dump (CLAUDE.md paste, ≥30 lines/≥600 words + imperative markers) is **rejected**, `goal_status=rejected`, reason `instruction_dump_without_local_goal_evidence` — this is exactly what prevents a pasted mandate from masquerading as "the user's goal." Deterministic idempotent id via the `_candidate_ref` sha256 pattern; never-downgrade guard on re-materialize.
- **Abandonment pairing:** `find_abandoned_sessions` (`server_insight_tools.py:299`) currently ranks purely on `terminal_state` (question_left/error_left/tool_left/agent_hanging). Extend its item payload with the session's `goal_text` + `goal_status` so an abandoned session surfaces *what objective was left unmet*, not just that it dangled. Pure additive read-join.

### (3) Migration
- Candidate side: index.db additive-derived (extend enrichment payload) → rebuild via `polylogue ops reset --index && polylogued run`, no numbered migration.
- Confirmed side: `AssertionKind.GOAL` is schema-free TEXT in user.db → **no numbered migration**, but regenerate `render openapi` + `render cli-output-schemas` (enum is embedded) and add a `user_audit` surface entry (`user_audit.py:23` maps kinds → surfaces; every kind needs one per the every-kind invariant). `scope_ref`/`author_ref` must use a **registered ObjectRef kind** (MEMORY gotcha: `insight:`/`session:`, not an invented prefix).

### (4) Test strategy
- **Recovery-digest regression is the anchor test** (see cross-cutting section below): a session whose first message is a pasted CLAUDE.md/mandate produces **no ACTIVE goal** — either rejected or CANDIDATE `inject:false`.
- Source-precedence test: `/goal` beats run_state heading beats first-human-message.
- Authorship gate: a Claude Code protocol `role=user` (`material_origin=runtime_protocol`) row is **not** mined as goal (only `human_authored`).
- Promotion state-machine test: mine → CANDIDATE; operator promote → ACTIVE; re-materialize does **not** downgrade the promoted assertion (mirror `test_archive_tiers_user_write.py` pattern).
- Abandonment-pairing test: abandoned session with a mined goal surfaces `goal_text` in `find_abandoned_sessions`.

### (5) Bead — acceptance criteria
`feat(insights): construct-gated goal candidates + AssertionKind.GOAL, paired with abandonment`
- AC1: goal mined from operator_command / run_state / first-human-message with source-labelled confidence; authorship-gated to `human_authored`.
- AC2: instruction-dump gate rejects pasted mandates; non-operator mined goals stay CANDIDATE `inject:false`.
- AC3: `AssertionKind.GOAL` added; openapi + cli-output-schemas + user_audit surface regenerated; operator promotion → ACTIVE, idempotent, never-downgrade.
- AC4: `find_abandoned_sessions` surfaces unmet `goal_text`/`goal_status`.

### (6) Top-3 risks
1. **Pasted-mandate-as-goal fabrication** (the #2482 failure mode for goals). Mitigation: instruction-dump gate + `human_authored`-only + CANDIDATE default; AC2 test.
2. **New AssertionKind breaks generated surfaces silently** — `render all --check` prints "sync OK" but exits 1 (CLAUDE.md gotcha). Mitigation: grep for "out of sync"; regenerate openapi/cli-schemas/user_audit in the same PR.
3. **Confidence inflation across sources** — a 0.4 first-message goal treated equal to a 0.9 `/goal`. Mitigation: source-keyed confidence bands, never a flat score; abandonment view shows `goal_source`.

---

## Unit C — `decision-object` (mined decisions + supersession edges, recursive-safety-gated)

### (1) Schema / DDL / tier / table-vs-VIEW
- **Extend the existing user.db assertion path, do not add a new tier.** `AssertionKind.DECISION` (`core/enums.py:419`) already exists (user-authored). `DecisionCandidate` (transforms.py:280) is already mined and mirrored via `upsert_transform_candidate_assertions` as `TRANSFORM_CANDIDATE` candidates. This unit **unifies** them: mined decisions become `AssertionKind.DECISION` at `AssertionStatus.CANDIDATE` (instead of the generic TRANSFORM_CANDIDATE), so user-authored and mined decisions share one queryable kind, distinguished by `author_kind` (`operator` vs `transform`) and `status`.
- **Supersession edges → reuse `AssertionStatus.SUPERSEDED`** (already in the enum, line 445) plus a `superseded_by` ref in the assertion `value` JSON (TEXT, no schema bump). A decision→superseding-decision edge is: promote new decision to ACTIVE, mark prior `SUPERSEDED` with `value.superseded_by = <new assertion_id>`. The edge relation is a **read-time VIEW** over assertion rows (`superseded_by` chain), not a new edge table.
- Tier: user.db (durable) for the assertion rows; the supersession-chain view is computed on read.

### (2) Derivation + construct-gate + recursive-safety algorithm
- Derivation: existing `_extract_decision_candidates` (transforms.py:1923) — `_DECISION_RE` ("decision|decided|choose|chosen:"), `_PRODUCT_DECISION_ANCHOR_RE` accept-gate, `_instruction_dump_rejection_reason` reject-gate. Reuse verbatim; only change the mirror target kind.
- **Construct gate:** identical to Unit B — mined = CANDIDATE, `inject:false`, `promotion_required:true`; only operator promotion moves to ACTIVE. A rejected/instruction-dump decision is stored with `status=rejected` and reason, never rendered as fact (this is already how `SuccessorContextEntry` treats rejected candidates as `caveat`/`rejected_candidate` at transforms.py:1177-1204).
- **Recursive-safety gate (the new part):** supersession chains must be acyclic. Mirror the **`TopologyEdgeStatus.quarantined`** cycle-break pattern used for `session_links` (CLAUDE.md lineage section): when promoting D_new to supersede D_old, walk the `superseded_by` chain from D_old; if it reaches D_new, refuse the edge and mark it `quarantined` (record the attempted edge, do not write the cycle). Also bound chain-recompose depth (no unbounded recursion on read). A decision cannot supersede itself; a superseded decision cannot re-supersede its superseder.

### (3) Migration
- No numbered migration (user.db TEXT column, values-only change). Regenerate `render openapi`/`render cli-output-schemas` only if new fields surface. `author_ref`/`scope_ref` must use registered ObjectRef kinds (`insight:session_digest_v0@vN`, `session:<id>`). Add/confirm `user_audit.py` DECISION surface entry (line 23 already maps `"decisions": AssertionKind.DECISION`).

### (4) Test strategy
- **Recovery-digest regression (shared anchor):** a mined "we decided to ship X" is CANDIDATE, `inject:false`, absent from any recall/context bundle until promoted. Assert `compose_context`/successor-context omits unpromoted decisions or marks them `rejected_candidate`/`caveat`.
- Cycle test: promoting D_c to supersede D_a where D_a→D_b→D_c already chains is refused/quarantined; no cycle persists; read-side chain-walk terminates.
- Unify test: user-authored and mined decisions both queryable under `AssertionKind.DECISION`, separable by `author_kind`.
- Never-downgrade: an operator-promoted (ACTIVE) decision is not reverted to CANDIDATE by re-materialization (user_write.py:1090 pattern).
- Supersession-view test: `superseded_by` chain view returns the active head and full lineage.

### (5) Bead — acceptance criteria
`feat(user): mined decision candidates unified into AssertionKind.DECISION with cycle-safe supersession`
- AC1: mined `DecisionCandidate` mirrors into `AssertionKind.DECISION` at CANDIDATE (`inject:false`), sharing the kind with user-authored decisions; distinguished by `author_kind`/`status`.
- AC2: supersession edges via `AssertionStatus.SUPERSEDED` + `value.superseded_by`; read-time chain view.
- AC3: recursive-safety — cycle attempts quarantined, not written; bounded read recompose.
- AC4: unpromoted mined decisions never injected into context/recall (recovery-digest regression passes).

### (6) Top-3 risks
1. **Mined decision masquerading as durable fact** (the #2482 incident, decision variant). Mitigation: CANDIDATE-default + instruction-dump reject gate + AC4 context-omission test.
2. **Supersession cycles corrupting the active-decision head** (mirrors the `branch_point_message_id` non-FK cascade hazard). Mitigation: quarantine-on-cycle + acyclicity invariant test; store `superseded_by` in value JSON, never a cascading FK.
3. **Merging mined into the user-authored DECISION kind blurs provenance** — an operator could mistake a mined candidate for their own decision. Mitigation: hard `author_kind`/`author_ref` separation + `status`-gated surfacing; mined never ACTIVE without explicit promotion.

---

## Cross-cutting: the candidate→verified state machine (mandatory, shared by B and C)

All mined intent flows through the **existing** lifecycle, never bypassing it:

```
mine → AssertionStatus.CANDIDATE, visibility=PRIVATE,
       context_policy={"inject": false, "promotion_required": true},
       deterministic id (sha256 candidate_ref), never-downgrade guard
   → operator judgment → ACCEPTED / REJECTED / DEFERRED
   → ACCEPTED promotes to ACTIVE (may set inject:true)
   → ACTIVE decision superseded → SUPERSEDED (+ superseded_by, cycle-checked)
```

This is not new infrastructure — it is `user_write.py:1002-1119` (`upsert_transform_candidate_assertions` / `upsert_pathology_findings_as_assertions`) applied to goal and decision. The invariant that must hold for every mined row: **a mined claim is never `inject:true` and never ACTIVE without explicit operator promotion.**

### The recovery-digest incident as the canonical shared test case
Per MEMORY (2026-06-29) and #2482: the recovery report once *fabricated* "PR #123 merged" by regex-mining prose with no authorship gating, rendering inference as fact. The fix was (a) read structured outcomes from keystone tool-result fields, not prose, and (b) demote text-mined items to *unverified candidates* carrying `raw_refs`. Both new units must ship a regression test built on that exact shape:

> A synthetic session containing prose like "we decided to ship X" / "goal: rewrite the parser" / a pasted CLAUDE.md mandate produces **candidate assertions only** — `inject:false`, `status=candidate` (or `rejected` for the instruction dump), each carrying `raw_refs` back to the source message — and **nothing** appears as ACTIVE/injected fact in any recall, context-preamble, successor-context, or `find_abandoned_sessions` surface until an operator promotes it.

Put this test in `tests/unit/storage/` (assertion side) and `tests/unit/insights/` (mining side); it is the single strongest guard that mined intent never masquerades as fact.

---

## Key file map (for the implementers)
- Phase/work-event derivation: `polylogue/archive/phase/extraction.py`, `polylogue/archive/session/extraction.py`
- Index row storage/DDL: `polylogue/storage/sqlite/archive_tiers/index.py:689-751`, `polylogue/storage/insights/session/storage.py:700-727`
- Insight registry (add unit metadata): `polylogue/insights/registry.py:454` (session_phases), `:415` (work_events)
- DSL unit sources: `polylogue/archive/query/{spec,unit_results,fields,expression}.py`
- Mining + candidates: `polylogue/insights/transforms.py:280-292` (DecisionCandidate), `:1812-1990` (run_state/decision/gate)
- Candidate→assertion state machine: `polylogue/storage/sqlite/archive_tiers/user_write.py:1002-1174`
- Enums: `polylogue/core/enums.py:399` (AssertionKind, add GOAL), `:437` (AssertionStatus, SUPERSEDED exists)
- Goal enrichment (existing #1687 surface): `polylogue/insights/archive_models.py:200-207` (SessionEnrichmentPayload)
- Abandonment tool: `polylogue/mcp/server_insight_tools.py:299`
- user_audit surface map: `polylogue/storage/sqlite/archive_tiers/user_audit.py:23`

**Verification note for all three:** derived-tier changes → `polylogue ops reset --index && polylogued run`, no numbered migration; any new module → `devtools render topology-projection && devtools render topology-status`; any AssertionKind/CLI/MCP surface → regenerate `render openapi` + `render cli-output-schemas` and grep output for "out of sync" (the "sync OK" exit-1 trap). Test via `devtools test <file>`, never blanket directory runs.

---

## [a4a1d85884af2a69d] Wave-5 Spec: migration-fuzz + scale-regression harness

I have full grounding. Here is the tight implementation spec.

---

# Wave-5 Spec: migration-fuzz + scale-regression harness

## Grounding (what already exists vs. what's missing)

| Surface | State | File |
|---|---|---|
| Durable migration runner | EXISTS — `migrate_archive_tier(conn, tier, *, backup_manifest)`; contiguous-chain check, `BEGIN IMMEDIATE`, per-step `PRAGMA user_version`, `PRAGMA quick_check`, rollback-on-fail | `polylogue/storage/sqlite/migration_runner.py` |
| Migration SQL (sparse) | EXISTS — `source/002_raw_capture_multimap.sql`, `user/004_user_settings.sql` (chain is sparse: fresh init writes `user_version=target`; migrations replay the gap for OLD archives at `target-1`) | `polylogue/storage/sqlite/migrations/{source,user}/` |
| Example migration tests | EXISTS — 6 example-based tests seeding `_create_source_v1`/`_create_user_v3` then migrating | `tests/unit/storage/test_durable_migrations.py` |
| Scale-regression probe + lane | EXISTS (bead 1xc.7) — 6 hardcoded checks; lane `scale-regression` runs `devtools workspace scale-regression --json`; ad-hoc commit-spy via `progress_callback` + `load_sync_batch` monkeypatch | `devtools/scale_regression_probe.py`, `devtools/validation_lane_catalog_contracts.py:406` |
| Tiered scale fixtures | EXISTS but CLEAN — small/medium/large, realistic distribution only; markers `scale_small/medium/large`, `scale(level)` registered | `tests/infra/scale_fixtures.py`, `pyproject.toml:129` |
| `literal_check(col, *get_args(Literal))` | EXISTS — embeds `typing.Literal` args into INDEX-tier CHECK DDL (lines 926–1009) | `archive_tiers/index.py`, `archive_tiers/common.py:18` |
| **Migration-fuzz property** | **MISSING** — no Hypothesis property over a *populated* old-shape corpus | — |
| **Adversarial scale generator** (collision/dup/giant knobs) | **MISSING** — "the 1xc.1 substrate"; `scale_fixtures` has no collision-density / dup-native / giant-artifact knobs | — |
| **Reusable WAL/commit-boundary spy** | **MISSING** — logic is inlined in the probe | — |
| **Metamorphic identity/idempotency/lineage laws** | **MISSING** as a property suite | — |
| **CHECK↔Literal drift property** | **MISSING** | — |

Design goal: promote the existing ad-hoc probe machinery into three reusable infra pieces (generator, spy, laws), then build the migration-fuzz and drift properties on top. Do **not** duplicate the 6 example tests — the property *generalizes* them over populated, fuzzed corpora.

---

## (1) Harness / infra design

Three new infra modules under `tests/infra/`, one property suite under `tests/property/`, one drift property under `tests/unit/storage/`.

### 1a. `tests/infra/scale_corpus.py` — adversarial scale-corpus generator (the missing 1xc.1 substrate)

A **shape-parameterized** synthetic corpus distinct from `scale_fixtures.py` (which is realistic-distribution and clean). This one injects the exact pathologies that hid the tier-1 bugs.

```
@dataclass(frozen=True)
class ScaleShape:
    sessions: int                    # N
    messages_per_session: int        # M (or a (min,max) via seeded rng)
    hash_collision_density: float    # fraction of sessions forced to share content_hash prefix
    dup_native_id_rate: float        # fraction reusing a native_id already emitted (across origins)
    giant_artifact_bytes: int        # inject >=1 single block/raw row of this size (e.g. 384MB class → scaled)
    lineage_fork_rate: float         # fraction that are prefix-sharing forks/resumes of an earlier session
    subagent_collision: bool         # emit two subagents sharing (tool_id) → run_ref fallback-collision class
    seed: int

def build_scale_corpus(root: Path, shape: ScaleShape) -> ScaleCorpusManifest: ...
```

- **Reuses** existing machinery, does not reinvent: `ParsedSession/ParsedMessage/ParsedContentBlock` + `write_parsed_session_to_archive` (as the probe already does), `SessionBuilder` from `tests/infra/storage_records.py`, and lineage shaping cribbed from `devtools/ingest_throughput_probe.py:_build_lineage_sessions`.
- **Deterministic** from `seed` (single `random.Random(seed)`); returns a `ScaleCorpusManifest` recording *logical* truth the laws assert against: `{logical_session_count, physical_session_count, expected_distinct_content_hashes, dup_native_id_groups, giant_session_ids, fork_edges}`.
- **Giant-artifact injection scaled, not literal**: default `giant_artifact_bytes` small enough for unit scope (~a few MB), a `scale_large`/lane profile bumps it. Never seed a literal 28GB/384MB row in unit scope (bead 1xc.1 pitfall).
- Exposes named presets: `SHAPE_UNIT` (tiny, all pathologies present at ≥1 instance — for markerless correctness), `SHAPE_SCALE_SMALL/MEDIUM/LARGE` (behind the `scale` marker).

### 1b. `tests/infra/commit_spy.py` — WAL/commit-boundary spy fixture

Generalizes the inlined `observe()` + `conn.commit` counting in the probe into a reusable fixture usable by both pytest and the devtools probe.

```
@dataclass
class CommitSpy:
    boundaries: list[CommitEvent]           # (seq, wal_pages, walltime, visible_row_snapshot)
    def wrap(self, conn) -> Connection       # patches conn.commit to record + PRAGMA wal_checkpoint(PASSIVE) read of wal frame count
    def max_wal_frames(self) -> int
    def mid_run_visible(self, table, key_col) -> list[set]   # snapshot via a 2nd read-only file: URI conn at each boundary

@pytest.fixture
def commit_spy() -> CommitSpy: ...
```

- Records: number of commit boundaries, peak WAL frame count between boundaries (`PRAGMA wal_checkpoint` / `pragma_wal_pages`), and — critically — a **read-only second connection** snapshot of a target table at each boundary (the probe's `mid_run_visible` trick). This is the substrate for the *no-single-transaction* and *no-empty-window* laws.
- The bounded-WAL law becomes: `spy.max_wal_frames() < single_transaction_frames * K` where `single_transaction_frames` is measured by a control run with the message budget forced to `inf`.

### 1c. `tests/infra/migration_corpus.py` — old-shape durable-corpus builder for fuzz

Factors the `_create_source_v1`/`_create_user_v3` helpers out of `test_durable_migrations.py` and makes them **row-populating and version-parameterized**, so the property can seed a *populated* archive at any `start_version ∈ [1, target)` and assert survival across the chain.

```
def build_old_durable_tier(
    path: Path, tier: ArchiveTier, *, at_version: int, rows: OldTierRows
) -> None
    # writes the historical DDL for `at_version` (from a small per-version DDL registry) + INSERTs rows,
    # sets PRAGMA user_version = at_version.

def snapshot_tier(conn) -> TierSnapshot   # row keysets per table, user_version, PRAGMA quick_check, integrity_check
```

- `OldTierRows` is produced from a Hypothesis strategy (below). Includes rows exercising the migration's own semantics — e.g. for source `002` (the multimap index) seed `raw_sessions` with **duplicate (origin, native_id)** and NULL `native_id` rows so the new partial index actually indexes something.

---

## (2) Generators + laws (pseudocode)

### 2a. Hypothesis strategies

```
# tests/infra/strategies/migration.py
old_tier_rows(tier) = fixed_dictionaries per historical table:
    source: raw_sessions rows with
        origin ∈ sampled(Origin values),
        native_id ∈ one_of(none(), text(min=1)),   # NULL-native + dup-native both reachable
        blob_hash = 32 bytes, source_index ∈ integers(0,3),
        # deliberately allow duplicate (origin, native_id) tuples across rows
    user: assertions rows with kind ∈ sampled(AssertionKind values),
        target_ref/scope_ref refs, context_policy_json ∈ sampled({'{"inject":false}', ...})
start_version(tier) = integers(1, target_version(tier) - 1)   # any resumable old shape

# scale shape strategy (bounded for CI)
scale_shape() = builds(ScaleShape,
    sessions=integers(3, 40), messages_per_session=integers(1, 30),
    hash_collision_density=floats(0, 0.5), dup_native_id_rate=floats(0, 0.5),
    giant_artifact_bytes=sampled([0, 1<<20, 4<<20]),
    lineage_fork_rate=floats(0, 0.4), subagent_collision=booleans(), seed=integers())
```

### 2b. Migration-fuzz property (the headline)

```
# tests/property/test_migration_fuzz.py
@given(tier=sampled([SOURCE, USER]), start=start_version(tier), rows=old_tier_rows(tier))
@settings(profile-aware, deadline=None)
def test_populated_old_corpus_migrates_without_loss(tmp, tier, start, rows):
    path = build_old_durable_tier(tmp, tier, at_version=start, rows=rows)
    before = snapshot_tier(open(path))
    manifest = write_valid_backup_manifest(tmp, tiers=[f"{tier.value}.db"])

    result = migrate_archive_tier(open(path), tier, backup_manifest=manifest)
    after = snapshot_tier(open(path))

    # LAW-M1 no row loss: every pre-existing row key survives (additive-only invariant)
    assert before.row_keys(table) ⊆ after.row_keys(table)  ∀ migration-unaffected tables
    # LAW-M2 version monotonic + terminal: after.user_version == target_version(tier)
    #        and result.applied_versions == range(start+1, target+1)
    # LAW-M3 constraint satisfaction: after.quick_check == "ok" and integrity_check == "ok"
    # LAW-M4 STRICT/CHECK still hold: re-inserting a known-good row succeeds; a known-bad row raises IntegrityError
    # LAW-M5 idempotent chain: applying migrate_archive_tier AGAIN is a no-op (applied_versions == ())
```

Plus two **negative** properties (fast, example-or-fuzzed) to prove the guardrails, not just the happy path:

```
# LAW-M6 missing-file gap: if a middle migration file were absent, the incomplete-chain MigrationError fires
#         (inject via monkeypatched _load_migrations dropping one step)
# LAW-M7 no-manifest refusal: migrate without a backup manifest → MigrationError, DB untouched (snapshot equal)
# LAW-M8 rollback atomicity: inject a failing statement into the last step → rollback,
#         user_version and rows unchanged (proves BEGIN IMMEDIATE + rollback path)
```

### 2c. Metamorphic laws over the scale corpus (identity / idempotency / lineage)

```
# tests/property/test_scale_metamorphic.py
@given(shape=scale_shape())
def test_identity_regeneration(shape):
    corpus = build_scale_corpus(root, shape)
    # LAW-I1 generated-id determinism: session_id == origin || ':' || native_id for every row
    #   message_id / block_id match their COALESCE formula → recomputed == stored (no orphan ids)
    # LAW-I2 dup-native collision safety: dup (origin,native_id) collapses to ONE canonical session,
    #   NOT a silent overwrite of a different origin's row (the run_ref #2464 class, generalized)
    # LAW-I3 hash-collision independence: sessions sharing a content_hash prefix still get distinct ids
    #   and distinct rows (no PK collapse from non-unique local coordinates)

def test_idempotency(shape):
    # LAW-D1 re-ingest with matching content_hash is skipped (row count + ids stable);
    #   differing hash updates in place, never duplicates a logical session

def test_lineage_no_double_count(shape):   # requires lineage_fork_rate > 0
    # LAW-L1 physical vs logical: sum(physical message rows) reflects divergent tails only;
    #   composed transcript length == parent_prefix + child_tail, NOT parent+child (no 2× replay)
    # LAW-L2 branch_point integrity: branch_point_message_id resolves; a parent full-replace
    #   (delete+reinsert) does NOT null it (the deliberate not-a-FK invariant)
```

### 2d. CHECK↔Literal drift property

```
# tests/unit/storage/test_check_literal_drift.py
LITERAL_BACKED_CHECKS = [   # one entry per literal_check() call site in archive_tiers/index.py
    ("session_runs", "harness", RunHarness),
    ("session_runs", "status",  RunStatus),
    ("observed_events", "kind", ObservedEventKind),
    ... # exhaustive; keep in lockstep with index.py:926-1009
]
def test_check_constraint_matches_literal(table, column, literal_alias):
    fresh = init index.db
    ddl = fetch CREATE TABLE sql from sqlite_schema for `table`
    allowed = parse the `column IN (...)` set out of the CHECK clause
    assert allowed == set(get_args(literal_alias))    # LAW-C1: SQL vocabulary == Python Literal args
    # LAW-C2 behavioral: every literal value inserts OK; a synthesized out-of-set value raises IntegrityError
# Guard against silent divergence: also assert the *count* of literal_check sites == len(LITERAL_BACKED_CHECKS)
# by grepping index.py, so a newly-added CHECK without a table row fails the test (drift tripwire).
```

---

## (3) Integration into `devtools verify`

- **Default gate (per-PR, testmon-selected):**
  - `test_migration_fuzz.py` and `test_check_literal_drift.py` are **plain unit/property** tests with a **capped Hypothesis budget** (`HYPOTHESIS_PROFILE=ci` → ~25 examples). They touch `migration_runner.py`/`index.py` DDL, so testmon selects them automatically on any migration/enum/CHECK change. No marker — they must run in the default gate.
  - `test_scale_metamorphic.py` runs at `SHAPE_UNIT` (tiny) **markerless** for correctness, and its `scale_small/medium/large` parametrizations carry the `scale` marker so the default gate's marker filter defers them (consistent with `scale_fixtures` policy, `TESTING.md:48`).
- **`--lab` gate:** `SHAPE_SCALE_MEDIUM` metamorphic runs (`@pytest.mark.scale("medium")`), mirroring the medium-tier fixture lane.
- **Optional scale-regression lane (extend, don't fork):** add the new invariants as `ScaleRegressionCheck`s to the **existing** `devtools/scale_regression_probe.py` so `devtools workspace scale-regression --json` and lane `scale-regression` gain: `migration_chain_no_row_loss`, `identity_regeneration`, `lineage_no_double_count`, `check_literal_no_drift`. Refactor the probe's inlined commit-observe to consume `tests/infra/commit_spy.CommitSpy` (single source of truth). Regenerate `devtools render quality-reference` + `render all --check` (grep for `out of sync`, per CLAUDE.md gotcha).
- **Marker registration:** the generic `scale(level)` marker already exists (`pyproject.toml:134`) — reuse it; no new marker needed. Add `stretch` shape usage under it (the registration already lists `small/medium/large/stretch`).
- **Nightly:** `SHAPE_SCALE_LARGE` under `scale_large`.

---

## (4) Test strategy

- **Mutation-first validation (the AC that matters):** each law must **fail against the pre-fix code**. Prove it by local mutation, don't just assert green:
  - migration LAW-M8 ↔ delete the `conn.rollback()` → property must catch dirty state.
  - LAW-M2 ↔ revert runner to skip the contiguous-chain check → M6 fails.
  - lineage LAW-L1 ↔ force full parent+child replay → double-count assertion fails.
  - drift LAW-C1 ↔ add a bogus value to a `Literal` without touching DDL → count/set mismatch fails.
  - commit-spy bounded-WAL ↔ set message budget to `inf` → `max_wal_frames` law fails (this is the 1xc.1 regression generalized to a *reusable* spy).
- **Determinism:** every generator seeded; property `@settings(derandomize=False, print_blob=True)` so CI failures are reproducible. Corpus manifest carries the seed.
- **Cost discipline:** unit-scope shapes bounded (≤40 sessions, giant artifact ≤4 MB); temp DBs under `/realm/tmp/polylogue-pytest` (TESTING.md). No literal large-archive in unit/property scope.
- **No overlap with `test_durable_migrations.py`:** keep the 6 example tests (they document specific v1→v2 / v3→v4 transitions); the property *generalizes* over populated fuzzed corpora and adds the negative-guardrail laws they lack.
- **Protected-file discipline:** additive only; touches none of the protected test files.

---

## (5) Bead breakdown (children under epic `polylogue-1xc`; 1xc.1 and 1xc.7 already closed)

**B1 — WAL/commit-boundary spy fixture (`tests/infra/commit_spy.py`)**
- AC: `CommitSpy` records commit-boundary count, peak WAL frame count, and per-boundary read-only visible-row snapshots. Probe `_check_message_budget_chunking` refactored to consume it (no behavior change; probe stays green). A control-run helper measures single-transaction WAL frames for ratio bounds. `devtools test tests/unit/... && devtools workspace scale-regression` green.

**B2 — Adversarial scale-corpus generator (`tests/infra/scale_corpus.py`) + strategy**
- AC: `build_scale_corpus(root, ScaleShape)` deterministic from seed; knobs for hash-collision density, dup-native-id rate, giant-artifact bytes, lineage-fork rate, subagent-collision. Returns `ScaleCorpusManifest` with logical-vs-physical truth. `SHAPE_UNIT` present with ≥1 instance of every pathology. Reuses `write_parsed_session_to_archive` / `SessionBuilder` / ingest-probe lineage shaping — no new write path. Unit smoke test asserts manifest matches seeded shape.

**B3 — Metamorphic identity/idempotency/lineage law suite (`tests/property/test_scale_metamorphic.py`)**
- AC: LAW-I1/I2/I3, D1, L1/L2 implemented over B2 corpus. `SHAPE_UNIT` markerless in default gate; `scale`-marked parametrizations for medium/large. Each law demonstrated to fail under a documented local mutation (dup-native overwrite, full lineage replay, PK-from-nonunique-coord). Green at unit shape.

**B4 — Old-shape migration corpus builder + fuzz property (`tests/infra/migration_corpus.py`, `tests/property/test_migration_fuzz.py`)**
- AC: `build_old_durable_tier` factors historical DDL out of `test_durable_migrations.py`, row-populating, version-parameterized. Property covers LAW-M1..M8 for SOURCE and USER over fuzzed populated corpora. Negative laws (no-manifest refusal, incomplete-chain, rollback atomicity) each fail under the corresponding runner mutation. Runs in default gate at `ci` Hypothesis profile; existing 6 example tests retained.

**B5 — CHECK↔Literal drift property (`tests/unit/storage/test_check_literal_drift.py`)**
- AC: `LITERAL_BACKED_CHECKS` exhaustively mirrors every `literal_check()` site in `index.py:926-1009`. LAW-C1 (SQL `IN`-set == `get_args(Literal)`) and LAW-C2 (behavioral insert accept/reject) pass. Drift tripwire: site-count assertion fails when a `literal_check` is added without a table row. Fails under a Literal-value-added mutation.

**B6 — Wire new invariants into the scale-regression lane + docs**
- AC: `scale_regression_probe.py` gains `migration_chain_no_row_loss`, `identity_regeneration`, `lineage_no_double_count`, `check_literal_no_drift` checks (consuming B1 spy / B2 corpus). `devtools workspace scale-regression --json` and lane `scale-regression` green with the added checks. `devtools render quality-reference` + `render all --check` clean (grep `out of sync`). Lane stays optional/scale tier, under `timeout_s`.

**B7 (optional) — Default/lab/nightly gate wiring + marker profiles**
- AC: metamorphic + migration-fuzz selected by testmon in default gate; `scale`-marked shapes deferred per marker policy; `--lab` runs medium, nightly runs large. `devtools verify --quick` and a `--lab` dry-run confirm selection. `TESTING.md` scale/property section updated to name the new suites.

Dependency order: B1 → {B2}; B2 → B3, B6; B4, B5 independent; B6 depends on B1+B2(+B3 checks); B7 last.

---

## (6) Top-3 risks

1. **Sparse-chain reality vs. fuzz assumptions.** The migration chain is *sparse* (source starts at `002`, user at `004`) because fresh init writes `user_version = target` and migrations only replay the gap for archives created at an older base. A naive `start_version ∈ [1, target)` strategy will generate versions with **no historical DDL** to build, and the runner's contiguous-chain check (`expected_versions = range(current+1, target+1)`) will legitimately raise `MigrationError` for gaps that have no file. **Mitigation:** the migration-corpus DDL registry must enumerate only the *real* historical shapes that have a migration path (drive `start_version` from a `sampled()` over versions that actually have a build recipe + a complete file chain to target), and treat "incomplete chain → MigrationError" as the *expected* outcome (LAW-M6), not a bug. Getting this boundary wrong makes the property either vacuous or falsely red.

2. **Adversarial-shape false positives masking real invariants.** Injecting duplicate `(origin, native_id)`, colliding content-hash prefixes, and subagent `tool_id` collisions can trip **legitimate** uniqueness/idempotency behavior that looks like a bug but is correct collapse (the docs are explicit that real Task subagents and dup-native across origins are legitimate). If the manifest's "logical truth" is computed wrong, laws I2/D1/L1 fail for the wrong reason. **Mitigation:** derive `ScaleCorpusManifest` logical counts from the *same* canonicalization rules the writer uses (content-hash idempotency, native-id coalescing), and validate the manifest itself with a small example test before trusting it as an oracle.

3. **Commit-spy portability / WAL-frame measurement fragility.** WAL frame counts depend on page size, checkpoint mode, and whether the harness runs on tmpfs vs. NVMe (`/realm/tmp` vs `/dev/shm`), and monkeypatching `conn.commit` interacts with the runner's own `BEGIN IMMEDIATE`/nested-savepoint gating (`commit_per_chunk = transaction_depth == 0` — 1xc.1 pitfall). An absolute WAL budget will be flaky. **Mitigation:** assert **ratios** against a same-host control run (budget=∞) as `scale_fixtures.py` already mandates, drive the top-level entrypoint (never a nested savepoint), and count *commit boundaries* + *mid-run visibility* (robust) as the primary signal, with peak-WAL as a secondary ratio check only.

No code or beads were modified. All file paths above are absolute-resolvable under `/realm/project/polylogue/`.

---

## [aba8e2f6f7ca863f0] Wave-5 Spec — the demo/proof engine (`*.polydemo`)

I have enough grounding. Here is the tight implementation spec.

---

# Wave-5 Spec — the demo/proof engine (`*.polydemo`)

Grounded in: `polylogue/demo/{constructs,models,verify,tour,seed,script}.py`, `polylogue/cli/commands/demo.py`, `devtools/claim_vs_evidence.py`, `polylogue/insights/{portfolio,pathology,postmortem}.py`, `polylogue/core/refs.py` (`EvidenceRef.format/parse/to_object_ref`), beads **polylogue-212** (demo-portfolio epic, compositionality rule, seed 1843, public+live variants) and **polylogue-sru / sru.1** (claim-vs-evidence to finding-grade; productize `followup_class` as query capability).

**Core problem this closes:** today `demo tour` hardcodes 4 CLI steps in Python (`tour.py:76-97`) and `claim_vs_evidence.py` is a 60KB bespoke report engine that violates 212's compositionality rule (shell/python is glue only). Neither emits a **content-addressed, citable finding** whose number provably resolves to structural evidence. This spec makes the demo a declarative document a stranger runs to reproduce a byte-identical citable artifact, with drift wired into CI.

---

## 1. The `.polydemo` format + finding schema

### 1.1 File format (`FILE.polydemo`)

A UTF-8 document: **YAML frontmatter** (budget + corpus binding) + **ordered steps** (product-primitive CLI/DSL invocations) + **assertions** + a **refusals** manifest. Parsed into `DemoScript` (new `demo/document.py`). No bespoke logic — every step is a `polylogue …` argv (compositionality rule); Python is sequencing/narration only.

```yaml
# --- frontmatter ---
polydemo: 1                        # format version (gate on mismatch)
id: claim-vs-evidence             # slug; owns artifact dir .agent/demos/<id>/
title: How often do agents proceed past a failed tool call?
corpus:
  seed: 1843                       # deterministic seed (demo.seed)
  datasheet_sha256: "b3f1…"       # REQUIRED: pins the seeded corpus (§1.3)
budget:
  first_result_s: 30               # reuse FIRST_RESULT_BUDGET_S default
  full_s: 420                      # reuse FULL_TOUR_BUDGET_S default
narration: |
  One completed multi-hour session; every number below drills to a
  tool_result outcome field, never prose.
---
# --- steps (ordered; argv after `polylogue`) ---
- step: archive-overview
  run: ["analyze", "--facets"]
  expect:
    exit: 0
    constructs: [multi_origin_sessions, tool_result_blocks]   # from DEMO_CONSTRUCTS

- step: silent-proceed-rate
  run: ["find", "actions where is_error:true | group by followup_class | count"]
  # ^ sru.1 product capability — DSL, not python
  finding: silent_proceed_rate                                # binds to a finding block below

# --- findings (citable claims) ---
findings:
  - id: silent_proceed_rate
    claim: "24.1% of failed tool calls are followed by a silent proceed (lower bound)."
    metric: {value: 0.241, unit: rate, kind: lower_bound}
    evidence_anchor: "blocks.tool_result_is_error"            # structural column, never prose
    produced_by: silent-proceed-rate                          # step id
    evidence_refs:                                            # EvidenceRef.format() strings
      - "codex-session:demo-1843-a::codex-session:demo-1843-a:4"
      - "claude-code-session:demo-1843-b::…:7"
    finding_id: "sha256:4c1a…"                                # EXPECTED value; CI recomputes & diffs

# --- refusals (anti-demo manifest; construct-validity honesty) ---
refusals:
  - question: "Was the agent *aware* it proceeded past the failure?"
    reason: "Awareness is not a structural field; only marker-based classification exists (ambiguous stays in denominator)."
  - question: "Did the silent proceed cause the final bug?"
    reason: "Causation is not represented; only temporal adjacency (next-turn) is."
```

### 1.2 `DemoFinding` schema (new Pydantic model in `demo/models.py`, sibling of `DemoVerifyResult`)

```python
class DemoFinding(ArchiveInsightModel):        # frozen, like PathologyFinding
    id: str                                    # local slug within the .polydemo
    claim: str                                 # human sentence
    metric: FindingMetric                       # {value, unit, kind: point|lower_bound|upper_bound}
    evidence_anchor: str                        # structural column/field name (validated allowlist)
    produced_by: str                            # step id that computed metric
    evidence_refs: tuple[EvidenceRef, ...]      # reuse core.refs.EvidenceRef
    corpus_datasheet_hash: str                  # copied from frontmatter at compute time
    finding_id: str                             # content address (§1.4)
```

### 1.3 Corpus datasheet (pins finding_id to a fixed world)

`seed_demo_archive` already produces `DemoSeedResult` (session_ids, message_count, construct_coverage). Add `corpus_datasheet()` in `demo/seed.py` returning a canonical dict:

```
{ seed: 1843,
  session_ids: sorted(DEMO_SESSION_IDS),
  message_count: <int>,
  constructs: { construct_id: observed, … } }   # from evaluate_demo_constructs
datasheet_sha256 = sha256(NFC(json.dumps(datasheet, sort_keys=True)))
```

Reuse `polylogue.core.hashing` NFC/sentinel canonicalization (same discipline as `pipeline/ids.py`). The datasheet hash is the fingerprint of the whole reproducible world; a finding is only valid against the corpus it was computed on.

### 1.4 `finding_id` = content address

```
finding_id = "sha256:" + hexdigest(
    NFC( canonical_json({
        "claim":            finding.claim,
        "metric":           finding.metric,          # value+unit+kind
        "evidence_anchor":  finding.evidence_anchor,
        "evidence_refs":    sorted(r.format() for r in evidence_refs),   # SORTED → order-independent
        "corpus_datasheet_hash": corpus_datasheet_hash,
    }) )
)
```

Properties: order-independent (sorted refs); world-pinned (datasheet hash); anchored (evidence_anchor in the hash — changing the number's provenance changes the id); **any drift in claim/metric/evidence/corpus flips the id → CI catches it** (§4 demo-as-CI-test).

---

## 2. Pipeline algorithms (pseudocode)

### 2.1 `polylogue demo run FILE` (new `demo/runner.py`, wired in `cli/commands/demo.py`)

```
run_polydemo(file, root, out_dir):
    doc = parse_polydemo(file)                       # DemoScript; gate polydemo==1
    assert_frontmatter_budget(doc.budget)

    # 1. deterministic world
    seed = await seed_demo_archive(root, force=True, with_overlays=True)   # seed 1843
    datasheet, datasheet_hash = corpus_datasheet(root)
    if datasheet_hash != doc.corpus.datasheet_sha256:
        FAIL("corpus datasheet drift", expected=doc…, actual=datasheet_hash)   # world changed

    # 2. construct-validity pre-render tripwire (§2.3) — BEFORE running steps
    tripwire_problems = construct_validity_tripwire(doc, root)
    if tripwire_problems: FAIL(tripwire_problems)

    # 3. execute steps as product primitives (like tour._run_cli_step)
    start = perf_counter(); first_result_s = 0
    for i, step in enumerate(doc.steps):
        r = subprocess([python,-m,polylogue, *step.run], env=tour_env(root))
        record_step(step, r)
        if i == 0: first_result_s = perf_counter() - start
        for construct_id in step.expect.constructs:
            assert_construct_ok(construct_id, coverage)           # reuse evaluate_demo_constructs
        assert step.expect.exit == r.returncode

    # 4. compute & gate findings
    findings_out = []
    for f in doc.findings:
        refs = [EvidenceRef.parse(s) for s in f.evidence_refs]
        resolver_gate(refs, root)                                 # §2.2 round-trip
        computed_id = compute_finding_id(f, datasheet_hash)
        if computed_id != f.finding_id:
            FAIL("finding_id drift", id=f.id, expected=f.finding_id, actual=computed_id)
        findings_out.append(DemoFinding(..., finding_id=computed_id))

    # 5. refusal invariant: no refused question may appear as a finding claim
    refusal_gate(doc.refusals, findings_out)                      # §2.4

    # 6. budgets + artifacts (reuse tour writers)
    problems += budget_problems(first_result_s, total_s, doc.budget)
    write_artifacts(out_dir, doc, findings_out, steps, datasheet)  # report.json/md, transcript, PUBLIC_REPRODUCTION.md, COLD_READER_GATE.md
    return DemoRunResult(ok=not problems, findings=findings_out, …)
```

### 2.2 Round-trip evidence-ref resolver gate

```
resolver_gate(refs, root):
    problems = []
    with ArchiveStore.open_existing(root, read_only=True) as arch:
        for ref in refs:
            obj = arch.resolve_ref(ref.to_object_ref())          # reuse existing resolve_ref surface
            if obj is None:
                problems.append(f"dangling evidence ref {ref.format()}")
            # round-trip: re-format resolved id must equal input (no silent coercion)
            elif obj.canonical_ref().format() != ref.format():
                problems.append(f"non-round-trip ref {ref.format()} -> {obj…}")
    if problems: FAIL(problems)
```

A finding whose citation does not resolve — or resolves to a *different* canonical id — is rejected. This is the anti-fabrication core (mirrors the sru lesson: the killed `missed_review` detector fabricated review state from prose).

### 2.3 Construct-validity pre-render tripwire

```
construct_validity_tripwire(doc, root):
    problems = []
    coverage = {c.construct_id: c for c in evaluate_demo_constructs(root)}
    for f in doc.findings:
        if f.evidence_anchor not in STRUCTURAL_ANCHOR_ALLOWLIST:   # §3
            problems.append(f"finding {f.id}: anchor {f.evidence_anchor} is not a structural column")
        # every finding must lean on at least one satisfied declared construct
        backing = anchors_to_constructs(f.evidence_anchor)         # anchor -> construct_ids
        if not any(coverage[c].ok for c in backing):
            problems.append(f"finding {f.id}: no satisfied construct backs anchor {f.evidence_anchor}")
    return problems
```

Fires **before** any number is rendered — a claim whose anchor is prose (not in `STRUCTURAL_ANCHOR_ALLOWLIST`, e.g. `blocks.text`) or unbacked by a satisfied `DemoConstruct` cannot be published.

### 2.4 Refusal gate (anti-demo manifest)

```
refusal_gate(refusals, findings):
    for r in refusals:
        for f in findings:
            if normalized_overlap(r.question, f.claim) > τ:       # cheap token-set / shared-anchor check
                FAIL(f"refused question '{r.question}' is answered by finding {f.id}")
```

Guarantees the honesty slide stays honest: you cannot list a question as "we cannot answer" and simultaneously ship a finding that answers it.

---

## 3. Structural-anchor allowlist (`demo/anchors.py`)

A closed set mapping honest structural evidence columns → the `DemoConstruct`s that prove they exist. Anchors NOT in this set are prose and rejected by the tripwire. Seeded from index-v16+ keystone columns:

| anchor | backing construct(s) |
| --- | --- |
| `blocks.tool_result_is_error` | `failed_tool_results`, `tool_result_blocks` |
| `blocks.tool_result_exit_code` | `tool_result_blocks` |
| `messages.material_origin` | `provider_usage_messages` |
| `messages.{input,output,cache_*}_tokens` | `provider_usage_messages` |
| `session_links.{link_type,inheritance}` | `prefix_sharing_links`, `subagent_links`, `continuation_links` |
| `session_profiles.terminal_state` | `unfinished_terminal_state_rows`, `error_terminal_state_rows` |
| `attachments.{acquisition_status,blob_hash}` | `acquired_attachment_rows` |

Explicitly **excluded**: `blocks.text`, `blocks.search_text` (prose — the whole point of sru).

---

## 4. Migration

1. **Additive, no schema change.** `.polydemo` files + `finding_id` are content over the existing derived tiers; no durable-tier migration, no `index.db` bump. The corpus datasheet is computed on read.
2. **`demo/anchors.py`, `demo/document.py` (parser), `demo/runner.py`, `demo/finding.py`** are new modules → **regenerate topology projection** (`devtools render topology-projection && devtools render topology-status`, commit `docs/plans/topology-target.yaml` + `docs/topology-status.md`) or `render all --check` fails.
3. **CLI:** add `demo run` subcommand in `cli/commands/demo.py` (mirror `tour_command`); update `render cli-output-schemas` + help snapshots (`tests/unit/cli/__snapshots__/test_help_snapshots.ambr`).
4. **Reframe `claim_vs_evidence.py` as the first real `.polydemo`** (satisfies 212 compositionality + sru.1). Author `demo/scripts/claim-vs-evidence.polydemo` whose steps are `find "actions where is_error:true | group by followup_class | count"` DSL (the sru.1 capability). The 60KB bespoke report engine's writers (`_write_public_reproduction`, `_write_cold_reader_gate`) become **generic runner artifact writers** in `demo/runner.py`; the calibration-specific logic that is genuinely not a product primitive is filed as a bead, not carried as glue. Keep `devtools/claim_vs_evidence.py` as a thin `demo run` shim until the DSL cut is proven, then retire (sru.1 design: "reduce … to a render preset … or retire it").
5. **`uvx polylogue demo run FILE`** works from a clean checkout because the demo tier is private-data-free and `POLYLOGUE_ARCHIVE_ROOT`-overridable (existing cloud-lane contract).

---

## 5. Test strategy

Inner loop is `devtools test <file>` (testmon), never blanket dirs. New tests:

- **`tests/unit/demo/test_finding_id.py`** — property (Hypothesis): `finding_id` is (a) invariant under `evidence_refs` permutation, (b) sensitive to any change in claim/metric/anchor/datasheet_hash. This is the load-bearing contract.
- **`tests/unit/demo/test_resolver_gate.py`** — a finding citing a fabricated ref (`codex-session:nope::…`) fails; a valid seeded ref round-trips; a ref resolving to a *different* canonical id fails.
- **`tests/unit/demo/test_construct_validity_tripwire.py`** — a finding anchored on `blocks.text` (prose) is rejected pre-render; an unbacked-construct anchor is rejected.
- **`tests/unit/demo/test_refusal_gate.py`** — a `.polydemo` that both refuses and answers a question fails.
- **`tests/unit/demo/test_polydemo_parse.py`** — frontmatter/version/budget validation; malformed docs raise typed errors.
- **`tests/unit/demo/test_demo_run_e2e.py`** — seed 1843 → `demo run claim-vs-evidence.polydemo` → `ok`, first_result within budget, artifact bytes stable (byte-identical `report.json` across two runs — the reproducibility claim).
- **Demo-as-CI-test** — add a `devtools verify` lane (extend `test_claim_vs_evidence.py`, already dirty in tree) that runs every checked-in `.polydemo`, recomputes each `finding_id`, and **diffs against the committed expected value**; drift breaks the build. Also `devtools verify doc-commands` over the demo's `run:` argv (212 acceptance).
- Use `frozen_clock`; temp DBs under `/realm/tmp/polylogue-pytest`. `demo run` is deterministic → no `datetime.now` in the pipeline.

---

## 6. Bead breakdown (children of polylogue-212 / follow-ups to sru.1)

**B1 — `.polydemo` format + parser (`demo/document.py`).**
AC: `DemoScript` parses frontmatter (version gate, budget, corpus.seed/datasheet_sha256), ordered steps with `run`/`expect`/`finding`, `findings`, `refusals`; malformed docs raise typed errors; parse round-trips (parse→serialize→parse stable). Test: `test_polydemo_parse.py`.

**B2 — corpus datasheet + `finding_id` content address (`demo/seed.py`, `demo/finding.py`, `demo/models.py`).**
AC: `corpus_datasheet(root)` deterministic over seed 1843; `DemoFinding` + `compute_finding_id` reuse `core.hashing` NFC canonicalization; property tests prove permutation-invariance and drift-sensitivity. Depends: B1.

**B3 — round-trip resolver gate + structural-anchor allowlist (`demo/anchors.py`, resolver gate).**
AC: dangling/non-round-trip refs rejected via existing `resolve_ref`; `STRUCTURAL_ANCHOR_ALLOWLIST` closed set with construct backing; prose anchors rejected. Depends: B2.

**B4 — construct-validity tripwire + refusal gate.**
AC: findings pre-render must have satisfied backing construct (reuse `evaluate_demo_constructs`); refused question answered by a finding fails. Depends: B3.

**B5 — `demo run FILE` command + runner + artifacts (`demo/runner.py`, `cli/commands/demo.py`).**
AC: `uvx polylogue demo run FILE` seeds, runs steps as product-primitive argv, enforces all gates, writes byte-identical `report.json`/`report.md`/`PUBLIC_REPRODUCTION.md`/`COLD_READER_GATE.md`; first-result + full budgets enforced (reuse `FIRST_RESULT_BUDGET_S`/`FULL_TOUR_BUDGET_S`); help snapshots + `render cli-output-schemas` + topology projection regenerated. Depends: B4.

**B6 — demo-as-CI-test lane.**
AC: `devtools verify` runs every checked-in `.polydemo`, recomputes `finding_id`s, diffs vs committed expected → drift fails build; `verify doc-commands` covers demo argv. Depends: B5.

**B7 — first real `.polydemo`: claim-vs-evidence on the sru.1 DSL (`demo/scripts/claim-vs-evidence.polydemo`), retire the bespoke engine.**
AC: the demo's `silent_proceed_rate` finding is produced by `find "actions where is_error:true | group by followup_class | count"` (sru.1 capability), not python; committed `finding_id` matches; `devtools/claim_vs_evidence.py` reduced to a `demo run` shim or retired; any genuinely non-glue calibration logic filed as its own bead. Depends: B5, B6, and sru.1 landed. Satisfies 212 compositionality rule.

*(Optional B8 — port a second 212 child, e.g. D1/D2, as a `.polydemo` to prove the format generalizes beyond one report.)*

---

## 7. Top-3 risks

1. **`finding_id` drift becomes CI noise, not signal.** Any legitimate corpus edit (seed change, a new `DemoConstruct`) flips the datasheet hash → every finding_id changes → mass expected-value churn. *Mitigation:* the datasheet hash is intentionally in the id (world-pinned honesty), but gate seed/construct edits behind an explicit "regenerate `.polydemo` expected values" devtools command + a review-visible diff, so drift is a deliberate act, not a surprise. Treat unexpected id drift as evidence the demo world silently changed.

2. **The resolver gate under-tests fabrication.** A ref can resolve to a real object yet the *claim* about it still be false (the number is real but mis-attributed). The gate proves the citation exists, not that it supports the metric. *Mitigation:* `evidence_anchor` + construct-validity tripwire narrow this (the number must come from a structural column backed by a satisfied construct), but true metric↔evidence soundness needs the sru.1 DSL to *compute* the metric from the same rows it cites — so B7 (finding produced by the DSL over the anchor column) is what actually closes the gap. Findings whose metric is authored by hand rather than by a `produced_by` step are the danger zone; consider forbidding hand-authored metrics once the DSL covers the cut.

3. **Compositionality erosion (212's core failure mode).** The path of least resistance is to let `run:` steps shell out to a python helper, recreating the `claim_vs_evidence.py` monolith inside a `.polydemo`. *Mitigation:* the parser should **only** accept `polylogue …` argv in `run:` (reject arbitrary shell); a step needing bespoke logic must fail the demo with "missing product primitive — file a bead," making the demo the forcing function for product algebra exactly as bead 212 mandates (the `agent_forensics.py → polylogue analyze` fold is the template).

---

Files a follow-up implementer will touch (all absolute): new `/realm/project/polylogue/polylogue/demo/{document,runner,finding,anchors}.py`; extend `/realm/project/polylogue/polylogue/demo/{models,seed}.py`; `/realm/project/polylogue/polylogue/cli/commands/demo.py`; author `/realm/project/polylogue/polylogue/demo/scripts/claim-vs-evidence.polydemo`; retire/shim `/realm/project/polylogue/devtools/claim_vs_evidence.py`; regenerate `docs/plans/topology-target.yaml`, `docs/topology-status.md`, CLI help snapshots. Reuse (do not reinvent): `core/refs.py:EvidenceRef`, `demo/constructs.py:evaluate_demo_constructs`, `tour.py` budgets/writers, `core/hashing.py`, and the sru.1 `followup_class` DSL capability.

---

## [a457a99ca5a0512f8] SPEC — Content-Variant Model + Alignment (polylogue-4smp / arso / 0v9p)

I have enough grounding (beads 4smp/arso/0v9p/mhx, `user.py` assertions DDL, `refs.py` ObjectRefKind, `index.py` session_links + generated-column patterns, DSL STRUCT_UNIT). Here is the spec.

---

# SPEC — Content-Variant Model + Alignment (polylogue-4smp / arso / 0v9p)

Grounding: variants are **transformed content artifacts, not assertions** (arso). They live over public refs, are rendered through Query × Projection × Render (4p1/jnj.1), and must never be confused with original evidence. Staleness reuses the existing content-hash idempotency substrate (`core/hashing.py`, `pipeline/ids.py`). Language detection is a **separate, rebuildable derived-fact layer** (0v9p) — detection alone never manufactures a variant.

## 1. Schema / DDL

### 1a. Refs (`core/refs.py`) — additive, no tier
Add two `ObjectRefKind` literals + `_OBJECT_REF_KINDS` entries: `"variant"`, `"variant-node"`. `variant:<id>` / `variant-node:<id>` resolve through the same `ObjectRef.parse/format`. Assertion refs are untouched; variants may target `assertion:<id>`, and assertions may target `variant:<id>`.

### 1b. Durable tier — `user.db` v4 → v5, numbered additive migration
`storage/sqlite/migrations/user/005_content_variants.sql`, one `PRAGMA user_version` step, behind a verified backup manifest (durable regime). Variants are authored/generated artifacts that cannot be rebuilt from source — they belong with `assertions` in the irreplaceable tier, but in **their own tables** (not the assertion table; wrong ontology per arso). CHECK constraints generated from Python enums via the existing `check(...)`/`literal_check(...)` helper so type ↔ SQL stay locked.

```sql
-- New Python enums (core/enums.py): VariantKind, VariantProvenance,
-- VariantStatus, CoverageLevel, VariantRelation, VariantGrain.

CREATE TABLE IF NOT EXISTS content_variants (
    variant_id           TEXT PRIMARY KEY,               -- opaque uuid, allocated by writer
    target_ref           TEXT NOT NULL,                  -- session:/message:/block:/assertion:/variant-node:
    kind                 TEXT NOT NULL CHECK ({check("kind", VariantKind)}),
        -- translation | transliteration | simplification | summary | caption | ocr
    provenance           TEXT NOT NULL CHECK ({check("provenance", VariantProvenance)}),
        -- mechanical | generative   (the honesty axis)
    source_language      TEXT,                            -- BCP-47 | 'und' | 'mul'
    target_language      TEXT,
    status               TEXT NOT NULL DEFAULT 'candidate'
                             CHECK ({check("status", VariantStatus)}),
        -- candidate | active | rejected | superseded | stale | orphaned
    coverage             TEXT CHECK ({check("coverage", CoverageLevel)}),
        -- complete | partial | sparse   (WRITER-COMPUTED, never caller-trusted)
    composition_policy_json TEXT DEFAULT '{}',
    author_ref           TEXT NOT NULL DEFAULT 'user:local',
    author_kind          TEXT NOT NULL DEFAULT 'agent',   -- agent | user
    evidence_refs_json   TEXT NOT NULL DEFAULT '[]',
    source_content_hash  TEXT,                            -- target fingerprint snapshot at authoring time
    supersedes_json      TEXT NOT NULL DEFAULT '[]',
    staleness_json       TEXT,                            -- drift evidence, set by daemon; NULL when fresh
    metadata_json        TEXT DEFAULT '{}',
    created_at_ms        INTEGER NOT NULL,
    updated_at_ms        INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_variants_target_kind   ON content_variants(target_ref, kind);
CREATE INDEX idx_variants_status_lang   ON content_variants(status, target_language);

-- Structured variant content at session/message/block/span/assertion-body grain.
CREATE TABLE IF NOT EXISTS variant_nodes (
    variant_node_id  TEXT PRIMARY KEY,
    variant_id       TEXT NOT NULL REFERENCES content_variants(variant_id) ON DELETE CASCADE,
    grain            TEXT NOT NULL CHECK ({check("grain", VariantGrain)}),
        -- session | message | block | span | assertion-body
    position         INTEGER NOT NULL,                    -- order within the variant; NOT a source-mapping
    content_text     TEXT NOT NULL,
    content_language TEXT,
    metadata_json    TEXT DEFAULT '{}',
    created_at_ms    INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_variant_nodes_variant ON variant_nodes(variant_id, position);

-- Alignment edges: source_ref -> variant_node. This is the semantic map.
CREATE TABLE IF NOT EXISTS variant_alignments (
    alignment_id        TEXT PRIMARY KEY,
    variant_id          TEXT NOT NULL REFERENCES content_variants(variant_id) ON DELETE CASCADE,
    variant_node_id     TEXT REFERENCES variant_nodes(variant_node_id) ON DELETE CASCADE,
        -- NULL iff relation = 'omits'
    source_ref          TEXT NOT NULL,                    -- block:/message:/assertion:...
    relation            TEXT NOT NULL CHECK ({check("relation", VariantRelation)}),
        -- translates | transliterates | simplifies | summarizes | omits | expands | reorders
    source_content_hash TEXT NOT NULL,                    -- per-edge fingerprint snapshot (staleness key)
    partial             INTEGER NOT NULL DEFAULT 0 CHECK (partial IN (0,1)),
    confidence          REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    metadata_json       TEXT DEFAULT '{}',
    created_at_ms       INTEGER NOT NULL,
    CHECK ((relation = 'omits') = (variant_node_id IS NULL))
) STRICT;
CREATE INDEX idx_alignments_variant ON variant_alignments(variant_id);
CREATE INDEX idx_alignments_source  ON variant_alignments(source_ref);
CREATE INDEX idx_alignments_node    ON variant_alignments(variant_node_id);
```

Cardinality falls out of the edge table with **no positional convention**:
- one-to-one → one edge (node ↔ source)
- one-to-many (`expands`) → many edges sharing `source_ref`, distinct `variant_node_id`
- **many-to-one** (`summarizes`) → many edges sharing `variant_node_id`, distinct `source_ref`
- **omitted** → edge with `variant_node_id IS NULL`, `relation='omits'` (acknowledged dark matter)
- **partial** → `partial=1` on the edge

### 1c. Rebuildable tier — `index.db` v24 → v25, canonical DDL edit (no migration chain)
Per-block prose-language facts (0v9p). Derived, so a schema mismatch rebuilds the tier; detection re-runs during materialize.

```sql
CREATE TABLE IF NOT EXISTS block_language_facts (
    block_id         TEXT PRIMARY KEY REFERENCES blocks(block_id) ON DELETE CASCADE,
    language         TEXT NOT NULL,                       -- BCP-47 | 'und'
    confidence       REAL,
    is_reliable      INTEGER NOT NULL DEFAULT 0,
    secondary_json   TEXT DEFAULT '[]',                   -- mixed-lang: [{lang,confidence,span}]
    detector         TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    detected_at_ms   INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_block_lang ON block_language_facts(language, is_reliable);
```
Message/session language are **VIEWs** rolling up block facts (dominant language + `is_mixed` flag) — no collapse to a single false session language. Detector library stays pluggable behind a `LanguageDetector` protocol (not part of the public contract).

User language state (durable, `user.db`): global preference in `user_settings["variant.preferred_languages"]`; per-target overrides as a new `AssertionKind.LANGUAGE_CORRECTION` (schema-free TEXT kind — but the enum is embedded in `render openapi` + `render cli-output-schemas`, regenerate). Override wins over detection without mutating source.

## 2. Write-time honesty invariants + staleness algorithm

Enforced at the repository write boundary (a new `VariantWriter` mixin) and mirrored by a `devtools lab policy variant-honesty` check. The **writer computes hashes and coverage — never trusts the caller.**

Invariants (reject on violation):
1. **coverage > 0** — a variant with `kind ∈ {summary, translation, simplification, transliteration, caption, ocr}` MUST have ≥1 non-`omits` alignment edge. *A summary with zero citation edges is rejected.*
2. `target_ref` resolves to an existing addressable ref.
3. Every edge `source_ref` lies **within the target's declared composition** (a cited block belongs to the target session/message) — can't cite outside the source.
4. `provenance` required; `generative` ⇒ durable+protected. `mechanical` variants (deterministic transliteration/OCR passthrough) may set a rebuildable hint but still persist.
5. `relation='omits'` ⟺ `variant_node_id IS NULL` (also DB CHECK).
6. `source_content_hash` on each edge is (re)computed by the writer at write time; caller-supplied values are ignored.
7. `coverage` level is derived by the writer from edges vs declared children; caller value ignored.

```
COVERAGE(variant):
  children  = declared_child_refs(variant.target_ref)   # session→messages/blocks, message→blocks, assertion→{itself}
  covered   = { e.source_ref for e in edges if e.relation != 'omits' }
  omitted   = { e.source_ref for e in edges if e.relation == 'omits' }
  accounted = covered ∪ omitted
  if children ⊆ covered:                    return 'complete'
  if children ⊆ accounted and covered ≠ ∅:  return 'partial'   # rest explicitly omitted
  else:                                      return 'sparse'
  # dark_matter(variant) = children − accounted   (unacknowledged gaps → rendered as dark matter)
```

Staleness — runs as a `ConvergenceStage` (`variant-staleness`, session-scoped `check_sessions`/`execute_sessions`, `false_means_pending` for backlog) triggered when a session's content hash changes on re-ingest. **Never auto-repaints.**

```
FINGERPRINT(ref):                             # deterministic, reuses core.hashing NFC normalization
  block          -> sha256(NFC(block.search_text))
  message        -> sha256(join(ordered child block fingerprints, role, material_origin))
  session        -> reuse pipeline/ids session content-hash (the idempotency hash)
  assertion      -> sha256(NFC(assertion.body_text ++ value_json))

RECOMPUTE_STALENESS(changed_session_ids):
  if rebuild_in_progress(session):           # index reset mid-flight
      return PENDING                          # defer via false_means_pending; do NOT orphan mid-rebuild
  for v in variants_touching(changed_session_ids):     # via target_ref OR any edge.source_ref
      drift = []
      for e in v.alignments:
          live = FINGERPRINT(e.source_ref)     # None => source no longer exists
          if   live is None:            drift.append((e, 'orphaned'))
          elif live != e.source_content_hash:  drift.append((e, 'stale'))
      if   any d is 'orphaned' in drift: new = 'orphaned'
      elif drift:                        new = 'stale'
      else:                              new = v.status   # unchanged; unrelated edits leave it 'active'
      if new != v.status:
          v.staleness_json = {drifted: drift, detected_at, prior: v.status}
          v.status = new                       # STATUS ONLY — variant text is never regenerated
  # Recovery is an explicit re-author: a new variant supersedes (supersedes_json), never in-place repaint.
```

This composes with existing idempotency: a re-ingest whose session hash matches is skipped ⇒ variants stay fresh; a differing hash rebuilds dependent insights **and** enqueues that session for `variant-staleness`.

## 3. Rebuild plan
- **user.db (durable):** additive numbered migration `user/005`, backup manifest verified first, one `user_version` bump. No rebuild — variants survive index/embeddings resets. Destructive changes would need copy-forward + explicit consent (not in scope).
- **index.db (rebuildable):** edit canonical DDL to v25 + a rebuild-plan entry (add `block_language_facts` + rollup views); **no upgrade helper** (`schema-versioning` policy rejects them). Deploy = `polylogue ops reset --index && polylogued run`; detection re-runs in materialize.
- **Cross-tier coupling:** after an index rebuild, source fingerprints are recomputed, so the `variant-staleness` stage re-runs over all sessions. Because user.db variants persist across the reset, this stage re-fingerprints and marks `stale`/`orphaned` where genuinely drifted — but must gate on `rebuild_in_progress` to avoid a false-orphan storm while the fresh index is still filling (Risk 2).
- **Topology projection:** any new module under `polylogue/` ⇒ `devtools render topology-projection && render topology-status`, commit `topology-target.yaml` + `topology-status.md`, else `render all --check` fails.

## 4. Test strategy
- **Model/property (`tests/property`, `tests/unit/storage`):** coverage math (complete ⟺ all children covered-or-omitted); many-to-one summary maps N `source_ref` → 1 node with **no positional hack**; partial labeled; omitted edges valid; `translates` assertion stays `variant of assertion:<id>` (not a projected original assertion).
- **Invariant tests:** reject zero-coverage summary; reject out-of-composition `source_ref`; assert writer recomputes `source_content_hash`/`coverage` and ignores caller values; DB CHECK `omits ⟺ node NULL`.
- **Staleness:** mutate a block → re-ingest → variant `→ stale`, text byte-identical, no repaint; delete source ref → `orphaned`; unrelated-session edit leaves variant `active`; **fingerprint stability** test — rebuild index with unchanged source ⇒ zero variants marked stale (Risk 1 guard); mid-rebuild ⇒ `PENDING`, never orphaned.
- **Query honesty (contract, adversarial-loop):** variant text never appears in `evidence_refs`, `read --view transcript` original-evidence lane, or default search results; `resolve_ref(variant:…)` returns transform provenance/`evidence_role='transform'`; search default excludes variant text unless `unit:variant`.
- **Language (0v9p):** mixed-language blocks preserved (secondary_json), low-confidence → `und`, user override wins over detection, **detection creates no variant**.
- **DSL:** `variants where kind:translation status:active | count` parses+executes; `coverage` unit returns the dark-matter gap set; `variant`/`variant-node` round-trip `ObjectRef.parse/format`.
- **Demo/fixture (4smp AC):** heavily-annotated session translated with transcript variants + variants of selected assertion annotations, clickable alignment back to source and to original assertions; assert dark-matter rendering shows uncovered children. Use the private-data-free `polylogue demo seed`/`verify` path.
- **MCP:** new tools (`create_variant`, `list_variants`, `resolve_variant_alignment`) ⇒ update `EXPECTED_TOOL_NAMES` + tool contract or discovery tests fail.
- **Lab policy:** `schema-versioning` green (user additive + backup manifest; index additive-derived); new `variant-honesty` lane green.

## 5. Bead breakdown (proposed — not created; refine children of arso/0v9p/rlsb)

| # | Bead | Acceptance |
|---|------|-----------|
| B1 | **Enums + refs + user.db v5 DDL/migration** | `VariantKind/Provenance/Status/CoverageLevel/Relation/Grain` enums; `variant`/`variant-node` in `ObjectRefKind`; `user/005` migration applies over a v4 archive behind backup manifest; STRICT tables + `omits⟺NULL` CHECK present; `schema-versioning` policy green. |
| B2 | **Repository/API read+write + honesty invariants** | `VariantWriter`/reader mixin on `SessionRepository`; writer recomputes hash+coverage; all 7 invariants enforced with unit tests incl. rejected zero-coverage summary and out-of-composition source; `resolve_ref` resolves variant/variant-node refs. |
| B3 | **Alignment + coverage semantics + dark-matter** | one/one-to-many/**many-to-one**/omitted/partial proven by tests; `COVERAGE()` + `dark_matter()` implemented; partial variants labeled; summary maps N→1 without positional convention. |
| B4 | **Staleness convergence stage** | `variant-staleness` `ConvergenceStage` (session-scoped, `false_means_pending`); FINGERPRINT reuses `core.hashing`; stale/orphaned set on drift, **text never regenerated**; fingerprint-stability + mid-rebuild-defer tests pass. |
| B5 | **Language-fact layer (0v9p)** | `block_language_facts` in index v25 + rollup views; pluggable `LanguageDetector`; mixed-lang preserved; `LANGUAGE_CORRECTION` override + `user_settings` preference; projection default `translate-if-source-not-preferred`; detection creates no variant. |
| B6 | **DSL units + query/read honesty + MCP (rlsb)** | `variant` STRUCT_UNIT in `expression.py:669`; `coverage` unit + gap projection wired in `unit_results`/`attached_units`; contract test: variant text never returned as original evidence on any surface; MCP tools + `EXPECTED_TOOL_NAMES`; `render openapi`/`cli-output-schemas`/`topology-projection` regenerated. |

Dependency order: B1 → B2 → {B3, B4} → B6; B5 parallel, feeds B6 projection defaults. B1–B4 refine `polylogue-arso`; B5 is `polylogue-0v9p`; B6 is `polylogue-rlsb`; UX (`d4zk`) consumes B2/B6.

## 6. Top-3 risks
1. **False-stale storms from fingerprint drift.** If `FINGERPRINT` normalization diverges even slightly from `pipeline/ids` idempotency hashing, every index rebuild marks all variants stale → constant repaint pressure and eroded trust in `active`. Mitigation: reuse `core.hashing` NFC path verbatim; the "rebuild with unchanged source ⇒ 0 stale" test is a required gate.
2. **Durable/rebuildable boundary orphaning.** Variants (durable user.db) reference source_refs whose content lives in the rebuildable index.db. A naive staleness pass during `ops reset --index && polylogued run` sees missing sources and mass-marks `orphaned` (data-destroying in effect). Mitigation: `variant-staleness` gates on `rebuild_in_progress` and defers via `false_means_pending`; only marks `orphaned` after a completed rebuild confirms genuine absence.
3. **Evidence contamination.** A single leak of variant text into an evidence projection, `evidence_refs`, or search-as-evidence breaks the core guarantee that transforms are never presented as original evidence. Mitigation: one choke point — variant reads carry `evidence_role='transform'`; an adversarial-loop contract test asserts across *every* read surface (CLI transcript, API, MCP, search) that variant/variant-node refs never surface in an evidence lane.

Key files to touch: `core/refs.py`, `core/enums.py`, `storage/sqlite/migrations/user/005_content_variants.sql`, `storage/sqlite/archive_tiers/user.py` + `index.py`, `storage/repository/__init__.py` (new mixin), `daemon/convergence_stages.py`, `archive/query/expression.py` (+ `unit_results.py`, `attached_units.py`), `insights/registry.py` (variant/coverage descriptors), `mcp/server_*.py` + `EXPECTED_TOOL_NAMES`.

---

## [af42c7c0cf308dad8] SPEC — Extract a narrow, versioned `polylogue.sdk` (external-consumer boundary)

I have the ground truth. Writing the spec.

---

# SPEC — Extract a narrow, versioned `polylogue.sdk` (external-consumer boundary)

## 0. Ground truth (what's actually there, corrects the premise)

The premise ("lynchpin bypasses the async facade entirely") is **half-true and already partly addressed** — the spec must build on, not ignore, what exists:

- **Async facade:** `polylogue/api/__init__.py` → `Polylogue(4 mixins)`; `api/archive.py` alone is **120 async methods** (~130 across mixins). `__all__ = ["ArchiveStats", "Polylogue", "select_pending_embedding_session_window"]` — leaks a substrate helper.
- **A sync facade *already exists*:** `polylogue/api/sync/SyncPolylogue` (`sessions.py` ~17 methods, `insights.py` ~19). Lynchpin's `polylogue_client.py` **already consumes it** (`from polylogue.api.sync import SyncPolylogue`, process-singleton via `lru_cache`). So "async-only is why they bypass" is stale — the sync bridge is a down payment, but it is **uncurated, unversioned, model-unfrozen, error-untyped, schema-unpinned**.
- **The real, live bypass is two things, not the facade absence:**
  1. **Raw sqlite reads coexisting with the sync client.** `lynchpin/sources/polylogue.py` does `sqlite3.connect(_default_polylogue_db_path())` in ≥5 places, incl. `PRAGMA table_info(...)`, table-existence probes, and — critically — **`SELECT MIN(created_at), MAX(created_at) FROM conversations`**. There is **no `conversations` table in the current split-tier index.db** (grep confirms; it's `sessions` now). Lynchpin's raw path is *already stale against schema v24* and only survives because that code path points at a legacy monolith db. This is exactly the fragility the SDK must delete.
  2. **Reimplemented models.** `lynchpin/sources/polylogue_models.py` redefines `SessionProfile`, `MessageRecord`, `ConversationTranscript`, `ConversationLineage`, `CostSummary`, `WorkEvent`, `DaySessionSummary`, `PolylogueReadiness`, `ChatDayActivity`, `WorkPattern` as its own frozen dataclasses — a parallel schema that drifts silently.
- **Public domain models have no stable home:** `Session`/`SessionSummary` in `archive/session/domain_models.py`, `Message` in `archive/message/models.py`, insight models scattered. There is **no `polylogue.models`**.
- **Schema pin:** `INDEX_SCHEMA_VERSION = 24` in `storage/sqlite/archive_tiers/index.py`. No external-consumer pin-and-warn.
- **Layering:** `docs/plans/layering.yaml` enforces only *no-backward-import* (substrate ⊄ surfaces). It has **no rule** that external consumers / non-`api` surfaces may not import substrate internals — the enforcement hook the SDK needs.
- **Contract pattern to extend:** `api/contracts/read_surface.py` already defines `@runtime_checkable` Protocol families over shared `SessionQuerySpec` + `surfaces.payloads` envelopes. The SDK conformance surface is a superset of this.

**Reframed thesis:** don't "extract a facade that's missing" — **promote the ad-hoc sync bridge into a curated, semver'd `polylogue.sdk` with a frozen `polylogue.models`, a typed error taxonomy, a schema pin-and-warn, and a returnable query object; then migrate lynchpin off raw-sqlite + reimplemented models onto it, and lock the boundary with a layering rule.**

---

## 1. Module layout + public surface

```
polylogue/
  sdk/
    __init__.py          # THE surface. Curated __all__ (~20 names). __version__ (semver, independent of pkg).
    client.py            # SyncClient (default) + AsyncClient. Thin re-home of api.sync + api.Polylogue.
    query.py             # Query — returnable, chainable, lazy. .count()/.list()/.iter()/.to_arrow()/.save()/.explain()
    errors.py            # Typed taxonomy (see below). One root PolylogueError.
    schema.py            # SDK_MIN_INDEX_SCHEMA / SDK_MAX_INDEX_SCHEMA pin; check_schema() pin-and-warn.
    _compat.py           # internal: adapts substrate objects → frozen models. NOT exported.
  models/
    __init__.py          # Frozen public models. Re-export (not redefine) the canonical domain models
                         # from archive/*, insights/*, wrapped/frozen for stability. Explicit __all__.
```

**Design stance:** `sdk/` is a *leaf adapter over `api/`* (which is already the declared adapter boundary over substrate). `models/` **re-exports** the existing canonical Pydantic/dataclass models — it does not fork them (forking is exactly lynchpin's mistake). Stability is enforced by a golden schema test, not by copying.

### `polylogue.sdk.__all__` (~20, the whole contract)

| Name | Kind | Backs onto (real) |
|---|---|---|
| `SyncClient` | class (default) | wraps `api.sync.SyncPolylogue` |
| `AsyncClient` | class | wraps `api.Polylogue` |
| `Query` | returnable query object | new, lowers to `SessionFilter`/`SessionQuerySpec` |
| `open(archive_root=…)` | fn | ctor sugar |
| `__version__` | str | SDK semver |
| `SDK_INDEX_SCHEMA_RANGE` | tuple | from `INDEX_SCHEMA_VERSION` |
| Read verbs (client methods, not top-level): `sessions()`, `session(id)`, `messages(id)`, `search()`, `stats()`, `readiness()`, `coverage()`, `session_profiles()`, `work_events()`, `cost_rollups()`, `day_summaries()`, `threads()` | — | existing sync mixin methods, renamed/narrowed |
| `PolylogueError` + taxonomy | exc | new |

The ~120-method async archive stays internal. The SDK exposes the **~20 methods lynchpin+oracle actually need** (evidenced: `list_summaries`, `bulk_get_messages`, `list_session_profile_insights`, `list_session_work_event_insights`, `list_archive_coverage_insights`, `list_work_thread_insights`, `insight_readiness_report`, `stats`, plus date-bounds/coverage the raw-sqlite path currently steals). Everything else remains reachable via `client._facade` **only** with a `SDK_UNSTABLE=1` escape hatch that emits a `UnstableAPIWarning`.

### `polylogue.models.__all__`

Frozen re-exports: `Session`, `SessionSummary`, `Message`, `Block`, `SessionProfile`, `WorkEvent`, `CostSummary`, `CoverageWindow`, `ThreadInsight`, `ReadinessReport`, `Origin` (enum), `MaterialOrigin` (enum). These are precisely lynchpin's 10 reimplemented dataclasses, sourced from the canonical originals.

---

## 2. Algorithms / ergonomics (pseudocode)

### 2a. Sync-over-async without loop hazards (the reason lynchpin bypassed)

```python
# sdk/client.py — reuse the existing, working bridge; don't reinvent
class SyncClient:
    def __init__(self, archive_root=None, db_path=None, *, check_schema=True):
        self._async = api.Polylogue(archive_root, db_path)   # existing facade
        if check_schema:
            schema.check_schema(self._async.backend)          # pin-and-warn (2c)
    # every read verb:
    def sessions(self, **kw) -> "Query":
        return Query(self, spec=SessionQuerySpec.from_kwargs(**kw))
    def _run(self, coro):
        return api.sync.bridge.run_coroutine_sync(coro)        # existing, loop-safe
```
Ground: `api/sync/bridge.py:run_coroutine_sync` already solves the loop problem the premise blames — SDK reuses it, adding nothing.

### 2b. Query as a returnable, lazy, chainable object

```python
class Query:                       # immutable; every op returns a new Query
    def where(self, **preds) -> Query:            # merges into SessionQuerySpec
    def since(self, when) -> Query:  def origin(self, o) -> Query:
    def with_units(self, *units) -> Query:        # maps to DSL `with <units>`

    # ---- terminals (materialize) ----
    def count(self) -> int:                        # pushes down to SQL COUNT, no row load
        return self._client._run(self._plan.count())
    def list(self) -> list[Session]:               # frozen models
    def iter(self) -> Iterator[Session]:           # streaming, bounded memory
    def first(self) -> Session | None:
    def to_arrow(self):                            # optional extra [arrow]; else raises FeatureUnavailable
        import pyarrow; return _rows_to_table(self.iter(), schema=self._arrow_schema())
    def save(self, name: str) -> SavedQueryRef:    # persists as user.db AssertionKind.saved_query
    def explain(self) -> str:                      # returns lowered SQL / plan, for debugging
```
`Query` lowers onto the **existing** `SessionFilter`/`SessionQueryPlan` (`archive/filter/filters.py`, immutable `SessionQueryPlan` already separates SQL-pushdown from post-filter) — the returnable object is a thin public skin, `count()` uses the plan's pushdown so it never loads rows. `save()` routes to `user.db` `assertions(kind=saved_query)`.

### 2c. Schema pin-and-warn

```python
# sdk/schema.py
SDK_INDEX_SCHEMA_RANGE = (24, 24)     # generated-checked against INDEX_SCHEMA_VERSION
def check_schema(backend) -> None:
    v = backend.user_version("index")
    lo, hi = SDK_INDEX_SCHEMA_RANGE
    if v < lo:  raise SchemaTooOldError(v, lo)      # hard: SDK newer than archive
    if v > hi:  warnings.warn(SchemaAheadWarning(v, hi))  # soft: archive rebuilt ahead; reads may miss columns
```
index.db is a *rebuildable derived tier* — a mismatch is a "rebuild + upgrade SDK" signal, never a migration. The SDK **reads** the pin, it never migrates. `SDK_INDEX_SCHEMA_RANGE` is a generated surface checked against `INDEX_SCHEMA_VERSION` by `render all --check`, so bumping index schema without bumping the SDK range fails CI.

### 2d. Error taxonomy

```
PolylogueError                       # root; catch-all for consumers
├─ ArchiveNotFoundError              # archive_root/index.db absent (replaces lynchpin's ad-hoc probes)
├─ SchemaTooOldError / SchemaAheadWarning
├─ SessionNotFoundError(session_id)
├─ QuerySpecError                    # re-home of existing archive QuerySpecError → public
├─ InsightUnavailableError           # insight not materialized yet (maps convergence_debt)
├─ FeatureUnavailableError           # e.g. to_arrow without [arrow] extra
└─ UnstableAPIWarning                # touching client._facade
```
Grounded: `QuerySpecError` already exists internally; the taxonomy promotes it and adds the not-found/schema/insight cases lynchpin currently handles by catching bare `sqlite3.OperationalError` / probing `sqlite_master`.

---

## 3. Migration / deprecation path

**Polylogue side (additive, no break):**
1. Land `polylogue/sdk/` + `polylogue/models/` re-exporting existing canonicals. `api.sync` stays as the internal impl `sdk.client` delegates to (no duplicate logic).
2. Add layering rule to `docs/plans/layering.yaml`: `target: <external consumers / sdk>` `disallow import from: storage, archive, insights, pipeline, sources internals` — SDK may import `api` + `models` only. Add a lint (extend the existing layering checker) that flags `from polylogue.storage…`/`polylogue.archive…` outside `api`/`sdk`/`models`.
3. Deprecation-shim the leak: keep `select_pending_embedding_session_window` in `api.__all__` for one minor, `DeprecationWarning` → move to a proper SDK method or drop.

**Lynchpin side (the actual payoff — separate repo, coordinated):**
4. Replace `polylogue_models.py` with `from polylogue.models import SessionProfile, WorkEvent, …`; delete the 10 forked dataclasses. Adapter shims only where lynchpin field names differ (`conversation_id` vs `session_id`).
5. **Delete every `sqlite3.connect(_default_polylogue_db_path())` block** in `sources/polylogue.py` (incl. the stale `FROM conversations` date-bounds query) → `client.coverage()` / `client.day_summaries()`. This removes the schema-archaeology (`PRAGMA table_info`, table-existence probes) entirely.
6. `polylogue_client.py` swaps `from polylogue.api.sync import SyncPolylogue` → `from polylogue.sdk import SyncClient` (same singleton/lru_cache pattern).

**Deprecation of `api.sync` public import:** keep importable one minor with `DeprecationWarning` pointing at `polylogue.sdk`; internal callers move immediately.

---

## 4. Test strategy

- **Golden public surface test** (`tests/unit/sdk/test_public_surface.py`): assert `set(polylogue.sdk.__all__)` and `set(polylogue.models.__all__)` exactly match a checked-in snapshot; any addition/removal forces an intentional edit. This is a *contract memorializer* (legitimate — it protects an external ABI, not a diff).
- **Model-freeze parity test:** for each `polylogue.models` type, assert field set + types == the canonical source model's, so a substrate rename can't silently drift the public model (the failure lynchpin has today).
- **Schema-pin generated-check:** `SDK_INDEX_SCHEMA_RANGE` upper bound must equal `INDEX_SCHEMA_VERSION`; wired into `render all --check`.
- **Query algebra laws** (reuse `tests/unit/cli/test_query_exec_laws.py` patterns): `count()` == `len(list())`; `where` chaining commutes where documented; `count()` issues no row-loading SQL (assert via `explain()`).
- **Sync bridge no-loop test:** `SyncClient` works inside and outside a running event loop (regression for the exact reason lynchpin bypassed).
- **Layering test:** lint asserts no `polylogue.storage/archive/insights` import outside `api|sdk|models`; a fixture module that violates it must fail.
- **Error taxonomy:** every taxonomy class reachable via a real trigger (missing archive, missing session, old schema, unmaterialized insight, `to_arrow` w/o extra).
- **Lynchpin cross-repo smoke** (in lynchpin CI, not polylogue): the migrated `sources/polylogue.py` produces byte-identical `MaterializedDataset` bounds vs the old raw-sqlite path over a fixture archive.
- Reuse `corpus_seeded_db` / `workspace_env`; `frozen_clock` for date-bounds tests.

---

## 5. Bead breakdown (acceptance criteria)

1. **`polylogue-sdk.1` — Frozen `polylogue.models` namespace.**
   AC: `polylogue/models/__init__.py` re-exports the 12 canonical types with explicit `__all__`; model-freeze parity test passes; no dataclass redefined (grep: models/ imports from archive/insights, defines no `class …(BaseModel/dataclass)`).
2. **`polylogue-sdk.2` — `polylogue.sdk` surface + `SyncClient`/`AsyncClient`.**
   AC: `sdk/__init__.py` `__all__` ≤ 22 names; `SyncClient` delegates to existing `api.sync` (no logic fork); golden public-surface snapshot test committed; `client._facade` access emits `UnstableAPIWarning`.
3. **`polylogue-sdk.3` — Typed error taxonomy.**
   AC: `sdk/errors.py` with root `PolylogueError` + 7 subclasses; each reachable by a real trigger test; internal `QuerySpecError` promoted/aliased, no bare `sqlite3.OperationalError` crosses the SDK boundary.
4. **`polylogue-sdk.4` — Returnable `Query` object.**
   AC: `Query.where/since/origin/with_units` immutable-chainable; terminals `count/list/iter/first/save/explain` over existing `SessionQueryPlan`; `count()` proven pushdown (no row load) via `explain()`; `save()` writes `user.db` `saved_query` assertion. `to_arrow()` gated behind `[arrow]` extra → `FeatureUnavailableError` otherwise.
5. **`polylogue-sdk.5` — Schema pin-and-warn.**
   AC: `SDK_INDEX_SCHEMA_RANGE` generated-checked == `INDEX_SCHEMA_VERSION`; `check_schema` raises on older, warns on ahead; `render all --check` fails if range drifts from index version.
6. **`polylogue-sdk.6` — Layering enforcement.**
   AC: `docs/plans/layering.yaml` rule + extended lint forbidding `polylogue.storage/archive/insights` imports outside `api|sdk|models`; violating fixture fails lint; `api.__all__` leak (`select_pending_embedding_session_window`) marked deprecated.
7. **`polylogue-sdk.7` — Migrate lynchpin off raw sqlite + forked models.** *(lynchpin repo; blocked by .1–.4)*
   AC: `lynchpin/sources/polylogue_models.py` deleted (or shim-only); zero `sqlite3.connect(_default_polylogue_db_path())` remaining in `sources/polylogue.py`; the stale `FROM conversations` query gone; lynchpin cross-repo smoke green; `polylogue_client.py` imports `polylogue.sdk`.
8. **`polylogue-sdk.8` — Deprecate `api.sync` public import + SDK semver doc.**
   AC: `api.sync` public import warns → `polylogue.sdk`; `sdk.__version__` set with a `docs/sdk-stability.md` policy (semver, model-freeze guarantee, schema-range promise); MEMORY/CLAUDE reference the SDK as the external boundary.

---

## 6. Top-3 risks

1. **Non-injective vocabulary at the frozen boundary.** `polylogue.models` freezes `Origin`, but the provider→origin retirement is *in progress* (`GEMINI`+`DRIVE` → `AISTUDIO_DRIVE`, non-injective; `project_origin_payload` is a live shim). Freezing an origin-typed public model now risks locking in a mid-migration vocabulary. **Mitigation:** the SDK must expose **`origin` only** (never `provider`), pin to `core.enums.Origin`, and the freeze test must reference that enum so a retirement bump is a deliberate, visible SDK-version event.

2. **Lynchpin's raw path is already stale — migration may surface a latent data gap, not a clean swap.** The `FROM conversations` query proves lynchpin reads a *legacy monolith*, not the split-tier index.db. Swapping to `SyncClient` points it at the *current* archive, which may return different date bounds / session counts (the real 16K-physical-vs-8.8K-logical lineage duplication). **Mitigation:** the cross-repo smoke (AC .7) must diff outputs and *expect* a documented delta, not byte-equality; treat divergence as correctness, not regression.

3. **Surface-slimming vs 120-method reality.** Consumers (oracle, future scripts) may need a method not in the curated ~20, and the `client._facade` escape hatch quietly becomes the new bypass — re-creating the exact problem. **Mitigation:** instrument `_facade` access (warning + optional telemetry) so demand for a method is *visible*; promote high-traffic escapes into `__all__` on a cadence rather than letting the hatch ossify. The ~20 surface is a floor to grow deliberately, not a fixed ceiling.

---

**Files grounding this spec (all absolute):** `/realm/project/polylogue/polylogue/api/__init__.py`, `/realm/project/polylogue/polylogue/api/sync/{__init__,bridge,sessions,insights}.py`, `/realm/project/polylogue/polylogue/api/contracts/read_surface.py`, `/realm/project/polylogue/polylogue/archive/session/domain_models.py`, `/realm/project/polylogue/polylogue/storage/sqlite/archive_tiers/index.py` (`INDEX_SCHEMA_VERSION = 24`), `/realm/project/polylogue/docs/plans/layering.yaml`; and on the consumer side `/realm/project/sinity-lynchpin/lynchpin/sources/polylogue_client.py`, `.../sources/polylogue.py` (raw sqlite incl. stale `FROM conversations`), `.../sources/polylogue_models.py` (10 forked models).

---

## [a6c00a3c034bafca9] Wave-5 Spec: Activation / Adoption Layer

I have enough grounding. Here is the tight implementation spec.

---

# Wave-5 Spec: Activation / Adoption Layer

**Beads:** `polylogue-d1y` (hooks install + liveness), `polylogue-3gd` (activation + measured adoption), `polylogue-pj8` (discoverability). Parent epic `s7ae` (coordination substrate), scheduler `37t.11`, compaction `gjg`.

**Grounding facts (verified in source):**
- Hooks flow: `polylogue-hook <Event>` → sidecar JSONL (`~/.local/share/polylogue/hooks/<harness>-<sid>.jsonl`) → daemon watcher → `ArtifactKind.HOOK_EVENT`, linked to parent session by `link_group_key` (`docs/hooks.md`). CC has 16 events / Codex 6; recommended-5 = `SessionStart Stop UserPromptSubmit PreToolUse PostToolUse`.
- ops.db is the disposable tier, `OPS_SCHEMA_VERSION = 1` (`storage/sqlite/archive_tiers/ops.py`) — no migration chain, rebuild-on-mismatch.
- Adoption is computable **live** from `blocks.tool_name` (`index.py:189`; `actions` view at `:328`): `block_type='tool_use' AND tool_name LIKE 'mcp__polylogue__%'`. No new table required.
- `polylogue init` already exists (config-only, `cli/commands/init.py`) — install extends the onboarding surface, does **not** duplicate it.
- Coordination envelope already shells `bd hooks list --json` (`coordination/envelope.py:346`) — reuse that `bd remember`/hook-liveness idiom, don't reinvent.
- `context/preamble.py::build_context_preamble_payload` already assembles a typed `ContextPreamble` (lineage + resume candidates + assertion guidance) for `read --view context` — the SessionStart brief is a **budgeted projection of this**, not a new composer.
- Recall packs already persist as `AssertionKind.recall_pack` in durable user.db (`save_recall_pack`/`get_resume_brief` MCP tools) — schema-free `TEXT`, **no migration** to add PreCompact writes.
- SessionStart recall hook is a **standalone shell script** at `~/.claude/hooks/sessionstart-polylogue-recall.sh` calling `polylogue --cwd-prefix ... read --all` — currently unowned by any polylogue command; install adopts it.

---

## 1. Schema / CLI surface

### CLI (new)

```
polylogue install [--harness claude-code|codex|all] [--events recommended|all|<csv>]
                  [--scope user|project] [--dry-run] [--force] [--uninstall]
    # idempotent settings wiring for hook capture + SessionStart brief + PreCompact recall.
    # Writes minimal diff; second run = zero diff. Backs up settings.json before write.

polylogue doctor [--json] [--window 7d] [--repo <path>]
    # health rollup:
    #   - hook liveness per harness×event (wired? observed-last-window?)
    #   - hook-covered vs post-hoc-discovered session ratio
    #   - adoption: mcp__polylogue__* tool diversity/rate per session, per relevant repo
    #   - "configured-but-zero-usage" diagnosis per repo
    #   exit 0 healthy / 1 degraded (broken-flow or zero-adoption-on-relevant-repo)

polylogue install status [--coverage]   # d1y's wired-vs-recommended table (alias into doctor's hook section)
```

Root filters (`--cwd-prefix`, `--origin`, `--window`) stay before the verb per the query-first floor. `install`/`doctor` are **plain subcommands** (not query verbs) — no did-you-mean floor applies.

### ops.db additions (disposable tier → bump `OPS_SCHEMA_VERSION 1 → 2`)

```sql
-- Heartbeat: every polylogue-hook invocation self-reports liveness (cheap, silent).
CREATE TABLE IF NOT EXISTS hook_liveness (
    harness        TEXT NOT NULL,         -- 'claude-code' | 'codex'
    event_type     TEXT NOT NULL,
    session_id     TEXT,                  -- native id from payload, may be null
    cwd            TEXT,
    observed_at_ms INTEGER NOT NULL,
    hook_version   TEXT,                  -- polylogue-hook version, catches stale scripts
    PRIMARY KEY (harness, event_type, observed_at_ms)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_hook_liveness_recent ON hook_liveness(harness, event_type, observed_at_ms DESC);

-- Doctor snapshots: each `doctor` run persists its verdict for trend + regression alerting.
CREATE TABLE IF NOT EXISTS doctor_snapshots (
    snapshot_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at_ms       INTEGER NOT NULL,
    window_ms         INTEGER NOT NULL,
    payload_json      TEXT NOT NULL DEFAULT '{}'   -- full DoctorReport (per-harness/event liveness, ratios, adoption)
) STRICT;
```

Everything else is **live-computed** (no materialized adoption table) — adoption/coverage are insight-registry descriptors over existing `blocks`, `actions`, `sessions`, and the source.db `HOOK_EVENT` artifacts.

### Config (`config.py`, additive, no schema)
`install_managed_events` (default recommended-5), `session_start_brief_token_budget` (default ~800), `precompact_recall_enabled` (default False, dogfood-gated), plus the existing `hooks_enabled`/`hooks_sidecar_dir`.

---

## 2. Install / measurement algorithms (pseudocode)

### `polylogue install` (idempotent, backed-up, per-harness adapter)

```
adapters = {claude-code: ClaudeSettingsAdapter, codex: CodexTomlAdapter}   # per-harness format
for harness in selected_harnesses:
    settings_path = adapter.locate(scope)          # ~/.claude/settings.json | ~/.codex/config.toml
    original = read_or_empty(settings_path)
    desired_hooks = build_hook_entries(events)     # capture events + SessionStart brief + PreCompact recall
    merged = adapter.merge(original, desired_hooks) # respects existing hooks; adds only missing polylogue entries
                                                    # match by command-substring 'polylogue-hook <Event>' — NEVER clobber foreign hooks
    diff = structural_diff(original, merged)
    if diff.empty: report "already wired"; continue
    if dry_run: print diff; continue
    backup(settings_path -> settings_path + ".polylogue-bak.<ts>")   # backup BEFORE write
    atomic_write(settings_path, merged)             # temp file + rename
    report diff
```

Key adapter rules:
- **Match key = normalized `polylogue-hook <Event>` command**, so re-runs and manual edits both converge (the `bd remember` idempotency idiom).
- SessionStart entry wires **two** commands: the capture hook AND the brief emitter (`polylogue read --view context --brief --budget N` — §6). PreCompact wires `polylogue recall save --precompact`.
- Recursive-safety: install stamps `POLYLOGUE_HOOK_SELF=1` guard into brief/recall commands; the daemon's own subprocess sessions and any polylogue-invoked agent set this env → capture hook no-ops (prevents self-ingest / brief-emits-into-brief loops). **Convergent recursive-safety anchor.**

### Hook-liveness heartbeat (inside `polylogue-hook`, ~1 write, fail-open)

```
on hook invocation(event, payload):
    write_sidecar(payload)                      # existing capture path (unchanged)
    try: ops.insert(hook_liveness, harness, event, sid, cwd, now_ms, HOOK_VERSION)  # best-effort, local timeout
    except: pass                                # never block the agent
```

### `doctor` — liveness × coverage × adoption

```
# (a) Hook liveness per harness×event
for (harness,event) in recommended_or_all:
    wired    = adapter.is_wired(harness, event)
    observed = ops.exists(hook_liveness where harness,event, observed_at_ms > now-window)
    state = { wired & observed: OK,
              wired & !observed: BROKEN_FLOW,       # <-- d1y health alert: wired but silent
              !wired & observed: UNMANAGED,
              !wired & !observed: ABSENT }

# (b) Hook-covered vs post-hoc-discovered ratio
sessions_in_window = index.sessions where origin in {claude-code, codex}, started > now-window
covered = sessions with >=1 HOOK_EVENT artifact (source.db, link_group_key join)
ratio = covered / sessions_in_window            # falling ratio while wired = silent regression

# (c) Adoption per relevant repo  (controls for irrelevant repos)
for repo in repos_active_in_window:
    sessions = sessions where git_repository_url|cwd-prefix == repo
    relevant = is_relevant(repo)                # see below
    poly_tool_uses = count blocks where block_type='tool_use'
                       and tool_name LIKE 'mcp__polylogue__%' and session in sessions
    tool_diversity = distinct tool_name (mcp__polylogue__*)
    adoption[repo] = { sessions, poly_tool_uses, tool_diversity, relevant }

# relevance control — do NOT flag repos where the substrate has nothing to offer:
is_relevant(repo) := (archive has >=K prior sessions for repo)      # there IS recall to use
                 AND (repo not in irrelevant_denylist)              # e.g. throwaway/scratch
                 AND (sessions_in_repo >= MIN_SESSIONS)

# (d) "Why isn't this used" diagnosis for configured-but-zero-usage repos
for repo where adoption.relevant and poly_tool_uses == 0:
    diagnose in priority order (first hit wins):
      MCP_NOT_REGISTERED   : no mcp__polylogue__* tool ever seen in ANY session on this machine  -> server not in agent's MCP registry
      HOOKS_BROKEN         : doctor(a) shows BROKEN_FLOW for this harness                          -> capture degraded, brief never injected
      BRIEF_EMPTY          : SessionStart brief for repo compiled to 0 items (no assertions/prior) -> nothing to surface
      NO_ACTIVATION_CH     : global CLAUDE.md activation chapter (3gd) absent from rendered agents  -> agent never told to reach for it
      HABIT_GAP            : all wiring healthy, brief non-empty, yet zero use                      -> behavioral; feeds 3gd chapter iteration
    emit {repo, diagnosis, evidence_refs}

persist DoctorReport -> ops.doctor_snapshots; alert if any BROKEN_FLOW or ratio regression vs prior snapshot.
```

---

## 3. Migration

| Change | Tier | Regime | Action |
|---|---|---|---|
| `hook_liveness`, `doctor_snapshots` tables | ops.db (disposable) | derived — **no numbered migration** | edit canonical `OPS_DDL`, bump `OPS_SCHEMA_VERSION 1→2`; mismatch triggers rebuild (`polylogue ops reset` / daemon recreate). ops is disposable → zero data-loss concern. |
| Adoption/coverage insight descriptors | index.db | additive-derived | register `InsightType` in `insights/registry.py` over existing columns; live-computed (no DDL). If a materialized cache is later wanted → additive index bump + `ops reset --index` rebuild. |
| PreCompact recall pack writes | user.db (durable) | **none** | `AssertionKind.recall_pack` is schema-free `TEXT` — no user-tier bump. |
| `install_managed_events` etc. | config | none | additive config keys. |

No durable-tier (source/user) numbered migration is required by this slice. `install`/`doctor`/adoption enum changes must regenerate `render openapi` + `render cli-output-schemas` + topology projection (new module) — run `devtools render all --check` (grep `out of sync`).

---

## 4. Test strategy

- **Install idempotency (core AC):** clean `settings.json` → `install --events recommended` wires 5 events; **second run yields byte-identical structural diff = empty**. Property test: `merge(merge(x)) == merge(x)` for arbitrary pre-existing foreign hooks (never clobbers non-polylogue entries). Backup file created exactly once per mutating run.
- **Per-harness adapters:** golden fixtures for CC JSON and Codex TOML; `--uninstall` is exact inverse (removes only polylogue entries).
- **Liveness alert (d1y AC):** seed `hook_liveness` rows for `SessionStart` but none for `PreToolUse` in-window → doctor reports `BROKEN_FLOW` for `PreToolUse`; wired+silent harness raises alert "within one session".
- **Coverage ratio:** seeded corpus with N sessions, M carrying `HOOK_EVENT` artifacts → ratio == M/N; regression vs prior snapshot trips degraded exit.
- **Adoption + relevance control:** seeded sessions with/without `mcp__polylogue__*` tool_use across a relevant repo (≥K prior sessions) and an irrelevant scratch repo → irrelevant repo **excluded** from zero-usage flags; relevant zero-usage repo produces the correct diagnosis branch (parametrized over all 5 diagnosis codes with the minimal seed that forces each).
- **PreCompact recall:** simulate PreCompact payload → a `recall_pack` assertion lands in user.db, resolvable via `resolve_ref` (citation-anchor invariant). Corpus-compaction convergence: repeated PreCompacts dedup by content-hash (cheap).
- **SessionStart brief budget:** property test — assembled brief never exceeds `session_start_brief_token_budget`; ordering is assertions-before-CLAUDE.md; deterministic (same inputs → byte-identical, matching `37t.11` invariant); every line carries a `resolve_ref`.
- **Recursive-safety:** with `POLYLOGUE_HOOK_SELF=1`, capture + brief hooks no-op (no sidecar write, no liveness row, no brief emission).
- Clock hygiene via `frozen_clock`; use `corpus_seeded_db`. Run through `devtools test`, not blanket pytest.

---

## 5. Bead breakdown (4–8, all under s7ae / children of d1y·3gd·pj8)

1. **install-core** (d1y.1) — `polylogue install` + per-harness adapters (CC JSON, Codex TOML), idempotent minimal-diff merge, backup, `--dry-run`/`--uninstall`, `POLYLOGUE_HOOK_SELF` recursive guard.
   *AC:* clean settings → recommended-5 wired; second run zero diff; foreign hooks preserved; backup written once; uninstall exact inverse.
2. **hook-liveness** (d1y.2) — `hook_liveness` table (ops bump 1→2) + best-effort heartbeat in `polylogue-hook` + `doctor` liveness section (wired×observed matrix) + daemon health alert on BROKEN_FLOW.
   *AC:* wired-but-broken script raises alert within one session; per-event observed-last-window table renders.
3. **coverage-ratio** (d1y.3 / relates 3uw) — hook-covered vs post-hoc-discovered ratio insight + `doctor_snapshots` trend + regression alert.
   *AC:* ratio renders per origin on live archive; seeded missed-session scenario trips regression.
4. **adoption-insight** (3gd.1) — `InsightType` counting `mcp__polylogue__*` tool_use per session/repo with diversity, live-computed; relevance control (prior-session threshold + denylist).
   *AC:* adoption renders per relevant repo; irrelevant repos excluded; baseline captured before activation chapter ships (measurable delta for 3gd).
5. **usage-diagnosis** (3gd.2) — `doctor` "why-zero-usage" branch (5 prioritized diagnosis codes with evidence refs).
   *AC:* each diagnosis code reachable from a minimal seed; MCP-not-registered vs hooks-broken vs empty-brief vs no-chapter vs habit-gap disambiguated.
6. **precompact-recall** (gjg child) — wire PreCompact via `install`; `polylogue recall save --precompact` writes a `recall_pack` assertion (+ optional context snapshot artifact); content-hash dedup.
   *AC:* real/simulated compaction lands a resolvable recall pack; repeated compactions dedup; behind config flag.
7. **sessionstart-brief** (37t.4 slice) — `read --view context --brief --budget N` emitting a token-budgeted, deterministically-ranked resume brief (assertions > CLAUDE.md), refs-not-dumps, over `build_context_preamble_payload`; adopt/replace the standalone recall shell script.
   *AC:* brief ≤ budget, assertions-first ordering, deterministic, every line resolve_ref-able; SessionStart hook calls one entrypoint.
8. **install-docs+render** (d1y.4) — regenerate cli-output-schemas / openapi / topology projection; update `docs/hooks.md` Setup to point at `polylogue install`; `render all --check` clean.
   *AC:* `render all --check` reports no `out of sync`; docs Setup section is the one-command path.

Sequence: 1→2→3 (capture/health spine); 4→5 (adoption spine, needs baseline before 3gd chapter); 6,7 parallel after 1 (both consume the install wiring); 8 last.

---

## 6. Top-3 risks

1. **Harness settings-format drift silently breaks the merge.** CC settings.json and Codex config.toml schemas are version-dependent (d1y explicitly warns). A stale template can clobber foreign hooks or wire a no-longer-recognized event name. *Mitigation:* per-harness adapter with a match-by-command-substring merge (never structural-position clobber), golden fixtures pinned to the current live schemas, `--dry-run` default in docs, and mandatory backup-before-write. Encode adapters, not one template.

2. **Hook recursion / self-ingest and critical-path latency.** SessionStart-emits-brief and PreCompact-saves-recall run polylogue *inside* the agent's own lifecycle; a polylogue-launched session firing capture hooks creates ingest loops, and any per-hook DB write sits on the tool-call critical path. *Mitigation:* `POLYLOGUE_HOOK_SELF=1` guard (convergent recursive-safety), heartbeat writes are best-effort with a hard local timeout and fail-open (no daemon = zero added latency), brief emission served from the daemon fast-path, never inline blocking.

3. **Adoption metric produces false alarms → the health signal gets ignored (the very failure 3gd exists to prevent).** Counting `mcp__polylogue__*` naively flags scratch repos, brand-new repos with no prior sessions, and MCP-not-registered machines as "adoption failures," and the number itself doesn't distinguish tool-not-available from tool-not-chosen. *Mitigation:* the relevance control (prior-session threshold + denylist + min-sessions) gates what counts; the 5-way diagnosis separates infrastructural causes (not-registered / hooks-broken / empty-brief / no-chapter) from the genuine behavioral `HABIT_GAP` — only the last is the signal 3gd should iterate the activation chapter against. Baseline must be captured **before** the chapter ships or the delta is unmeasurable.

---

## [ac7fb1f744bd134ca] Cross-Machine Sync Spec — durable-tier reconciliation for a single-writer archive

I have everything needed. Ground-truth confirmed. Writing the spec.

---

# Cross-Machine Sync Spec — durable-tier reconciliation for a single-writer archive

Grounded in `storage/sqlite/archive_tiers/{types,bootstrap,source,user,source_write,user_write,migration_runner}.py` and `pipeline/ids.py`. **No code/beads edited.**

## 0. The invariant this must not break

Each machine runs its own `polylogued` as the **sole SQLite writer** over its own archive root. "Sync" is therefore **not** a shared-filesystem or shared-DB scheme — it is periodic **reconciliation of durable-tier rows** into each machine's local archive, applied *through that machine's own writer*, followed by a local derived-tier rebuild. No process ever writes another machine's `.db` file directly.

Durability classes are already declared in `bootstrap.py:ARCHIVE_TIER_SPECS` and are the whole basis of the design:

| Tier | ver | `durability` | `backup_required` | Sync verdict |
| --- | --- | --- | --- | --- |
| `source.db` | 2 | `irreplaceable` | yes | **SYNC** — content-hash union |
| `user.db` | 4 | `human` | yes | **SYNC** — assertion natural-key merge |
| `index.db` | 24 | `rebuildable` | no | **NEVER copy** — rebuild locally |
| `embeddings.db` | 1 | `expensive_rebuild` | yes | **NEVER copy** — rebuild locally (incremental) |
| `ops.db` | 1 | `disposable` | no | **NEVER copy** — machine-local cursors |

## 1. Sync model + what-syncs-what by tier

**Topology:** N peers, each authoritative for the sessions *it* captured. Sessions are naturally near-disjoint by machine (each captures its own `~/.claude/projects`, `~/.codex/sessions`, …). Sync is a **selective content-hash pull**, not `rsync`.

**source.db (union of raw acquisitions):**
- `raw_sessions` PK = `raw_id = deterministic_raw_session_id(origin, source_path, source_index, blob_hash, native_id)` (`source_write.py:130`). This is content-addressed via `blob_hash = sha256(payload)`.
- Consequence: identical bytes → identical `raw_id` → idempotent no-op on union. Different bytes → different `raw_id` → both retained. **Safe union with zero cross-machine coordination.**
- `blob_refs` keyed `(blob_hash, ref_type, ref_id)` — content-addressed, dedup-safe.
- Also synced: `raw_artifacts`, `raw_hook_events`, `history_sidecars` (all deterministic-id or content-hash keyed). `otlp_spans`, `pending_blob_refs`, `gc_generations` are **machine-local runtime** — do NOT sync (spans are telemetry; pending/gc are in-flight lease state for the local writer only).

**user.db (merge of human assertions):**
- `assertions` PK = `assertion_id`, which is **deterministic content-derived** (`user_write.py:_deterministic_id` over kind + target semantics — e.g. `assertion_id_for_session_tag(session_id, tag, source)`). The *same logical assertion on two machines yields the same id* → natural-key merge, no UUID duplication.
- The only real conflict: two machines write **different** `value_json`/`body_text`/`status` under the *same* `assertion_id` (same session tagged with different metadata). Resolved by `updated_at_ms` + `supersedes_json` (§2).
- `user_settings` keyed `setting_key` → last-writer-wins by `updated_at_ms`.

**Derived tiers (index/embeddings/ops): never transmitted.** Reasons: (a) `index.db` rows reference `raw_sessions`/blocks that the peer's union produces locally anyway; copying creates dangling references; (b) `embeddings.db` is a `vec0` virtual table whose rowids bind to local block identity; (c) all three carry WAL/SHM — copying a live file is a torn read; (d) schema-version mismatch (`index=24`) already triggers a local rebuild by design (`bootstrap.py:initialize_archive_database` raises "move it aside and rebuild"). The system is *built* to rebuild these — lean on it.

### Why naive `rsync -a <archive_root>/` is unsafe (the conflict boundary)
1. **Wholesale replace, not union** — rsync overwrites `source.db`, discarding the receiver's own captured sessions. Union semantics require row-level merge, which rsync cannot do.
2. **Live WAL/SHM torn copy** — `source.db-wal`/`-shm` copied mid-write yield a corrupt or half-committed DB; the single-writer daemon is actively appending.
3. **Split-brain derived tiers** — copying `index.db`/`embeddings.db` imports another machine's rebuildable state whose internal refs don't match the receiver's `source.db`, and whose `PRAGMA user_version` may differ → hard failure or silent corruption.
4. **Single-writer violation** — writing the file under the daemon breaks the "main process is sole writer" contract that blob GC leases and FTS triggers depend on.

**The conflict boundary is exactly:** row-level union of `{raw_sessions, blob_refs, raw_artifacts, raw_hook_events, history_sidecars}` in source.db + natural-key merge of `{assertions, user_settings}` in user.db. Everything outside that boundary is local-only.

### The `.well-known`-style manifest + selective pull
Each peer exposes (over the existing daemon HTTP surface, read-only) `/.well-known/polylogue-sync/manifest.json`:
```
{
  "archive_id": "<stable per-archive uuid, minted once>",
  "machine": "laptop",
  "tiers": {
    "source":  {"version": 2, "count": N, "digest": "<merkle root over sorted raw_id>"},
    "user":    {"version": 4, "count": M, "digest": "<merkle root over (assertion_id, updated_at_ms)>"}
  },
  "buckets": { "source": [ {"prefix":"00","digest":...,"count":...}, ... 256 buckets by raw_id[:2] ],
               "user":   [ ... by assertion_id[:2] ] }
}
```
Pull protocol (peer B reconciling from peer A):
1. Compare tier `version` — mismatch on a **durable** tier aborts sync (needs migration, not merge).
2. Compare top-level `digest`; if equal, done.
3. Diff the 256 buckets; for each differing bucket, `GET /.well-known/polylogue-sync/rows?tier=source&bucket=NN` returning `(raw_id, blob_hash, …, blob_bytes_url)`.
4. Fetch only rows whose `raw_id`/`assertion_id` B lacks (or, for user, whose `updated_at_ms` is newer). Blob bytes fetched by `blob_hash` on demand, dedup-checked.
5. Apply through B's writer (§2), then trigger B's local rebuild (§3).

Bucketing by id-prefix keeps a 38 GB / ~16 K-session archive to a bounded delta scan; the digest short-circuits the common "nothing changed" case.

## 2. Merge / conflict algorithms (pseudocode)

```
# ---- source.db: content-hash union (idempotent, conflict-free) ----
def merge_source(local_conn, remote_rows, fetch_blob):
    for row in remote_rows:                       # row.raw_id is content-addressed
        if not local_has_blob(local_conn, row.blob_hash):
            blob = fetch_blob(row.blob_hash)       # verify sha256(blob)==blob_hash BEFORE insert
            assert sha256(blob) == row.blob_hash   # reject on mismatch (integrity gate)
            stage_blob(local_conn, row.blob_hash, blob)
        # PK collision => identical content by construction => harmless
        local_conn.execute("INSERT OR IGNORE INTO raw_sessions (...) VALUES (...)", row)
        for bref in row.blob_refs:
            local_conn.execute("INSERT OR IGNORE INTO blob_refs (...) VALUES (...)", bref)
        merge_children(local_conn, row.raw_artifacts, row.raw_hook_events, row.history_sidecars)
    # NOTE: use INSERT OR IGNORE, not OR REPLACE — a matching raw_id means matching
    # blob_hash means identical bytes; REPLACE would needlessly churn and could clobber
    # locally-advanced parse/validation columns (parsed_at_ms etc.) with remote NULLs.

# ---- user.db: assertion natural-key merge (LWW + supersession) ----
def merge_user(local_conn, remote_assertions, remote_settings):
    for a in remote_assertions:                    # a.assertion_id is deterministic/content-derived
        local = local_conn.get_assertion(a.assertion_id)
        if local is None:
            insert(local_conn, a);                 continue
        if a.assertion_id in local.supersedes_chain: continue   # local already superseded remote
        if local.assertion_id in a.supersedes_json:  replace(local_conn, a); continue
        # same id, divergent payload => last-writer-wins on updated_at_ms,
        # tie-break on stable archive_id order (deterministic across peers)
        if (a.updated_at_ms, a.archive_id) > (local.updated_at_ms, local.archive_id):
            replace(local_conn, a)
    for s in remote_settings:                       # user_settings keyed setting_key
        local = local_conn.get_setting(s.setting_key)
        if local is None or s.updated_at_ms > local.updated_at_ms:
            upsert(local_conn, s)
```
Properties: source union is **commutative + idempotent** (content-addressed keys). User merge is **deterministic and convergent** across peers because both the natural key and the LWW tie-break (`updated_at_ms`, then `archive_id`) are peer-independent — every machine converges to the same set regardless of pull order. `AssertionKind.CORRECTION`/`SUPPRESSION` rows are just assertions, so operator corrections sync for free.

## 3. Rebuild plan on the peer

After a durable-tier merge on machine B, its `index.db`/`embeddings.db` are stale w.r.t. the newly-unioned `source.db`. Two paths:

**A. Incremental (default, cheap):** the daemon's existing convergence pipeline already handles new `raw_sessions`. Merge stages new rows with `parsed_at_ms IS NULL` (the `idx_raw_sessions_parse_ready` partial index picks them up). Then `polylogued run` runs **acquire-skip → parse → materialize → index**, and the `DaemonConverger` does FTS repair + **embedding catch-up in bounded windows** (ops.db tracks `embedding_catchup_runs`). Because content-hash idempotency means only *genuinely new* sessions get parsed/embedded, embeddings' `expensive_rebuild` cost is paid **only on the delta**, not the whole archive — this is why we don't need to copy `embeddings.db` despite its cost class.

**B. Full rebuild (schema drift only):** if the pulled tier version ≠ local expected (e.g. peer on a newer `source` schema), sync aborts before merge; operator runs the durable migration (§ below) first, or for derived drift: `polylogue ops reset --index && polylogued run` (the sanctioned derived-tier rebuild — `bootstrap.py` literally raises telling you to move-aside-and-rebuild).

**Ordering guarantee:** merge source.db → merge user.db → local converge. user.db assertions target `session_id`s that must exist post-source-merge for injection/context, so source precedes user; derived rebuild runs last.

**Durable schema-version skew:** durable tiers use additive numbered migrations behind a **verified backup manifest** (`migration_runner.DURABLE_MIGRATION_TIERS = {source, user}`). Sync must **refuse to merge across a durable-version gap** — a v2→v3 source row may carry columns the receiver's schema lacks. Peers must be on equal durable `user_version` before row exchange; bootstrap already backs up before migrating.

## 4. Test strategy

- **Property (Hypothesis, `tests/property`):** union is idempotent + commutative — `merge(merge(A,B)) == merge(A,B)`, `merge(A,B) == merge(B,A)` over generated `raw_sessions`/`blob_refs`/`assertions` (reuse `SessionBuilder`, schema-driven strategies). This is the load-bearing correctness proof.
- **Convergence:** three-peer fixture, random pull order, assert all three archives reach byte-identical `source.db` row-sets and identical assertion sets (deterministic LWW).
- **Idempotency-under-reingest:** apply the same remote delta twice → no new rows, no churned `parsed_at_ms` (guards the `OR IGNORE` vs `OR REPLACE` choice).
- **Conflict resolution:** same `assertion_id`, divergent `value_json`, differing `updated_at_ms` → newer wins; equal `updated_at_ms` → `archive_id` tie-break is stable both directions.
- **Blob integrity gate:** corrupt blob whose bytes ≠ `blob_hash` is rejected pre-insert (security boundary — belongs in `tests/unit/security/`).
- **Derived-tier rebuild:** post-merge `polylogued run` (against synthetic fixtures, per cloud-lane rules) reconstructs index/FTS/embeddings for merged sessions; FTS row count matches block count.
- **Manifest/bucket diff:** equal archives → empty delta (digest short-circuit); one differing session → exactly one differing bucket, one row pulled.
- **Anti-tests:** assert derived-tier files are **never** written by the sync path (guard against accidental `index.db` transmission).
- Use `frozen_clock` for all `updated_at_ms` logic; temp DBs under `/realm/tmp/polylogue-pytest`.

## 5. Bead breakdown (6 issues)

1. **Sync manifest + digest surface.** Add read-only `/.well-known/polylogue-sync/{manifest,rows,blob}` to the daemon HTTP surface; per-tier merkle digest + 256 id-prefix buckets over source/user. *AC:* equal archives report equal top-level digest; single new session flips exactly one bucket digest; endpoint exposes zero derived-tier data; new MCP/HTTP surface reflected in generated schemas.
2. **source.db content-hash union merge.** `merge_source` with `INSERT OR IGNORE`, blob integrity verification, children (artifacts/hook_events/sidecars). *AC:* idempotent + commutative property tests green; corrupt-blob rejected; `parsed_at_ms` not clobbered on re-merge; `otlp_spans`/`pending_blob_refs`/`gc_generations` excluded.
3. **user.db assertion + settings merge.** Natural-key LWW with `supersedes_json` respect and `(updated_at_ms, archive_id)` tie-break. *AC:* three-peer convergence identical regardless of order; supersession chain honored; settings LWW; CORRECTION/SUPPRESSION rows merge.
4. **Selective-pull client + durable-version guard.** Bucket-diff pull loop; abort on durable `user_version` skew with actionable message; wire source-then-user-then-converge ordering. *AC:* pulls only missing/newer rows; version gap aborts before any write; blob fetched by hash on demand + deduped.
5. **Post-sync incremental rebuild trigger.** After merge, mark new raw rows parse-ready and drive `DaemonConverger`; embeddings catch-up on delta only. *AC:* merged sessions searchable via FTS post-run; embed catch-up touches only new blocks; no full-archive re-embed; derived files never received over the wire.
6. **`polylogue sync` CLI/operator command + archive_id minting.** `sync pull <peer>` / `sync status` (dry-run delta preview from manifest); mint stable per-archive `archive_id` once. *AC:* dry-run shows delta counts without writing; single-writer preserved (sync routes through daemon, not direct file write); help/CLI-reference regenerated.

(Optional 7th if scope grows: signed manifest / peer auth — durable data crossing the wire warrants an integrity+authn story beyond blob-hash verification.)

## 6. Top-3 risks

1. **`source_path` in the `raw_id` key is machine-local.** `deterministic_raw_session_id` folds in `source_path` (e.g. `/home/sinity/.claude/projects/...`). If two machines share a username *and* capture byte-identical payloads, ids collide harmlessly (same `blob_hash`); if paths differ, the **same logical session gets two `raw_id`s** → union keeps both → the lineage layer sees near-duplicate sessions. Mitigation: sync is honest about physical union; logical dedup already exists (session-lineage model, #2467) and operates downstream — but cross-machine duplicates will *inflate* that. Must be tested against the lineage composer, not just row counts.
2. **Single-writer violation via the merge path.** The merge must execute inside the daemon's writer, never as a side process opening `source.db` concurrently — otherwise blob-GC leases and FTS triggers race. Risk is a well-meaning `sync` CLI that opens the DB directly. Mitigation: route all merges through the daemon (bead 6 AC); guard with a test that the sync path holds no direct write connection.
3. **Durable schema-version skew silently corrupting on merge.** A peer on a newer durable `source`/`user` schema can emit rows/columns the receiver can't represent; merging across the gap is data corruption of an *irreplaceable* tier. Mitigation: hard abort on version mismatch before any write (bead 4), reuse the existing backup-manifest-gated migration path, and require equal durable `user_version` as a precondition — never auto-migrate a durable tier as a side effect of sync.

**Files that anchor this spec:** `polylogue/storage/sqlite/archive_tiers/bootstrap.py` (durability classes), `.../source.py` + `.../source_write.py:130` (content-addressed `raw_id`), `.../user.py` + `.../user_write.py:156-215` (deterministic `assertion_id`), `.../migration_runner.py:16` (`DURABLE_MIGRATION_TIERS`), `polylogue/pipeline/ids.py` (session content hash), `polylogue/daemon/convergence.py` (rebuild driver).

---

## [a3c5a37c5e2e30063] Wave-5 Spec — Prompt/Meta-Workflow Distillery

I have enough grounding. Here is the implementation spec.

---

# Wave-5 Spec — Prompt/Meta-Workflow Distillery

Induce 5–8 general **parametrized meta-prompts** (params: `repo`, `task_type`, `risk_tier`) from the operator's highest- and lowest-value past sessions, each carrying (a) the pattern it generalizes, (b) citation-anchors to the sessions it's distilled from, (c) a falsifiable A/B test. Store as `user.db` objects behind a recursive-safety gate (candidate until judged).

**The central grounding win:** the entire feature needs **zero durable SQL migration**. `AssertionKind` is free `TEXT` (no CHECK), `USER_SCHEMA_VERSION` stays 4, and the existing candidate→judgment lifecycle (`TRANSFORM_CANDIDATE`/`JUDGMENT`, `AssertionStatus.CANDIDATE`, `record_candidate_judgment`) already *is* the recursive-safety gate. We reuse it verbatim.

---

## 1. Schema / DDL + tier

No new tables. Two storage touchpoints, both already exist:

**A. Distilled prompts → `user.db` `assertions` (durable tier, USER_SCHEMA_VERSION unchanged).**
Row shape (via existing `upsert_assertion`, `polylogue/storage/sqlite/archive_tiers/user_write.py:901`):

| column | value |
| --- | --- |
| `kind` | **new** `AssertionKind.PROMPT_TEMPLATE = "prompt_template"` (free TEXT — no schema bump) |
| `status` | `AssertionStatus.CANDIDATE` on birth → `ACCEPTED`/`REJECTED`/`DEFERRED` via judgment |
| `scope_ref` | `distillery:prompt_distillery@v1` |
| `target_ref` | `prompt-template:<slug>` (**new registered `ObjectRef` kind** — NOT NULL, cannot be a session) |
| `key` | `<repo>/<task_type>/<risk_tier>` (the cell coordinate) |
| `value_json` | `{params:{repo,task_type,risk_tier}, template_text, pattern, ab_spec, distilled_from:[session_ids], distiller_version, uses_llm}` |
| `body_text` | the rendered meta-prompt text |
| `evidence_refs_json` | the **citation-anchors**: `["session:<id>", "evidence:<session:msg:block>", …]` (existing evidence-ref machinery) |
| `context_policy_json` | `{"inject": false, "promotion_required": true}` (mirrors `upsert_transform_candidate_assertions`, user_write.py:1052) |
| `confidence` | mining confidence (REAL, existing column) |
| `visibility` | `private` (default) |
| `author_kind`/`author_ref` | `distillery` / `distillery:prompt_distillery@v1` |

Deterministic `assertion_id` via a new `assertion_id_for_prompt_template(...)` helper mirroring `assertion_id_for_transform_candidate` (user_write.py:215) — hash over `(cell, template_text, sorted evidence_refs, distiller_version)` so rebuild is idempotent.

**B. A/B verdicts → wire the already-reserved `AssertionKind.PROMPT_EVAL = "prompt_eval"`** (enums.py:426, declared but no producer today; already in the `user_audit` map, user_audit.py:30). One PROMPT_EVAL row per template evaluation, `target_ref = assertion:<template_id>`, `value_json = {verdict, sample_n, baseline_metric, treatment_metric, evidence_refs}`.

**C. Mining substrate = read-only over existing derived rows (index tier, rebuildable) — no new index table.** Sources:
- `session_profiles` (index.py:799): `terminal_state`+`terminal_state_confidence` (success/stall signal), `workflow_shape`+`workflow_shape_features_json`, `tool_use_count`, `tool_calls_per_minute`, `total_cost_usd`/`total_credit_cost`, `repo_names_json`, `tags_json`/`auto_tags_json`, `substantive_count`, `work_event_count`, `phase_count`.
- `SessionDigest.tool_summaries` / `RunProjection` (`insights/transforms.py`, `insights/run_projection.py`): tool-call shapes, `handler_kind` (shell/git/github/test), `status` (ok/failed/unknown), `RunStatus` (completed/failed/unknown), `ObservedEvent` (test_passed/failed, command_*).
- `AssertionKind.CORRECTION` rows (`storage/insights/feedback/__init__.py:159`): corrections the operator issued, `target_ref = insight:<session_id>`.
- `PathologyFinding`s (`insights/pathology.py`): `wasted_loop`, `stale_context` — the deterministic stall detectors.

---

## 2. Mining algorithms (pseudocode)

```
# Deterministic-first. LLM only phrases the generalization (uses_llm flag honest).
# Everything measurable is derived without an LLM so evidence is reproducible.

def distill_prompts(archive, *, distiller_version) -> list[PromptTemplateCandidate]:
    sessions = load_profiles(session_profiles)              # + join digests/corrections/pathologies

    # --- derive the two axes not stored as columns ---
    for s in sessions:
        s.task_type  = classify_task_type(s)                # workflow_shape + tag/auto_tags + tool-mix + repo
        s.risk_tier  = classify_risk_tier(s)                # destructive handler_kinds (git push/reset, schema),
                                                            #   cost band, tool_use_count, pathology presence
        s.outcome    = score_outcome(s)                     # SUCCESS | STALL | CORRECTED (see below)

    cells = group_by(sessions, key=(repo, task_type, risk_tier))

    candidates = []
    for cell in cells:
        succ  = [s for s in cell if s.outcome == SUCCESS  and s.terminal_state_confidence >= θ_conf]
        stall = [s for s in cell if s.outcome in (STALL, CORRECTED)]
        if len(succ) < N_MIN_SUCC or len(stall) < N_MIN_STALL:   # honesty floor — no induction on thin evidence
            continue

        # feature contrast: what did winners do that stallers didn't?
        succ_shape  = aggregate_tool_shape(succ)    # ordered handler_kind n-grams, plan-before-act, test cadence
        stall_shape = aggregate_tool_shape(stall)
        correction_themes = cluster_correction_bodies(stall)   # what the operator repeatedly had to correct
        pathology_mix     = counts_by_kind(stall)              # wasted_loop / stale_context frequency

        delta = contrast(succ_shape, stall_shape, correction_themes, pathology_mix)
        if delta.is_empty(): continue

        template_text = render_meta_prompt(cell.key, delta)    # deterministic skeleton;
                                                               # optional LLM rephrase → uses_llm=True
        candidates.append(PromptTemplateCandidate(
            params      = {repo, task_type, risk_tier},
            template    = template_text,
            pattern     = delta.describe(),                    # the generalization, in words
            evidence    = citation_anchors(succ[:k] + stall[:k]),   # session/evidence refs — REQUIRED, ≥1
            ab_spec     = build_ab_spec(cell),                 # §falsifiability below
            confidence  = min(delta.strength, evidence_coverage(cell)),
        ))
    return rank(candidates)[:8]                                # top 5–8 by (confidence × cell volume)


def score_outcome(s):
    if s.terminal_state in CONVERGED_STATES and not s.pathologies and not s.corrections:
        return SUCCESS
    if s.corrections or s.terminal_state in STALL_STATES:
        return CORRECTED if s.corrections else STALL
    return UNKNOWN            # excluded from both exemplar sets

def build_ab_spec(cell):
    # Falsifiable: a query predicate for future in-scope sessions, a baseline, a decision rule.
    return {
      "applies_to": query_expr(f"repo:{cell.repo} task_type:{cell.task_type} risk:{cell.risk_tier}"),
      "baseline":   { "success_rate": cell.hist_success_rate,
                      "correction_rate": cell.hist_correction_rate,
                      "median_cost": cell.hist_median_cost },
      "metric":     "success_rate_minus_correction_rate",
      "min_sample": N_AB,
      "decision":   "SUPPORTED if treatment - baseline > ε over ≥N_AB sessions; "
                    "REFUTED if treatment <= baseline; else INSUFFICIENT_EVIDENCE",
    }
```

```
# A/B evaluation — the PROMPT_EVAL producer (later pass, over sessions that ran with the template).
def evaluate_prompt_template(template):
    used     = future_sessions matching template.ab_spec.applies_to AND tagged provenance=template_id
    if len(used) < template.ab_spec.min_sample:
        verdict = "INSUFFICIENT_EVIDENCE"          # never fabricate a win under the floor
    else:
        treat = metric(used); base = template.ab_spec.baseline
        verdict = decide(treat, base, template.ab_spec.decision)
    write PROMPT_EVAL assertion(target=assertion:template_id, value={verdict, sample_n, base, treat, evidence})
```

`classify_task_type` / `classify_risk_tier` are the only novel derivations (not stored columns); both are pure functions with unit tests. LLM induction is optional and gated — deterministic feature extraction is mandatory and is the evidence.

---

## 3. Migration

**No durable SQL migration. No `USER_SCHEMA_VERSION` bump. No index-tier rebuild** (mining reads existing derived tables; distilled objects live in `user.db`).

Changes are enum + generated-surface only:
1. Add `PROMPT_TEMPLATE` to `AssertionKind` (enums.py:407); confirm `PROMPT_EVAL` present (it is).
2. Register `prompt-template` as an `ObjectRef` kind (`core/refs.py`) — required for `target_ref`/`scope_ref` normalization (memory: refs must use a registered kind, cf. the `insight:` vs `pathology-detector:` gotcha).
3. Add both kinds to the `user_audit` map (`archive_tiers/user_audit.py`) — the **every-kind audit invariant** requires a surface entry.
4. Regenerate embedded surfaces (kinds leak into generated schemas): `devtools render openapi` + `devtools render cli-output-schemas`, then `devtools render all --check` — **grep for `out of sync`, don't trust the tail line** (exits 1 while printing per-surface `sync OK`).
5. New module under `polylogue/insights/` → regenerate topology projection (`devtools render topology-projection && devtools render topology-status`) or `render all --check` fails.
6. New MCP tools → update `EXPECTED_TOOL_NAMES` + tool contract.

---

## 4. Test / validation strategy

- **Determinism/idempotency:** same corpus ⇒ identical `assertion_id`s and value_json; second run is a no-op upsert. Mirror `test_crud`-style checks.
- **Evidence-completeness invariant:** pydantic `model_validator` requiring ≥1 evidence ref on every `PromptTemplateCandidate` (exactly like `SessionDigest`/`ToolSummary` in transforms.py). Property test: no template without citation-anchors.
- **Citation-anchor validity:** every `evidence_refs` entry resolves to a real session/message/block (via `resolve_ref`).
- **Recursive-safety property (the load-bearing test):** *no `PROMPT_TEMPLATE` with `status=candidate` ever appears in* `compile_context` / `compose_context_preamble` output. Assert `inject:false` + `promotion_required:true` on birth; assert promotion requires an explicit `JUDGMENT` row.
- **Judgment lifecycle:** `record_candidate_judgment` on a template writes a `JUDGMENT` superseding the candidate and only then may an `ACCEPTED`/ACTIVE row become injectable — reuse existing candidate-review tests.
- **Falsifiability honesty:** evaluator returns `INSUFFICIENT_EVIDENCE` (not a win) below `min_sample`; baseline computed from historical cell rate, never assumed. Synthetic-outcome unit tests for all three verdicts.
- **Provider/privacy:** any public payload passes through `project_origin_payload` (no provider leakage); `_redact_local_path` applied to repo paths in template text; `visibility=private`, `export_eligible=False`.
- **`user_audit` every-kind invariant** green for both new kinds.
- Verification loop: `devtools test <files>` (testmon inner loop), `devtools render all --check` (grep out-of-sync), `mypy --strict`. Never blanket-run directories.

---

## 5. Bead breakdown (acceptance criteria)

1. **Foundation: kinds + ref + generated surfaces** (no SQL). *AC:* `PROMPT_TEMPLATE` added + `PROMPT_EVAL` wired; `prompt-template` ObjectRef registered; `user_audit` lists both kinds; `render all --check` green (grepped for `out of sync`); mypy green.
2. **Deterministic mining engine** — cell clustering + success/stall/corrected outcome scoring + `classify_task_type`/`classify_risk_tier` over `session_profiles`+`RunProjection`+`CORRECTION`+pathology. *AC:* pure function (reads only); honesty floors (`N_MIN_*`) enforced; determinism property test; task-type/risk-tier derivations unit-tested with fixtures.
3. **Candidate induction + `user.db` write** — mirror `upsert_transform_candidate_assertions`. *AC:* templates written `status=CANDIDATE`, `inject:false`, `promotion_required:true`; deterministic idempotent `assertion_id`; ≥1-evidence-ref validator; params+pattern+ab_spec in value_json.
4. **Recursive-safety gate + promotion** via existing `record_candidate_judgment`. *AC:* property test — candidate never injected into context compile; judgment promotes to ACCEPTED; injection possible only after explicit accept.
5. **Falsifiable A/B evaluator (`PROMPT_EVAL` producer).** *AC:* verdict ∈ {SUPPORTED, REFUTED, INSUFFICIENT_EVIDENCE} with evidence refs; honest baseline; below-floor ⇒ INSUFFICIENT_EVIDENCE (no fabrication); provenance-tagging of template-influenced sessions.
6. **Surfaces** — `InsightType` registry entry + MCP tools (list/get/judge distilled prompts) + `analyze` verb. *AC:* `EXPECTED_TOOL_NAMES` + tool contract updated; discovery tests pass; origin-projected payloads; CLI surfaces templates with citations.
7. *(optional)* **Daemon convergence stage** — bounded mining on quiet window using `false_means_pending` → `convergence_debt`. *AC:* bounded per-run; no writer contention (main process stays sole writer); debt retried.
8. *(optional)* **Docs + topology** — `docs/` entry + topology-projection regen. *AC:* `render topology-status` green.

---

## 6. Top-3 risks

1. **Construct-validity / counterfactual fabrication (highest).** "A prompt that *would have* produced a better outcome" is a causal claim from observational data — confounded by task difficulty, model-version drift, and corpus era. Overstating it repeats the memory's "recovery report fabricated PR #123" failure. *Mitigation:* candidate-only birth; falsifiable A/B with honest baselines and an INSUFFICIENT_EVIDENCE floor; never inject unjudged; `pattern`/`confidence` framed as hypothesis, not fact.
2. **Recursive contamination (Ouroboros).** A promoted template injected into future sessions, then those sessions mined again → self-reinforcing loop that launders a weak prompt into apparent success. *Mitigation:* the candidate-until-judged gate plus **provenance tagging** — exclude template-influenced sessions from a template's own exemplar/success set; the A/B evaluator measures them only as *treatment*, never as fresh evidence for induction.
3. **Privacy / provider + local-path leakage.** Templates distilled from real operator sessions carry repo paths, secret-shaped commands, and provider tokens. *Mitigation:* `_redact_local_path` on all path-bearing template text, `project_origin_payload` on every public surface, `visibility=private` + `export_eligible=False` by default; a lint/property test asserting no raw absolute path or provider token survives into a `prompt_template` body.

**Key file anchors:** `polylogue/core/enums.py:399` (AssertionKind/Status), `storage/sqlite/archive_tiers/user.py:12` (assertions DDL, v4), `storage/sqlite/archive_tiers/user_write.py:901/1002/1297` (upsert/candidate/judgment), `storage/sqlite/archive_tiers/index.py:799` (session_profiles), `insights/transforms.py:280` (DecisionCandidate pattern), `insights/pathology.py` (deterministic-detector template), `storage/insights/feedback/__init__.py:159` (CORRECTION rows), `insights/registry.py:70` (InsightType descriptor).

---

## [aae19bba204887c04] Wave-5 spec: three missing insight units

I have the grounding I need. Here is the implementation spec for the three units, grounded in the live schema.

---

# Wave-5 spec: three missing insight units

## Grounding (verified against live source)

- **embeddings.db** (`storage/sqlite/archive_tiers/embeddings.py`, schema v1, rebuildable): `message_embeddings` vec0 vtable is **message-level**, 1024-dim Voyage, with aux `+session_id +origin`; plus `message_embeddings_meta`, `embedding_status`. **No stored per-session centroid exists** — "similar sessions" is a per-message `MATCH` fanout (`_PER_MESSAGE_K=20`) deduped to sessions (`daemon/similarity.py`). Any session-vector unit must mean-pool at derivation time.
- **index.db** (v24, rebuildable — no migration chain, bump DDL + rebuild) already carries: `session_links` (lineage edges; `LinkType` = continuation/sidechain/subagent/branch/fork/resume/repaired; PK `src_session_id,dst_origin,dst_native_id,link_type`; `resolved_dst_session_id`), `threads`+`thread_sessions` (the **lineage** "logical-session" thread — this is what the `threads` MCP tool / `ThreadInsightQuery` / `list_thread_insights` / registry `name="threads"` serve), `session_profiles` (`repo_paths_json`, `repo_names_json`, `logical_session_id`, `source_name`, `canonical_session_date`), `repos`/`session_repos`/`session_working_dirs`/`session_commits`. `neighbor_candidates.py` already scores token-overlap + temporal-closeness + shared-repo with tuned weights — the cross-origin precursor.
- **user.db** (v4, durable — additive numbered migration under `storage/sqlite/migrations/user/NNN_*.sql`, one `PRAGMA user_version` step, behind backup manifest): unified `assertions` (kind `TEXT`, no CHECK — vocab grows freely; existing `AssertionKind.WORKSPACE_NOTE` backs `save_workspace`) + `user_settings` ("settings are state, not epistemic claims").
- **Query units** (`archive/query/metadata.py`): `QueryUnitName` = message/action/block/assertion/file/run/observed-event/context-snapshot; `PROJECTION_QUERY_UNITS = {action,assertion,file,message}`. Adding a DSL-queryable unit = new `QueryUnitDescriptor`.
- **gizmo_id is NOT surfaced into index today** — it lives in source.db raw payload (`raw_provider_payload.gizmo_id == g-p-<id>` for ChatGPT projects). `project` membership needs a prerequisite extraction step.
- Adding any `polylogue/` module → regen topology projection or `render all --check` fails. New MCP tool → `EXPECTED_TOOL_NAMES` + contract. New `AssertionKind` → regen `render openapi` + `render cli-output-schemas`.

---

## Unit 1 — `topic-cluster`

**(1) Schema / tier.** Tier: **index.db** (rebuildable from embeddings.db ∪ session_profiles). **Tables**, not VIEWs (clustering is a whole-corpus, non-incremental snapshot). New DDL in `archive_tiers/index.py`, index version v24→v25:

```sql
CREATE TABLE topic_clusters (
    cluster_id        TEXT PRIMARY KEY,          -- content-addressed: hash(sorted member session_ids)
    run_id            TEXT NOT NULL,             -- clustering pass identity
    algo              TEXT NOT NULL CHECK(algo IN ('hdbscan','kmeans')),
    label             TEXT NOT NULL DEFAULT '',  -- top structural TF-IDF terms (no LLM, no prose-mining)
    label_method      TEXT NOT NULL DEFAULT 'tfidf',
    session_count     INTEGER NOT NULL CHECK(session_count >= 0),
    first_seen_date   TEXT, last_seen_date TEXT,
    centroid_json     TEXT,                      -- optional stored centroid for reuse by unit 3
    materialized_at   TEXT NOT NULL DEFAULT '',
    search_text       TEXT NOT NULL DEFAULT ''
) STRICT;
CREATE TABLE session_topic_memberships (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    cluster_id  TEXT REFERENCES topic_clusters(cluster_id) ON DELETE SET NULL,  -- NULL = HDBSCAN noise (-1)
    probability REAL NOT NULL DEFAULT 1.0 CHECK(probability BETWEEN 0 AND 1),
    PRIMARY KEY(session_id)
) STRICT;
```

**(2) Derivation.** (a) Mean-pool + L2-normalize each session's `message_embeddings` rows into a session centroid (filter to authored/substantive `material_origin` to avoid tool-spam dilution). (b) Cluster: **HDBSCAN** on normalized centroids (cosine≈euclidean on the unit sphere), `min_cluster_size` from config; **k-means fallback** (fixed seed → deterministic) when HDBSCAN wheel absent or n < threshold. (c) Label = top TF-IDF terms over member `session_profiles.search_text` — structural, never prose-mined; local-LLM narration is a deferred follow-up. (d) `cluster_id` = hash of sorted member set for cross-run stability; carry label continuity by mapping old→new clusters on membership Jaccard. Timelines/"what have I worked on lately" = `session_topic_memberships ⋈ session_profiles.canonical_session_date` bucketed by ISO week.

**(3) Migration.** Derived tier: **no migration chain** — edit canonical DDL, bump index user_version, rebuild plan `polylogue ops reset --index && polylogued run`. New `TopicClusterStage` in `daemon/convergence_stages.py` after embed catch-up, whole-corpus recompute on quiet window, `false_means_pending` batching (mirror embed's `check_many`).

**(4) Tests.** Property: every embedded session lands in exactly one membership row (cluster or NULL-noise); partition covers all; k-means determinism under fixed seed; label is structural (assert no substring leaks from tool_result prose). Contract: disabled/absent-embeddings returns graceful status mirroring `similarity.py` (`disabled`/`unavailable`/`not_embedded`), never a fake empty cluster. Rebuild idempotency.

**(5) Bead.** `feat(insights): topic-cluster unit over session embeddings`. **AC:** (i) `topic_clusters`+`session_topic_memberships` DDL @ index v25; (ii) HDBSCAN with k-means fallback, config-driven `min_cluster_size`; (iii) structural TF-IDF labels, zero prose-mining; (iv) converger stage with quiet-window deferral; (v) topic-timeline read surface + `find`/MCP exposure answering "what have I worked on lately"; (vi) graceful embeddings-disabled contract.

**(6) Risks.** (1) HDBSCAN instability across runs churns cluster ids → timeline discontinuity — mitigate with content-addressed ids + Jaccard label carry-over. (2) Mean-pooling multi-topic long sessions dilutes signal — filter by `material_origin`, consider per-message soft membership later. (3) Whole-corpus recompute cost on the ~38 GB / 8.8k-logical-session archive — bound like embed (windowed, deferred), never inline-block ingest.

---

## Unit 2 — `project`

**(1) Schema / tier.** Split by durability. **Durable identity in user.db** (project identity is *state*, not an epistemic claim — same rationale as `user_settings`; typed membership rules don't fit one assertion `value_json`). New dim tables, user v4→v5:

```sql
CREATE TABLE projects (
    project_id    TEXT PRIMARY KEY,   -- deterministic from slug
    name          TEXT NOT NULL, slug TEXT NOT NULL UNIQUE,
    created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
) STRICT;
CREATE TABLE project_membership_rules (
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    rule_kind  TEXT NOT NULL CHECK(rule_kind IN ('repo_id','cwd_prefix','worktree_path','gizmo_id','session_pin')),
    rule_value TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    PRIMARY KEY(project_id, rule_kind, rule_value)
) STRICT;
```

**Resolved membership is a derived VIEW in index.db** (`v_project_sessions`): union of sessions matched by each rule — `repo_id` via `session_repos`; `cwd_prefix`/`worktree_path` via `session_working_dirs` (**segment-aligned** prefix, not substring); `gizmo_id` via a prerequisite index column/table `session_project_ids(session_id, gizmo_id)` extracted from source.db raw payload at materialize time; `session_pin` direct. Rollups (`project × origin`, `project × ISO-week`: cost/sessions/tokens) computed over the VIEW. Cross-reference existing `WORKSPACE_NOTE` workspaces for a compat/upgrade path.

**(2) Derivation.** Membership = `DISTINCT session_id` across rule matches (a session matching multiple rule kinds counts once). Rollups aggregate `session_profiles` cost/token columns grouped by resolved membership × `source_name`(→origin projection) × week.

**(3) Migration.** **Durable**: additive `migrations/user/005_projects.sql`, single `PRAGMA user_version` 4→5, behind verified backup manifest. **Derived**: gizmo extraction is an additive-derived index change (parse/materialize + index column) → index rebuild; the VIEW ships in canonical DDL. `devtools lab policy schema-versioning` must see the user step as additive-only.

**(4) Tests.** cwd-prefix boundary: `/foo` must NOT match `/foobar` (segment-aligned). Multi-rule session counted once. **Durable survival**: create project + rules, `ops reset --index`, rebuild → identity + rules intact, membership re-resolves. gizmo `g-p-` normalization; GEMINI/DRIVE non-injective origin guard. Backup-manifest gate present.

**(5) Bead.** `feat(identity): durable project unit aggregating repo ∪ worktree ∪ cwd ∪ gizmo`. **AC:** (i) `projects`+`project_membership_rules` in user.db via additive v5 migration + backup manifest; (ii) prerequisite gizmo_id extraction into index; (iii) `v_project_sessions` VIEW with segment-aligned cwd matching; (iv) cross-repo/cross-origin rollups; (v) survives index rebuild; (vi) MCP/CLI surface; workspace compat noted.

**(6) Risks.** (1) cwd-prefix substring false positives — enforce path-segment alignment. (2) gizmo not yet surfaced from source.db → hard prerequisite; respect non-injective GEMINI/DRIVE→AISTUDIO_DRIVE collapse. (3) Durable-rules/derived-membership split: editing a rule must invalidate the derived VIEW/rollup cache — needs converger invalidation hook, else stale membership.

---

## Unit 3 — `cross-origin-thread`

**(1) Schema / tier.** Tier: **index.db**, rebuildable. **Tables** (pairwise scoring needs stored edges; O(n²)-ish, not a VIEW). Name `cross_origin_threads` to avoid collision with the existing **lineage** `threads` table:

```sql
CREATE TABLE cross_origin_thread_edges (
    src_session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    dst_session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    embed_sim   REAL, entity_overlap REAL, temporal_score REAL,
    score       REAL NOT NULL, hard_signal INTEGER NOT NULL CHECK(hard_signal IN (0,1)),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(src_session_id, dst_session_id)
) STRICT;
CREATE TABLE cross_origin_threads (
    thread_id TEXT PRIMARY KEY, session_count INTEGER NOT NULL,
    origins_json TEXT NOT NULL DEFAULT '[]', start_time TEXT, end_time TEXT,
    label TEXT NOT NULL DEFAULT '', materialized_at TEXT NOT NULL DEFAULT ''
) STRICT;
CREATE TABLE cross_origin_thread_sessions (
    thread_id TEXT NOT NULL REFERENCES cross_origin_threads(thread_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    PRIMARY KEY(session_id)
) STRICT;
```

**(2) Derivation.** Candidate pairs blocked by shared repo/time-bucket (never full cross-join). Keep a pair iff **origins differ** AND **no resolved `session_links` edge exists either direction** (this is the "logical conversation with NO replay session_link" constraint — subtract lineage). Score = weighted sum (reuse `neighbor_candidates` weights) of: embedding-sim (session centroids from unit 1, or per-message MATCH fanout) ≥ τ; shared-entity (`session_repos.repo_id` ∪ `session_commits.commit_sha` ∪ shared file ∪ project/gizmo); temporal adjacency within window W. **Hard false-merge floor**: edge requires `hard_signal=1` (shared-entity or high embed-sim) — temporal-only never merges. Group via union-find over surviving edges; **cap transitive merge** (hub-session guard: reject a merge that bridges two otherwise-disjoint high-mass components through a single low-degree pivot). Formalizes the `threads` MCP tool as a sibling cross-origin mode/insight (new `InsightType` + MCP contract + `EXPECTED_TOOL_NAMES`).

**(3) Migration.** Derived index bump + rebuild plan; `CrossOriginThreadStage` in converger, ordered **after** embed + topic-cluster (depends on centroids).

**(4) Tests.** **Exclusion invariant**: no `cross_origin_thread_edges` row coincides with any resolved `session_links` pair (property test over lineage fixtures). **Differing-origin invariant**. **False-merge floor**: temporal-only pair with no entity/embed overlap does NOT produce an edge. **Hub guard**: fixture where one pivot would over-merge two threads stays split. Determinism; graceful degradation to entity+temporal when embeddings disabled (explicit reduced-confidence marker, not silent garbage).

**(5) Bead.** `feat(insights): cross-origin-thread — logical conversation across providers`. **AC:** (i) three tables @ index bump; (ii) edge = differ-origin ∩ ¬lineage ∩ score≥floor ∩ hard_signal; (iii) union-find grouping with transitive-merge cap; (iv) blocked candidate generation (no full cross-join); (v) `threads` tool/insight cross-origin surface + contract + `EXPECTED_TOOL_NAMES`; (vi) exclusion + false-merge invariants tested; (vii) embeddings-disabled degradation.

**(6) Risks.** (1) Transitive over-merge via hub sessions → one giant meaningless thread — hard floor + hub guard + component-mass cap. (2) Embedding dependency: must degrade to entity+temporal (marked) not silently emit low-quality merges. (3) O(n²) candidate blowup on the large archive — mandatory repo/time-bucket blocking before scoring.

---

## Cross-cutting sequencing

Build order **1 → 2 (gizmo prereq) → 3** (unit 3 consumes unit-1 centroids and unit-2 project entities). All three add `polylogue/` modules → run `devtools render topology-projection && devtools render topology-status`. Units 1 & 3 are index-only rebuildable (edit DDL, no upgrade helper — `schema-versioning` policy rejects them). Unit 2 is the only durable-tier touch → additive numbered user migration + backup manifest. Consider whether each warrants a `QueryUnitDescriptor` (DSL `topic`/`project`/`cross-origin-thread where …`) versus insight-registry-only exposure; recommend insight-registry + MCP first, DSL unit as a fast-follow once field sets stabilize.

---

## [a255ed3e49b833e02] Spec: Collapse ~130 MCP tools → ~15-verb algebra

I have the full grounding. Returning the spec.

---

# Spec: Collapse ~130 MCP tools → ~15-verb algebra

Grounded in `mcp/server.py` (role-gated registration), `mcp/server_tools.py` / `server_mutation_tools.py` / `server_maintenance_tools.py` / `server_insight_tools.py` / `server_context_tools.py`, `mcp/server_resources.py`, `mcp/server_prompts.py`, `tests/infra/mcp.py::EXPECTED_TOOL_NAMES`, `tests/unit/mcp/test_envelope_contracts.py::TOOL_CONTRACT`, `archive/query/expression.py`, `core/enums.py::AssertionKind`.

**Real surface today:** `EXPECTED_TOOL_NAMES` is a 95-entry static allowlist; `register_insight_tools` adds ~15 more `list_<insight>` tools dynamically from `INSIGHT_REGISTRY` → ~110–130 registered tools. Role-gating is registration-time only (`role_allows`, `read<write<admin`): read gets query/read/context/insight; write adds `register_mutation_tools`; admin adds `register_maintenance_tools`. FastMCP resource subscription / `list_changed` is **not wired today** (no `subscribe`/`notify` in `mcp/`); the `ingest_cursor` table in `ops.db` (`daemon/cursor_lag_status.py:136`) is the natural change-generation source.

---

## 1. Target tool set + EXPECTED_TOOL_NAMES delta

**Target algebra (9 tools; budget ≤15):**

| Verb | Role | Absorbs |
|---|---|---|
| `query(expression, …)` | read | search, query_units, list_sessions, archive_list/search_sessions, facets, stats, get_stats_by, provider_usage, aggregate_sessions, threads, tool_usage, cost_rollups, session_costs, usage_timeline, archive_coverage, session_profiles/phases/tag_rollups/work_events, session_tool_timing, tool_call_latency_distribution, workflow_shape_distribution, find_stuck/abandoned/resume_candidates, neighbor_candidates, blackboard_list, list_assertion_claims, list_corrections, list_tags/marks/annotations/saved_views/recall_packs/workspaces/read_view_profiles + every dynamic `list_<insight>` |
| `get(ref, view=…)` | read | get_session, get_session_summary, get_messages, get_session_tree, get_session_topology, get_logical_session, archive_get_session, session_profile, session_latency_profile, raw_artifacts, get_metadata, get_pathologies, get_postmortem_bundle, resolve_ref (`view∈{transcript,messages,tree,topology,logical,summary,profile,latency,raw,metadata,pathologies,postmortem}`, mirrors CLI `read --view`) |
| `explain(subject, …)` | read | explain_query_expression, explain_import, query_completions, insight_rigor_audit, action_affordances, readiness_check, embedding_status, embedding_preflight, archive_debt |
| `context(…)` | read | compile_context, build_context_image, compose_context_preamble, get_resume_brief |
| `correlate(…)` | read | correlate_session, correlate_sessions, compare_sessions, find_similar_sessions |
| `coordinate(view, …)` | read | agent_coordination (keep; distinct bounded envelope) |
| `assert(kind, target, body, …)` | write | add_tag, add_mark, bulk_tag_sessions, save_annotation, set_metadata, record_correction, save_saved_view, save_recall_pack, save_workspace, blackboard_post |
| `retract(kind, ref, …)` | write | remove_tag, remove_mark, delete_annotation, delete_metadata, delete_saved_view, delete_recall_pack, delete_workspace, clear_corrections |
| `maintenance(action, …)` | admin | maintenance_preview/execute/status/list, rebuild_index, update_index, rebuild_session_insights, delete_session (destructive; `action="delete_session", confirm=true`) |

**EXPECTED_TOOL_NAMES delta:** remove all 95 static entries **except** none survive verbatim; the dynamic insight tools also drop. Net: `−95 static −~15 dynamic`, `+9` (`query, get, explain, context, correlate, coordinate, assert, retract, maintenance`). `TOOL_CONTRACT` shrinks to 9 rows: `query`→`("envelope",{items,total,limit,offset})` (+`{items,total,group,metric}` when aggregate), `get`→`single_object`, `explain`/`context`/`correlate`/`coordinate`→`single_object`, `assert`/`retract`/`maintenance`→`operation_result`.

**Resources (progressive disclosure + read surface):** keep `polylogue://stats|sessions|tags|readiness` + templates `session/{id}`, `messages/{id}`, `session-tree/{id}`, `origin/{name}/recent`; **add** `polylogue://tools` (verb taxonomy + retired→new map), `polylogue://schema` (query DSL grammar/fields), `polylogue://views` (saved views as subscribable list). All gain `resources/subscribe` + `list_changed` off `ingest_cursor`.

**Prompts:** keep 6 static as `query` presets; **add dynamic** prompts generated per `AssertionKind.SAVED_QUERY` / `RECALL_PACK` row (each a slash-command that runs its stored `query`). `EXPECTED_PROMPT_NAMES` becomes `static ∪ dynamic`.

---

## 2. Collapse mapping (pseudocode)

```python
# query() — universal read/aggregate over the Lark DSL
async def query(expression, *, limit=10, offset=0, **field_filters):
    ast = parse_unit_source_expression(expression)         # expression.py
    if ast is None:                                         # bare/legacy → session source
        ast = build_session_query_request(query=expression, **field_filters)
    pipe = ast.pipeline()                                   # Session|UnitSource + stages
    # terminal action drives shape: rows | count | (NEW) facet | profile
    return envelope(execute_pipeline(pipe, limit, offset))

# insight tools become presets, NOT tools:
#   list_cost_rollups        → query("sessions | group by model | with cost")
#   session_phases           → query("session:<id> | phases")   (with-projection)
#   tool_usage               → query("actions | group by tool | count")

# get() — resolve_ref + read_view fan-in
async def get(ref, *, view="summary", limit=20, offset=0):
    sid = resolve_session_id(ref)                           # was resolve_ref
    return READ_VIEWS[view](sid, limit, offset)             # transcript/tree/raw/...

# assert()/retract() — one write path over unified assertions table
WRITABLE = ASSERTION_KINDS - {PATHOLOGY, SUPPRESSION_system}   # system-emitted excluded
async def assert(kind, target, body=None, *, context_policy=None, id=None):
    k = AssertionKind.from_string(kind)
    if k not in WRITABLE: return error("kind not user-assertable", code="forbidden_kind")
    row = upsert_assertion(kind=k, scope_ref=target, payload=body,
                           context_policy_json=context_policy or {"inject": False})
    if k in {SAVED_QUERY, RECALL_PACK}: mcp.notify_prompt_list_changed()   # dynamic prompt
    return operation_result(row)

async def retract(kind, ref):
    return operation_result(delete_assertion(AssertionKind.from_string(kind), ref))
```

---

## 3. Migration path (role-gating coherent)

Registration stays a role filter — only the split changes:

```python
def register_tools(mcp, hooks):
    register_read_verbs(mcp, hooks)        # query, get, explain, context, correlate, coordinate
    if role_allows(hooks.role, "write"):
        register_write_verbs(mcp, hooks)   # assert, retract
    if role_allows(hooks.role, "admin"):
        register_admin_verbs(mcp, hooks)   # maintenance (incl. delete_session)
```

- `assert`/`retract` are **single-gated at the verb** (write); no per-kind gate needed because the `WRITABLE` allowlist is validation, not authorization — a read client never sees the verb. This is the key coherence win: 15 write tools with 15 gates → 2 tools, 1 gate, 1 allowlist.
- `delete_session` moves **up to admin** (it deletes durable archive content, unlike assertion writes) — a deliberate tightening; flag in the bead if the operator wants it left at write.
- Dynamic prompts derived from saved-view assertions are **read-safe** (they only run `query`); creating them is the side effect of a write `assert`, so the write gate still owns creation.
- Rollout: land verbs alongside old tools behind `POLYLOGUE_MCP_ALGEBRA=1`; flip default; one-release alias shim (bead 8) maps old names→verbs emitting `{"deprecated": "use query(...)"}`; then delete. Regenerate `render openapi`, `render cli-output-schemas`, `render mcp-reference`, topology projection.

---

## 4. Test strategy

- **Collapse-equivalence goldens (load-bearing):** parametrize over every retired tool; assert `new_verb(mapped_params)` envelope == old tool envelope against `seeded_db`/`corpus_seeded_db`. No tool is deleted until its golden passes. This is what prevents capability loss on the continuity surface.
- **TOOL_CONTRACT completeness gate stays:** `test_every_registered_tool_is_classified` already fails on unclassified tools — it enforces the shrink is intentional. Update the 9 rows.
- **Role-set assertions:** `build_server(role=r)._tool_manager._tools` == exact expected set for read/write/admin.
- **assert/retract round-trip:** parametrized over `WRITABLE` kinds — assert → `query(kinds=…)`/`list_assertion_claims` returns it → retract removes; assert of `PATHOLOGY` → `forbidden_kind`.
- **Subscription:** subscribe resource → advance `ingest_cursor` (fixture) → assert `notifications/resources/list_changed` fires; unsubscribe silences.
- **Dynamic prompts:** `assert(kind="saved_query", …)` → prompt appears in `_prompt_manager._prompts` and executes its `query`; `retract` → gone. Loosen `EXPECTED_PROMPT_NAMES` to `static ⊆ registered`.
- **DSL projection coverage:** property test that each former insight has an equivalent `query` expression (facet/profile terminal actions).
- Verify via `devtools test tests/unit/mcp/` + `render all --check` (grep `out of sync`).

---

## 5. Bead breakdown (acceptance)

1. **`query()` universal read/aggregate verb.** AC: every retired search/list/facet/stats/aggregate/insight/`find_*`/`threads` tool reproduced via `query` equivalence golden; dynamic `list_<insight>` registration removed; TOOL_CONTRACT+EXPECTED updated. **Depends on grammar bead below if projections missing.**
2. **DSL terminal-action extension (`| facet`, `| profile`).** AC: `QueryUnitTerminalAction` gains facet/profile (reserved slots noted in `expression.py:282`); facets/session_profile/latency expressible without a bespoke tool. (Prereq for #1's non-aggregate insights.)
3. **`get(ref, view)` universal single-record read.** AC: 13 retired getters reproduced via `view`; `resolve_ref` internalized; CLI `read --view` parity.
4. **`assert`/`retract` over unified assertions.** AC: round-trip per `WRITABLE` kind; `PATHOLOGY`/system kinds rejected; write-role gate; equivalence goldens vs 18 old mutation tools; `delete_session`→admin.
5. **Subscribable resources + `list_changed` off `ingest_cursor` + taxonomy resources.** AC: subscribe fires on ingest; `polylogue://tools|schema|views` added; MCP capabilities advertise subscription.
6. **Dynamic prompts from saved views/recall-packs.** AC: assert→prompt appears+executes, retract→gone; 6 static kept as presets.
7. **`explain`/`context`/`correlate`/`coordinate`/`maintenance` consolidation + docs regen.** AC: role gates correct; equivalence goldens; `render openapi|cli-output-schemas|mcp-reference|topology` clean; MEMORY `EXPECTED_TOOL_NAMES` gotcha updated.
8. **(optional) One-release alias shim.** AC: old names dispatch to verbs with `deprecated` marker behind flag; removed on close.

---

## 6. Top-3 risks

1. **Silent capability loss on the continuity surface.** MCP (not the API) is the agent-continuity surface (~130 heavily-used tools). A 130→9 collapse can drop a param or envelope field agents rely on. *Mitigation:* per-retired-tool equivalence goldens **before** deletion; TOOL_CONTRACT completeness test as the gate.
2. **`query()` DSL can't yet express all reads.** The Lark grammar covers `where`/`| group by | count`/`with`/`near:` (aggregates), but **not** facet histograms, similarity/correlation, or single-object profiles. Forcing them in bloats the grammar. *Mitigation:* extend the reserved `QueryUnitTerminalAction` vocabulary (bead 2) as a hard dependency of bead 1; keep `correlate` as its own verb rather than a DSL stage.
3. **Ecosystem breakage + role-gate regression.** Renaming breaks saved agent/MCP client configs; folding 15 write tools into `assert` risks a kind escaping its gate (a read client asserting). *Mitigation:* single write-gate + `WRITABLE` allowlist; version the `polylogue://tools` taxonomy resource so clients rediscover the algebra; optional alias shim (bead 8) for one release.

**Landing lands at 9 tools — well under the ~15 budget**, with resources (subscribable + taxonomy) and dynamic prompts absorbing the read/saved-query surface the old flat tool list carried.

---

## [a12b38bfb555a358c] Grounded in `pricing.py`, `cost_compute.py`, `subscription_pricing.py`, `timing.py`, `arch

Grounded in `pricing.py`, `cost_compute.py`, `subscription_pricing.py`, `timing.py`, `archive_summaries.py`, `session_summaries.py`, `cost_enrichment.py`. Confirmed the physical-vs-logical grain split in the day summarizer and multiple dead/degraded cost paths. Red-team list:

- **Lineage cost double-count in day/week summaries** — `summarize_day` (session_summaries.py:100-116) sums `total_cost_usd`/`total_messages`/`total_words`/`total_duration_ms` over *every physical* `SessionProfile` while `logical_session_count` dedups on `logical_session_id`; forks/resumes/acompact replay parent content (#2467: 16K physical vs 8.8K logical, ~32% dup) so cost-per-logical inflates ~32% and the numerator/denominator sit on different grains. Detect: `SUM(total_cost_usd)` over all profiles vs over one-per-`logical_session_id`. — Fix: reduce cost/token/word/message aggregates on logical grain (canonical-tail row per logical id), not raw physical sum; bead 4ts/2467.

- **`total_wall_duration_ms` additive across overlapping physical sessions** — concurrent subagents/forks run in parallel real-time but `summarize_day` sums each physical `wall_duration_ms`, so a day can report >86.4M ms/24h; idle-vs-parallel wall time is conflated. Detect: per-day total_wall > wallclock span. — Fix: interval-union on logical grain, not sum.

- **Exact provider-reported cost path is dead → subscription view always $0** — `_session_level_estimate` unconditionally returns `None` (pricing.py:615-617) and `estimate_message_cost` only ever yields `status="priced"` (never `"exact"`), so `estimate_session_cost`'s exact branch and `compute_session_cost`'s exact early-return (cost_compute.py:36) are unreachable; every Claude Code session is re-priced from the catalog and `total_credit_cost`/`total_subscription_equivalent_usd` stay 0. Detect: `session_costs` on any claude-code session → credit=0, sub=0. — Fix: wire provider-reported cost rows into `_session_level_estimate`; populate `basis.subscription_equivalent_usd` on the exact path.

- **`cost_enrichment` downgrades provider-reported EXACT to catalog-priced** — guard at cost_enrichment.py:54 only protects a stored estimate when the *re-derived* one is not in `("exact","priced")`; when stored=`exact` and re-derived=`priced` it proceeds and replaces the provider truth with an api-equivalent catalog estimate — directly contradicting the module docstring ("never downgraded"). Detect: cost insight `total_usd` ≠ stored `session_profiles.cost_usd` for provider-reported sessions. — Fix: keep stored whenever it is `exact` and re-derived is merely `priced`.

- **Missing credit rates for current Opus (4-7/4-8) → subscription spend silently 0** — `MODEL_CREDIT_RATES` only has 4-5/4-6/sonnet/haiku (subscription_pricing.py:84-105) while `_CURATED_PRICING` carries 4-7/4-8; `get_credit_rate("claude-opus-4-8")=None` → `credit_cost=0`, `sub_equivalent=0` for the most-used model. Detect: credit totals collapse after each model bump. — Fix: add rates for every curated Anthropic model + a lab-policy test asserting `MODEL_CREDIT_RATES.keys() ⊇` curated anthropic keys.

- **Subscription $/credit hardcodes the Pro tier** — `sub_equivalent = credit_cost/21_700_000*20.0` (cost_compute.py:133) bakes in Pro (21.7M credits, $20); a Max 20x operator ($200/361.1M) has a different $/credit, so subscription-equivalent USD is off by ~1.6× and never varies by tier. Detect: change plan, sub-equivalent unchanged. — Fix: parametrize by the configured tier from `user_settings`/`SUBSCRIPTION_TIERS`.

- **Codex disjoint-lane double-count in catalog pricing** — `_cost_components` prices `input_tokens` at full input rate *and* `cache_read_tokens` at the cache rate (pricing.py:403-424), and `billable_tokens` sums all four lanes (pricing.py:67); Codex reports input *inclusive* of cached (~96%) so if the parser also fills `cache_read_tokens` the same tokens are billed twice (the 7.69× class fixed once in `3938bc6c2`, easy to reintroduce). Detect: codex `input_tokens ≈ cache_read_tokens + small`; catalog cost ≫ provider-reported. — Fix: assert lane disjointness at parse (price `max(input-cache,0)` for inclusive providers).

- **Model prefix-fallthrough misprices adjacent variants** — `_normalize_model` falls back to longest `startswith` (pricing.py:344-347): `gpt-4.1`→`gpt-4` ($30/$60), any `-thinking`/`-audio`/`-preview` suffix→base model. Detect: `normalized_model` differs from raw provider model with a large price delta. — Fix: require exact or dated-canonical match; replace the `startswith` fallback with a curated alias table and emit `no_price` otherwise.

- **Unknown model prices to $0 with `reported` confidence** — `estimate_cost` returns `0.0` when `PRICING.get(norm) is None` (pricing.py:438-439); `compute_session_cost` still stamps `cost_confidence="reported"` when tokens were provider-reported, so an unpriced model contributes $0 and no missing-reason to the aggregate. Detect: sessions with `tokens>0` but `api_cost=0` and empty `missing_reasons`. — Fix: propagate a `no_price` reason and force `cost_confidence="mixed"`.

- **`CostModelBreakdown.session_count` is mislabeled and frozen at 1** — set to 1 on first message and copied unchanged on merge within a session (pricing.py:688-698); any cross-session rollup that sums this field under/over-counts sessions-per-model. Detect: `sum(per_model.session_count)` ≠ distinct sessions. — Fix: rename to `message_count` or recompute at rollup grain.

- **Estimated timing conflates idle with thinking/output** — `compute_session_timing` charges every inter-message gap to thinking/output/tool by the *next* message's type (timing.py:109-127, provenance `sort_key_estimated`); a user who walked away before an assistant "output" message inflates `output_duration_ms`/`thinking_duration_ms`. Detect: `computed_total_ms` diverges from event-measured `tool_active_duration_ms`. — Fix: cap gaps or prefer event-window measurement; label these as upper bounds.

- **`median_agent_response_ms` uncapped while user side is capped** — `_message_response_latencies` caps user-response gaps at 30 min (timing.py:294) but leaves agent-response gaps uncapped (timing.py:289-293); an overnight-open-then-resumed session injects a giant human→assistant gap into the agent-latency distribution. Detect: agent median ≫ user median with a fat tail. — Fix: symmetric cap or drop pairs that cross an idle boundary.

- **provider→origin collapse merges distinct runtimes' cost** — cost is keyed on `session_profiles.source_name`/provider then projected via `project_origin_payload`, but GEMINI and DRIVE both collapse to `AISTUDIO_DRIVE` (non-injective, per CLAUDE.md), so `cost_rollups` grouped by origin sums two runtimes into one bucket and per-runtime cost is unrecoverable. Detect: origin-grouped cost for `aistudio-drive` = Gemini-CLI + Drive combined. — Fix: key cost rollups on `Source` (family+runtime_root), project to origin only at the output boundary; beads 9e5.8/2qx.

- **Per-origin session share is a Simpson trap (physical grain)** — `bucket.origins[source_name] += row.session_count` (archive_summaries.py:121) counts *physical* sessions per origin while cross-origin totals elsewhere dedup logically, so `sum(providers.values()) ≠ logical_session_count` and fork-heavy runtimes (Claude Code) are over-weighted in "share of sessions/cost by provider." Detect: providers dict sums to physical, not logical, count. — Fix: dedup per-origin on `logical_session_id` too.

- **Cross-midnight resume double-books cost across days; week rollup compounds** — day bucketing assigns each *physical* profile to its own `bucket_day` and `summarize_week` re-sums day totals (session_summaries.py `summarize_week`, archive_summaries.py:165-187); a logical session whose replayed physical rows span midnight contributes cost to two day buckets, and the already-physical-inflated day cost flows straight into the week. Detect: a logical session with messages on two dates appears in both days' `total_cost_usd`. — Fix: assign each logical session's cost to a single canonical day before day/week aggregation.

GPT-pro prompt stubs:

- **[A]** "Given Python cost code where `estimate_session_cost` re-prices from a catalog per message and never reads provider-reported session totals (`_session_level_estimate` returns None), and an enrichment guard that replaces stored `exact` with catalog `priced`: enumerate every path by which a provider-reported EXACT cost is lost or downgraded, and design a precedence lattice (provider_reported ≻ catalog_priced ≻ heuristic) with an invariant test suite that proves exact is never silently replaced."

- **[DR]** "Research Anthropic Claude Code subscription credit accounting (Pro vs Max 5x/20x) as of mid-2026: exact per-model credit-per-token rates, whether cache reads are free, whether cache-write bills at input rate, and the credit-pool→USD conversion per tier. Produce a cit, per-model table I can validate `MODEL_CREDIT_RATES` and the hardcoded `21_700_000 / $20` divisor against."

- **[DR]** "Research provider token-usage semantics for OpenAI Codex/gpt-5.x, Anthropic, and Gemini: for each, does reported `input`/`prompt_tokens` *include* cached tokens, and does `output` *include* reasoning tokens? Give the exact field inclusion/exclusion rules so a cost estimator can bill input, cache_read, cache_write, and reasoning as disjoint lanes without double-counting."

---

## [a0d575c8fe1e3aa6b] OPERATOR DAILY-WORKFLOW JOURNEYS (lane 5/16). Each: today's sequence → where it breaks → s

OPERATOR DAILY-WORKFLOW JOURNEYS (lane 5/16). Each: today's sequence → where it breaks → smallest fix.

- **"Resume what I was doing yesterday in this repo"** — Today: `polylogue --cwd-prefix $PWD find "since:1d" then read` + `find_resume_candidates`/`get_resume_brief` (MCP). Break: the sessionstart hook only *lists* 3 sessions; there is no single motion that says "here is the divergent tail you left, the open loop, and the next action." Fix: `polylogue resume` verb = cwd-scoped `find_resume_candidates` → auto-selects most-recent unfinished logical session → prints resume_brief (last authored user turn, last assistant action, unresolved tool errors). One command, no MCP round-trip. — NEW

- **"Postmortem the session where the tests kept failing"** — Today: `find "tool_result_is_error:true AND pytest" then read` then `get_postmortem_bundle` per id. Break: you must already know *which* session; error-density ranking isn't a query surface. Fix: `find_stuck_sessions`/`get_pathologies` promoted to a CLI verb `analyze --pathology stuck-tests` that ranks sessions by repeated same-tool error runs (keystone `tool_result_is_error` v16) and emits the postmortem bundle inline. — NEW

- **"Prep a GPT-pro bundle for topic X"** — Today: `find "X" | with messages` → hand-assemble; or `build_context_image`/`compile_context`. Break: output is archive-shaped, not a paste-ready external-LLM brief; no token budget, no provenance footer. Fix: `read --view bundle --budget 60k` = corpus-compaction (dedup lineage prefixes via logical-session recompose) + per-included-session citation anchor footer, single markdown blob sized to a target context window. — NEW (builds on corpus-compaction + episode unit)

- **"Cite the session that motivated this PR"** — Today: `resolve_ref`/`add_mark` then paste a URL. Break: no stable citation token that survives index rebuild and resolves back to the exact message/block; canonical-URL projection exists for chatgpt/claude.ai but not for local claude-code sessions. Fix: `polylogue cite <query>` → returns a `resolve_ref`-backed anchor (`session_id:message_id`) + one-line quote, formatted for a commit trailer (`Cited-Session:`) that MCP `resolve_ref` can rehydrate. Anchors are computed ids, so rebuild-stable. — NEW (citation-anchor theme)

- **"Audit last month's spend by model and project"** — Today: `cost_rollups`/`session_costs`/`provider_usage` (MCP) or `analyze` cost. Break: subscription-credit vs API-equivalent are conflated; per-model partitioning bug (#2472); stale Codex rows inflate (#2469 memory). Fix: `analyze cost since:30d group by model,repo` as a first-class measure-algebra pipeline emitting BOTH views (API-equiv + subscription-credit) side by side, with a stale-row guard note. — polylogue-#2472 / NEW

- **"Onboard a new repo so its sessions get captured + recalled"** — Today: sessions land in `~/.claude/projects/<slug>` automatically; recall depends on `--cwd-prefix` matching. Break: no confirmation that a new repo's cwd-prefix is being ingested/recalled; silent if slug-mangling breaks the prefix match. Fix: `polylogue onboard [--cwd $PWD]` = readiness_check scoped to this cwd-prefix → reports "N sessions seen, hook wired, recall returns M" and writes a workspace_note assertion so the repo is a known recall target. — NEW (builds on readiness/coverage surfaces)

- **"Find the 3 things I abandoned and should finish"** — Today: `find_abandoned_sessions` (MCP). Break: abandonment is per-session; it doesn't cluster by *topic* or rank by how-close-to-done, and there's no CLI verb. Fix: `analyze abandoned --top 3` ranks by (recency × unresolved-loop × proximity-to-a-successful-sibling-fork), dedups lineage so a fork chain counts once, emits resume_brief for each. — NEW (episode unit + resume convergence)

- **"What did I decide about X across every session"** — Today: `find "X"` + manual scan, or `list_assertion_claims`/`list_marks` if you tagged as you went. Break: decisions live in prose, not as retrievable objects, unless you remembered to `add_mark`. Fix: `analyze decisions X` mines `material_origin=assistant_authored` turns following an operator command for decision-shaped statements → surfaces as *candidate* `AssertionKind.decision` rows (unverified, honest — no fabrication, per recovery-digest lesson) you can promote. — NEW (findings-as-objects)

- **"Continue that session but in a fresh branch of thought"** — Today: `continue` verb / `get_logical_session`. Break: `continue` resumes linearly; there's no "spawn a sibling from message N with the prefix as context" that records the fork as a first-class `session_links` edge from the operator side. Fix: `continue <id> --from <message_ref> --fork` writes a `spawned-fresh`/`prefix-sharing` link so the operator's manual fork is captured in the same lineage graph the parser builds. — NEW (lineage as durable object)

- **"Show me the oracle-digest inputs but from the archive, not scattered files"** — Today: `oracle` concatenates rawlog + git + thoughtspace + issues + lynchpin. Break: it does NOT pull the day's actual AI-session activity — the richest signal — because there's no compact per-day session digest. Fix: `polylogue digest --date D` = one-paragraph-per-logical-session summary (`get_session_summary`) for that cwd/day, shaped for `oracle` to concatenate. Closes the loop between the two evidence planes the operator already runs. — NEW

- **"Diff how two attempts at the same task went"** — Today: `compare_sessions`/`correlate_sessions` (MCP). Break: no CLI verb; comparison is structural, not outcome-framed (which attempt cost less, erred less, finished). Fix: `analyze compare <A> <B>` → measure-algebra table: authored turns, tool errors, cost, terminal state, divergence point. Uses lineage recompose so shared prefix isn't double-counted. — NEW (measure-algebra + episode)

- **"Recall the last time I hit this exact error"** — Today: `find "near:\"<error text>\"" then read`. Break: FTS is unicode61 no-stemmer, so error strings with paths/hashes match poorly; embeddings (`find_similar_sessions`) are session-level not block-level. Fix: `find --error "<paste>"` routes to block-level semantic neighbor search (`neighbor_candidates`) scoped to `tool_result_is_error:true` blocks, returning the fixing turn that followed. — NEW (builds on citation-anchor + keystone v16)

- **"Package a session as a shareable postmortem write-up"** — Today: `get_postmortem_bundle` → hand-edit. Break: bundle is data, not narrative; the retired sanitize/export cluster (chatlog≠spec) tried this and was paused. Fix: fold into `read --view postmortem` (NOT a separate export flow, per operator doctrine): problem → what-was-tried → resolution → citations, drawn from `session_phases`/`session_work_events`, redaction-free. — Ref chatlog-not-spec / NEW

- **"Save this query so tomorrow's standup is one keystroke"** — Today: `save_saved_view`/`save_recall_pack` (MCP), no CLI. Break: queries-as-objects exist in user.db but the daily-driver CLI can't create/run them; operator re-types the pipeline each morning. Fix: `find ... then save <name>` and `polylogue run <name>` — makes the saved-view a durable, runnable object on the surface the operator actually lives in. — NEW (queries-as-objects convergence)

- **"Check the archive is actually keeping up before I trust a recall"** — Today: `polylogued status` / `embedding_status` / `archive_debt`. Break: three separate probes; no single freshness verdict scoped to "the repo I'm in right now." Fix: `polylogue fresh` = cwd-scoped one-liner: last-ingested session age, convergence_debt count, embed catch-up lag → green/amber/red, so recursive-safety (don't cite a stale corpus) is a reflex, not an audit. — NEW (recursive-safety theme)

GPT-pro prompt stubs:

- **[A]** "Here is Polylogue's CLI verb set (find/read/analyze/mark/select/delete/continue) and its ~130 MCP tools. Design a *minimal* verb+flag grammar that turns the 15 operator journeys above into ≤6 fluid commands, reusing existing surfaces (resume_brief, postmortem_bundle, cost_rollups, saved_view, resolve_ref). Show the mapping journey→command and flag which need NEW substrate vs pure CLI adaptation."

- **[DR]** "Deep-research how mature dev-tools (Sourcegraph, Raycast, `atuin`, `fish`/`atuin` history, Warp AI, `gh`) expose 'resume/recall/cite my past work' as one-keystroke journeys. Extract the interaction patterns (candidate-ranking, citation tokens, saved-query objects, freshness signals) and map each to a concrete Polylogue verb, citing sources."

- **[A]** "Given content-hash idempotency + lineage recompose (parent-prefix + divergent-tail) + computed ids, specify a rebuild-stable *citation anchor* format (`session_id:message_id`) and a `Cited-Session:` commit trailer + MCP `resolve_ref` rehydration contract. Address: index.db rebuilds, lineage re-parenting, and quoting a block inside a shared prefix."

---

## [a7f009259099da1a7] Release / Versioning / Schema-Evolution as product discipline

Grounded in the schema-versioning regimes, `storage/sqlite/migrations/{source,user}/`, `migration_runner.py`, and beads b5l/20d.15/f2qv.5/3tl.13/83u. Here is the tight list.

## Release / Versioning / Schema-Evolution as product discipline

- **Publish a written semver contract for an archive tool** — define what MAJOR/MINOR/PATCH *mean* here: MAJOR = durable-tier (source/user) shape change requiring copy-forward + consent; MINOR = new derived insight / CLI verb / MCP tool (auto-rebuild, no data risk); PATCH = fix. The current release-please flow maps conventional subjects to bumps but never states the durability semantics a user reads a version number *for*. — NEW
- **Decouple app-version from the five tier `user_version`s in the release story** — a user upgrading polylogue needs a table: "app 3.x reads source v2/user v4/index v24"; surface it in `polylogue status` and CHANGELOG so an upgrade's blast radius is legible before running the daemon. — NEW
- **Make "your durable data survives every upgrade" a tested release gate, not a doctrine sentence** — a CI/`devtools verify --lab` lane that boots the current binary against a pinned *old* source.db + user.db fixture (v1..vN) and asserts clean read + additive migration to head, per durable tier version. Migration files exist but there's no forward-compat corpus proving the promise. — NEW (extends 3tl.13)
- **Backup-manifest verification as a release gate** — the doctrine says durable changes go "behind a verified backup manifest," but nothing in the release checklist forces a restore drill. Add `devtools lab policy backup-drill`: take manifest → restore to scratch → checksum-parity → open archive → run smoke query, gating any PR that bumps a durable `user_version`. — NEW (near 83u)
- **Land blue-green index generations (b5l) as the headline upgrade-UX feature** — generation-suffixed `index.gN.db` + ops pointer so a schema bump serves the old generation while the new converges; kills the observed 20-40 min degraded window. This is *the* "upgrades are non-events" release story; ship it and market it in the version notes. — polylogue-b5l
- **Rebuild-throughput SLO as a versioned budget** — b5l's double-write window is only trivial if bulk replay is fast; adopt 20d.15's ≥100 rows/s / <5min rebuild target as a *release-gated* maintenance-tier SLO with a bench regression check, so no release silently regresses rebuild time. — polylogue-20d.15
- **Generalize the self-healing version-gate pattern (f2qv.5) to every materialized projection** — `provider_usage`/`session_profiles` version-gate and rebuild on mismatch; make that a *uniform derived-surface contract* (each insight declares a projection_version; a mismatch triggers targeted re-materialize, not a whole-index reset). Turns minor derived-schema bumps into invisible auto-migrations. — polylogue-f2qv.5
- **Rollback story for durable migrations, explicitly** — additive-only migrations have no down-step, so "rollback" = restore-from-manifest + downgrade binary. Document and *test* that path (old binary must reject a source.db it can't read with a clear message, not corrupt it). Guard: `user_version > max_known` → refuse-to-write with a "your archive is newer than this polylogue" error. — NEW
- **Forward-incompatibility guard on durable tiers** — a downgraded binary opening a *newer* durable DB is the real data-loss vector (writes against an unknown-shape table). Add a hard version ceiling check at bootstrap for source.db/user.db that refuses writes above the compiled max, with a restore hint. — NEW (near migration_runner.py)
- **Migration property/fuzz testing** — Hypothesis-generate old-shape durable rows, apply the numbered migration chain, assert no row loss / constraint-satisfaction / id-column regeneration. Migrations are hand-written SQL touching the irreplaceable tier; they deserve the same property harness as parsers (`test_parsers_props.py` is the model). — NEW
- **A "dry-run upgrade" preflight command** — `polylogue upgrade --check` that reports, without mutating: which tiers will migrate additively, which will rebuild (and est. time from 20d.15 telemetry), whether a backup manifest is current, and whether disk headroom fits a blue-green second generation. Upgrade UX = no surprises. — NEW
- **Release notes auto-derive the operator's "what breaks / what auto-migrates" section** — extend release-please rendering so any diff touching `migrations/{source,user}/` or a canonical derived DDL emits a mandatory "Upgrade impact" block (durable=backup+consent, derived=auto-rebuild ETA). Prevents a silent durable bump shipping without operator-facing narrative. — NEW (extends 3tl.13)
- **Backup rotation + generation retention as first-class ops policy** — blue-green leaves stale `index.gN.db` and durable-migration backups; define retention (keep last K generations, reap after grace), surface disk cost in `status`, and gate release on the reaper being exercised. An archive tool that grows unbounded on every upgrade breaks the durability promise economically. — NEW (b5l scope tail)
- **Version-stamp exports and MCP payloads** — postmortem bundles / sanitized reads / recall packs should carry the app+tier version they were produced under, so a consumer (or future-you) can tell whether a stored artifact predates a semantic reparse. Continuity surface honesty across versions. — NEW
- **A schema-change classification gate in PR CI** — the docs already enumerate five change classes (metadata-only / index-only / additive-derived / additive-durable / semantic-reparse). Make the PR author *declare* the class in a machine-checked field and have `devtools lab policy schema-versioning` verify the declared class matches the diff (e.g. a durable-tier DDL edit declared "index-only" fails). Stops misclassified durable changes from skipping the backup gate. — NEW (extends existing lab policy)
- **"Semantic reparse" as a distinct, announced release event** — the one class that *does* change stored meaning without a version-number telling the user; give it an explicit `polylogue reparse` operation with its own release-note category and a before/after coverage diff, so a content-meaning change is never conflated with a routine derived rebuild. — NEW

## GPT-pro prompt stubs

- **[A]** "You are designing the versioning contract for a *local single-writer archive tool* (not a service): five SQLite tiers keyed by durability — two durable+irreplaceable (additive numbered migrations, backup-gated), three derived (rebuilt from source, no migration chain). Given an app semver, propose the precise mapping from {MAJOR,MINOR,PATCH} to {durable-shape change, derived-schema bump, insight addition, bugfix}, the operator-facing 'upgrade impact' taxonomy, and the minimal set of runtime guards (version ceilings, refuse-to-write conditions) that make 'your durable data survives every upgrade' a *checkable* invariant rather than a slogan. Enumerate the failure modes a naive implementation misses."

- **[DR]** "Survey how mature local-first / embedded-datastore tools (SQLite-based apps, Datasette, litestream, Dolt, CRDT stores, Obsidian, git-annex, Time Machine, ZFS-snapshot workflows) handle: (a) schema migration of irreplaceable user data, (b) forward/backward version incompatibility guards, (c) zero-downtime rebuild of *derived* indexes while serving reads (blue-green / generation-pointer patterns), and (d) backup/restore as a release gate. Extract concrete, transferable patterns and anti-patterns for an additive-durable + rebuildable-derived split, and rank them by fit for a single-writer daemon architecture."

- **[DR]** "What are the empirically-known ways SQLite schema migrations corrupt or lose irreplaceable data (generated-column regeneration, CHECK-from-types drift, FK cascade-during-replace hazards, partial-migration crash recovery, WAL/checkpoint interplay, user_version desync), and what test strategies (property/fuzz on old-shape corpora, restore drills, forward-incompat ceilings) actually catch them before release? Include real post-mortems where an additive-only migration still lost data."

---

## [a46cbd9f5589a3f01] Config & runtime-preferences doctrine — ideas

Grounded in the w8db epic (y4c spine + 3xx/y8w/6kh/1jc bundles) and fnm.12. Below are additive ideas that build past the existing inventory rather than restating it — leaning on the convergent themes (settings-as-objects, recursive-safety, construct-validity).

## Config & runtime-preferences doctrine — ideas

- Unify `user_settings` into the `assertions` table as `AssertionKind.setting` (scoped key/value in the JSON), not a parallel registry — settings inherit provenance, `context_policy_json`, audit, suppression, and the judgment queue for free, and "your archive, your defaults" becomes one durable tier with one query surface instead of two — polylogue-y4c
- Make `settings` a queryable DSL unit source (`settings where scope:codex | group by key`) so preferences are first-class findings-as-objects — the operator can audit their own config with the same grammar they query sessions, and agents can read the active pref set server-side — NEW
- `config explain <key> [--scope <ctx>]` renders the full scope-chain resolution (global→repo→origin→surface, plus flag>env>toml>db>default) showing which layer/row won and, for learned defaults, the evidence assertion — generalizes 1jc's "why is this my default?" into a standing legibility surface — polylogue-y4c
- Recursive-safety gate on learned defaults: the detector must exclude invocations that were themselves shaped by a learned/proposed default (tag each such span at emit time) — otherwise accepting a proposal manufactures the very dominance signal that re-proposes it; the 30d aggregate would ratchet on its own output — polylogue-1jc
- Learned-default staleness decay: an accepted learned default whose supporting evidence no longer holds (you stopped using codex) re-enters the judgment queue for revocation as a low-priority candidate — preferences are time-bounded judged claims, not permanent state; closes the loop symmetrically (propose AND retire) — polylogue-1jc
- CSS-cascade-style pin/`!important` on a scope level: an operator can pin a global or per-repo default so no narrower scope AND no learned proposal can silently override it — protects deliberate choices from auto-adaptation drift, the recursive-safety guarantee the operator needs before trusting 1jc default-on — NEW (relates polylogue-y4c/1jc)
- Add an `actor` axis to the scope chain (operator | agent:<id> | harness) so an agent's learned/defined view prefs live in their own namespace and never clobber the operator's — generalizes fnm.12's `agent:@` macro namespace into the whole preference model; keeps "your defaults" the operator's, not the last agent's — NEW
- One `presets` namespace collapsing query macros + view presets + column sets + fold profiles + context specs behind a single `@name` reference — "my coding view" becomes a saved bundle of (query-scope + view + 1lm budget + x7d columns), composable anywhere; generalizes fnm.12 beyond predicate-groups — polylogue-fnm.12 (relates y8w)
- Config-registry validation lane (`devtools lab policy config-registry`): every key must carry doctrine metadata — ≥2 legitimate values, a stated 90%-default, an owner surface — or the render/verify gate rejects it; makes y4c's doctrine executable instead of a prose page, construct-validity for config itself — polylogue-y4c
- Dead-knob audit over the operator's own telemetry: flag keys that have only ever resolved to one value across all real usage as deletion candidates — the doctrine says "expect deletions"; a recurring audit turns "no pointless configurability" from an aspiration into evidence-driven pruning — NEW (relates polylogue-1jc telemetry)
- `config set --preview` dry-runs the change against a sample read/list/query and shows before/after diff before committing — a knob you can preview is a knob whose effect is legible; makes adding a pref cheaper to justify and reduces the doctrine's fear of pointless configurability — polylogue-3xx
- Preference-provenance marker in every non-default-shaped output (generalizing 6kh's "trailing 90d — all: to widen" footer to view/fold/columns/limit) so an agent consuming a rendered result knows it is operator-shaped, not the canonical corpus — construct honesty: implicit scope/preset must never masquerade as ground truth — polylogue-6kh (relates y8w)
- Cross-surface parity test asserting CLI, MCP, and webui resolve the same key+scope to the same effective value — fnm.12/y4c promise "resolves server-side"; a lane that seeds a pref and reads it through all three surfaces proves the single-writer/single-resolver contract instead of trusting it — polylogue-y4c
- toml/Nix-module ↔ db-pref drift detector: if a key that belongs in the runtime-pref tier is also set in polylogue.toml or the HM module (migration residue), `config effective` flags the misplacement and points to the DB path — enforces the deployment-vs-runtime split y4c's audit establishes, keeps Nix parity honest — polylogue-y4c
- Portable preference bundle (`config export`/`import`) carrying only runtime-pref rows, deployment keys excluded by construction — "your archive, your defaults" survives a machine move or seeds a fresh install; the assertions-table home (idea 1) makes this a filtered assertion export, reusing existing sanitized-export machinery — NEW (relates polylogue-y4c)
- Config-as-context toggle: settings rows with `context_policy_json.inject:true` let an agent see the operator's active reading/format prefs so it renders answers in the operator's preferred style (relative timestamps, dialogue view, terse rows) — ties config into the context-scheduler theme; gated per-key so only presentation prefs inject, never deployment — NEW (relates polylogue-37t.11)

## GPT-pro prompt stubs

- **[A]** "Design the precedence and conflict-resolution semantics for a multi-axis runtime-preference resolver where a preference key resolves through both a *scope* chain (global → per-repo → per-origin → per-surface) and an *actor* chain (operator → agent → harness), layered over the classic override order (invocation-flag > env > file > db-pref > code-default). Specify: the total order that makes resolution deterministic, how pinning/`!important` at one level interacts with a narrower level and with machine-proposed 'learned' values, and the exact data a `config explain <key>` must emit to make any resolved value auditable. Compare against Git config, CSS cascade, and Kubernetes admission/defaulting; note failure modes."

- **[DR]** "Survey how mature local-first / developer tools separate *deployment configuration* (paths, ports, secrets, declarative infra) from *user runtime preferences* (view defaults, UX toggles), and where each is stored — file vs OS keychain vs embedded DB vs sync service. Cover Git, VS Code (settings.json vs workspace vs sync), fish/atuin, Kubernetes, Nix/Home-Manager, JetBrains, and browser sync. For each: storage substrate, scope/precedence model, live-reload vs restart, provenance/'why is this set' UX, and validation. Extract a rubric for deciding which tier a given knob belongs to and what makes a knob worth having at all."

- **[DR]** "Investigate 'learned defaults' / preference-inference systems that observe usage and adapt configuration (editor adaptive UI, MRU/frecency ranking, adaptive autocomplete, recommender-driven settings, Windows adaptive menus). Focus on the feedback-loop and drift risks: how systems avoid reinforcing a preference they themselves induced, how they gate silent adaptation vs explicit confirmation, staleness/decay of learned state, and user trust/legibility. Surface concrete guardrails (support/dominance thresholds, self-exclusion of induced signal, revocation, capped open suggestions) and cite empirical findings on when adaptive defaults help vs frustrate."

---

## [aa972997b9020b43c] Grounded in `daemon/metrics.py` (Prometheus surface), `daemon/health.py` (tiered `DaemonHe

Grounded in `daemon/metrics.py` (Prometheus surface), `daemon/health.py` (tiered `DaemonHealth`), `daemon/status_snapshot.py` (fresh/stale), the ops.db tables (`ingest_attempts`, `convergence_debt`, `cursor_lag_samples`, `daemon_stage_events`, `daemon_events`, `embedding_catchup_runs`, `otlp_spans`/`otlp_telemetry`), plus the existing cursor-lag/convergence-debt alert+anomaly modules. Point-alerting is mature; steady-state throughput/SLO/self-dashboard framing is the gap.

- **Synthetic runtime meta-session ("polylogue:self")** — materialize daemon_events + stage_events + convergence outcomes into a real archive session so the daemon's own behavior is queryable/searchable/readable through the exact same `find`/`read`/timeline surfaces as any Claude session (dogfoods the reader, gives operators one narrative) — NEW
- **Ingest-latency SLO: capture→durable-row p50/p95/p99** — cursor_lag_samples measures file-tail lag but not end-to-end acquire→committed-row time; join ingest_attempts start/end per source-family and publish an SLO with an explicit target (e.g. p95 < 90s for live Codex/Claude Code) so "are we keeping up with live capture?" has a number, not a vibe — NEW
- **Convergence-lag SLO distinct from ingest-lag** — debt count is a level, not a rate; add derived-model freshness age (oldest unconverged session's index/embedding/FTS timestamp vs now) per stage, with a burn target, so deferred insights that never drain surface as SLO violation rather than a static backlog gauge — NEW
- **One-glance health verb `polylogue status --health` (or `doctor`)** — collapse DaemonHealth tiers + the three lag/debt SLOs + status_snapshot freshness into a single green/amber/red line with the top offending signal, so the operator question "is polylogue healthy?" resolves in one command instead of reading /metrics — bead: extend existing status surface
- **Backlog-growth trend detection (slope, not threshold)** — convergence_debt and embedding backlog alerts fire on absolute count; add a rolling-window slope estimator over cursor_lag_samples/debt so a monotonically *growing* backlog that hasn't yet crossed the warn line is flagged as "degrading" before it's an incident — NEW
- **Embedding backlog burn-down ETA** — from embedding_catchup_runs throughput (rows/window) and current unembedded count, compute projected drain time; alert when ETA diverges (never converges / grows), the classic silent-degradation mode for the rebuildable embeddings tier — NEW
- **FTS drift as a first-class steady-state signal** — metrics already has `_fts_trigger_presence` and `_fts_freshness_ready`; promote contentless-FTS row-count divergence (blocks vs FTS rows) into a continuously-sampled drift gauge with a debt-style alert, since FTS is trigger-maintained and can silently skew without an error — bead near fts_startup
- **Per-source-family throughput table (rows/hr, bytes/hr, backlog)** — pivot ingest_attempts + storage_route counts by Origin family into a steady-state throughput board so operators see which provider is stalled or flooding, mirroring sinex's source_health/source_throughput ergonomics — NEW
- **/metrics as a versioned product with a contract test** — 1857-line metrics.py emits many series defensively (table-existence guards); add a golden metric-name/labels snapshot + a `render metrics-reference` doc so series can't silently disappear on schema bumps and downstream dashboards don't break — bead: contract-test the metric surface
- **Self-dashboard artifact (single HTML, theme-aware)** — a `devtools` or daemon-served static dashboard rendering the SLO trio, backlog trends, throughput board, and health tiers from ops.db in one page; the "product" framing of observability for a single-writer local system where nobody runs Grafana — NEW
- **otlp/daemon_events retention + cardinality budget** — otlp_spans/telemetry and daemon_events grow unbounded in the disposable ops tier; define a retention/rollup policy (raw window + aggregated older) so the observability substrate doesn't itself become the disk-pressure incident, with a metric for ops.db self-size — NEW
- **Stage-timing profile from daemon_stage_events** — surface per-ConvergenceStage duration distribution (which stage dominates wall-time, which regresses release-over-release) so convergence slowdowns are attributable to a stage rather than "daemon feels slow"; reuses the session latency-profile pattern already in insights — NEW
- **Liveness-vs-quiet disambiguation signal** — a daemon that's idle because capture is quiet looks identical to one that's stalled; combine watcher activity + last-successful-attempt age + hot-file quiet-deferral state into an explicit "idle-healthy vs stalled" classifier so zero-throughput doesn't false-alarm or hide a real stall — NEW
- **Ingest success-rate / attempt-outcome SLO** — ingest_attempts has status; publish rolling success-rate and a repeated-failure ("stuck source") signal per family, distinct from lag: a source can be low-lag but silently failing every attempt — bead near find_stuck_sessions/live_ingest_attempt_progress
- **Convergence "never drains" watchdog on false_means_pending debt** — the `false_means_pending` trick pushes remaining work into convergence_debt as retryable; add a per-item age/retry-count ceiling so an item that keeps deferring forever (poison row) is escalated as pathological rather than perpetually "deferred until quiet" — bead near convergence_debt_status
- **Health/SLO history persisted for regression baselining** — health checks compute consecutive_failures but SLO values aren't retained as a timeseries; write periodic SLO snapshots to ops.db so "is today slower than last week's baseline?" is answerable, feeding the slow-degradation detectors above with their own reference line — NEW

GPT-pro prompt stubs:

- **[A]** "You are designing steady-state SLOs for a single-writer local ingestion daemon (no Grafana/Prometheus operator, one machine). Given these signals — file-tail cursor lag, acquire→committed-row latency, per-stage convergence debt with retry deferral, embedding backlog with a catch-up-run throughput, FTS trigger drift — propose a minimal SLO set (target, measurement window, burn-rate alert) that distinguishes 'keeping up with live capture' from 'silently falling behind', and specify which are ratios vs levels vs rates and why. Avoid alert-on-absolute-count anti-patterns."
- **[DR]** "Survey how local/single-node data pipelines and embedded databases expose self-observability without external monitoring stacks (SQLite-based tools, litestream, restic, syncthing, Prometheus textfile-collector patterns, systemd service health). Extract concrete patterns for: exposing a self-dashboard from the same store the tool manages, detecting slow backlog growth before threshold breach, and retention/cardinality budgets for self-telemetry that lives in the same disposable tier."
- **[DR]** "Research the 'application as its own observable dataset' pattern — systems that ingest their own runtime events back into their primary store as first-class queryable records (event-sourced audit logs, SQLite-as-metrics, 'dogfooding the reader'). What are the failure modes (feedback loops, unbounded growth, self-ingest storms), and what guardrails keep a synthetic self-session from polluting user-facing queries or corrupting throughput accounting?"

---

## [af119c90d6cf8eb02] ACTIVATION & ADOPTION — 16 ideas (build on wave-1: analytics-one-measure-away, citation-an

ACTIVATION & ADOPTION — 16 ideas (build on wave-1: analytics-one-measure-away, citation-anchors, construct-validity, corpus-compaction, recursive-safety gate):

- `polylogue install` one-command harness wiring — writes idempotent, backed-up settings.json hook blocks for all 16 Claude Code + 6 Codex events (not just today's recall+2 agent hooks), prints a diff, re-runs safely after a harness update; manual per-machine settings.json surgery is why only the starter subset is ever wired — polylogue-d1y
- Hook-liveness heartbeat + `polylogue doctor` — every installed hook emits a heartbeat into ops.db; doctor flags any expected event that hasn't fired in N sessions and reports the hook-covered vs post-hoc-discovered session ratio, so silent capture degradation (moved script, broken PATH) becomes a visible number instead of quiet decay — polylogue-d1y
- Adoption measured FROM the archive itself — polylogue ingests the very sessions that call its MCP tools, so count `mcp__polylogue__*` tool_use blocks per session/repo/week and define adoption-rate = (sessions that called polylogue) / (sessions in polylogue-configured repos). "Is it used" becomes a live insight, not a hope — NEW (analytics-one-measure-away)
- PreCompact hook auto-saves a recall pack — before compaction discards context, snapshot open threads/decisions/last-failure into a `recall_pack` assertion keyed to repo+session, so continuity survives the compaction boundary by default rather than being manually re-established — polylogue-d1y / polylogue-3gd
- Replace the dumb SessionStart list with a real resume brief — current hook does `read --all` on cwd path-match (pure-chat sessions miss entirely); inject `get_resume_brief`/`find_resume_candidates` structured preamble (decisions + open loops + last failure) for the cwd instead — polylogue-d1y / polylogue-pj8
- A `polylogue` harness SKILL.md — teach idioms the way the `beads` skill teaches `bd`: an intent→tool routing table for the 5-7 canonical intents ("what was I doing here", "postmortem last failure", "what did we decide about X", "what failed and was never acknowledged", "find the session that touched file Y") with worked MCP call examples — polylogue-pj8
- `what_now(intent)` intent-router MCP tool — one entrypoint that maps a natural-intent string to the right 1-3 of the 130 tools and executes them, collapsing the tool-selection search space so agents don't have to know the namespace to get value — polylogue-pj8
- Primary-tool tiering / progressive disclosure — tag ~10 tools as "primary" in their descriptions and list them in a canonical `getting_started` MCP prompt, de-emphasizing the other ~120; the model's tool-selection attention is finite and 130 flat tools induce paralysis — polylogue-pj8 / polylogue-3gd
- "Why isn't this used" diagnosis report — for repos where polylogue IS configured but tool-calls ≈ 0, emit the gap plus likely cause (hook not firing / skill not loaded / CLAUDE.md never names the tool), turning adoption failure into a diagnosable object rather than a mystery — polylogue-3gd (construct-validity-as-substrate)
- Config-as-corpus so the activation layer's own effectiveness is measurable — ingest CLAUDE.md/skills/hook configs as source family, then correlate config-version × tool-call-rate: did the CLAUDE.md revision that named polylogue actually raise usage? Closes the loop from investment to evidence — polylogue-7aw
- Assertions as the injected continuity substrate (assertions > CLAUDE.md) — SessionStart injects only repo-scoped assertions with `context_policy.inject:true`; agents write handoff/decision/blocker assertions that auto-resurface next session — targeted, durable, low-token, and default-on — polylogue-3gd / polylogue-37t
- Stop-hook blackboard auto-post — write each session's outcome/handoff to the blackboard on end, so parallel and subsequent agents see prior work without being told; coordination substrate becomes used-by-default instead of opt-in-and-forgotten — polylogue-3gd
- Adoption canary as a recursive-safety gate — a `devtools status`/CI lane that asserts the recommended hooks are wired AND firing on the operator's own machine; fail loudly when the substrate this repo builds is inert on the repo's own author (dogfood gate) — polylogue-d1y / NEW (recursive-safety)
- Citation-anchored injection — every injected preamble line carries a resolvable ref (session_id:message) an agent can `resolve_ref` to expand; injection earns trust (and thus use) because each claim is traceable, not asserted — polylogue-pj8 / polylogue-3gd (citation-anchors)
- Token-budgeted preamble compiler — `compose_context_preamble` with a hard token budget that ranks decisions/open-loops/assertions by recency×relevance and compacts to fit; 3gd authorizes 10-50K, but unranked bulk injection trains agents to ignore the preamble — budget+rank keeps signal density high — polylogue-3gd (corpus-compaction)
- A/B activation experiment rail — with config versioned (7aw) and adoption measured (archive self-instrumentation), run the activation layer AS an experiment: variant preambles → measured continuity/tool-call outcomes → keep the winner, making the large token investment empirical rather than faith-based — polylogue-3gd / polylogue-7aw

GPT-pro prompt stubs:

- [A] "Given a ~130-tool MCP server whose sessions are themselves archived by that server, design the adoption-measurement layer: what per-session/per-repo signals distinguish 'agent knew about and used the tool' from 'tool was invisible', how to compute an adoption-rate that controls for repos where the tool is irrelevant, and what dashboard/insight surface turns it into an actionable weekly number. Include the failure taxonomy for zero-usage-despite-configured repos."
- [A] "Design a progressive-disclosure / intent-routing scheme for an LLM agent facing 130 flat MCP tools so the right 3-5 land in attention for each of 7 canonical intents. Compare: primary-tool tagging in descriptions, a single what_now(intent) router tool, MCP prompts, and a harness skill. Give selection-accuracy and token-cost tradeoffs and a recommended layering."
- [DR] "Survey how coding-agent harnesses (Claude Code, Codex, Cursor, Aider, Continue, OpenHands) expose and auto-wire hooks/lifecycle events and context-injection at SessionStart/PreCompact, how each handles cross-machine/cross-harness install idempotency and liveness detection, and what patterns exist for measuring whether injected context actually changes agent behavior. Cite primary docs; extract a portable one-command-install + heartbeat design."