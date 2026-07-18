# EVIDENCE — terminal continuity gate

## Authority hierarchy

1. The attached working-tree archive is code authority.
2. Current source and typed interfaces override stale plans or old path names.
3. Complete relevant Beads records define acceptance intent and known residual scope; their tracker status is not changed by self-report.
4. Provider-native orchestration artifacts determine Workflow membership and provenance.
5. Public MCP route rows determine what the sparse operator walk can discover and cite.
6. Independent git/GitHub/Beads receipts determine observed effects.
7. Reconciliation judgments evaluate claim support without converting claims into effects.
8. Synthetic fixture constants are independent known answers only; they are not production evidence for a real incident.

## Snapshot evidence

`polylogue-manifest.json` in the attached archive reports:

```text
generated_at: 2026-07-18T013442Z
branch: master
commit: bf8191b3f56aa40da8f271df7f3385c712825497
dirty: true
```

The supplied working-tree delta was preserved as local baseline commit `3a23389823b9a78fe03f497ee719ac9af670d815`. Its changes were:

```text
browser-extension/package-lock.json     added (ignored input state)
polylogue/archive/query/unit_results.py modified
polylogue/daemon/http.py                modified
polylogue/hooks/__init__.py             modified
```

`PATCH.diff` contains none of those paths. It is generated against the exact reconstructed dirty tree, not against a guessed clean checkout.

## Repository instruction findings

`AGENTS.md` / `CLAUDE.md` establishes these load-bearing constraints:

- semantics belong in substrate/product layers; surfaces adapt;
- current source wins when plans are stale;
- MCP is the continuity surface;
- `material_origin`, not role alone, is the authoredness authority;
- actions are a derived view and exact-session work should remain physically selective;
- use focused verification rather than blanket suites;
- adding a new parallel scenario/devloop framework is an anti-pattern;
- acceptance criteria must be reported as satisfied, deferred, or still blocking rather than inferred from tests alone.

The implementation follows those constraints: it extends the existing continuity registry/replay and existing production parsers/graph/repository/query seams.

## Beads findings

### `polylogue-z9gh.7` — open, P0

The Bead calls for the sole terminal black-box gate. Its exact incident acceptance is four Workflow invocations over one resumed run, 50 call keys, 91 attempts, 65 result records over 49 completed keys, one unresolved key, final structured result, and exclusion of 38 unrelated child sessions. It also requires claim/material/call/attempt/effect scope separation, uncertainty-bearing git/PR/Beads effects, lossless paging/cancellation/SLO measurements, a cold-model run, mutations, and a per-Bead disposition.

Its latest readiness note requires two separated lanes: privacy-safe deterministic fixture/cold-model replay in CI and explicitly authorized live-scale replay with redacted receipts. The package implements the deterministic lane, the cold-receipt validator, and the authorization/redaction lane, but no external cold model or live archive was available.

### `polylogue-t8t` — in progress, P0

The Bead owns eight declarations (seven generic flows plus the incident), sparse wording, independent known answers, discovery state, evidence refs, plan equivalence, paging/cancellation/resource bounds, stop conditions, failure taxonomy, original-attempt grading, and six named route mutations. It explicitly says z9gh.7 must consume this catalog rather than duplicate the terminal walk.

The all-refs archive contained newer t8t source commit `1963ef875a20b960509460e250c0e594f8384ae2`, which was imported and then repaired so the incident no longer starts from exact fixture identities.

### `polylogue-z9gh.9.1` — in progress, P0

The Bead owns the canonical query transaction, exact-once continuation state, result/query refs, physical page construction, cancellation, bounded work, and surface parity. Its notes report that MCP already accepts/validates continuation but HTTP/API continuation input remains a residual parity gap. This package consumes the MCP transaction and does not claim to close that broader Bead.

### `polylogue-2qx.2` — open, P1

The Bead requires coordinator streams, run-state JSON, journals, 91 transcript/meta pairs, adopt manifests, exact 4/50/91/65/49/1 census, raw provenance, exclusion of 38 unrelated children, positive generated-vs-human classification, and explicit degraded links. It also requires a semantic reparse plan and real-source integration evidence.

The deterministic corpus and terminal verifier exercise the artifact family, parser, inventory, pairing, graph, and provenance contracts. They do not quantify live rows or prove the real private run.

### `polylogue-1vpm.6.2` — open, P1

The Bead requires claims, observed effects, and evaluated AC satisfaction to remain three distinct facts; direct refs and independent authorities take precedence over time/file overlap; uncertainty and snapshots must survive; and bidirectional traversal must work. Its full scope also includes real git/GitHub/complete-Beads history, branch/squash/correction cases, the private incident's 25-open-P1 baseline, and seeded production queries.

The terminal fixture proves the core graph/reconciliation/persistence contract with one supported, one partial, and one unresolved effect. It does not claim the remaining real-history cases.

## Source findings

### Existing mechanisms consumed

- `polylogue/archive/query/transaction.py` and the MCP `query_units` route already own canonical request identity, opaque continuation, page envelopes, result refs, totals, and archive-epoch validation.
- `polylogue/storage/sqlite/archive_tiers/archive.py::_action_relation_for_query` already owns exact-session action relation selection.
- `polylogue/sources/origin_specs.py` and Claude orchestration parsers already declare/admit the relevant artifact classes.
- The Claude Code parser already owns positive `MaterialOrigin` classification.
- `polylogue/insights/work_evidence.py` and `work_reconciliation.py` already own typed nodes/edges/effects/judgments.
- `SessionRepository` already owns graph persistence and traversal.

The new terminal module composes those owners; it does not duplicate their logic.

### Sparse discovery contradiction found and repaired

The imported t8t incident route's later queries contained the exact synthetic run/coordinator identifiers from the fixture. That could prove query behavior but not discovery from sparse operator clues.

Repair:

- add one public sparse discovery message and query;
- add `ContinuityBindingProjection` with `single`/`regex_single` reducers;
- bind run and coordinator from the returned message;
- materialize every later expression from those bindings;
- keep expected identities only in the independent oracle;
- assert in unit/integration tests that only the sparse route is initially authored.

### Selective-SQL contradiction found and repaired

Two existing tests assumed a global-first action-relation mutation would necessarily exceed 50,000 SQLite VM steps. On this environment, current SQLite pushes the exact session predicate through the global view, so the mutation returned the same semantic answer in only zero to a few hundred measured steps. The old performance threshold no longer distinguished the mutation.

Repair:

- validate the production relation-selection contract directly (`bounded_actions`, bounded CTE, three target-session parameters);
- retain real query execution, exact semantic census, cleanup, and VM receipt as secondary evidence;
- update both stale canaries without deleting them;
- add a terminal mutation that monkeypatches `_action_relation_for_query` to the global relation and requires `selective_sql_plan_amplification` / `plan`.

### SLO observation

The standalone incident stayed within call, page-byte, total-byte, elapsed-time, and process-RSS guidance. MCP process restart/resume took 4.515 seconds against a 1-second guidance target. The implementation records the advisory and preserves the valid exact result, matching the mandate constraint that SLOs guide paging/resume and do not arbitrarily reject valid work.

## History findings

| Commit | Relevance |
|---|---|
| `fd7b3549292927fbd69e0cb07dff9a1205d8e6c8` | Interruptible/admission-controlled query execution; consumed rather than recreated. |
| `ed44be18f448c31f9fa5b9289c75da7eee99b131` | Current MCP tool/query declarations used for discovery. |
| `9163d0134f3d334960e4c249c96c5671919a9a06` | Shared bounded query transaction and continuation behavior. |
| `4053787ab547299a4402e33e05f73d04840c74c3` | Prior continuity transaction replay and mutation seam. |
| `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` | Claude artifact/work-evidence/effect continuity precursor. |
| `1963ef875a20b960509460e250c0e594f8384ae2` | Current t8t all-scenario real stdio replay imported into this draft. |

## Standalone receipt evidence

Official stdio replay over the deterministic corpus produced:

```text
protocol_version=2025-11-25
server_name=polylogue
server_version=1.28.1
tool_count=64
scenario_count=8
passed=8
failed=0
incident pages=6
incident restart_count=1
incident exact_enumeration_verified=true
terminal calls=50
terminal attempts=91
terminal results=65
terminal completed_call_keys=49
terminal unresolved_call_keys=1
terminal final_results=1
terminal excluded_unrelated_children=38
terminal unrelated_children_admitted_to_graph=0
selective_relation=bounded_actions
cold_model.status=unavailable
```

The authorized sanitized redaction smoke found zero raw occurrences of the archive path, fixture ID, run ID, coordinator ID, or sparse query marker in the serialized live-lane receipt.

## Unavailable or deliberately unclaimed evidence

- External cold-model behavior and competence.
- Operator live-scale archive receipt.
- Actual private incident identifiers and corpus volume.
- Live daemon/browser/NixOS behavior and secrets.
- Real GitHub checks for the cited PR.
- Beads closure evidence or current tracker mutation.
- Full 1vpm.6.2 real-history matrix and 2qx.2 live semantic-reparse census.
- Repository-wide Mypy completion and full test suite.

These remain explicit limitations. No synthetic receipt or structured agent result is promoted into tracker truth or real repository effect evidence.
