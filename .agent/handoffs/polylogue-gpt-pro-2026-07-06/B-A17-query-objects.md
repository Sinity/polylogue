## [assistant 326]
bash -lc cd /mnt/data/poly_tree/polylogue && 
for range in '1,125' '126,225'; do echo '--- refs.py '$range; nl -ba polylogue/core/refs.py | sed -n $range'p'; done
printf '\n--- enums assertion kinds/status ---\n'; nl -ba polylogue/core/enums.py | sed -n '390,450p'
printf '\n--- user DDL ---\n'; nl -ba polylogue/storage/sqlite/archive_tiers/user.py | sed -n '1,52p'
printf '\n--- user_write upsert assertion ---\n'; nl -ba polylogue/storage/sqlite/archive_tiers/user_write.py | sed -n '900,1030p'
printf '\n--- user_write candidate judgement funcs ---\n'; nl -ba polylogue/storage/sqlite/archive_tiers/user_write.py | sed -n '1150,1385p'
printf '\n--- api candidate judge/list ---\n'; nl -ba polylogue/api/archive.py | sed -n '2070,2215p'
printf '\n--- api saved query/view/workspace? ---\n'; nl -ba polylogue/api/archive.py | sed -n '5030,5105p'

## [assistant 374]
bash -lc cd /mnt/data/poly_pkg/polylogue 2>/dev/null || cd /mnt/data/polylogue_pkg/polylogue 2>/dev/null || exit 0; python3 - <<'PY'
import json, textwrap, pathlib
p=pathlib.Path('polylogue-beads-export.jsonl')
if not p.exists(): p=pathlib.Path('/mnt/data/poly_pkg/polylogue/polylogue-beads-export.jsonl')
issues=[]
for line in p.read_text().splitlines():
    d=json.loads(line)
    if d.get('_type')=='issue': issues.append(d)
for ident in ['polylogue-fnm.13','polylogue-4p1','polylogue-4p1.1','polylogue-t46','polylogue-37t.12','polylogue-fnm.12','polylogue-fnm.9']:
    b=next((x for x in issues if x.get('id')==ident), None)
    if b:
        print('\n###', ident, 'status', b.get('status'),'priority',b.get('priority'),'type',b.get('issue_type'), b.get('title'))
        for fld in ['description','design','acceptance_criteria']:
            v=b.get(fld) or ''
            print(fld+':', textwrap.shorten(v.replace('\n',' | '), width=1100, placeholder=' ...'))
PY

## [assistant 390]
According to a document from 2026-07-05, the A17 task is the right next convergence point: make `query`, `result_set`, and `finding` durable public objects, not just transient execution artifacts or opaque assertion payloads. The prompt explicitly asks for `query:<hash>`, `AssertionKind.FINDING`, result-set snapshots, a query-edge DAG, a `StandingQueryStage`, findings-as-tests, DDL, tier placement, `resolve_ref`, CLI surfaces, and recursive-loop failure modes. fileciteturn2file0

My defended recommendation: **add first-class query/result-set tables plus a `FINDING` assertion kind, while keeping findings inside the existing assertion lifecycle.** Queries and result-set manifests belong in durable `user.db`; hot/cache membership can live in derived `index.db`; pinned/watch/finding membership must be durable. The runner-up I reject is “just store everything as assertions.” Assertion-only is tempting because `user.db.assertions` is already flexible, but it gives you opaque JSON blobs, weak foreign-key/index behavior, no clean `query:<hash>`/`result_set:<hash>` resolver, and no good substrate for set-algebra operands.

Assumption to verify locally: I am reading the uploaded snapshot, not live master. The prompt itself warns the snapshot is dated. fileciteturn2file0

## Layer 1 — today’s substrate

[evidence] Today, `ObjectRefKind` does not include `query`, `result_set`, or `finding`. The accepted kinds include `session`, `message`, `block`, `file`, `commit`, `insight`, `run`, `context-snapshot`, `observed-event`, `assertion`, `saved_view`, `recall_pack`, `tool-call`, `subagent-report`, and GitHub refs, but no query/result/finding object kinds. `ObjectRef.parse` supports `kind:id` with opaque ids and only special block qualifiers. `polylogue/core/refs.py:8-38`, `polylogue/core/refs.py:76-113`.

[evidence] `user.db` already has the right durable assertion substrate. The unified `assertions` table has `scope_ref`, `target_ref`, `key`, `kind`, `value_json`, `body_text`, `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, `context_policy_json`, `supersedes_json`, and timestamps. `user.py` explicitly treats this tier as the place for user overlays including annotations, corrections, tags, saved views, recall packs, workspaces, and notes. `polylogue/storage/sqlite/archive_tiers/user.py:7-31`.

[evidence] `AssertionKind` already includes `SAVED_QUERY`, `JUDGMENT`, `PROMPT_EVAL`, `TRANSFORM_CANDIDATE`, and `PATHOLOGY`, but not `FINDING`. It already has candidate lifecycle states: `candidate`, `accepted`, `rejected`, `deferred`, `superseded`, `deleted`, etc. `polylogue/core/enums.py:399-428`, `polylogue/core/enums.py:437-447`.

[evidence] Current saved queries are assertion-backed saved views: `upsert_saved_view()` writes `kind=AssertionKind.SAVED_QUERY`, `target_ref=saved_view:<id>`, `key=name`, and `value=query`. That matches A17’s diagnosis: the saved query is effectively a mutable-name assertion value, not a first-class query object. `polylogue/storage/sqlite/archive_tiers/user_write.py:567-588`.

[evidence] The candidate→judge→promote lifecycle already exists and should be reused. Deterministic pathology findings are mirrored as private, non-injected `PATHOLOGY` candidate assertions with `context_policy={"inject": False, "promotion_required": True}`; promoted candidates are never silently downgraded by rebuilds. `judge_assertion_candidate()` accepts/rejects/defers/supersedes a candidate, promotes on accept/supersede, writes a `JUDGMENT` assertion, and records `supersedes`. `polylogue/storage/sqlite/archive_tiers/user_write.py:1060-1119`, `polylogue/storage/sqlite/archive_tiers/user_write.py:1245-1321`, `polylogue/storage/sqlite/archive_tiers/user_write.py:1338-1370`.

[evidence] Assertions are already queryable as a unit. Assertion structural fields include author, author_kind, author_ref, body, context, evidence, key, kind, scope, target, status, value, visibility, etc.; the `assertion` query unit is registered for both `exists assertion(...)` and terminal `assertions where ...`. `polylogue/archive/query/metadata.py:507-525`, `polylogue/archive/query/metadata.py:713-717`, `polylogue/archive/query/metadata.py:793-805`.

[evidence] The read algebra baseline is already `Query × Projection × Render`: `QueryProjectionSpec` is `SelectionSpec × ProjectionSpec × RenderSpec`, with evidence families, body policies, render formats, and render destinations in a storage-free surface contract. The swarm brief frames this as `4p1`, with query grammar/pipeline, projection, render, retrieval lanes, and the `fnm` roadmap. `polylogue/surfaces/projection_spec.py:1-7`, `polylogue/surfaces/projection_spec.py:82-146`, `polylogue/surfaces/projection_spec.py:149-183`; fileciteturn2file8

[evidence] Set-algebra is designed but not implemented. `docs/design/query-set-algebra.md` is explicitly “design — hold,” owned by `polylogue-fnm.13`, anchored to `polylogue-4p1`. It defines set operations over query result sets and recommends Design A: `| intersect (SUBQUERY)`, `| union (SUBQUERY)`, `| except (SUBQUERY)` as binary pipeline stages. It also requires same-grain identity, rank-honest union via RRF, left-rank intersect/except, and EXPLAIN sub-plan nodes. `docs/design/query-set-algebra.md:1-10`, `docs/design/query-set-algebra.md:47-59`, `docs/design/query-set-algebra.md:61-73`, `docs/design/query-set-algebra.md:95-113`, `docs/design/query-set-algebra.md:148-194`, `docs/design/query-set-algebra.md:198-211`.

[evidence] The composer design already wants committed queries to become recall entries with query text, resolved spec, result fingerprint, timestamp, and optional macro name; it also wants set-op operand previews and live cardinalities. That is almost the UI-level version of A17, but not yet the storage/object migration. fileciteturn2file2

[evidence] The daemon convergence substrate already supports session-scoped stage repair. `ConvergenceStage` has optional `check_sessions` and `execute_sessions`, and `DaemonConverger.converge_sessions()` uses them stage-by-stage. Default stages are FTS, embed, and insights; FTS/insights already wire session-scoped functions and `false_means_pending=True`. `polylogue/daemon/convergence.py:61-84`, `polylogue/daemon/convergence.py:414-498`, `polylogue/daemon/convergence_stages.py:245-256`, `polylogue/daemon/convergence_stages.py:542-562`.

[evidence] Tier semantics strongly constrain the design. Reset treats `index.db`, `embeddings.db`, and `ops.db` as rebuildable; `source.db` is durable evidence; `user.db` is irreplaceable and preserved unless explicitly destroyed with `--include-user-db`. `ops reset --index` deletes only the rebuildable index tier, and the follow-up is `ops maintenance rebuild-index`. `polylogue/cli/commands/reset.py:31-45`, `polylogue/cli/commands/reset.py:249-259`, `polylogue/cli/commands/reset.py:347-371`, `polylogue/cli/commands/reset.py:429-431`, `polylogue/cli/commands/maintenance.py:1463-1468`.

## Layer 2 — near-term migration

[proposal] The near-term migration should be additive, user-tier first, and resolver-driven. Do not start by changing set-algebra execution. First make queries/findings/result sets addressable objects, then let set-algebra and the composer use them.

### 2.1 ObjectRef changes

[proposal] Add these public ref kinds:

```python
ObjectRefKind += Literal[
    "query",
    "query_run",
    "result_set",
    "finding",
    "analysis",
]
```

[proposal] `query:<hash>` and `result_set:<hash>` should use full ObjectRef parsing. `finding:<hash>` should resolve to an assertion row whose `kind == FINDING`. `assertion:<id>` should continue to work, but `finding:<hash>` is the public human-facing alias.

[proposal] Extend refs to support content-anchored evidence suffixes for session/message/block refs:

```text
session:<session_id>@<content_hash>
message:<message_id>@<content_hash>
block:<block_id>@<content_hash>
```

The parser should only treat `@...` as an anchor if the suffix is a valid hash shape, e.g. 64 hex chars. The resolver must verify the anchor against the current row content hash and return `resolved=true, stale_anchor=true` or `resolved=false, reason=content-hash-mismatch` rather than silently resolving to changed evidence.

[evidence] This is directly aligned with the existing NFC content-hash doctrine: `hash_text()` NFC-normalizes and SHA-256 hashes text; docs state that NFC/NFD inputs should produce identical `content_hash`. `polylogue/core/hashing.py:14-23`, `docs/internals.md:408-419`.

### 2.2 Query object identity

[proposal] `query:<hash>` is keyed by the canonical **planned AST after macro expansion**, not by the raw query text.

Canonicalization rules:

```text
1. Expand macros before hashing.
2. Parse into the typed query/pipeline AST.
3. Normalize all strings to NFC before hashing.
4. Canonicalize commutative Boolean nodes:
   AND(a,b) == AND(b,a)
   OR(a,b)  == OR(b,a)
5. Do not reorder non-commutative constructs:
   pipeline stages, seq(...), except, sort, limit, offsets, ranked left operand.
6. Preserve exact quoted phrases.
7. Normalize equivalent field aliases into canonical field names.
8. Include retrieval lane, grain, projection-affecting source terms, and rank policy.
9. For relative-time queries, hash the dynamic query definition, while query_run/result_set stores the resolved absolute execution bounds.
10. Hash deterministic JSON with sorted keys and compact separators.
```

[proposal] The query hash input should be something like:

```json
{
  "schema": "polylogue.query-plan.v1",
  "grain": "session",
  "pipeline": [...],
  "predicate": {...},
  "macro_expansions": [
    {"name": "mine", "query_ref": "query:abc...", "expanded_hash": "abc..."}
  ],
  "rank_policy": "left-rank|rrf-k60|structural",
  "relative_time_policy": "dynamic"
}
```

Then:

```text
query_ref = "query:" + sha256(nfc(canonical_json))
```

[evidence] The current parser already has serializable AST/payload pieces: pipeline stages have `to_payload()` methods; explanations include `predicate`, `ast`, `lowering_plan`, selected units, execution legs, and plan descriptions. `polylogue/archive/query/expression.py:274-286`, `polylogue/archive/query/expression.py:430-475`, `polylogue/archive/query/expression.py:588-616`.

[evidence] The set-algebra design already requires exact identity keys per grain and says set-ops combine by identity key, not by fuzzy row equivalence. `docs/design/query-set-algebra.md:61-73`.

### 2.3 User-tier DDL

[proposal] Add a user migration, likely `migrations/user/005_queries_findings.sql`, and bump `USER_SCHEMA_VERSION` from 4 to 5.

I would use the full ref as the primary key, because `target_ref` and `scope_ref` already store full public refs.

```sql
-- Durable query definitions. Content-addressed by planned AST after macro expansion.
CREATE TABLE IF NOT EXISTS queries (
    query_ref              TEXT PRIMARY KEY CHECK(query_ref LIKE 'query:%'),
    query_hash             TEXT NOT NULL UNIQUE,
    hash_algorithm         TEXT NOT NULL DEFAULT 'sha256:nfc-json:query-plan-v1',
    plan_schema            TEXT NOT NULL DEFAULT 'polylogue.query-plan.v1',

    original_text          TEXT,
    display_text           TEXT,
    planned_ast_json       TEXT NOT NULL,
    normalized_spec_json   TEXT NOT NULL DEFAULT '{}',
    macro_expansions_json  TEXT NOT NULL DEFAULT '[]',

    grain                  TEXT NOT NULL DEFAULT 'session',
    retrieval_lane         TEXT,
    rank_policy            TEXT,
    dynamic_params_json    TEXT NOT NULL DEFAULT '{}',

    created_at_ms          INTEGER NOT NULL,
    updated_at_ms          INTEGER NOT NULL,
    author_ref             TEXT NOT NULL DEFAULT 'user:local',
    author_kind            TEXT NOT NULL DEFAULT 'user'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_queries_grain_updated
ON queries(grain, updated_at_ms);

-- Mutable human names / saved-query pointers.
-- This is the git-branch model: name is mutable; query_ref is immutable.
CREATE TABLE IF NOT EXISTS query_names (
    name                   TEXT PRIMARY KEY,
    query_ref              TEXT NOT NULL REFERENCES queries(query_ref) ON DELETE RESTRICT,
    scope_ref              TEXT,
    watch                  INTEGER NOT NULL DEFAULT 0 CHECK(watch IN (0,1)),
    watch_policy_json      TEXT NOT NULL DEFAULT '{}',
    last_result_set_ref    TEXT,
    supersedes_json        TEXT NOT NULL DEFAULT '[]',
    created_at_ms          INTEGER NOT NULL,
    updated_at_ms          INTEGER NOT NULL,
    author_ref             TEXT NOT NULL DEFAULT 'user:local'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_query_names_query_ref
ON query_names(query_ref);

CREATE INDEX IF NOT EXISTS idx_query_names_watch
ON query_names(watch, updated_at_ms);

-- Durable result-set manifest. Membership may be in index cache or durable user rows.
CREATE TABLE IF NOT EXISTS result_sets (
    result_set_ref         TEXT PRIMARY KEY CHECK(result_set_ref LIKE 'result_set:%'),
    query_ref              TEXT NOT NULL REFERENCES queries(query_ref) ON DELETE RESTRICT,

    grain                  TEXT NOT NULL,
    corpus_epoch           TEXT NOT NULL,
    corpus_fingerprint     TEXT NOT NULL,
    result_hash_algorithm  TEXT NOT NULL DEFAULT 'sha256:query-corpus-members-v1',

    member_count           INTEGER NOT NULL CHECK(member_count >= 0),
    membership_merkle_root TEXT NOT NULL,
    ordered_rank_hash      TEXT,
    exactness              TEXT NOT NULL CHECK(exactness IN ('exact','sampled','capped','estimate')),
    persistence            TEXT NOT NULL CHECK(persistence IN ('cache','watch','pinned','finding','cohort')),
    member_storage         TEXT NOT NULL CHECK(member_storage IN ('index','user','blob','none')),

    source_query_run_ref   TEXT,
    created_at_ms          INTEGER NOT NULL,
    updated_at_ms          INTEGER NOT NULL,
    created_by_ref         TEXT NOT NULL DEFAULT 'user:local',
    payload_json           TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE INDEX IF NOT EXISTS idx_result_sets_query_epoch
ON result_sets(query_ref, corpus_epoch, created_at_ms);

CREATE INDEX IF NOT EXISTS idx_result_sets_persistence
ON result_sets(persistence, created_at_ms);

-- Durable members only for frozen/watch/finding/cohort snapshots.
-- Cache-only snapshots keep members in index.db.
CREATE TABLE IF NOT EXISTS result_set_members (
    result_set_ref         TEXT NOT NULL REFERENCES result_sets(result_set_ref) ON DELETE CASCADE,
    member_key             TEXT NOT NULL,
    member_ref             TEXT NOT NULL,
    rank                   INTEGER,
    score                  REAL,
    member_content_hash    TEXT,
    sort_key_json          TEXT,
    payload_json           TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(result_set_ref, member_key)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_result_set_members_ref
ON result_set_members(member_ref);

-- Query graph. Edges are durable because they explain derivation.
CREATE TABLE IF NOT EXISTS query_edges (
    edge_id                TEXT PRIMARY KEY,
    src_query_ref          TEXT NOT NULL REFERENCES queries(query_ref) ON DELETE CASCADE,
    dst_query_ref          TEXT NOT NULL REFERENCES queries(query_ref) ON DELETE CASCADE,
    edge_kind              TEXT NOT NULL CHECK(edge_kind IN (
                              'operand-of',
                              'refines',
                              'supersedes',
                              'derived-from',
                              'same-as'
                            )),
    edge_payload_json      TEXT NOT NULL DEFAULT '{}',
    source_ref             TEXT,
    created_at_ms          INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_query_edges_src
ON query_edges(src_query_ref, edge_kind);

CREATE INDEX IF NOT EXISTS idx_query_edges_dst
ON query_edges(dst_query_ref, edge_kind);
```

[proposal] Add `FINDING` to `AssertionKind` and to `ASSERTION_CLAIM_KINDS`.

```python
class AssertionKind(...):
    ...
    FINDING = "finding"
```

```python
ASSERTION_CLAIM_KINDS = (
    ...
    AssertionKind.PATHOLOGY,
    AssertionKind.FINDING,
)
```

[evidence] This is the smallest extension to the existing judgment queue because claim kinds already drive candidate review reads. `polylogue/storage/sqlite/archive_tiers/user_write.py:1520-1545`.

### 2.4 Index-tier cache DDL

[proposal] Result-set cache membership belongs in `index.db`, but only as a rebuildable acceleration layer. This is not the source of truth for published findings.

```sql
CREATE TABLE IF NOT EXISTS result_set_cache_members (
    result_set_ref         TEXT NOT NULL,
    member_key             TEXT NOT NULL,
    member_ref             TEXT NOT NULL,
    rank                   INTEGER,
    score                  REAL,
    member_content_hash    TEXT,
    sort_key_json          TEXT,
    cached_at_ms           INTEGER NOT NULL,
    PRIMARY KEY(result_set_ref, member_key)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_result_set_cache_member_ref
ON result_set_cache_members(member_ref);

CREATE TABLE IF NOT EXISTS result_set_cache_state (
    result_set_ref         TEXT PRIMARY KEY,
    query_ref              TEXT NOT NULL,
    corpus_epoch           TEXT NOT NULL,
    member_count           INTEGER NOT NULL,
    membership_merkle_root TEXT NOT NULL,
    cached_at_ms           INTEGER NOT NULL
) STRICT;
```

[proposal] The rule is:

```text
cache/preview/run result set:
  manifest optional in ops.db or user.db, members in index cache.

watch result set:
  manifest in user.db, members durable in user.db or blob, cache copy in index.

finding/cohort/pinned result set:
  manifest in user.db, exact member keys durable in user.db or content-addressed blob.
```

This resolves `ops reset --index`: a reset drops derived cache rows, but durable query definitions, finding assertions, result-set manifests, and pinned/finding/watch members survive in `user.db`. If a result set was cache-only, `resolve_ref(result_set:...)` says “cache missing; rerun query to rematerialize” rather than pretending the snapshot still exists.

[evidence] This directly follows the reset contract: index/embeddings/ops are rebuildable and deleted by `reset --database`; user.db is irreplaceable and preserved unless explicitly opted into deletion. `polylogue/cli/commands/reset.py:35-45`, `polylogue/cli/commands/reset.py:347-371`.

### 2.5 Result-set id and Merkle root

[proposal] Use the id the prompt suggests, but make grain and hash algorithm explicit so member-key collisions cannot cross grains:

```text
result_set_id_input =
  canonical_json({
    "schema": "polylogue.result-set.v1",
    "query_ref": query_ref,
    "grain": grain,
    "corpus_epoch": corpus_epoch,
    "members": sorted(member_keys)
  })

result_set_ref = "result_set:" + sha256(result_set_id_input)
```

[proposal] Compute two roots:

```text
membership_merkle_root:
  Merkle(sorted(sha256(grain || NUL || member_key || NUL || content_hash?)))

ordered_rank_hash:
  sha256(canonical_json([
    {"rank": 1, "member_key": "...", "score": ...},
    ...
  ]))
```

Why two? Membership equality is not rank equality. Set-algebra mostly cares about membership; reports and UX may also care that the top-N/ranking shifted.

### 2.6 Finding lifecycle

[proposal] Add:

```python
def assertion_id_for_finding(
    *,
    target_ref: str,
    key: str,
    value: object,
    evidence_refs: Sequence[str],
    detector_ref: str,
) -> str:
    ...
```

[proposal] Add:

```python
def upsert_findings_as_assertions(
    conn,
    findings: Sequence[FindingCandidate],
    *,
    now_ms: int | None = None,
) -> list[ArchiveAssertionEnvelope]:
    ...
```

Use the pathology pattern exactly:

```text
kind              = FINDING
status            = CANDIDATE
visibility        = PRIVATE
context_policy    = {"inject": false, "promotion_required": true}
author_kind       = "detector" | "agent" | "analysis"
target_ref        = query:<hash> | session:<id> | insight:<id> | finding:<id>
scope_ref         = analysis:<id> | query:<hash> | insight:<detector>@vN
evidence_refs     = content-anchored refs + result_set refs + query refs
```

[proposal] `finding:<hash>` should resolve to the underlying `FINDING` assertion. The underlying `assertion:<id>` remains valid, but CLI/docs should prefer `finding:<hash>`.

Suggested `value_json` shape:

```json
{
  "schema": "polylogue.finding.v1",
  "finding_kind": "query-delta|finding-drift|measure|pathology|claim-vs-evidence",
  "measure": "count",
  "n": 12,
  "statistic": {
    "op": "eq",
    "value": 12,
    "unit": "sessions"
  },
  "query_ref": "query:...",
  "result_set_ref": "result_set:...",
  "baseline_result_set_ref": "result_set:...",
  "delta": {
    "added": 3,
    "removed": 1,
    "unchanged": 8
  },
  "expected": {
    "measure": "count",
    "op": "==",
    "value": 12
  }
}
```

[proposal] Machine-generated findings default to candidate. Only operator or an explicitly trusted promotion policy can make them active. That preserves the existing “machine claim → judgment → active assertion” epistemic boundary.

### 2.7 Query edges

[proposal] `query_edges` should be emitted from the query planner/explain machinery, not retrofitted by string parsing.

Set-op example:

```text
query:Q_parent = query for: find auth | intersect (test)
query:Q_left   = query for: find auth
query:Q_right  = query for: test

edge: Q_left  --operand-of{op:"intersect", side:"left"}-->  Q_parent
edge: Q_right --operand-of{op:"intersect", side:"right"}--> Q_parent
```

[proposal] Direction conventions:

```text
operand-of:   operand/src -> combined/dst
derived-from: child/new -> parent/source
refines:      refined/new -> broader/old
supersedes:   new -> old
same-as:      query refs that canonicalized after migration/import
```

[evidence] Set-algebra EXPLAIN was already designed to show two sub-plans joined by a set-op node, which is exactly where `operand-of` edges should be emitted. `docs/design/query-set-algebra.md:198-211`.

### 2.8 StandingQueryStage

[proposal] Add a fourth default convergence stage after insights:

```python
make_standing_query_stage(archive_root, index_db_path, user_db_path)
```

and extend:

```python
make_default_convergence_stages(db_path):
    return (
        make_fts_stage(db_path),
        make_embed_stage(db_path),
        make_insights_stage(db_path),
        make_standing_query_stage(...),
    )
```

[proposal] It should use the existing session-scoped stage pattern:

```python
def check_sessions(session_ids: Sequence[str]) -> set[str]:
    # For each watched query, determine whether the scoped predicate fingerprint
    # could be affected by these sessions. Return session ids whose membership
    # needs re-evaluation.

def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
    # Re-materialize affected watch:true queries.
    # Compare old vs new result_set merkle/member keys.
    # Emit query-delta candidate findings.
    # Store fresh snapshot and last_result_set_ref.
```

[evidence] This fits the current convergence API: `ConvergenceStage` already has `check_sessions`/`execute_sessions`, and `converge_sessions()` retries stages with session ids without resolving back to source files. `polylogue/daemon/convergence.py:61-84`, `polylogue/daemon/convergence.py:414-498`.

[proposal] `query-delta` should be a `FINDING` candidate, not an automatic active assertion:

```text
target_ref      = query:<hash>
kind            = finding
key             = query-delta/<query_hash>/<old_epoch>/<new_epoch>
status          = candidate
context_policy  = {"inject": false, "promotion_required": true}
evidence_refs   = [query:<hash>, old_result_set_ref, new_result_set_ref, added/removed session anchors]
```

### 2.9 Findings-as-tests

[proposal] A promoted `FINDING` with `value.expected` becomes an invariant. Do not mutate it on drift. Re-run it and emit a new candidate finding:

```text
original finding:
  finding:F1
  target_ref=query:Q
  value.expected={"measure":"count","op":"==","value":12}

stage re-runs query:Q
actual count = 14

new candidate:
  kind=FINDING
  key=finding-drift/F1/<epoch>
  target_ref=finding:F1
  value={
    "finding_kind":"finding-drift",
    "expected":{"measure":"count","op":"==","value":12},
    "actual":{"measure":"count","value":14},
    "baseline_result_set_ref":"result_set:old",
    "current_result_set_ref":"result_set:new"
  }
```

[proposal] This makes findings testable without making findings self-modifying. Drift is a new claim awaiting judgment.

### 2.10 Read/write/API/CLI surface

[proposal] `PolylogueService` should get these through the existing `query/read/act/status` shape, not as a dozen one-off methods. B8 already recommends a single `PolylogueService` with `query`, `read`, `preview`, `complete`, `act`, `status`, plus `facets`. fileciteturn2file10

Near-term facade/API additions:

```python
async def record_query(req: RecordQueryRequest) -> QueryPayload
async def materialize_result_set(req: MaterializeResultSetRequest) -> ResultSetPayload
async def save_query_name(req: SaveQueryNameRequest) -> QueryNamePayload
async def record_finding(req: RecordFindingRequest) -> AssertionClaimPayload
async def rerun_finding(req: RerunFindingRequest) -> FindingTestResultPayload
```

But on the thin service:

```text
act(kind="record-query", ...)
act(kind="materialize-result-set", ...)
act(kind="save-query-name", ...)
act(kind="record-finding", ...)
act(kind="judge-assertion", ...)   # existing lifecycle
```

[proposal] CLI:

```bash
polylogue query record 'repo:polylogue AND tool:bash'
# => query:abc...

polylogue query save mine 'repo:polylogue origin:claude-code-session' --watch
# => name @mine -> query:abc...

polylogue find 'repo:polylogue' --json
# response contains query_ref, query_run_ref, result_set_ref

polylogue read query:abc...
# show canonical query, names, latest snapshots, edges, findings, actions

polylogue read result_set:def...
# show member count, grain, epoch, merkle, sample, query_ref, projection options

polylogue read finding:ghi...
# show finding assertion, target, expected/actual, evidence, judgment state

polylogue findings queue
polylogue judge accept finding:ghi... --reason 'validated against evidence pack'
polylogue standing-query list
polylogue standing-query run @mine
```

[evidence] Direct ref reads already exist in the CLI, but they currently go through `resolve_ref` and support JSON-only direct refs. That is enough to extend rather than invent a new command family. `polylogue/cli/query_verbs.py:1122-1128`.

[proposal] `resolve_ref()` extensions:

```python
if object_ref.kind == "query":
    return _resolve_query_object_ref(...)
if object_ref.kind == "result_set":
    return _resolve_result_set_object_ref(...)
if object_ref.kind == "finding":
    return _resolve_finding_object_ref(...)
```

[evidence] `resolve_ref()` already dispatches on `session`, `message`, `block`, `assertion`, and runtime object refs, and returns `PublicRefResolutionPayload` with payload, refs, caveats, and actions. `polylogue/api/archive.py:2713-2750`, `polylogue/surfaces/payloads.py:1741-1772`.

## Layer 3 — full direction

[proposal] The full direction is not merely “save query history.” It is a typed analysis graph:

```text
query_definition
  -> query_run
  -> result_set
  -> projection/render artifact
  -> finding assertion
  -> judgment
  -> standing query / finding-as-test / report
```

This is the substrate that lets Polylogue analyze itself, let external agents annotate results, and let findings become durable objects that are rerunnable, refutable, and citable.

[proposal] Full object model:

```text
query:<hash>          immutable content-addressed query plan
query_run:<id>        execution event against archive epoch
result_set:<hash>     materialized relation snapshot
finding:<hash>        assertion-backed claim over target/evidence
analysis:<id>         DAG tying query runs, result sets, findings, artifacts
report:<id>           rendered artifact derived from analysis/finding set
cohort:<id>           named dynamic or frozen result-set family
```

[proposal] `query_run` should probably live in `ops.db` by default, with promotion into `user.db` only when cited, named, watched, or used by an analysis. The full system should not permanently store every preview keystroke. The composer can record committed runs; previews remain ephemeral unless promoted.

[proposal] The full direction should make `result_set:<hash>` a legal set-algebra operand:

```text
result_set:abc... | except (query:def...) | group by model | count
query:abc... | intersect (@strict_delegate_prompts)
```

[proposal] This also clarifies “cohort.” A cohort is a named result-set family. A dynamic cohort points at `query:<hash>` or `@name`; a snapshot cohort points at `result_set:<hash>`. The earlier saved-cohort idea is really a named layer over this result-set machinery. The previous swarm’s `E22 saved-cohort primitive` was stopped, but the A17 migration supplies its missing substrate. fileciteturn2file15

[proposal] In the full design, `FINDING` is one member of the assertion family, not a special report table. A finding can target a query, result set, session, insight, analysis, or another finding. That gives you self-auditing without needing a separate “findings DB.”

[proposal] The important extension beyond A17 is an `analysis_run` layer. A complex external agent analysis should not be forced into YAML alone. YAML/prompt is the recipe; database objects record what actually happened:

```text
analysis:a1
  step 1: query:q1 -> result_set:r1
  step 2: evidence_pack artifact:e1
  step 3: imported annotation batch -> findings f1..fN
  step 4: aggregate query:q2 -> result_set:r2
  step 5: report artifact:rep1
```

This is how the “Fable subagent rhetoric” demo, claim-vs-evidence demos, context compaction audits, and methodology evals become repeatable rather than one-off prompt runs.

## Recursive-loop failure modes and guards

[proposal] These loops are real enough that they should be designed against in v1.

1. **Query→finding→query self-justification.** A watched query over `assertions where kind:finding` emits a finding, which changes the query’s own result set, which emits another finding. Guard: generated findings default `candidate` and `inject:false`; standing queries exclude their own `scope_ref` by default unless `allow_self_scope:true`.

2. **Finding-as-test self-target loop.** A drift finding targets original finding F1, then the invariant stage sees the drift finding as evidence for F1 and re-emits drift. Guard: finding-tests only rerun active findings with explicit `value.expected`; drift findings do not themselves become tests unless separately promoted with `expected`.

3. **Macro identity instability.** `@mine` expands differently after the macro pointer moves, causing the same text to hash to a different query. Guard: `query:<hash>` always hashes the expanded planned AST; `query_names` carries mutable human name history and `supersedes`, like a git branch.

4. **Relative-time ambiguity.** `since:7d` has stable query definition but changing execution bounds. Guard: `query:<hash>` stores dynamic relative AST; `query_run`/`result_set` stores resolved absolute bounds.

5. **Set-op self-edge cycles.** A query can reference a result set produced by itself or a macro that expands to itself. Guard: query planner detects cycles across `query_edges` and macro expansion depth before materialization.

6. **Supersedes cycles.** Mutable names or findings can supersede in loops. Guard: `query_edges`/assertion supersedes insertion checks DAG acyclicity for `supersedes` and `derived-from`.

7. **Index-reset false drift.** A watch query’s previous derived snapshot vanished under `ops reset --index`, so re-run looks like “everything changed.” Guard: watched/finding result sets pin membership in `user.db` or blob; cache-only result sets cannot produce deltas after reset.

8. **Content-anchor rot.** `session:X` now points at different content after reingest. Guard: evidence refs for promoted findings use `session:X@hash`; resolver verifies and reports mismatch.

9. **Candidate promotion feedback into context.** A finding promoted active and injected into context changes future agents, who emit annotations that validate the finding. Guard: `FINDING` default `context_policy.inject=false`; explicit injection requires separate operator policy and should be visible in analysis provenance.

10. **Rank drift mistaken for membership drift.** Result membership unchanged but union ranking changes. Guard: store both `membership_merkle_root` and `ordered_rank_hash`.

11. **Mixed-grain false equality.** `message:m1` and `session:s1` accidentally share member-key text. Guard: grain is included in result-set id, Merkle leaves, DDL, and set-op type checks. The set-algebra doc already recommends fail-closed mixed-grain policy for v1. `docs/design/query-set-algebra.md:126-140`.

12. **Evidence laundering through findings.** Query A uses finding F as evidence; F was derived from query A. Guard: findings include `source_query_ref`/`result_set_ref`; report renderer warns on circular evidence ancestry.

## Migration sequence

[proposal] Implement in this order.

1. Add `ObjectRefKind` values and resolver stubs for `query`, `result_set`, `finding`; return unresolved payloads until tables land.

2. Add user migration for `queries`, `query_names`, `result_sets`, `result_set_members`, `query_edges`; add index cache tables.

3. Add `AssertionKind.FINDING`, claim-kind listing, deterministic finding ids, and `upsert_findings_as_assertions()` cloned from pathology flow.

4. Add query canonicalization and hashing. Start with session-grain compact/Boolean queries; reject set-op hashing until fnm.13 parser nodes exist.

5. Migrate saved views: for each `SAVED_QUERY` assertion, compile/hash query value into `queries`; update saved query assertion or `query_names` to point at `query:<hash>`.

6. Extend `resolve_ref` and `read --ref` for query/result_set/finding.

7. Add result-set materialization and Merkle roots to committed query runs. Do not record every preview yet.

8. Add query-edge emission from set-algebra EXPLAIN/planner once fnm.13 lands.

9. Add `StandingQueryStage` using session-scoped convergence hooks.

10. Add findings-as-tests over promoted `FINDING` assertions with explicit `value.expected`.

11. Then wire composer commits: every committed query returns `query_ref`, `query_run_ref`, `result_set_ref`; naming a query updates `query_names`/saved-query pointer. This is the durable version of the C10 recall-entry idea. fileciteturn2file2

## Open questions for the operator

1. Should `query_names` be a real table, or should mutable names remain `SAVED_QUERY` assertions whose `target_ref` points at `query:<hash>`? My recommendation: table for operational lookup, mirrored assertion for overlay compatibility only.

2. For watched queries, should full member lists always be pinned in `user.db`, or should large lists spill to content-addressed blobs with only a manifest in `user.db`?

3. Should `finding:<hash>` be a first-class ref kind, or is `assertion:<id>` enough? My recommendation: add `finding:<hash>` because operators and reports should not have to remember that findings are assertion rows.

4. What is the first acceptable `corpus_epoch` implementation: a cheap global index epoch, or a more expensive scoped predicate-fingerprint epoch? My recommendation: global epoch first, scoped epoch fast-follow.

5. Should `result_set:<hash>` be accepted directly in the DSL v1, or only through `read/resolve` until set-algebra lands?

6. How strict should content anchors be: hard failure on hash mismatch, or resolved-with-stale-warning? My recommendation: hard failure for promoted finding evidence; warning for exploratory reads.

## What’s missing

I did not run tests or inspect live master. The code line references are from the uploaded package snapshot, and A17 itself warns that the snapshot is dated. fileciteturn2file0

I did not inspect an implemented `docs/design/query-set-algebra.md` branch beyond the packaged design file; fnm.13 is still design/hold in this snapshot. `docs/design/query-set-algebra.md:1-4`.

I did not inspect whether a post-upload branch already added query/result-set/finding objects. Based on this package, they are absent from `core/refs.py`, `AssertionKind`, and `user.py`.

I did not produce a patch. The DDL and surfaces above are migration design, not applied code.
