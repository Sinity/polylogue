# Canonical Judgment Transaction — Handoff

## Delivery identity

This package is an apply-ready implementation against Polylogue snapshot commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` (`fix(repair): harden raw authority convergence (#3046)`), captured from snapshot branch `master` with no pre-existing working-tree patch. The implementation worktree used branch `feature/assertions/judgment-transaction-r01` only as a local delivery branch.

The merged lifecycle authority remains commit `5aa34e6c5d231c952529174febe99b2a58f4da07` (`feat(assertions): add reviewed candidate judgment flow (#2791)`). This patch extends that authority; it does not create a second transition engine, a second queue, or another promotion path.

`PATCH.diff` changes 36 files: 1,744 insertions and 601 deletions. It introduces no user-tier schema change and no migration.

## Mission completed

The implementation closes the reproduced `polylogue-41ow` preservation race, completes the bounded evidence and queue-health residuals from `polylogue-37t.12`, removes the duplicate `mark candidates` public workflow after moving its useful behavior to root `polylogue judge`, and prepares a genuine operator-executed `polylogue-mrxt` canary with a stable retry identity and exact verification procedure.

## Existing PR #2791 guarantees retained

PR #2791 already supplied the authoritative candidate transition machinery. This patch retains these properties and tests them through the same storage functions:

- machine-authored claims are candidates with `inject=false` and `promotion_required=true`;
- default acceptance creates one active, non-injected assertion;
- injection requires the explicit review-authority input;
- reject and defer record durable judgments without promotion;
- exact retries are idempotent while changed decisions conflict;
- bulk judgments use one outer transaction with per-item SAVEPOINT isolation, input deduplication, partial success for malformed/conflicting refs, and rollback on unexpected batch failure;
- MCP judgment remains bound to the `assertion-review` capability rather than ordinary mutation discovery;
- root `polylogue judge` is the canonical operator workflow.

## Additions in this patch

### 1. `polylogue-41ow`: TOCTOU-safe assertion writes

`upsert_assertion` previously read an existing status before obtaining the SQLite writer slot, then performed `INSERT ... ON CONFLICT` later. An automated producer could therefore read `candidate`, block while an operator accepted the row, and then overwrite the accepted status back to `candidate`.

The repair is at the shared user-tier write chokepoint:

- standalone writes configure the canonical write profile and start `BEGIN IMMEDIATE` before the preservation read;
- nested writes use a uniquely named SAVEPOINT and a zero-row `UPDATE assertions ... WHERE 0` before the preservation read, forcing a deferred outer transaction to reserve the SQLite writer slot without committing caller-owned work;
- the status-preservation `SELECT`, policy constraint, write, and readback occur under that same reservation;
- standalone failures roll back the owned transaction; nested failures roll back and release only the owned SAVEPOINT;
- `judge_assertion_candidate` uses the same reservation helper and delegates to the existing locked lifecycle implementation, so there is still one judgment authority;
- the shared helper applies `WRITE_CONNECTION_PROFILE.busy_timeout_ms` and enables foreign keys when SQLite permits it.

The regression uses two real SQLite connections. It pauses the automated writer after it has acquired the writer reservation, verifies that the second operator connection cannot interleave, then releases the first writer and verifies the legal serial outcome: the later operator acceptance wins and remains accepted. Removing the reservation, moving it after the read, or removing terminal-status preservation makes this test fail. A trigger-induced write failure proves complete rollback.

### 2. Upsert identity semantics aligned and documented

| Writer | Logical identity | Exact retry | Changed payload |
|---|---|---|---|
| `upsert_assertion` | caller/producer-supplied `assertion_id` | updates the same row | updates the same row; automated writes preserve terminal judgment state |
| `upsert_suppression` | session id | converges | updates suppression reason/mode |
| `upsert_mark` | target type + target id + mark type | converges | updates label/metadata |
| `upsert_session_tag_assertion` | session id + normalized tag + source | converges | updates metadata for that logical tag |
| `upsert_session_metadata_assertion` | session id + normalized key | converges | updates that metadata key |
| `upsert_annotation` with explicit id | caller-provided annotation id | converges | updates that annotation |
| `upsert_annotation` without explicit id | target type + target id + body hash | converges | appends a distinct annotation when body changes |
| `upsert_correction` | target type + target id + correction type | converges | updates that correction type |
| `upsert_saved_view` | explicit id, otherwise stable name-derived id | converges | updates the named view |
| `upsert_recall_pack` | explicit id, otherwise stable name-derived id | converges | updates the named pack; this replaces the former name-plus-payload identity |
| `upsert_workspace` | explicit id, otherwise stable name-derived id | converges | updates the named workspace |
| `upsert_blackboard_note` | explicit id, otherwise scope + body hash | converges | appends a distinct note when body changes |
| transform/pathology/finding producers | deterministic provenance/content identity | converges | changed provenance/content becomes a new candidate |
| comparative judgment writer | existing `judgment_id` supplied by comparative builder | converges only when the same id is reused | unchanged by this patch; the bead notes that wall-clock-dependent ids still need a real caller contract |

Query-first `mark --note` now uses a deterministic session/body annotation id instead of generating a UUID for every retry. Terminal/MCP candidate capture also accepts an actor-scoped `idempotency_key`: exact replay returns the original lifecycle row even after judgment, while reusing the key with changed content raises a conflict before overwrite.

### 3. Evidence disclosure on the canonical review projection

`list_assertion_candidate_reviews` now resolves at most five evidence refs per candidate and returns typed, bounded previews through the shared payload used by CLI and MCP. Each review item exposes:

- a one-line claim summary bounded to 480 characters;
- source/author identity and candidate age;
- target and scope already carried by the candidate payload;
- evidence total and omitted counts;
- up to five previews with `resolved`, `missing`, `unsupported`, `pending`, or `error` state;
- bounded title, excerpt, reason, open commands, and hrefs.

One resolver failure degrades only that evidence preview. It does not fail the candidate list or expand an unbounded transcript.

Root `polylogue judge --review --format json` and MCP `list_assertion_candidate_reviews` serialize the same fields.

### 4. Queue health as a non-destructive product

The new facade method `assertion_candidate_queue_health()` reads durable candidate state and operations telemetry without mutating the queue. It reports:

- pending and lifecycle-status counts;
- counts by candidate kind and producer/source;
- oldest/newest pending timestamps and oldest age;
- candidates older than 60 days as retained, visible backlog;
- latest standing-query producer status and age;
- scheduler heartbeat state and age;
- failed/deferred convergence debt;
- caveats when telemetry is unavailable.

State classification is explicit:

- `healthy-empty`: zero pending only when a recent successful producer run and a fresh scheduler heartbeat are both observable;
- `empty-unverified`: zero pending without enough producer/scheduler evidence;
- `pending`: observable recent backlog;
- `stale-pending`: retained backlog older than 60 days;
- `producer-stalled`: failed producer, stale/stopped scheduler, or durable producer debt;
- `unavailable`: a surface-level fallback when health collection itself fails.

No expiry path promotes or deletes reviewed history. Old candidates remain `retained-visible`.

The product is available through:

```bash
polylogue judge --status
polylogue judge --status --format json
polylogue status --format json
polylogue ops status --format json
```

Daemon status includes `assertion_candidate_queue`; direct status renders the same summary. The facade route catalog classifies the method as an archive-routed user-tier/operations-telemetry read.

### 5. One public operator lifecycle

The `mark candidates` group and its list/review/accept/reject/defer/supersede commands were removed from `polylogue/cli/query_verbs.py`. Their useful behavior now lives on root `polylogue judge`:

```bash
polylogue judge --list
polylogue judge --review
polylogue judge --status
polylogue judge --accept assertion:CANDIDATE_ID
polylogue judge --reject assertion:CANDIDATE_ID
polylogue judge --defer assertion:CANDIDATE_ID
polylogue judge --supersede assertion:CANDIDATE_ID --replacement-kind summary --body 'replacement claim'
```

The command supports repeated decision options for explicit multi-ref batches, `--accept-all-of-kind`, target/kind/status/time filters, actor/reason fields, explicit `--inject`, text/JSON output, and the standard `--json` alias. Interactive skip or editor cancellation now writes a durable defer instead of silently abandoning state.

Click registration, action contracts, shell completions, generated CLI reference, generated product workflows, deterministic-output contracts, help snapshots, and search documentation now identify root `judge` as the owner. Ordinary query-first `mark` remains limited to session overlays.

## Prepared `polylogue-mrxt` operator canary

The canary must be run against the operator's real archive with a genuinely authored observation. The script below creates and judges through shipped CLI routes; its SQL is read-only verification after those production actions.

Save as `run-mrxt-canary.sh`, make it executable, set `MRXT_BODY` to a real observation from the referenced session, and pass a real archived `session:<id>` ref.

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${MRXT_BODY:?Set MRXT_BODY to a genuine operator-authored observation from the target session}"
SESSION_REF="${1:?usage: run-mrxt-canary.sh session:<id> [idempotency-key]}"
KEY="${2:-mrxt-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
ACTOR_REF="user:operator"
REASON="mrxt genuine operator canary acceptance"
TMPDIR_CANARY="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_CANARY"' EXIT

polylogue --plain note "$MRXT_BODY" \
  --kind lesson \
  --ref "$SESSION_REF" \
  --idempotency-key "$KEY" \
  --format json > "$TMPDIR_CANARY/capture.json"

CANDIDATE_ID="$(python - "$TMPDIR_CANARY/capture.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "candidate", payload
assert payload["context_policy"] == {"inject": False, "promotion_required": True}, payload
print(payload["assertion_id"])
PY
)"
CANDIDATE_REF="assertion:${CANDIDATE_ID}"

polylogue --plain judge \
  --target-ref "$SESSION_REF" \
  --candidate-status candidate \
  --review \
  --format json > "$TMPDIR_CANARY/pending-review.json"

CANDIDATE_REF="$CANDIDATE_REF" python - "$TMPDIR_CANARY/pending-review.json" <<'PY'
import json, os, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = [item for item in payload["items"] if item["candidate_ref"] == os.environ["CANDIDATE_REF"]]
assert len(rows) == 1, (os.environ["CANDIDATE_REF"], payload)
row = rows[0]
assert row["candidate"]["status"] == "candidate", row
assert row["evidence_total_count"] >= 1, row
assert len(row["evidence_previews"]) <= 5, row
PY

polylogue --plain judge \
  --accept "$CANDIDATE_REF" \
  --reason "$REASON" \
  --actor-ref "$ACTOR_REF" \
  --format json > "$TMPDIR_CANARY/judgment.json"

RESULTING_REF="$(python - "$TMPDIR_CANARY/judgment.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["applied_count"] == 1, payload
assert payload["failed_count"] == 0, payload
item = payload["items"][0]
assert item["outcome"] == "applied", item
assert item["result"]["candidate"]["status"] == "accepted", item
assert item["result"]["judgment"]["decision"] == "accept", item
print(item["result"]["judgment"]["resulting_assertion_ref"])
PY
)"

polylogue --plain note "$MRXT_BODY" \
  --kind lesson \
  --ref "$SESSION_REF" \
  --idempotency-key "$KEY" \
  --format json > "$TMPDIR_CANARY/replay.json"

CANDIDATE_ID="$CANDIDATE_ID" python - "$TMPDIR_CANARY/replay.json" <<'PY'
import json, os, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["assertion_id"] == os.environ["CANDIDATE_ID"], payload
assert payload["status"] == "accepted", payload
PY

if polylogue --plain note "${MRXT_BODY} [changed-content conflict probe]" \
  --kind lesson \
  --ref "$SESSION_REF" \
  --idempotency-key "$KEY" \
  --format json > "$TMPDIR_CANARY/changed-conflict.txt" 2>&1; then
  cat "$TMPDIR_CANARY/changed-conflict.txt" >&2
  echo "changed content unexpectedly reused the idempotency key" >&2
  exit 1
fi
grep -Fq "idempotency_key conflicts" "$TMPDIR_CANARY/changed-conflict.txt"

polylogue --plain judge \
  --target-ref "$SESSION_REF" \
  --candidate-status accepted \
  --review \
  --format json > "$TMPDIR_CANARY/accepted-review.json"

CANDIDATE_REF="$CANDIDATE_REF" RESULTING_REF="$RESULTING_REF" \
python - "$TMPDIR_CANARY/accepted-review.json" <<'PY'
import json, os, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = [item for item in payload["items"] if item["candidate_ref"] == os.environ["CANDIDATE_REF"]]
assert len(rows) == 1, payload
judgment = rows[0]["latest_judgment"]
assert judgment["decision"] == "accept", judgment
assert judgment["resulting_assertion_ref"] == os.environ["RESULTING_REF"], judgment
PY

SESSION_REF="$SESSION_REF" CANDIDATE_REF="$CANDIDATE_REF" RESULTING_REF="$RESULTING_REF" \
python - <<'PY'
import asyncio
import json
import os
import sqlite3

from polylogue.api import Polylogue
from polylogue.context.compiler import ContextSpec
from polylogue.paths import archive_root

session_ref = os.environ["SESSION_REF"]
candidate_ref = os.environ["CANDIDATE_REF"]
resulting_ref = os.environ["RESULTING_REF"]

async def verify_surfaces() -> None:
    async with Polylogue(archive_root=archive_root()) as poly:
        review = await poly.list_assertion_candidate_reviews(
            target_ref=session_ref,
            statuses=("accepted",),
            limit=20,
        )
        rows = [item for item in review.items if item.candidate_ref == candidate_ref]
        assert len(rows) == 1, review
        row = rows[0]
        assert row.latest_judgment is not None
        assert row.latest_judgment.decision == "accept"
        assert row.latest_judgment.resulting_assertion_ref == resulting_ref

        candidate = await poly.resolve_ref(candidate_ref)
        result = await poly.resolve_ref(resulting_ref)
        assert candidate.resolved is True, candidate
        assert result.resolved is True, result

        image = await poly.compile_context(
            ContextSpec(seed_refs=(session_ref,), read_views=())
        )
        assert candidate_ref not in image.assertion_refs
        assert resulting_ref not in image.assertion_refs

asyncio.run(verify_surfaces())

root = archive_root()
with sqlite3.connect(root / "user.db") as conn:
    conn.row_factory = sqlite3.Row
    candidate_id = candidate_ref.removeprefix("assertion:")
    result_id = resulting_ref.removeprefix("assertion:")
    rows = conn.execute(
        """
        SELECT assertion_id, target_ref, kind, status, evidence_refs_json,
               context_policy_json, supersedes_json, value_json
        FROM assertions
        WHERE assertion_id IN (?, ?)
           OR target_ref = ?
        ORDER BY kind, assertion_id
        """,
        (candidate_id, result_id, candidate_ref),
    ).fetchall()

candidate_rows = [row for row in rows if row["assertion_id"] == candidate_id]
result_rows = [row for row in rows if row["assertion_id"] == result_id]
judgment_rows = [row for row in rows if row["kind"] == "judgment"]
assert len(candidate_rows) == 1 and candidate_rows[0]["status"] == "accepted", rows
assert len(result_rows) == 1 and result_rows[0]["status"] == "active", rows
assert len(judgment_rows) == 1, rows

result = result_rows[0]
assert json.loads(result["context_policy_json"]) == {"inject": False}, result
assert json.loads(result["supersedes_json"]) == [candidate_ref], result
assert json.loads(result["evidence_refs_json"]) == [session_ref, candidate_ref], result
judgment = json.loads(judgment_rows[0]["value_json"])
assert judgment["candidate_ref"] == candidate_ref, judgment
assert judgment["resulting_assertion_ref"] == resulting_ref, judgment
assert judgment["inject_authorized"] is False, judgment

print(json.dumps({
    "candidate_ref": candidate_ref,
    "resulting_assertion_ref": resulting_ref,
    "judgment_assertion_id": judgment_rows[0]["assertion_id"],
    "context_policy": {"inject": False},
    "verified": True,
}, sort_keys=True))
PY

printf 'Canary receipts:\n  candidate: %s\n  resulting assertion: %s\n  key: %s\n' \
  "$CANDIDATE_REF" "$RESULTING_REF" "$KEY"
```

Expected durable effects, in order:

1. `polylogue note` creates one private candidate row with stable evidence ref, `inject=false`, and the actor-scoped retry identity.
2. `polylogue judge --review` exposes that row and its bounded evidence disclosure.
3. `polylogue judge --accept` records one accepted candidate, one judgment assertion/receipt, and one active resulting assertion whose evidence adds the candidate ref and whose `supersedes` contains the candidate ref.
4. Because `--inject` was not supplied, the resulting assertion records `{"inject": false}` and default context compilation excludes both candidate and result.
5. Replaying the capture returns the same candidate id and accepted state; it does not create another row.
6. A changed body with the same key must fail with `idempotency_key conflicts with a different assertion candidate capture`.

Exact MCP read checks for the operator's authenticated MCP client are:

```text
list_assertion_candidate_reviews(target_ref=SESSION_REF, statuses=["accepted"], limit=20)
resolve_ref(ref=CANDIDATE_REF)
resolve_ref(ref=RESULTING_REF)
```

The first call requires the archive-evidence read policy and must expose the same judgment/result refs. The two `resolve_ref` calls must resolve the same durable candidate and active assertion observed by CLI/API. Judgment tools remain available only under `assertion-review`; ordinary mutation discovery must not expose them.

## Changed files

Production and control-plane files:

```text
devtools/render_cli_reference.py
devtools/render_product_workflows.py
docs/cli-reference.md
docs/product/workflows.md
docs/search.md
polylogue/api/archive.py
polylogue/cli/click_app.py
polylogue/cli/command_inventory.py
polylogue/cli/commands/judge.py
polylogue/cli/commands/note.py
polylogue/cli/commands/status.py
polylogue/cli/query_group.py
polylogue/cli/query_verbs.py
polylogue/daemon/status.py
polylogue/mcp/server_mutation_tools.py
polylogue/operations/action_contracts.py
polylogue/product/workflows.py
polylogue/storage/sqlite/archive_tiers/archive.py
polylogue/storage/sqlite/archive_tiers/user_write.py
polylogue/surfaces/payloads.py
```

Tests and generated snapshots:

```text
tests/unit/api/test_assertion_candidate_evidence_disclosure.py
tests/unit/api/test_assertion_candidate_queue_health.py
tests/unit/api/test_facade_contracts.py
tests/unit/cli/__snapshots__/test_help_snapshots.ambr
tests/unit/cli/__snapshots__/test_terminal_snapshots.ambr
tests/unit/cli/test_assertion_candidates.py
tests/unit/cli/test_cli_action_contracts.py
tests/unit/cli/test_completion_matrix.py
tests/unit/cli/test_deterministic_output.py
tests/unit/cli/test_judge_command.py
tests/unit/cli/test_note.py
tests/unit/mcp/test_assertion_judgment_tools.py
tests/unit/mcp/test_candidate_capture_tool.py
tests/unit/product/test_query_action_workflows.py
tests/unit/storage/test_archive_tiers_assertion_write_through.py
tests/unit/storage/test_archive_tiers_assertions.py
```

## Acceptance matrix

| Requirement | Result | Evidence |
|---|---|---|
| `BEGIN IMMEDIATE` preservation/write boundary | Implemented | `_assertion_write_transaction`; two-connection regression |
| Nested caller transaction remains owned by caller | Implemented | SAVEPOINT + no-op write reservation; rollback test |
| Terminal operator judgment cannot be silently reverted | Implemented | forced interleaving ends accepted |
| Competing decisions retain conflict/idempotency semantics | Preserved | existing lifecycle and bulk tests |
| Canonical busy timeout/profile | Implemented at shared writer | `WRITE_CONNECTION_PROFILE` helper |
| Upsert identities classified/aligned | Implemented | table above; recall-pack and annotation tests |
| Bounded evidence disclosure | Implemented | typed payloads, five-preview cap, per-ref failure isolation |
| Queue health/status | Implemented | facade, root judge, direct status, daemon status |
| 60-day behavior is non-destructive | Implemented | old rows reported `retained-visible`; persistence test |
| Root `judge` sole public judgment CLI | Implemented | duplicate group removed; registry/completion/docs tests |
| Machine candidates remain candidate/non-injected | Preserved and tested | storage and canary-route tests |
| Accept/reject/defer/supersede lifecycle | Preserved and tested | storage, CLI, MCP tests |
| Bulk valid/idempotent/malformed/conflicting refs | Preserved and tested | existing per-item SAVEPOINT test |
| Stable canary retry identity | Implemented | terminal/MCP key, changed-content conflict |
| Genuine live operator canary | Prepared, not executed | exact script above; no live archive access |
| user.db schema stability | Preserved | no schema or migration files changed |

## Apply order

From a clean checkout at the named snapshot:

```bash
git switch --detach 536a53efac0cbe4a2473ad379e4db49ef3fce74d
git apply --check PATCH.diff
git apply PATCH.diff
```

Then run the focused commands in `TESTS.md`. Generated documentation should remain synchronized:

```bash
python -m devtools.render_cli_reference --check
python -m devtools.render_product_workflows --check
```

## Verification and limitations

`PATCH.diff` was generated directly from the authoritative base and passed `git apply --check` in a clean detached worktree at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. Ruff, mypy over 14 changed production modules, compileall, generated-document checks, and the focused storage/API/CLI/MCP/status/daemon suites described in `TESTS.md` were executed.

No live archive, daemon process, MCP transport session, browser surface, deployment, credentials, or operator-authored event was available. The live `polylogue-mrxt` canary is therefore deliberately unexecuted. A complete repository-wide test run did not finish within the execution environment's per-command ceiling; two broad aggregate attempts timed out without a reported failure and are not counted as passes. The terminal PTY snapshot suite is environmentally unstable in this container because an unrelated Tokio worker intermittently writes a `Bad file descriptor` panic into the PTY; clean snapshot files contain no such text, but the latest isolated PTY run reported three snapshot mismatches for that external stderr noise.

The remaining engineering value is a small certification repair, not a substantial second implementation pass: run the full suite in the repository's normal devshell, run the PTY snapshots without the Tokio runtime fault, and execute the supplied canary on the operator's live archive. A substantial redesign is not indicated by the completed tests or source audit.
