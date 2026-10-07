# Archive Backup and Restore Boundaries

The fresh-start reset moves previous Polylogue core state aside intact and constructs fresh Source, User, Audit, Index and Ops tiers through the current constructor. It does not import previous core tiers or their physical authority receipts. External source files explicitly declared for intake may be copied into the new intake roots and ingested, including original hook and browser capture payloads stored beside the old tiers. Preserve the originals and their declared layouts; source copies do not carry old database or capture-registry authority. Explicitly selected purchased Embeddings may be preserved from a compatible sealed backup; matching content/model outputs must not be regenerated. The preparation witness restores only the selected `embeddings.db` bytes into an otherwise absent scratch root, then runs the ordinary owned constructor and embedding lifecycle startup. Startup creates destination-owned generation metadata; old occurrence references do not supply fresh occurrence bindings. This is separate from the complete-core verified restore operation below, which still refuses partial durable packages.

Polylogue stores one archive root as a split SQLite file set plus a
content-addressed blob store. Backups must preserve the tiers by durability
class instead of treating the archive root as one anonymous cache directory.

## Archive Root Layout

The configured archive root contains these durable paths:

| Path | Durability | Backup policy |
| --- | --- | --- |
| `source.db` | Raw acquisition evidence and source observations. | Back up. This is the rebuild root for parsed/indexed data. |
| `index.db` | Parsed sessions, messages, FTS/search indexes, graph rows, and derived read models. | Rebuildable from `source.db`; include in full evidence backups for faster restore, but cache-exclude profiles may omit it. |
| `embeddings.db` | Vector rows, embedding status, and catch-up metadata. | Back up when present. It is rebuildable, but expensive and may require provider cost. |
| `user.db` | Human/user/agent overlays stored as assertions, immutable annotation schema definitions and batch provenance, settings, and context-delivery receipts. | Always back up. This tier is irreplaceable user state. |
| `audit.db` | Append-only mutation authority, authorizations, attempts, receipts, and continuity heads. | Always back up. Full-evidence backups require it. |
| `ops.db` | Daemon cursors, attempts, convergence debt, stage events, and operational telemetry. | Disposable for ordinary restore profiles, but required by the exact full-evidence tier contract. |
| `source-declared-absent.json` | Operator-authored declared-absent blob hashes for a pre-generation `source.db`. | Copy with `source.db`; never derive or replace it with GC observations. |
| `blob/` | Content-addressed binary payloads keyed by SHA-256. | Back up referenced blobs with `source.db`/`user.db`; do not prune by age alone. |

`polylogue ops maintenance backup-plan --output-format json` is the
machine-readable inventory for tier filenames, backup boundaries, and missing
tiers; `polylogue ops status --format json` reports each tier's expected and
found version. Run them before backup automation rather than hard-coding only
the files that happen to exist locally.

## Backup Profiles

Use these profiles when choosing what to copy:

| Profile | Include | Exclude | Use case |
| --- | --- | --- | --- |
| Full evidence | All six archive tiers: `source.db`, `index.db`, `embeddings.db`, `user.db`, `ops.db`, and `audit.db`, plus referenced `blob/`. | Temporary SQLite `*-wal`/`*-shm` only after a clean checkpoint. | The fastest restore with raw evidence, read models, vectors, overlays, audit authority, and operational state. |
| User overlays | `user.db` and any assertion/note evidence blobs referenced by user-owned rows. | `index.db`, `ops.db`, rebuildable search/derived models. | Protect irreplaceable human/agent state before resets or schema rebuilds. |
| Rebuildable-cache exclude | `source.db`, `user.db`, referenced `blob/`, optionally `embeddings.db`. | `index.db`, `ops.db`, derived/cache artifacts. | Small backup that can rebuild parsed/indexed data locally. |
| Diagnostics bundle | `ops.db`, `backup-plan` JSON, `daemon-workload-probe` JSON, logs, and readonly status outputs. | Private raw blobs unless explicitly needed for the incident. | Bug reports and incident triage without over-sharing archive contents. |

When SQLite WAL files are present, either stop the daemon or run an explicit
checkpoint before copying. Copying only `*.db` while an uncheckpointed `*-wal`
contains recent writes creates an incomplete backup.

### Pre-generation source assertions

A source tier before the GC-generation migration may carry an optional
`source-declared-absent.json` beside `source.db`:

```json
{
  "format": "polylogue-source-declared-absent-v1",
  "freeze_authority": "polylogue-2x6xu",
  "source_db_sha256": "<sha256 of source.db>",
  "declared_absent_blob_hashes": ["<64 lowercase hex characters>"]
}
```

This is a durable metadata-only change: it adds no SQLite schema object and
has no lifecycle migration. A sidecar is required because an audit row may not
exist on the restored tier, while an additive migration cannot run before the
backup gate it would unblock. GC member outcomes are observations and cannot
provide operator intent. The verifier re-projects source references from the
restored `source.db`, checks the declaration against that projection, and
requires the effective reference scope to remain non-empty. The sidecar is
included in the signed backup artifact inventory, so changing it invalidates
the backup receipt.

The declaration applies only to source-owned blob hashes. It does not excuse
missing hashes referenced by `index.db` attachments. A `full_evidence` backup
therefore cannot attest when an index attachment is missing, even if that hash
also appears in the source declaration. Restore or otherwise resolve every
missing index attachment before relying on that profile.

After the source generation migration creates `source_generations` and
`source_items`, a retained sidecar makes verified backups fail closed. Remove
`source-declared-absent.json` from the archive root after the migration and
before the next verified backup; it is valid only for the pre-generation
source tier.

## Changing a configured archive root

Changing a configured archive root is a restore into a new root, not an
in-place transition: create and verify a `full_evidence` backup, then restore
it at the new root and let the daemon converge. `ArchiveLocation` refuses a
file set copied to a different root while its `index.db` and active-generation
tier links still resolve absolutely into the old one.

Active index pointer targets must resolve inside the configured archive root, with one exception for a target that is also the resolved target of the configured `index.db` symlink used by a symlink farm. Copied archives are refused by `ArchiveLocation`: an out-of-root pointer is admitted only when every durable tier at the root is also a symlink, so a symlink-preserving copy — which keeps real durable files inside itself — does not resolve into the archive it was copied from.

## Explicit verified restore

A verified backup is evidence, not an operational archive at its copied inodes.
Restore it through the declared daemon operation into a destination that does
not exist:

```bash
polylogue ops maintenance restore-verified-backup \
  --backup-dir /verified/backup --destination /new/archive --format json
```

`maintenance.restore_verified_backup` verifies the complete signed package,
blob closure, and original released train bindings. The production archive
population owner creates the destination with the immutable six-tier v1
baseline. A package whose durable tiers sit below the runtime versions
populates its exact rows before the destination's own numbered trains run; a
package at the runtime versions needs none. SQLite backup preserves
destination-owned inodes.
Startup checks that destination's actual train authority. Original
format and train receipts remain byte-for-byte detached provenance under
`.archive-population-provenance`; they are never rebound or admitted by
ordinary startup as authority for copied files.

A restore requires the complete Source, User, and Audit core. Overlay and
diagnostics profiles remain verified recovery evidence but receive a typed
`restore_partial_durable_core` refusal for operational restoration.
Backup acquisition retains stale Index and Ops as authenticated SQLite evidence
without serving their read models. Their physical integrity and supported version
remain required; ordinary query readers still refuse stale derived identity.
Index and Ops that are omitted or carry an earlier derived identity are
new empty tiers requiring convergence. Stale copied derived files remain
detached evidence under `original-derived`; their stamps are never admitted
or rewritten. `requires_convergence` names these tiers and operational
admission remains degraded until convergence. Omitted Embeddings are
`unrestored_purchased_tiers`, with degraded operational admission: the
canonical constructor's empty tier does not recover purchased vectors.
The returned `restored_tiers` excludes omitted and replaced derived tiers.
The backup itself remains immutable throughout restore.

## Runtime admission after restore

Use the installed runtime's explicit verified restore operation for supported
current-format backups. It applies declared durable evolution on the new
owned destination; a raw file copy does not grant startup authority. Unknown
versions or noncanonical schema shapes receive a typed refusal while the
original package remains intact. Previous core tiers and their authority receipts
remain salvage evidence. The explicitly selected purchased Embeddings exception
uses ordinary fresh-root startup as described above; it does not import a core
archive or grant copied durable receipts startup authority.

Verify the completed destination through production status and query routes.
Report field-query readiness and FTS availability separately: a restored
Index may require daemon convergence, and a newly derived empty Index has no
replayed sessions yet. Purchased vectors and missing referenced blobs remain
explicit gaps in the restore result.

## Rollback custody and qualification

For ordinary archive replacement or removal, preserve a full-evidence copy outside the path that will be recreated. Include all six tiers and every referenced blob. Record the copy's location, the selected runtime commit and executable version, and a manifest of the preserved files outside Git. Keep the original `user.db`; an export of selected rows is supplementary evidence, not a replacement.

For the fresh-start reset, preserve the previous Polylogue state intact outside the new archive. Its core tiers remain salvage evidence and are not rollback or readback inputs. Independently preserve the explicitly selected embedding backup, verify its current schema and exact content/model reuse through the preparation witness, and keep the original backup unchanged.

A copied archive is custody evidence, not active authority for its new inodes.
Create and verify a complete backup package, use the explicit restore operation
for a fresh destination, and qualify that destination through its normal
startup and read routes. Keep the original backup unchanged. Matching schema
versions alone does not authenticate copied physical train receipts.

Archive roots may contain absolute symlinks to generation or tier files. Moving the root aside does not preserve those files independently: a link can still resolve through the original path after that path is recreated. Before relying on a moved copy, inventory and preserve the resolved targets as part of the custody copy, or repair links in a separate copy and verify that every target resolves within that copy. Do not modify the sole preserved archive to repair its links.

Qualification requires the production read route against the completed destination. Check status, then run a representative field/origin query. Report field-query and FTS readiness separately; stale search indexes can remain unavailable until daemon convergence. A raw SQLite open or file listing establishes neither runtime compatibility nor query readiness. Never migrate the sole preserved copy as part of qualification.

Changing a configured archive root is a restore into a new root, not an in-place transition. Create and verify a full-evidence backup, restore it at the new root, and let the daemon converge. `ArchiveLocation` refuses an out-of-root active-generation pointer unless it resolves through the configured index symlink in the supported symlink-farm layout.

## Restore Rules

Restore into an isolated archive root first:

```bash
polylogue ops maintenance restore-verified-backup \
  --backup-dir /verified/backup --destination /realm/tmp/work/restore-check
export POLYLOGUE_ARCHIVE_ROOT=/realm/tmp/work/restore-check
polylogue ops maintenance backup-plan --output-format json
polylogue ops status --format json
```

Then verify the restored root before pointing the daemon at it:

```bash
polylogue ops diagnostics workload --json
polylogue ops doctor --format json
polylogue find pytest then read --view summary
```

Restore expectations:

- `user.db` survives every `polylogue ops reset`, including `--database` and
  `--all`: a reset deletes only `index.db` and `ops.db`.
- Assertion candidates, accepted/rejected/deferred judgments, and promoted
  active assertions all live in `user.db`. Rebuilding `index.db` from
  `source.db` must not turn rejected or deferred inference candidates back into
  actionable user assertions, and editing assertion metadata is outside the raw
  session content-hash boundary.
- `index.db` may be rebuilt from `source.db` when schema versions change.
- `embeddings.db` contains purchased vectors. Restore it when present; omitted
  vectors remain unrestored and require repurchase, never raw replay.
- `ops.db` does not decide archive correctness; restore it only when preserving
  daemon history matters.
- A restored blob store is valid only when referenced blobs still match their
  SHA-256 paths and source resolution plus active-index attachment ownership.
- Blob backup includes the exact union of referenced and publication-reserved
  bytes. `blob-inventory.json` records every hash and size;
  `blob-reference-evidence.json` separately records source resolution and
  active-index attachment ownership. Verification re-hashes restored bytes
  and checks the independent attachment evidence rather than accepting an
  equal file count.
- A backup package is complete by itself. When a referenced source blob is
  missing from the live store, backup replays its acquisition source and
  copies the payload into the package only when it is the blob's exact
  SHA-256 and size (a ZIP member only after the same ZIP admission
  acquisition applies); `recoverability_proofs` records those recovered
  hashes. Which source windows can hold a raw's bytes (the file prefix, a
  recorded append window, a pre-offset append after its preceding full
  observation, or the ZIP member) is decided in one place,
  `storage/source_blob_restoration.retained_blob_source_candidates`, which
  raw derivation also reads when it restores an absent blob before
  preparation. Verification and the migration backup gate derive the required
  blob set from the package's own `source.db`, `index.db`, reservations and
  declared-absent sidecar and never read an acquisition file, so a package
  missing any required blob is refused even if its receipt was signed.
- The blob namespace authority marker is deliberately excluded from ordinary
  archive-file-set backups, including `full_evidence`. Those backups copy the
  source tier and referenced bytes as independently verified artifacts, not
  one atomic source-plus-namespace unit. Restoring a pending GC intent with a
  newly created or separately restored blob root must therefore block on the
  absent or different marker. An operator may terminalize that exact intent
  through the authorized offline recovery route; it never rebinds the intent.
  Preserve a marker only in a restore format that proves source.db and the
  physical namespace were captured and restored as one bound unit.

## Blob GC Safety Boundary

Blob garbage collection is dry-run-first work. A safe GC report must prove:

- the candidate has no current canonical source or active-index attachment
  owner and no publication reservation;
- the candidate is older than the generation/age defense-in-depth gate
  (`MIN_AGE_S`; see `docs/internals.md` "GC concurrency model");
- a durable intent bound to the observed namespace exists before unlink, and
  the final source/index liveness and reservation recheck remains clear. The
  final object observation and unlink are performed through no-follow root
  and shard directory handles so a root swap cannot redirect the effect into
  a replacement namespace;
- the report names exact candidate counts and references before deletion.

Do not delete blobs based only on filesystem age, directory mtime, or one
tier's absence. Canonical liveness resolves source and active-index owners.

## Quarterly Restore Drill Runbook (polylogue-4be)

First real drill executed 2026-07-27. This is the repeatable procedure for the
next quarterly run — an untested backup is a hypothesis, not a capability, so
this drill must actually restore bytes from the real backup stores into a
scratch location and verify them, not just check that a job "ran".

**Safety invariants**: restore only into a fresh scratch directory (never the
live archive root, never `polylogued.service`'s data, never this repo's
`.beads/`); treat borg repos and reflink source directories as read-only for
the whole drill; delete the scratch restore once verified.

### 1. Durable tier from Borg (`source.db`/`user.db`)

The operator's `sinnix` host backs up `/realm` into the Borg repo at
`/outer-realm/backup/borg-realm-v2` (btrbk snapshot -> `borgbackup-job-realm`
drain, see sinnix `modules/backup.nix`). List archives and restore only the
target files (not a whole directory — a directory extract can pull in a
multi-GB blob store and stall):

```bash
export BORG_PASSCOMMAND="cat /run/agenix/borg-passphrase"
export BORG_CACHE_DIR=/persist/root/.cache/borg
REPO="file:///outer-realm/backup/borg-realm-v2"

# List recent archives (needs root — repo dir is 0700 root:root)
sudo env BORG_PASSCOMMAND="$BORG_PASSCOMMAND" BORG_CACHE_DIR="$BORG_CACHE_DIR" \
  borg list --last 5 --format '{archive}{NL}' "$REPO"

ARCHIVE="<pick latest realm-realm.* archive>"
mkdir -p /realm/tmp/restore-drill-$(date +%Y%m%d)/borg-restore
cd /realm/tmp/restore-drill-$(date +%Y%m%d)/borg-restore

# Extract only the specific durable-tier file paths inside the archive —
# never extract a whole directory without checking its size first.
sudo env BORG_PASSCOMMAND="$BORG_PASSCOMMAND" BORG_CACHE_DIR="$BORG_CACHE_DIR" \
  borg extract --list "$REPO::$ARCHIVE" \
  "<relative/path/to>/user.db" "<relative/path/to>/source.db"

sudo chown -R "$USER":"$USER" /realm/tmp/restore-drill-$(date +%Y%m%d)
```

Verify:

```bash
sqlite3 <restored>/user.db "PRAGMA integrity_check;"     # must print exactly "ok"
sqlite3 <restored>/user.db "PRAGMA user_version; SELECT count(*) FROM assertions;"
sqlite3 <restored>/source.db "PRAGMA integrity_check;"
sqlite3 <restored>/source.db "PRAGMA user_version; SELECT count(*) FROM raw_sessions;"

# Sane-lag comparison against the live archive (restored counts must be <=
# live counts, and the gap should track the age of the chosen archive):
archive_root="${POLYLOGUE_ARCHIVE_ROOT:?set the configured archive root}"
sqlite3 "$archive_root/user.db" "SELECT count(*) FROM assertions;"
sqlite3 "$archive_root/source.db" "SELECT count(*) FROM raw_sessions;"
```

**Negative control (deliberately corrupted restore must fail loudly)** —
flip a few bytes past the SQLite header and confirm `integrity_check` reports
corruption, not `ok`:

`user.db` is the irreplaceable tier and holds the operator's own assertions, so
the working copy goes to a fresh owner-only directory rather than a fixed
`/tmp` name that another local account could pre-create or read:

```bash
work="$(mktemp -d)"            # 0700, unique per run
cp <restored>/user.db "$work/corrupt-test.db"
python3 - "$work/corrupt-test.db" <<'PY'
import sys
with open(sys.argv[1], 'r+b') as f:
    f.seek(4096); d = f.read(64); f.seek(4096)
    f.write(bytes(b ^ 0xFF for b in d))
PY
sqlite3 "$work/corrupt-test.db" "PRAGMA integrity_check;"  # must report errors, exit 11
rm -rf "$work"
```

2026-07-27 result: extracting `inbox/polylogue-backups/polylogue-archive-20260710T162633Z/{user,source}.db`
(a receipt-verified pre-deploy backup snapshot, not the live-tier path — see
gap below) from archive `realm-realm.20260727T163001+0200` took **29.4s**.
Both files passed `integrity_check = ok`; `user.db` carried 1 assertion at
schema `user_version=4`, `source.db` carried 17,839 `raw_sessions` rows at
`user_version=3` — both older than live (`user_version=10`/95 assertions and
`user_version=13`/41,233 raw_sessions respectively), consistent with this
snapshot's 17-day age. The corruption negative control correctly failed with
`database disk image is malformed (11)`.

**CRITICAL FINDING — the durable tier then under review had NO Borg coverage.**
The configured archive root (resolved from `POLYLOGUE_ARCHIVE_ROOT`) was a nested
Btrfs subvolume at the time of the drill. Its location is configuration, not a
fixed runtime path; inspect the resolved root before repeating this evidence.
btrbk/Borg snapshot the **parent** `/realm` subvolume only; a nested
subvolume shows up as an **empty directory** in every snapshot and archive —
confirmed directly: `borg list <latest realm archive> db/polylogue` returns
only the empty directory entry itself, zero children. This is the exact same
class of gap `sinex`'s blob repository hit before `borgbackup-job-sinex-blobs`
was added, and that `state/machine-telemetry`/`db/machine-telemetry` hit
before `machine-telemetry-sqlite-backup.service` was added (see sinnix
`modules/services/machine-telemetry.nix`). Polylogue's durable tiers have no
equivalent dedicated job. The drill above only worked because an older,
already-durable pre-deploy backup snapshot happened to sit under
`/realm/inbox/polylogue-backups/` (itself not a nested subvolume, so it *is*
covered) — the actual live `user.db`/`source.db` files have been unbacked-up
since 2026-07-06. Tracked as a new bead; see notes on polylogue-4be.

### 2. Beads workspace from reflink snapshot

Pre-migration reflink snapshots of this repo's Dolt-backed Beads workspace
live under `/realm/tmp/beads-backup-<repo>-<pid>/`. Restore is a plain
`cp --reflink=always` (fast, copy-on-write, no borg involved) into scratch,
then open it directly with the `dolt` CLI — no need to reconstruct a full
`.beads/` tree, the noms data directory alone is a valid Dolt database:

```bash
cp --reflink=always -a /realm/tmp/beads-backup-polylogue-<pid>/polylogue \
  /realm/tmp/restore-drill-$(date +%Y%m%d)/beads-restore

cd /realm/tmp/restore-drill-$(date +%Y%m%d)/beads-restore
dolt sql -q "show databases;"                 # must list the restored db
dolt sql -q "use \`beads-restore\`; show tables;"
dolt sql -q "use \`beads-restore\`; select count(*) from issues;"
dolt sql -q "use \`beads-restore\`; select count(*) from dolt_log;"  # commit history intact
```

2026-07-27 result: reflinking `/realm/tmp/beads-backup-polylogue-0007/polylogue`
(290 MB apparent, birth 2026-07-13) took **0.007s** (confirms true reflink,
not a byte copy). The restored database opened cleanly under `dolt` 2.1.9,
listed all 26 expected tables (`issues`, `dependencies`, `wisps`, `events`,
...), reported **713 issues** and **6,274** `dolt_log` commits. Sane-lag
check: live `bd count` currently reports 1,108 issues — restored count is
lower and consistent with 14 days of growth since the snapshot.

### Cleanup

Delete the scratch restore once both verifications are captured — do not
leave multi-hundred-MB restored copies in `/realm/tmp/`:

```bash
rm -rf /realm/tmp/restore-drill-$(date +%Y%m%d)
```

The destination is reserved exclusively and carries an unfinished-population
marker until exact schema, rows, blob files, and ordinary startup admission
have passed. Other archive readers and writers refuse that destination while
population is pending. An interrupted restore retains its partial directory
and marker as evidence; it is not automatically repaired or resumed. A new
restore requires a different, absent destination.


The machine restore exchange has a 300-second response budget. Once its first
possible filesystem effect has been admitted, expiry returns `indeterminate`
and the accepted restore continues; it does not cancel progressing work.
`operation.await` or `operation.status` with the same request ID and principal
reads its exact terminal result while that daemon remains live. Completed unbound
results transfer atomically to private runtime scratch before the live exchange
retires; they are not subject to progress-buffer expiry. A transfer fault keeps
the original terminal future owned and visible through `terminal_custody_error`;
new mutation admission pauses retryably until that transfer succeeds. Runtime
shutdown drains workers before removing its result scratch. These files are not Audit receipts and do
not grant restart authority.
An await may return a running observation or progress frame first; consumers
follow its state and progress cursors until the terminal outcome.
An expired await poll still authenticates the reference and reads its actual
lifecycle once. Expiry ends the wait and returns that snapshot; it does not
turn an accepted mutation into a failed operation or cancel its work.
This unbound filesystem operation does not mint an Audit machine-request
receipt. After a daemon crash, an unfinished destination therefore remains
fenced evidence, not a claim of durable terminal success.
