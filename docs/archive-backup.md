# Archive Backup and Restore Boundaries

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
| `ops.db` | Daemon cursors, attempts, convergence debt, stage events, and operational telemetry. | Disposable. Include only in diagnostics bundles or incident snapshots. |
| `blob/` | Content-addressed binary payloads keyed by SHA-256. | Back up referenced blobs with `source.db`/`user.db`; do not prune by age alone. |

`polylogue ops maintenance archive-plan --output-format json` is the machine-readable
inventory for tier filenames, expected versions, backup-required tiers, and
missing blockers. Run it before backup automation rather than hard-coding only
the files that happen to exist locally.

## Backup Profiles

Use these profiles when choosing what to copy:

| Profile | Include | Exclude | Use case |
| --- | --- | --- | --- |
| Full evidence | `source.db`, `index.db`, `embeddings.db`, `user.db`, referenced `blob/`, and optional `ops.db` snapshot. | Temporary SQLite `*-wal`/`*-shm` only after a clean checkpoint. | Fastest complete restore with raw evidence, read models, vectors, and overlays. |
| User overlays | `user.db` and any assertion/note evidence blobs referenced by user-owned rows. | `index.db`, `ops.db`, rebuildable search/derived models. | Protect irreplaceable human/agent state before resets or schema rebuilds. |
| Rebuildable-cache exclude | `source.db`, `user.db`, referenced `blob/`, optionally `embeddings.db`. | `index.db`, `ops.db`, derived/cache artifacts. | Small backup that can rebuild parsed/indexed data locally. |
| Diagnostics bundle | `ops.db`, `archive-plan` JSON, `daemon-workload-probe` JSON, logs, and readonly status outputs. | Private raw blobs unless explicitly needed for the incident. | Bug reports and incident triage without over-sharing archive contents. |

When SQLite WAL files are present, either stop the daemon or run an explicit
checkpoint before copying. Copying only `*.db` while an uncheckpointed `*-wal`
contains recent writes creates an incomplete backup.

## Offline archive-root relocation

An inode-preserving filesystem move is the only supported way to change a configured archive root without restoring or rebuilding it. Stop the daemon, move the complete root without copying its database files, and retain the verified `full_evidence` backup made at the old root. Then point `POLYLOGUE_ARCHIVE_ROOT` at the destination and create the bound plan:

```bash
POLYLOGUE_ARCHIVE_ROOT=/new/archive/root \
  polylogue ops maintenance archive-root-relocation plan \
  --old-root /old/archive/root \
  --backup-manifest /path/to/verified-full-evidence/manifest.json \
  --output /safe/operator/location/relocation-plan.json --output-format json
```

Apply only the exact self-hash printed in that plan:

```bash
POLYLOGUE_ARCHIVE_ROOT=/new/archive/root \
  polylogue ops maintenance archive-root-relocation apply \
  --plan /safe/operator/location/relocation-plan.json \
  --authorize PLAN_SHA256 --output-format json
```

The route reads every SQLite file immutably and refuses copied files, WAL sidecars, missing HMAC authority for the old path, changed bytes/schema/version/tier inventory, fresh-bootstrap authority, source-continuity trains, or any non-released source train. It records both configured and resolved paths. A configured `index.db` active-generation symlink is permitted only through the existing `ArchiveLocation` resolver, and the plan binds the resolved generation rather than a shadow index path. Apply writes no SQLite rows, blobs, or sidecars. It CAS-revises only released source train manifests and records a prepared then committed receipt under `.maintenance-state/archive-root-relocations/`. A prepared receipt blocks daemon startup and prints the exact resume command. Live application and post-move observation remain operator evidence, outside this code path.

## Restore Rules

Restore into an isolated archive root first:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-restore-check
polylogue ops maintenance archive-plan --output-format json
polylogue ops status --format json
```

Then verify the restored root before pointing the daemon at it:

```bash
polylogue ops diagnostics workload --json
polylogue ops doctor --format json
polylogue find pytest then read --view summary
```

Restore expectations:

- `user.db` must survive ordinary `polylogue ops reset --database` and
  `polylogue ops reset --all`; deleting it requires the explicit
  `--include-user-db` opt-in.
- Assertion candidates, accepted/rejected/deferred judgments, and promoted
  active assertions all live in `user.db`. Rebuilding `index.db` from
  `source.db` must not turn rejected or deferred inference candidates back into
  actionable user assertions, and editing assertion metadata is outside the raw
  session content-hash boundary.
- `index.db` may be rebuilt from `source.db` when schema versions change.
- `embeddings.db` may be rebuilt, but restore it when possible to avoid
  provider cost and delay.
- `ops.db` does not decide archive correctness; restore it only when preserving
  daemon history matters.
- A restored blob store is valid only when referenced blobs still match their
  SHA-256 paths and `source.db`/`user.db` references.
- Blob backup includes the exact union of referenced and publication-reserved
  bytes. `blob-inventory.json` records every hash and size; verification
  re-hashes restored bytes rather than accepting an equal file count.

## Blob GC Safety Boundary

Blob garbage collection is dry-run-first work. A safe GC report must prove:

- the candidate blob is not referenced by `source.db.raw_sessions`;
- the candidate has neither a durable reference nor a publication reservation;
- the candidate is older than the generation/age defense-in-depth gate
  (`MIN_AGE_S`; see `docs/internals.md` "GC concurrency model");
- the report names exact candidate counts and references before deletion.

Do not delete blobs based only on filesystem age, directory mtime, or absence
from `index.db`. `source.db` is the authority for raw evidence reachability.

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
sqlite3 /realm/db/polylogue/user.db "SELECT count(*) FROM assertions;"
sqlite3 /realm/db/polylogue/source.db "SELECT count(*) FROM raw_sessions;"
```

**Negative control (deliberately corrupted restore must fail loudly)** —
flip a few bytes past the SQLite header and confirm `integrity_check` reports
corruption, not `ok`:

```bash
cp <restored>/user.db /tmp/corrupt-test.db
python3 -c "
with open('/tmp/corrupt-test.db', 'r+b') as f:
    f.seek(4096); d = f.read(64); f.seek(4096)
    f.write(bytes(b ^ 0xFF for b in d))
"
sqlite3 /tmp/corrupt-test.db "PRAGMA integrity_check;"   # must report errors, exit 11
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

**CRITICAL FINDING — the live durable tier currently has NO Borg coverage.**
`/realm/db/polylogue` (where `source.db`/`user.db` actually live; `/realm/data/captures/polylogue/*.db`
are symlinks to it) was converted to its own nested Btrfs subvolume on
2026-07-06 (`btrfs subvolume list /realm` shows `ID 3862 ... path db/polylogue`).
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
