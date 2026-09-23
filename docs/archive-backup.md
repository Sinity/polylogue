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
| `audit.db` | Append-only mutation authority, authorizations, attempts, receipts, and continuity heads. | Always back up. Full-evidence backups require it. |
| `ops.db` | Daemon cursors, attempts, convergence debt, stage events, and operational telemetry. | Disposable for ordinary restore profiles, but required by the exact full-evidence tier contract. |
| `source-declared-absent.json` | Operator-authored declared-absent blob hashes for a pre-generation `source.db`. | Copy with `source.db`; never derive or replace it with GC observations. |
| `blob/` | Content-addressed binary payloads keyed by SHA-256. | Back up referenced blobs with `source.db`/`user.db`; do not prune by age alone. |

`polylogue ops maintenance archive-plan --output-format json` is the machine-readable
inventory for tier filenames, expected versions, backup-required tiers, and
missing blockers. Run it before backup automation rather than hard-coding only
the files that happen to exist locally.

## Backup Profiles

Use these profiles when choosing what to copy:

| Profile | Include | Exclude | Use case |
| --- | --- | --- | --- |
| Full evidence | All six archive tiers: `source.db`, `index.db`, `embeddings.db`, `user.db`, `ops.db`, and `audit.db`, plus referenced `blob/`. | Temporary SQLite `*-wal`/`*-shm` only after a clean checkpoint. | The fastest restore with raw evidence, read models, vectors, overlays, audit authority, and operational state. |
| User overlays | `user.db` and any assertion/note evidence blobs referenced by user-owned rows. | `index.db`, `ops.db`, rebuildable search/derived models. | Protect irreplaceable human/agent state before resets or schema rebuilds. |
| Rebuildable-cache exclude | `source.db`, `user.db`, referenced `blob/`, optionally `embeddings.db`. | `index.db`, `ops.db`, derived/cache artifacts. | Small backup that can rebuild parsed/indexed data locally. |
| Diagnostics bundle | `ops.db`, `archive-plan` JSON, `daemon-workload-probe` JSON, logs, and readonly status outputs. | Private raw blobs unless explicitly needed for the incident. | Bug reports and incident triage without over-sharing archive contents. |

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
missing index attachment before using that profile for audit
adoption.

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

## Runtime Pin for a Restored Archive

A restored file set is not readable on its own. Durable tiers carry a
`PRAGMA user_version` and derived tiers carry a stamped schema identity; a
runtime that matches neither refuses to open the archive instead of patching
it. The complete rollback artifact is therefore the archive files plus the
commit that can read them. Pinning that commit costs no write to the archive,
while migrating the tiers forward mutates the copy being kept as the fallback.

Read the versions the commit has to match out of the archive itself:

```bash
for tier in source user audit; do
  printf '%s ' "$tier"
  sqlite3 "file:$POLYLOGUE_ARCHIVE_ROOT/$tier.db?mode=ro" 'PRAGMA user_version;'
done
sqlite3 "file:$(readlink -f "$POLYLOGUE_ARCHIVE_ROOT/index.db")?mode=ro" 'PRAGMA user_version;'
```

The candidate is the newest first-parent `master` commit whose
`ARCHIVE_VERSION_BY_TIER` in
`polylogue/storage/sqlite/archive_tiers/__init__.py` maps every tier to those
numbers. This is the single version authority; the durable tier modules do not
declare parallel `SOURCE_SCHEMA_VERSION`, `USER_SCHEMA_VERSION`, or
`AUDIT_SCHEMA_VERSION` constants. Tiers migrate on independent schedules, so
an archive whose durable tiers were migrated at different times may have no
commit that matches every tier. Pin on the tiers the restore has to read, and
record which tier is left unopenable and what that costs.

**This rule searches history, and history has a discontinuity.** The durable
tier lineage was reset to `ARCHIVE_FORMAT_FLOOR_VERSION = 1`, while the
current authority maps source to `SOURCE_TIER_VERSION = 4`, user to
`USER_TIER_VERSION = 2`, and audit to `AUDIT_TIER_VERSION = 2`.
An archive carrying pre-reset numbers therefore has **no post-reset commit that
matches it**, and the rule above resolves only into pre-reset history. Read the
numbers out of the archive first and check which side of the reset they are on
before searching; a search that returns nothing is the expected answer for a
pre-reset archive, not a missing commit.

Build the candidate in its own checkout and confirm the executable names it:

```bash
uv sync --frozen
./.venv/bin/polylogue --version   # 0.3.0+<short sha>
```

The build hook requires git metadata. A tree exported without `.git` builds
only when `polylogue/_build_info.py` is present, carrying `BUILD_COMMIT` and
`BUILD_DIRTY`.

Verify through the production read route. Opening the tier files with `sqlite3`
proves the bytes are intact and says nothing about whether the runtime accepts
the archive:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/restored/archive/root
polylogue ops maintenance archive-plan --output-format json  # expected vs found user_version, per tier
polylogue status                                             # each tier reports vN/N ok
polylogue --origin ORIGIN find 'FIELD:VALUE' then select --format json
```

Expect text search to be unavailable. An archive stopped before its search
index converged refuses FTS queries with `Search index is incomplete` while
field, origin and date filters answer normally, and the index returns only
after `polylogued run` converges it. Report field-query readiness and search
availability separately rather than as one readiness claim.

Reading through the pinned runtime writes nothing durable; only `ops.db`, the
disposable tier, is touched. When the archive is the only copy, capture a
size/mtime/ctime/sha256 manifest of the durable tiers before and after the read
and compare the two, rather than assuming the read was clean.

## Pre-wipe rollback packet

The rollback packet for the 2026-09-12 fresh-restart campaign is a public pointer to private operator-held artifacts. It contains no archive bytes, transcripts, assertions, or blob payloads in Git. The executable pin is first-parent commit `c8ba64157ea1a8eeed175c9a80229aedfa818820` (2026-08-10, `fix(replay): block readiness on incomplete parser census (#3903)`). Keep that checkout available with the aside archive root, all six tier files, and the referenced `blob/` directory.

Three facts about that pin, each rechecked 2026-09-22:

* **The pin is an ancestor of `master` and nothing points at it.** `git cat-file -t` resolves it and `git merge-base --is-ancestor <pin> origin/master` succeeds, but `git tag --points-at <pin>` is empty. The pin lives only in this prose and in reachability from `master`; it survives a prune because `master` contains it, not because anything names it. A named ref would make it independently discoverable.
* **The audit tier is outside the four-tier match.** The pin's window is source 30, user 10, index 67, embeddings 4, ops 1. Audit is not in that list: the pin declares audit version 1 while a live archive at that window carries `user_version = 2`. A rollback checkout at this pin therefore faces audit-tier skew, and what it costs must be reported rather than assumed away.
* **Reproducing the pin from a live pre-reset archive no longer works**, for the floor-reset reason recorded above. Take the pin as recorded here; do not expect to re-derive it.

The prior rollback drill also exported 107 user assertions (18 columns) and
verified a canonical row-list SHA-256 round trip. That export is private
operator evidence, not a Git artifact and not a substitute for retaining
`user.db`; recheck its count and digest against the aside snapshot before
authorizing a restore.

Before a wipe, the operator must recheck the private packet and record its exact archive-root path, commit, executable version, and a size/mtime/ctime/sha256 manifest outside Git. The packet is ready only when all six files are present: `source.db`, `index.db`, `embeddings.db`, `user.db`, `audit.db`, and `ops.db`, together with the referenced blobs. The independently exported user assertions file is additional evidence, not a replacement for `user.db`.

### Assembling a packet is not the only route

The 2026-09-20 operator simplification: for the fresh restart the operation is
**stop the daemon, move `/realm/state/polylogue` aside as one intact root,
start the daemon, and let it converge into the recreated root**. Everything in
the root moves together, inbox symlinks and referenced blob material included,
and nothing durable is written to obtain the rollback. It therefore cannot fail
halfway by migrating the only copy, and no packet has to be assembled or copied
first. Keep the aside path and the pinned checkout together. The packet
procedure above remains the route for a rollback that must live somewhere other
than the original path.

Two mechanical notes for whoever runs the move:

* The daemon must be stopped first. `.archive-ownership.lock` and live `-shm`/`-wal` files are present while it runs.
* Symlinks inside the root use absolute self-referential targets (`index.db` into `.index-generations/gen-*/`, and the tier links inside that generation pointing back at the root). After a move they still name the original path, which is the path the daemon recreates. Relativize them, or point `POLYLOGUE_ARCHIVE_ROOT` at the aside path when reading it. This only matters if someone actually reads the aside copy.

Use the pinned checkout and the production read route for the proof. A raw SQLite open or a file listing alone does not establish rollback readiness:

```bash
export POLYLOGUE_ARCHIVE_ROOT="<operator-retained rollback root>"
./.venv/bin/polylogue ops maintenance archive-plan --output-format json
./.venv/bin/polylogue status
./.venv/bin/polylogue --origin ORIGIN find 'FIELD:VALUE' then select --format json
```

The rollback pin is expected to provide field, origin, and date queries while FTS remains stale until daemon convergence. Report FTS as degraded rather than claiming complete search readiness. Do not migrate, re-adopt audit receipts, reconcile a reserved migration train, or start a daemon against the only archive as part of this preparation. If the private packet or its recorded production-route receipt cannot be rechecked, stop and leave the wipe unauthorized.

Restore in place. An archive root can be a symlink farm whose `index.db` and
active-generation tier links are absolute, so a file set copied to a different
root resolves back into the old one and `ArchiveLocation` refuses it. Changing
the root goes through the restore route above.

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
  SHA-256 paths and source resolution plus active-index attachment ownership.
- Blob backup includes the exact union of referenced and publication-reserved
  bytes. `blob-inventory.json` records every hash and size;
  `blob-reference-evidence.json` separately records source resolution and
  active-index attachment ownership. Verification re-hashes restored bytes
  and checks the independent attachment evidence rather than accepting an
  equal file count.
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
