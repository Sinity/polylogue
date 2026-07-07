# Polylogue Archive Inventory — 2026-06-30

Purpose: prevent future agents from confusing historical/stale archive roots
with the current target.

## Active Runtime

- Prod `polylogued.service`: inactive during this cleanup.
- Repo-local `polylogued-devloop.service`: active, running
  `/realm/project/polylogue/.venv/bin/polylogued run`.
- Active XDG archive root: `/home/sinity/.local/share/polylogue`.

Latest probe during catch-up:

| root | tier | schema | count |
| --- | --- | ---: | ---: |
| `/home/sinity/.local/share/polylogue` | `index.db sessions` | 18 | 2,722 |
| `/home/sinity/.local/share/polylogue` | `index.db messages` | 18 | 263,697 |
| `/home/sinity/.local/share/polylogue` | `source.db raw_sessions` | 1 | 2,729 |
| `/home/sinity/.local/share/polylogue` | `ops.db ingest_cursor` | 1 | 1,185 |

Interpretation: this is an active rebuilding/catch-up archive, not the old
steady-state archive.

## Recent Dev Archive

Root: `/realm/tmp/polylogue-dev/archive`

| tier | schema | count |
| --- | ---: | ---: |
| `index.db sessions` | 18 | 4,302 |
| `index.db messages` | 18 | 1,380,539 |
| `source.db raw_sessions` | 1 | 4,304 |
| `ops.db ingest_cursor` | 1 | 4,110 |

Exported devloop transcripts prove this same path previously held a v16 archive
with 12,991 sessions / 4.1M messages. That state was real, but it appears to
have been overwritten by later reset/rebuild/schema work.

## Historical Backups

Useful for archaeology, not current runtime targets:

| root | schema | sessions | messages | note |
| --- | ---: | ---: | ---: | --- |
| `/home/sinity/.local/share/polylogue/archive-db-backups/polylogue-archive-20260626T004605Z` | 8 | 16,308 | 5,689,470 | pre-current schema |
| `/realm/tmp/polylogue-replay-probe-index-v8-raw-replay-20260625T053604Z-forcefalse-2` | 8 | 15,553 | 2,940,484 | quarantined from `/realm/tmp` |
| `/home/sinity/.local/share/polylogue/archive-db-backups/index-v7-pre-v8-20260624T195918Z` | 7 | 16,371 | 4,404,075 | pre-current schema |

## Snapshot Availability

Btrfs snapshots under `/realm/.btrfs/snapshot` only cover 2026-06-30 01:45 and
later, too late to recover the earlier v16 12,991-session dev archive if it was
overwritten before then.

## Obsolete DB Quarantine

Old schema-v8 replay probes and June 25 full-ingest benchmark DB directories
were moved from `/realm/tmp` to:

`/realm/inbox/polylogue-obsolete-db-quarantine/2026-06-30`

The old 19 GiB `/realm/tmp/polylogue-db-repair-20260622T120023` workspace was
also moved there. Quarantine size after this pass: 68 GiB. Nothing was deleted.
The goal is to prevent stale DB roots from being mistaken for live archives.
