Summary

Ran the disposable archive rehearsal against `/realm/state/polylogue-rehearsal` without mutating the canonical archive.

Problem

The quiescent copy completed only after excluding disappearing SQLite sidecars. The clone opened with source v30 and index v67, but the copied active-pointer metadata still referred to `/realm/state/polylogue`, and the copied index generation has no `blocks_fts` table. The production preflight and every migration command fail closed on incomplete search state. The full-evidence backup copied all six tier databases and wrote blob evidence, but verification remained in uninterruptible I/O and produced no verification receipt before the run was stopped.

Solution

No product code was changed. The active symlink in the disposable clone was corrected to its local generation for inspection. The failed first copy was retained under `/realm/tmp/work/polylogue-74kj3-copy-failed-20260827`; the interrupted redundant rsync was retained under `/realm/tmp/work/polylogue-74kj3-rsync-interrupted-20260827`.

Verification

`systemctl --user is-active polylogued.service` -> `inactive`; `systemctl --user is-enabled polylogued.service` -> `masked-runtime`.

`cp -a --reflink=auto /realm/state/polylogue /realm/state/polylogue-rehearsal` -> exit 1 after 7m18.65s: four vanished `source.db`/`user.db` WAL/SHM sidecars.

`nix develop --accept-flake-config --command polylogue ops backup --output-dir /realm/tmp/work/polylogue-74kj3-full-backup --profile full_evidence --verify` -> stopped after tier snapshots and blob evidence, with no verification receipt.

`POLYLOGUE_ARCHIVE_ROOT=/realm/state/polylogue-rehearsal .venv/bin/polylogue ops maintenance rebuild-index --preflight --output-format json` -> exit 1: `Error: Search index is incomplete`.

The same preflight gate blocked source migration, user migration, and audit adoption before SQL.

Residual risk

The rehearsal did not reach durable migrations, audit adoption, derived rebuild, clean source generation, daemon convergence, quiet-window proof, or a valid conservation report. The partial backup and scratch trees require operator cleanup after the report is filed.
