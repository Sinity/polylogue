# Codex review guide

Guide version: v1. End every finding you post with the line `Review-guide: v1`.

You review a single-writer archive. Its durable tiers (`source.db`, `user.db`,
`audit.db`) are irreplaceable; its derived tiers are rebuilt from them. The
costly failures are data silently lost or duplicated, a durable reference that
re-points, a build that never converges, and work reported as done when it was
skipped. Weight your attention there.

## Writing a finding

- Give the concrete input (file shape, member, sequence, interruption point)
  and the wrong observable outcome at the reviewed head. Name the function
  that decides it.
- If the defect sits in a decision that another site also makes, name the
  other site and the single owner that should hold the decision. For example,
  `classify_pre_acquisition` in `sources/live/batch_support.py` owns
  pre-acquisition exclusion for both intake and the cold-build baseline.
- If the changed file is in the derived-identity closure
  (`devtools schema closure <file>`), say so: the fix must land before a
  rebuild starts. Do not ask for a version bump or a reparse declaration; the
  identity is computed from the AST closure.

## Severity

P1 (merge blocker):

- durable-tier data loss or corruption, or a durable reference that can
  re-point (message and block identity, the `pipeline/ids.py` hash partition);
- input silently dropped, truncated, or duplicated while the run reports
  success or complete enumeration;
- a writable open or mutation that bypasses the daemon writer route;
- a build, promotion, or convergence loop that can never finish (livelock,
  permanent exclusion of a transient fault, a restart that cannot resume);
- a change that leaves a consumer in the repository broken (see
  "Compatibility and complete changes");
- a compatibility path that keeps two authorities alive;
- a surface that decides `ok` without the rows or measurements that justify
  it.

P2: devtools and test-harness defects, status, progress and metric accuracy,
docs drift, performance that does not grow with archive size, and test
weakness where the gate still runs. A test-only or docs-only diff is P2 unless
it makes a required gate vacuous.

## Checks that are easy to miss (apply to every diff)

Area-specific rules live in the `## Code Review Rules` section of the nested
`AGENTS.md` beside the code they govern (`polylogue/storage/`,
`polylogue/sources/`, `polylogue/daemon/`, `devtools/`). The checks below
apply everywhere.

1. **Accounting of skips.** For any loop, filter, or early `continue` or
   `return` over inputs (files, ZIP members, records, events), each skipped
   item must leave a typed disposition: a refusal, an exclusion reason, or a
   retryable fault. A skip that is not recorded while the caller records
   enumeration as complete is P1. Check fallbacks, such as an unknown provider
   or origin selecting a narrower filter.
2. **Transient versus permanent.** A read fault (EACCES, `SQLITE_BUSY`,
   `SQLITE_CANTOPEN`, a missing root, a partial write) stays retryable. It must
   not be recorded as "not ours", excluded, or converged. Look for
   `except sqlite3.Error` or `except OSError` that returns a negative answer.
3. **One decision, one owner.** If the diff adds a predicate that decides the
   same thing as an existing one (what is a session, what is excluded, what is
   retained, what is converged), flag the divergence and name the owner. Two
   classifiers that can disagree are a defect even when they agree on today's
   fixtures.
4. **Scale.** Flag per-chunk, per-event, or per-open work that costs
   O(archive): an unfiltered `fetchall`, a full-table scan, a six-tier
   bootstrap, an fsync or FULL-synchronous commit per event, re-reading or
   re-hashing the same file. The defect is cost that grows with the archive
   while the unit of work stays fixed.
5. **Removed symbols.** When the diff removes or renames a function,
   attribute, or keyword, check the tests and stubs that name it
   (`monkeypatch.setattr` targets, fakes with fixed signatures).
6. **Outcome.** Row-bearing operations decide one `outcome` in
   `surfaces/outcome.py`. `ok` over zero returned rows, over an unmeasured
   component, or while convergence is incomplete is wrong (`degraded` or
   `empty`). Exit codes follow the outcome.

## Compatibility and complete changes

No compatibility shims means every change is done completely, not that
breakage is acceptable.

- **A real finding.** A change that alters or removes an interface, command,
  config key, schema, route, or file format without updating every consumer in
  the same change. Consumers include callers, CLI and MCP surfaces, tests,
  docs, generated references, configs, and hooks. Name the consumer that was
  not updated. It is P1 when a caller in the repository is broken.
- **The required remedy.** Update every consumer and delete the predecessor in
  the same change.
- **Also a defect.** Any compatibility path present in the diff: a shim, a
  legacy alias, a fallback to old behaviour, a dual read or dual write, a
  deprecated wrapper kept for callers, or migration or carry-forward of prior
  archive state. It is P1 when it keeps two authorities alive (old and new
  readers, keys, or routes); otherwise P2.
- **Noise.** A finding whose remedy is to keep the old path alongside the new
  one (a shim, alias, fallback, dual read or write, or deprecated wrapper). A
  finding that asks to migrate, upgrade, or carry forward prior archive state,
  receipts, cursors, or generations, or to read a previous format. The archive
  starts fresh with every tier at `user_version=1`; earlier state is inert
  salvage evidence (AGENTS.md, "Storage tiers"). Do not post these.

## Limits

A limit that refuses or truncates valid input is a defect. Bounding the work
per call (paging, streaming, a cursor) is the correct shape; a cap on what can
be accepted is not. Do not recommend new caps, shorter timeouts on work that is
making progress, or truncating retained text. A timeout that abandons and
restarts in-progress work is a livelock, and that is the defect. Only a real
physical limit is acceptable, and its refusal is typed.

## Do not flag

- Requests to keep old behaviour beside new behaviour, or to migrate or
  upgrade prior archive state (see above).
- Requests to bump schema, parser, or materializer versions, or to declare a
  reparse.
- New caps, smaller timeouts, or truncation as a remedy.
- Test strictness beyond the anti-vacuity condition the test names.
- Scenarios that need the environment corrupted below its own integrity
  contract (lockfile, provision stamp, environment digest).
- Publication text, PR bodies, task metadata, and the changelog.
- A finding already answered by a commit or a stated refutation, unless the
  answer is wrong.
