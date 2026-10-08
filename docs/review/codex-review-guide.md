# Codex review guide

You review a single-writer archive. Its durable tiers (`source.db`, `user.db`,
`audit.db`) are irreplaceable. `index.db` and `ops.db` are rebuilt from them;
`embeddings.db` is expensive to rebuild because vectors are purchased again and
no replay route replaces it, so a change that deletes or replaces it is a
destructive change. The
costly failures are data silently lost or duplicated, a durable reference that
re-points, a build that never converges, and work reported as done when it was
skipped. Weight your attention there.

## Review the changed behavior

Review the changed delta and the contracts it affects. Report concrete
findings together, including relevant sibling sites. Reuse prior review of
unchanged code rather than restarting a full review after each small fix.
Scale depth to consequence: durable data loss, destructive operations and
purchased-vector preservation warrant stronger scrutiny than ordinary edits.
Known failures and untested areas stay explicit; a perfect unrelated suite
is not a prerequisite for a routine fix. Required hosted reviews remain in
force.

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
  rebuild starts. Do not ask to bump the derived identity or declare a
  reparse for it; that identity is computed from the AST closure. A manual
  invalidation token in the same file (see "Do not flag") still needs its
  bump.

## Severity

P1 (merge blocker):

- durable-tier data loss or corruption, or a durable reference that can
  re-point (message and block identity, the `pipeline/ids.py` hash partition);
- deleting or reinitializing a populated `embeddings.db` outside a declared,
  backup-gated route (its vectors are purchased again);
- input silently dropped, truncated, or duplicated while the run reports
  success or complete enumeration;
- a writable open or mutation of a live archive that bypasses the daemon
  writer route without archive-bound custody or the exact owned offline
  destination authority;
- operator archives, transcripts, private exports, local databases, or
  receipts committed in any diff, including a docs-only or test-only one;
- a build, promotion, or convergence loop that can never finish (livelock,
  permanent exclusion of a transient fault, a restart that cannot resume);
- a change that breaks a caller in the repository (see "Compatibility and
  complete changes"); a missed docs or generated reference alone is P2;
- a compatibility path that keeps two authorities alive;
- a surface that decides `ok` without the rows or measurements that justify
  it.

P2, when no P1 item applies: devtools and test-harness defects, status, progress and metric accuracy,
docs drift, performance that does not grow with archive size, and test
weakness where the gate still runs. A test-only or docs-only diff is P2 unless
it makes a required gate vacuous or crosses the public boundary.

## Invariant reference

Use the relevant items to inspect changed behavior. No item-by-item narration,
15+6 report, or full-list restart is required for each result or small fix.
Area rules live in the `## Code Review Rules` section of
the nested `AGENTS.md` beside the code (`polylogue/storage/`,
`polylogue/sources/`, `polylogue/daemon/`, `devtools/`).

1. **Siblings and owners.** Where else does the rule this diff adds or fixes
   apply: every route, caller, branch, and record it governs (live and
   baseline replay, grouped, ZIP, append and source-only paths, sync and
   async, CLI and MCP; every record rather than the first or a prefix)? If
   the diff adds a predicate that decides something an existing one decides
   (what is a session, excluded, retained, converged, a valid option), flag
   the divergence and name the single owner; two classifiers that can
   disagree are a defect even when they agree on today's fixtures. Example:
   foreign-origin validation added to live ZIP intake but not to baseline
   replay or append deltas; a hand parser of pytest options that misses
   clustered short flags.
2. **The fix itself.** Read each fix in the diff as new code. Does it add a
   defect of its own: a new exception another caller does not catch, a new
   marker another reader matches by substring, new state that a retry or
   restart does not restore? Example: a new refusal type raised in one path
   escapes uncaught through the baseline caller.
3. **Memory and limits.** Does any read, cache, or buffer grow with the input
   (an unfiltered `fetchall`, `list()` of a stream, a whole member or
   transcript decoded at once, a set of every ID, a cache with no byte
   bound)? The remedy streams, pages, or spills; a cap, truncation, or fixed
   timeout on valid input or progressing work is itself a defect ("Limits").
   Example: composing a whole transcript to run one query per message.
4. **Cost per unit.** Does per-page or per-item work grow with the archive:
   OFFSET or growing-cursor paging, a page cut before the filter or sort,
   per-row queries, a quadratic loop, re-reading or re-hashing the same
   bytes, a six-tier bootstrap or FULL-synchronous commit per event? The
   defect is cost that grows with the archive while the unit of work stays
   fixed. Are pages read from one snapshot? Example: a correlated prefix
   count per row where one linear renumbering pass suffices.
5. **Interruption.** For each loop, wait, and multi-step state change,
   wherever it lives (daemon, operations, storage): is cancellation checked
   inside it, and what does a cancel, deadline, kill, or restart between
   steps leave behind (a stage marked skipped or converged that never ran, a
   cursor advanced before its commit, a session committed before its cursor,
   a claim never released, a resumed candidate treated as fresh, a request
   that looks settled although its work never ran)? Example: reconciling
   prepared sessions without polling cancellation.
6. **Ordering and races.** Between check and use, can a writer, a file
   change, or a concurrent task change what was checked? Is publication
   atomic, are locks taken in one order, and are compared times from one
   clock? Example: validating a path, then parsing bytes re-read from it.
7. **Failure classification.** A read fault (EACCES, `SQLITE_BUSY`,
   `SQLITE_CANTOPEN`, a root that vanishes mid-read, a partial write) stays
   retryable; it is never recorded as "not ours", excluded, or converged
   (look for `except sqlite3.Error` or `except OSError` that returns a
   negative answer). A deterministic failure is not retried forever. A typed
   refusal reaches every caller and is recorded, not swallowed by a broad
   `except` or matched by substring. An optional source whose root is not
   installed is a declared exclusion (`absent_root`), not a fault.
8. **Silent outcomes.** In an enumeration or admission loop whose caller
   claims complete processing, does every skipped item leave a typed
   disposition (refusal, exclusion reason, retryable fault)? A helper that
   selects from data it leaves intact is not covered. An unrecorded skip
   while the caller records enumeration as complete is P1. Check fallbacks,
   such as an unknown provider or origin selecting a narrower filter.
   Row-bearing operations decide one `outcome` in `surfaces/outcome.py`:
   `ok` over zero rows, an unmeasured component, or incomplete convergence
   is wrong, and exit codes follow the outcome. Metrics, counts, and receipts
   describe what ran.
9. **Cleanup on failure.** When a step fails or refuses after acquiring
   something (a blob, reservation, claim, scratch file, debt row, cache
   entry), is it released or rolled back on every exit path? Example: a blob
   published before member validation refuses it.
10. **Identity and keys.** Does each hash, ID, and cache or receipt key cover
    exactly its declared inputs, stay stable under insertion, deletion, and
    reordering, stay NFC-, surrogate- and BOM-safe, and avoid collisions
    across namespaces? A reverse lookup from Origin to Provider that picks
    one of several matches without independent evidence is wrong (the AI
    Studio and Drive mapping is non-injective; refuse, or use a declared hint
    such as `family_hint` of `provider_from_origin` in `core/sources.py`).
    Example: a fallback ID keyed by row position that renames on deletion.
11. **Lossless transforms.** Does every copy, retry, replay, rerun, or
    regeneration keep the fields and state it does not mean to change?
    Example: rerunning failed tests drops caller-loaded plugins; a copied
    prefix loses its variant coordinates.
12. **Degenerate inputs.** Does each input handle zero, empty, one, absent,
    non-finite, and oversized values? Example: a sample that selects zero
    files crashes; a non-positive worker count spins.
13. **Consumers.** Is every consumer of a changed interface, command, config
    key, schema, route, or file format updated, and the predecessor deleted
    ("Compatibility and complete changes")?
14. **Tests and stubs.** Does each test drive the production route, stay
    isolated from host state, and fail when the behaviour it names is
    removed? When the diff removes or renames a function, attribute, or
    keyword, do the tests and stubs that name it (`monkeypatch.setattr`
    targets, fakes with fixed signatures) follow? Example: a test patching a
    helper the receiver no longer calls.
15. **Trust boundaries.** Do secrets stay out of argv, logs, and payloads;
    are files created owner-only where they hold private data; is each
    authentication or ownership check made on the value that is used?

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
  deprecated wrapper kept for callers, or migration or carry-forward of state
  from before the fresh-start reset. An additive numbered durable-tier
  migration for a schema change made after the reset is ordinary evolution,
  not a compatibility path. It is P1 when it keeps two authorities alive (old and new
  readers, keys, or routes); otherwise P2.
- **Noise.** A finding whose remedy is to keep the old path alongside the new
  one (a shim, alias, fallback, dual read or write, or deprecated wrapper). A
  finding that asks to migrate, import, or carry forward state from before the
  fresh-start reset (archives, receipts, cursors, generations) or to read its
  format. The archive starts fresh with every tier at `user_version=1`; the
  earlier state is inert salvage evidence (AGENTS.md, "Storage tiers"). Do
  not post these. This is separate from ordinary durable-tier evolution, which
  uses additive numbered migrations as AGENTS.md describes.

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
- Requests to bump the AST-computed derived identity or to declare a reparse
  for it. This covers only the computed identity. A manually maintained
  version or invalidation token (such as `RESUME_BRIEF_MATERIALIZER_VERSION`
  or `_PARSER_FINGERPRINT` in `sources/live/watcher.py`) that a change
  requires but leaves unchanged is a finding.
- New caps, smaller timeouts, or truncation as a remedy.
- Filesystem enumeration (installed trees, example databases, executables) as
  a cache or receipt key, in devtools or test infrastructure; receipts and
  caches are keyed on declared inputs.
- Test strictness beyond the anti-vacuity condition the test names.
- Scenarios that need the environment corrupted below its own integrity
  contract (lockfile, provision stamp, environment digest).
- Publication text, PR bodies, task metadata, and the changelog.
- A finding already answered by a commit or a stated refutation, unless the
  answer is wrong.
