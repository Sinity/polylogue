# The Honesty Anti-Demo: Minute-by-Minute Multi-Source Reconstruction

## Claim

REFUSED. "Reconstruct exactly what the operator was doing, minute-by-minute,
on a given day — correlating AI chat sessions with desktop window focus,
shell commands typed outside any captured agent session, and browser tabs
viewed — into one fused timeline" is **not supported** by the current
Polylogue archive.

## Corpus

Polylogue's own schema (`polylogue/storage/sqlite/archive_tiers/index.py`
and sibling tier DDL files) plus the seeded demo corpus. This is a
structural refusal, not a corpus-size limitation — the claim fails on
schema grounds and would fail identically against the full live archive.

## Method

1. Identify what session-to-external-activity correlation actually exists
   in the schema.
2. Search for any table carrying desktop window-focus, raw shell history,
   or browser-tab telemetry across every archive tier.
3. Refuse the claim if neither exists at the required granularity, naming
   the specific gap rather than narrowing the claim quietly.

## Findings

**What exists**: `session_commits` (per-session git commit correlation,
`detection_type` one of `time_window|file_overlap|explicit_ref|
origin_reported`, confidence-scored `0..1`) and `session_repos`
(session-to-repo/branch linkage). Both are real, and both are *session*-
grained — they say "this session touched this repo/commit," never "this
number of shell commands ran between 14:02 and 14:03."

**What does not exist**: no table in any Polylogue tier (source.db,
index.db, embeddings.db, user.db, ops.db) holds desktop window-focus
events, raw shell command history independent of a captured agent session,
or browser tab/navigation telemetry. Confirmed by direct schema grep across
every `archive_tiers/*.py` DDL file (see `run.log`) — zero matches.

**Where this data actually lives**: window-focus (ActivityWatch), shell
history (Atuin), and browser history are captured by a *separate* system
(sinity-lynchpin), not by Polylogue. Polylogue is scoped to AI session
archives; cross-system correlation with Lynchpin's telemetry is a distinct,
undecided product question, not a missing query on data Polylogue already
holds.

## Specimens

See `evidence.ndjson` for the exact grep commands and their (zero-match)
output.

## Counterexamples

None — the claim's negative result *is* the finding. There is no adjacent
easier claim substituted here; the demo does not quietly narrow "minute-by-
minute multi-source" down to "session-to-commit" and declare success.

## Limits

- This refusal is about Polylogue's own architecture as of this commit. If
  a future bead decides to ingest Lynchpin telemetry into Polylogue (or
  build a federated cross-system query layer), this verdict would need to
  be re-run and could change.
- **No existing bead currently owns cross-system (Polylogue+Lynchpin)
  timeline fusion.** This is itself an honest finding: rather than invent a
  plausible-sounding bead reference, this report states plainly that the
  capability gap has no tracked owner yet. The closest adjacent, already-
  scoped primitives are `session_commits`/`session_repos` (session-grained
  git correlation, already shipped) and the general query-objects/result-set
  direction work (`polylogue-rxdo` family) that could eventually be a
  foundation for richer cross-source composition, but neither is a
  reference for "the bead that supplies minute-by-minute cross-system
  reconstruction" because no such bead exists.

## Reproduce

```bash
grep -n "CREATE TABLE" polylogue/storage/sqlite/archive_tiers/index.py | grep -iE "session_commits|session_repos"
grep -n "CREATE TABLE" polylogue/storage/sqlite/archive_tiers/index.py | grep -iE "window|focus|shell_history|browser_tab|activitywatch"
grep -rln "window_focus|shell_history|browser_tab|activitywatch" polylogue/storage/sqlite/archive_tiers/*.py
```

See `run.log` for the exact recorded output.
