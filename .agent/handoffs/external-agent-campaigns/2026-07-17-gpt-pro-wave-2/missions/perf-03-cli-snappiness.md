Title: "CLI snappiness: startup profile, lazy import discipline, and daemon-first fast reads"

Result ZIP: `perf-03-cli-snappiness-r01.zip`

## Mission

Make `polylogue` FEEL instant at the shell. Three sub-missions, evidence
first:

1. **Startup profile**: measure and fix cold `polylogue --help`, bare
   command-floor error, and simple read startup. Use `python -X
   importtime` against the snapshot (your container can run this — the CLI
   entry is `polylogue/cli/click_app.py`; install the package from the
   snapshot源). Produce the import-cost table; identify heavyweight imports
   reachable from the root path (pydantic model modules, lark grammar
   compilation, storage engines, insight registries…). Fix with lazy
   patterns the repo already uses (Click lazy subcommand registration
   exists — see `click_command_registration.py`; note the known gotcha that
   lazy commands hide flags from doc tooling unless `cmd.get_params(ctx)`
   is used). Target: root dispatch + help under ~150ms interpreter-time in
   your container; state measured numbers honestly.
2. **Grammar/model warm cost**: if Lark grammar compilation or large
   pydantic schema building lands on every invocation, move it behind
   demand or a versioned on-disk cache (Lark supports serialized grammar
   caching) — with a staleness key (grammar file hash) and a test proving
   cache invalidation on grammar change.
3. **Daemon-first reads**: when `polylogued` is healthy, interactive reads
   should hit the daemon's warm HTTP surface instead of opening cold SQLite
   (page-cache-cold reads on a 38GB archive are the latency floor
   otherwise). Specify + implement detection (existing daemon
   discovery/health probe in the CLI status path — reuse), per-command
   opt-in for the hot path (start with list/read/search/status), timeout +
   silent fallback to direct SQLite, and a `--no-daemon` escape hatch.
   Coordinate vocabulary with the QueryTransaction envelope
   (`archive/query/transaction.py`) — the daemon routes already speak it.

## Constraints

- No behavior changes to command semantics; the strict command floor
  (#1842) stays exactly as is.
- Lazy-import refactors must keep `mypy --strict` green and not break the
  generated CLI reference (`devtools render cli-reference` machinery);
  list every render regeneration needed.
- Measure in-container, label host-dependent numbers `unverified`, and
  ship the measurement script so the integrator can reproduce on the real
  machine.

## Deliverable emphasis

HANDOFF.md: import-cost table before/after, exact lazy-loading changes,
grammar-cache design, the daemon-first read path design (detection,
fallback, per-command coverage), measured numbers with honest container
caveats, and the reproduction script.
