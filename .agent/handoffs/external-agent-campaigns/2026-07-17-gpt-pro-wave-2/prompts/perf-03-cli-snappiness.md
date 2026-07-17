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


---

## Context and authority

You are a long-running ChatGPT Pro engineering worker. A recent Polylogue
project-state archive will be attached. Retrieve and inspect it broadly; do not
assume attachment bytes consume your active prompt context. The attached
snapshot is the code authority. This prompt defines your mission. Repository
instructions and complete relevant Beads records define constraints and intent;
later Beads notes may supersede older descriptions. Current source wins when a
stale plan names paths or APIs that no longer exist.

Start by reporting the snapshot commit/branch/dirty-patch identity you found and
the source, tests, Beads, and history you inspected. Follow dependencies beyond
the obvious files when they affect the production route. Do not invent an API,
test helper, product contract, or parallel framework to make the task easy.

## Working contract

- Produce the largest internally coherent implementation draft that fits the
  mission. Prefer one real end-to-end behavior over disconnected scaffolding.
- Preserve Polylogue's substrate-first architecture and existing typed
  interfaces. Small production seams are allowed only when real production
  behavior needs observation or control.
- Write concrete production changes and real-route tests. A test must name the
  production dependency it exercises and the representative implementation
  mutation/removal that should make it fail.
- Do not delete existing tests or helpers. Identify proposed dominated
  deletions separately for independent local certification.
- Use your container and run meaningful self-contained checks when possible.
  Never claim access to the operator's live daemon, browser, archive, secrets,
  NixOS deployment, or current worktree. Mark those checks `unverified`.
- If the full scope is unsafe, complete the strongest coherent subset and make
  the remaining decisions and exact continuation steps explicit. Do not return
  placeholders, ellipses, pseudocode presented as code, or a generic plan in
  place of implementation.

## Deliverable

Create the exact `Result ZIP` named near the top of this prompt under
`/mnt/data/`. Do not include the supplied repository/project-state archive or
other copied inputs in the result. The finished ZIP must be attached to the
conversation through a working, user-clickable download link. Work left only
in an internal shell directory, temporary notebook, scattered sandbox files,
or prose is not delivered.

The ZIP must contain:

- `HANDOFF.md`: mission, snapshot identity, inspected evidence, mechanism,
  decisions, changed files, acceptance matrix, apply order, risks, and exact
  verification performed/remaining;
- `PATCH.diff`: one apply-ready unified diff against the named snapshot;
- `TESTS.md`: test design, production dependencies, anti-vacuity mutation,
  commands, and honest execution results;
- `EVIDENCE.md`: relevant source/Bead/history findings and any contradictions;
- `FILES/`: complete replacements only where they materially disambiguate the
  patch; omit it when unnecessary.

Before answering, reopen the ZIP, list and validate its members, compute its
SHA-256 and byte size, and confirm that `PATCH.diff` has no placeholders or
copied source snapshot. Your final chat response must begin with a substantive
operator-readable report of what you did and why. It must also state important
limitations, missing or unverified work, and how much additional value another
iteration could plausibly add—distinguishing a small repair from a substantial
second pass. Then report verification and risks and give a prominent working
link to the exact `/mnt/data/` ZIP. A bare download receipt is not acceptable.

## Continuation protocol

Do not perform a separate adversarial review unless the user explicitly asks
for one. If the user asks to **iterate** or **continue**, preserve valid prior
work, perform the highest-value remaining implementation/research pass, and
publish a new cohesive package revision with the same complete structure—not a
loose supplemental patch. Explain exactly what changed, what improved, what
still remains, and whether another iteration is likely to pay off.

If the user explicitly asks for an **adversarial review**, attack your prior
result against the original mission and current attached authority: search for
unsupported claims, invented or stale APIs, missing call sites, composition
failures, unsafe assumptions, vacuous tests, patch/apply defects, incomplete
acceptance criteria, and evidence that would falsify the design. Preserve work
that survives. Then repair every legitimate finding you can, regenerate the
entire cohesive package as the next revision, and report findings, repairs,
remaining disputes, and the value of another adversarial/implementation pass.
