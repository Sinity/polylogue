Title: "WebUI v2 workspace: TypeScript + Preact + Vite scaffold with daemon-served SSR, wheel/Nix packaging, and zero CDN"

Result ZIP: `webui-01-workspace-scaffold-r01.zip`

## Mission

Polylogue's current web reader is Python-embedded JavaScript served by the
daemon (`polylogue/daemon/http.py` + `web_shell_*.py` satellites) and is a
declared rewrite boundary: do not polish it. Build the replacement workspace
foundation the ratified direction specifies: a TypeScript + Preact + Vite
workspace where the daemon serves semantic SSR HTML plus typed JSON, and
Preact provides progressive islands (NOT an SPA-only reader).

Deliver, as one coherent implementation draft against the attached snapshot:

1. A `webui/` workspace (or the path current source conventions suggest):
   Vite config, TypeScript strict config, Preact, vitest; pnpm/npm lockfile
   discipline; NO external CDN/network requests at runtime — all assets
   bundled and served by the daemon.
2. The daemon integration seam: how built assets are discovered and served
   (hashed filenames, immutable caching), how SSR HTML is produced for a
   first route (a minimal archive-overview page is enough), and how islands
   hydrate against typed JSON endpoints that already exist. Concrete Python
   changes to `daemon/http.py`-adjacent code kept minimal and clearly seamed.
3. Packaging: built assets shipped inside the wheel/sdist (MANIFEST/包
   data-files route) and a note on the Nix flake packaging step; a build
   command a CI job can run (`npm ci && npm run build` shape); dev-mode story
   (Vite dev server proxying the daemon) documented.
4. Tests: vitest component smoke for the first island; a Python test that the
   daemon serves the built entry and SSR shell on the new route.

## Read-first routing

- Current shell to eventually replace (do not extend it): `polylogue/daemon/
  http.py` (~4.6k LOC, 49 routes) + the `web_shell_*.py` satellites
  (web_shell.py, _reader, _realtime, _selection, _semantic_cards,
  _coordination, _lineage, _attachments, _paste, _provenance, _similar,
  _workspace). Bead `polylogue-kchb` documents the satellite pattern and
  decomposition intent — your serving seam should extend `http.py`'s
  dispatcher idiom, not bypass it.
- Beads (in the snapshot's `.beads/issues.jsonl`): `t46.8` (the six-tool
  MCP cutover running in parallel — web JSON and MCP should converge on the
  QueryTransaction read contract), `4p1` (Query × Projection × Render read
  algebra), `z9gh.9.1` (bounded resumable reads: the continuation/result-ref
  envelope in `polylogue/archive/query/transaction.py`).
- Packaging precedent: how the wheel/sdist currently ships data files
  (check `pyproject.toml` + MANIFEST handling) and the Nix flake outputs.

## Constraints

- Read `docs/architecture.md`, the daemon HTTP source, and Beads `t46.8`
  context enough to keep the read surface aligned with the (concurrently
  landing) six-tool/QueryTransaction read contract — client code should
  consume continuation-based paged JSON, never assume unbounded payloads.
- Do not delete the existing web shell in this job (later verticals replace
  routes as they land); do mark exactly which files the eventual deletion
  covers.
- No secrets, no telemetry, no external fonts/scripts.

## Deliverable emphasis

HANDOFF.md must state: exact tree added, daemon seam diff summary, build/dev
commands, packaging notes, what the next seven vertical jobs (session list,
search, transcript, insights/status, cost, design system, client contracts)
should import from this scaffold, and every decision you made that the
integrator could reasonably overturn.


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
