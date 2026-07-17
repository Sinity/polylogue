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
