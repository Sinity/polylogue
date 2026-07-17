Title: "Config resolution closure: make the documented 5-layer precedence actually true (fd2s + cxlk + the 9gh1 epic)"

Result ZIP: `misc-01-config-closure-r01.zip`

## Mission

Close the gap between `config.py`'s documented 5-layer resolution
(defaults → site TOML → user TOML → `POLYLOGUE_*` env → CLI, with
provenance) and runtime reality. The dogfood-2 config investigation found
three structurally distinct failures (epic `polylogue-9gh1`; children
`fd2s`, `cxlk`; read all three fully in `.beads/issues.jsonl` — fd2s
carries the decided architecture):

1. **Two parallel config systems** (`fd2s`, P1, severest): the legacy
   env-only `Config`/`get_config()`/`IndexConfig` (+ `paths.py` + direct
   `os.environ` reads) coexists with the newer 5-layer
   `PolylogueConfig`/`load_polylogue_config`. Settings inventoried with a
   `toml_path` (implying full precedence) but resolved through the legacy
   system silently IGNORE site/user TOML — `archive_root` itself is dead
   in TOML for the runtime Config every real consumer uses;
   `voyage_api_key` in TOML never reaches embedding execution.
   Decided fix: `load_polylogue_config` is the SOLE five-layer resolver;
   introduce an immutable `ResolvedRuntimeConfig`/`ResolvedArchivePaths`
   projection at CLI/daemon/MCP/API/maintenance composition roots and
   INJECT it; `Config`/`get_config`/`IndexConfig` become compatibility
   projections of the already-resolved object that may NOT re-read env,
   cwd, `Path.home`, or `polylogue.paths`.
2. **Nested-table merge is full-replace** (`cxlk`, P2): `_merge_toml`
   (~config.py:1216–1228) replaces whole tables — a user-layer
   `[health.convergence_debt] default_error=X` silently drops the site
   layer's `default_warning` and family overrides. Fix: deep-merge for the
   nested-table inventory entries incl. the families sub-dict
   (`subscription_plans` array-of-tables full-replace is CORRECT TOML
   semantics — leave it, per the bead).
3. **Bypassing callers**: ~20-31 files read `POLYLOGUE_*`/env directly.
   Inventory them all (grep), migrate each to the resolved projection, and
   add the regression guard the epic's AC demands: a fixture-driven test
   that would FAIL if any of the three bug shapes recurs for a NEW setting
   (e.g. an inventory-driven test asserting every toml_path setting
   actually responds to a TOML fixture through its real consumer).

Verification of the epic's headline AC: site/user TOML archive-root change
must alter the archive used by daemon, CLI, MCP, API, and maintenance, all
reporting the same resolved tier paths (test via isolated XDG fixtures —
`workspace_env` fixture family in `tests/infra/`).

## Constraints

- A prior external attempt exists: the wave-1 `beads-01` package
  (`polylogue-config-closure.zip`, in `.agent/handoffs/.../beads/results/
  beads-01/` if present in your snapshot) — inspect it, adjudicate what's
  reusable, and say what you took vs rejected. Do not assume it was sound.
- Migration must not change RESOLVED behavior for settings that currently
  work via env (env layer keeps its precedence slot); the change is that
  TOML layers start working — call out any setting whose effective value
  could change for a user relying on current buggy behavior.
- `.claude/settings.json` sets `POLYLOGUE_ARCHIVE_ROOT=/tmp/...` for cloud
  shells — that env override must keep winning over TOML (it's layer 4);
  add it as an explicit test case.
- Keep provenance/redaction reporting working (`polylogue config`-style
  surfaces; find the current provenance renderer and extend to the
  resolved projection).

## Deliverable emphasis

HANDOFF.md: the composition-root injection map (every root, what it
receives), the bypass-caller inventory table (file → setting → migration),
deep-merge semantics, behavior-change census (settings whose effective
value changes), beads-01 adjudication verdict, and the regression-guard
design.
