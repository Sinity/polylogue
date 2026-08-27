Summary

- Add `devtools verify webui` as the canonical project-level typed WebUI check.
- Keep package scripts as the low-level owner and return a typed green/red/blocked-env result.
- Record the current typed-versus-legacy capability inventory and index it in generated docs.

Problem

The dispatch packet named a WebUI verification route, but the command catalog had no such route. The typed surface also lacked a committed capability ledger, making the remaining legacy-only cells implicit.

Solution

The new command runs `npm run check` from `webui/`, supports JSON output, and propagates package failures. The capability matrix records each independently valuable browser capability and identifies the remaining legacy-only routes without claiming parity.

Verification

- `uv run devtools test tests/unit/devtools/test_webui_package_scripts.py` — 3 passed.
- `uv run devtools render all --check` — all generated surfaces sync.
- `uv run devtools verify oracle-integrity` — 1159 scanned, 0 invalid reachability, 30 baselined.
- `uv run devtools verify --quick` — all quick steps passed.
- `uv run devtools verify webui --json` — reached WebUI tests but remains red on the inherited public-origin count mismatch.
- Focused daemon tests — 161 passed, 1 inherited failure for missing `/api/demo/augment` route metadata.

Residual risk

This lane does not complete the product cutover. Topology/provenance, attachments/pastes, assertions/selection, and compare/similar/workspace remain legacy-only in the matrix; legacy modules and tests are intentionally retained pending their typed replacements. The WebUI check currently runs under Node 24.18.1 although the package declares Node 22.
