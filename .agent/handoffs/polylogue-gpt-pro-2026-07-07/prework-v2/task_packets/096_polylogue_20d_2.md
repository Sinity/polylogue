# 096. polylogue-20d.2 — Defer heavy imports off the CLI startup path

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

~2s import tax per invocation; also the residual cold cost when the daemon path is absent. Candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive. Measure first: python -X importtime -c 'from polylogue.cli.click_app import main'. Covers the old help-latency and find-select-cold items; add the help-latency devtools budget check as the regression gate.

## Existing design note

Measure first: python -X importtime -c 'from polylogue.cli.click_app import main' 2>&1 | sort -t'|' -k2 -rn | head -30. Known heavy candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive, storage imports pulled at command-module import time. Mechanics: the repo already uses lazy Click commands (see bd memory: lazy cmds hide flags — use cmd.get_params(ctx) in tests); push heavy imports inside command bodies / module __getattr__; keep a leaf path-resolution module import-light for the daemon fast-path handshake. Regression gate: a devtools help-latency budget check (targeted `polylogue <cmd> --help` under a fixed budget) so drift fails loudly. Prior evidence: nested help 5-9s (import/reset/maintenance archive-read/analyze tools); warm find-select ~1.7s vs cold spikes.

## Acceptance criteria

- `python -X importtime -c 'from polylogue.cli.click_app import main'` shows surfaces/payloads and api/archive no longer imported on the `polylogue <cmd> --help` path. Verify: importtime diff before/after.
- A new devtools help-latency budget check runs targeted `polylogue <cmd> --help` invocations under a fixed budget (e.g. <700ms cold, citing the 20d.14 cold-CLI budget) and fails loudly on drift.
- Nested helps (import / reset / maintenance archive-read / analyze tools) drop from the observed 5-9s to under the budget. Verify: measured before/after under the new budget check.

## Static mechanism / likely defect

Issue description localizes the mechanism: ~2s import tax per invocation; also the residual cold cost when the daemon path is absent. Candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive. Measure first: python -X importtime -c 'from polylogue.cli.click_app import main'. Covers the old help-latency and find-select-cold items; add the help-latency devtools budget check as the regression gate. Design direction: Measure first: python -X importtime -c 'from polylogue.cli.click_app import main' 2>&1 | sort -t'|' -k2 -rn | head -30. Known heavy candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive, storage imports pulled at command-module import time. Mechanics: the repo already uses lazy Click commands (see bd memory: lazy cmds hide flags — use cmd.get_params(ctx) in tests); push heavy impor…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.

## Implementation plan

1. Measure first: python -X importtime -c 'from polylogue.cli.click_app import main' 2>&1 | sort -t'|' -k2 -rn | head -30.
2. Known heavy candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive, storage imports pulled at command-module import time.
3. Mechanics: the repo already uses lazy Click commands (see bd memory: lazy cmds hide flags — use cmd.get_params(ctx) in tests)
4. push heavy imports inside command bodies / module __getattr__
5. keep a leaf path-resolution module import-light for the daemon fast-path handshake.
6. Regression gate: a devtools help-latency budget check (targeted `polylogue <cmd> --help` under a fixed budget) so drift fails loudly.
7. Prior evidence: nested help 5-9s (import/reset/maintenance archive-read/analyze tools)

## Tests to add

- Acceptance proof: `python -X importtime -c 'from polylogue.cli.click_app import main'` shows surfaces/payloads and api/archive no longer imported on the `polylogue <cmd> --help` path.
- Acceptance proof: Verify: importtime diff before/after.
- Acceptance proof: A new devtools help-latency budget check runs targeted `polylogue <cmd> --help` invocations under a fixed budget (e.g.
- Acceptance proof: <700ms cold, citing the 20d.14 cold-CLI budget) and fails loudly on drift.
- Acceptance proof: Nested helps (import / reset / maintenance archive-read / analyze tools) drop from the observed 5-9s to under the budget.
- Acceptance proof: Verify: measured before/after under the new budget check.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
