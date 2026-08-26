# CLI interaction oracle deletion ledger

This ledger records the consolidation for `polylogue-07o1e`. It is a test
plane inventory, not a product UX approval record.

## Owners

| Concern | Canonical owner | Retained compatibility surface |
| --- | --- | --- |
| Click command paths and capability cells | `tests/infra/cli_interaction.py` | `polylogue/cli/command_inventory.py` remains the product tree walker |
| Query/argv/property inputs | `tests/infra/strategies/cli.py` | Existing mutation/output strategies remain because they cover different contracts |
| Real process and PTY capture | `tests/infra/pty_cli.py` | Existing snapshots use the same runner |
| PTY event schedules | `tests/infra/pty_scenarios.py` | No second process runner is introduced |
| Terminal layout/accessibility semantics | `tests/infra/terminal_cells.py` | VHS tapes remain bounded human exemplars only |
| Visual exemplars | `devtools/visual_vhs.py` | `devtools/render_visual_tapes.py` remains the single generated-tape command |
| CLI/UDS performance workloads | `tests/benchmarks/cli_profile.py` | Existing benchmark tests are route-specific consumers |
| Managed benchmark entrypoint | `devtools/cli_interaction_profile.py` | `devtools bench daemon-operation` remains a narrower compatibility command |

No screenshot approval store, model-results database, parallel scenario
registry, or second snapshot authority was added. The existing byte snapshots
remain only for protocol/error exemplars; semantic layout laws use normalized
cells.

## LOC accounting

Measured from the feature diff after generated documentation was refreshed:

| disposition | lines |
| --- | ---: |
| gross deleted | 32 |
| relocated | 0 |
| added | 1127 |
| net maintained | 1095 |

The additions are the reusable oracle primitives, their self-tests, the named
profile manifest, and this ledger/index entry. The 32 deleted lines remove
duplicated shell-set and benchmark wiring; existing PTY/VHS/snapshot helpers
were shared in place rather than copied. A future CLI slice should consume
these owners and update this ledger when it retires a legacy helper.

## Retired or shared routes

- Shell names now come from `tests/infra/cli_interaction.py`; the completion
  tests only supply Click shell classes.
- Cold CLI and daemon route measurements share `cli_profile.py` metric names
  and the `bench cli-interaction` managed entrypoint.
- `pty_cli.py` remains the sole subprocess/PTY implementation; eventful tests
  use its event hook rather than a sibling runner.
- VHS is not a layout oracle. It produces three bounded exemplars, while
  `terminal_cells.py` owns width, wrap, clipping, focus, hyperlinks,
  control-safety, and non-color meaning.
