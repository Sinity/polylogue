# Lane D raw-authority restart proof — tests

## Test design

The new test module is `tests/unit/devtools/test_raw_authority_restart_proof.py`. It exercises production repair behavior rather than a parallel state machine.

| Test | Production dependencies exercised | Anti-vacuity property |
| --- | --- | --- |
| `test_raw_authority_restart_proof_reaches_conserved_two_census_fixed_point` | Active archive bootstrap, `ArchiveStore.write_raw_payload`, production parser census, candidate closure, component ordering, immutable plan publication, application writer, per-plan outcome commit, finalization, interrupted recovery, durable census reader, application-receipt validator, blocker ledger, and two-census fixed-point rule | Fails if any boundary is not reached, plan IDs/order/components drift, an outcome is lost/duplicated, an executed receipt is invalid, a blocker or planned census remains, terminal partition changes, or the second fixed-point receipt is absent |
| `test_raw_authority_restart_proof_rejects_broken_ledger_conservation` | Durable `source.db` census-plan algebra and `_audit_case` | Mutates one finalized selected outcome from recorded to unrecorded; the audit must reject it |
| `test_raw_authority_restart_proof_rejects_postcondition_mutation_after_crash` | Production exact-postcondition recovery, stale-plan rejection, durable blocker creation, and fail-closed repair result | Deletes one membership witness after application but before outcome commit; recovery must record `rejected_stale`, open a blocker, and prevent convergence |
| `test_raw_authority_restart_proof_cli_and_catalog` | CLI argument handling and central command catalog | Fails if the command is not registered with the intended module/examples or CLI propagation changes |

Representative implementation mutations/removals that should make the successful proof fail include:

- Remove or bypass `recover_interrupted_raw_authority_censuses`: the before-commit and resumed-batch cases leave planned censuses or duplicate/retry the wrong plans.
- Treat `parsed_at_ms` or aggregate counts as execution proof instead of validating the application receipt: the postcondition-mutation test would cease to create a blocker, which is a failure.
- Remove `outcome_recorded = 0` from the guarded outcome update or allow duplicate updates: the exactly-once census algebra or terminal endpoint count fails.
- Let `finalize_raw_authority_census` publish with pending selected outcomes: the before-commit boundary no longer remains recoverable and the census-row audit fails.
- Drop carried-forward accounting for unselected plans or silently omit an expanded sibling: preview conservation, plan count, component sizes, membership closure, or immutable plan identity fails.
- Accept one empty dry run as fixed point: the harness requires a first non-fixed quiescent receipt and a second predecessor-linked fixed receipt.
- Remove exact source/index/session/hash fields from application receipts: the production validator fails at the crash boundary or terminal audit.

## Fault matrix and observed behavior

The final dispatcher run used:

`python -m devtools workspace raw-authority-restart-proof --workdir /mnt/data/lane-d-proof-package --keep --json`

Proof ID: `raw-authority-restart-proof:a114d2360d08b2d323281ca0`

Common topology and limits:

- Six retained raw bundles.
- Twelve durable membership rows.
- Four immutable components with sizes `[1, 1, 2, 2]`.
- Default `raw_artifact_limit=None`; no test-only reduction.
- Production application envelope `1,073,741,824` bytes.
- Production parser-census component limit `25`.
- Final endpoint partition in every case: two `executed`, one `terminal`, one `deferred`.
- Final inventory digest: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
- Final residual digest: `6ab22a5de6879f31b9aed726d238eccdb85d4bab186bf0b5dcca01b664c79e69`.

| Boundary | Durable state at injected crash | Recovery/result |
| --- | --- | --- |
| `before_outcome_commit` | One planned census, four selected plans, zero recorded selected outcomes; one newly applied executed receipt already passes the production validator | One restart pass reports one recovered census, drains all six raw candidates, preserves one terminal endpoint per initial plan, then emits two matching quiescent dry runs |
| `after_outcome_commit_before_census_finalization` | One planned census, four selected plans, all four outcomes recorded; both executed receipts pass the production validator | Restart finalizes the complete interrupted census without new candidate replay, then emits two matching quiescent dry runs |
| `during_resumed_batch` | First crash leaves four selected/zero recorded; restart recovers it and a second apply census crashes on `solo-two` with three selected/zero recorded while one interrupted census already exists | Final restart recovers the second application, drains remaining work, retains exactly two interrupted census receipts, and reaches the same terminal partition and fixed point |

The first two cases finish with fixed-point census IDs `census:5:4f53cda18c2baa0c:6ab22a5de6879f31` and `census:6:4f53cda18c2baa0c:6ab22a5de6879f31`. The resumed-batch case finishes with sequences 6 and 7 because it has one extra interrupted apply census. IDs are deterministic for this synthetic corpus, but the proof contract relies on full durable fields and predecessor identity, not string shape alone.

## Commands and final results

All commands below ran from the captured snapshot with the repository's existing `.venv`. Disk-backed basetemps were used because this container has only 64 MiB of `/dev/shm`.

### New harness and static checks

- `.venv/bin/ruff check devtools/raw_authority_restart_proof.py tests/unit/devtools/test_raw_authority_restart_proof.py`
  - Result: passed.
- `.venv/bin/mypy devtools/raw_authority_restart_proof.py tests/unit/devtools/test_raw_authority_restart_proof.py`
  - Result: success, no issues in two source files, against the captured dirty snapshot.
- `.venv/bin/pytest -q --basetemp=/mnt/data/pytest-lane-d-restart-v2 tests/unit/devtools/test_raw_authority_restart_proof.py`
  - Result: `4 passed in 4.25s`.
- The generated `PATCH.diff` was applied to a separate copy of the captured dirty snapshot and the same Ruff/mypy/new-test sequence was rerun there.
  - Result: Ruff passed; mypy passed; `4 passed in 4.62s`.

### Production-route regressions

- `.venv/bin/pytest -q --basetemp=/mnt/data/pytest-lane-d-ledger-final tests/unit/storage/test_raw_authority_ledger.py`
  - Result: `22 passed in 10.81s`.
- `.venv/bin/pytest -q --basetemp=/mnt/data/pytest-lane-d-repair-final tests/unit/storage/test_repair.py -k raw_materialization`
  - Result: `38 passed, 17 deselected in 11.79s`.
- `.venv/bin/pytest -q --basetemp=/mnt/data/pytest-lane-d-scale-final tests/unit/devtools/test_raw_authority_scale_proof.py`
  - Result: `21 passed in 5.32s`.
- `.venv/bin/pytest -q --basetemp=/mnt/data/pytest-lane-d-catalog-a tests/unit/devtools/test_command_catalog.py tests/unit/devtools/test_render_devtools_reference.py`
  - Result: `9 passed in 0.96s`.
- `.venv/bin/pytest -q --basetemp=/mnt/data/pytest-lane-d-main-final tests/unit/devtools/test_devtools_main.py`
  - Result: `8 passed in 0.69s`.

An earlier combined catalog invocation accidentally named nonexistent `tests/unit/devtools/test_main.py`; pytest reported file-not-found. It was corrected to the authoritative `test_devtools_main.py`, which passed. This invocation error is not a product or patch failure.

### Command/reference checks

- `.venv/bin/python -m devtools --list-commands`
  - Result: registered command present.
- `.venv/bin/python -m devtools workspace raw-authority-restart-proof --workdir /mnt/data/lane-d-proof-package --keep --json`
  - Result: passed all three cases with proof ID `raw-authority-restart-proof:a114d2360d08b2d323281ca0`.
- `.venv/bin/python -m devtools render devtools-reference --check`
  - Result: synchronized reference confirmed in final packaging verification.
- `git apply --check PATCH.diff`
  - Result: passed against both clean `bf8191b3...` and the captured dirty baseline; the patch was also actually applied and tested in a separate captured-baseline copy.

### Managed wrapper limitation

`.venv/bin/python -m devtools test tests/unit/devtools/test_raw_authority_restart_proof.py` exited `125` before collection with:

`verify: only 64 MiB free in /dev/shm; refusing disk-backed pytest`

This is the wrapper's intentional containment gate, not a test failure. Raw focused pytest remained in the same locked environment and used explicit disk-backed basetemps.

## Not executed or not verified

- Full `devtools verify` and `devtools verify --all`.
- Full non-integration test suite.
- `nix flake check` or packaged/Nix command execution.
- Real subprocess `SIGKILL`, power-loss, filesystem/WAL fault injection, or systemd restart behavior.
- July-15-sized synthetic replay/resource envelope and daemon-health measurements.
- Any operator daemon, browser, secrets, deployed package, current worktree, backup, or live archive.
- Any live dry run, apply, cursor change, replay, SQL repair, evidence deletion, or authorization.

These remain explicit local-lane continuation work; none is inferred from the compact proof.
