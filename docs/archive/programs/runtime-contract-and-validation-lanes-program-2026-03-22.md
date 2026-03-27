# Polylogue Runtime Contract And Validation Lanes Program

Date: 2026-03-22
Status: executed subprogram
Role: executed closure slice for the remaining testing/runtime frontier covering root machine CLI contracts, query/TUI hardening, and explicit validation lanes

See also:

- `testing-reliability-expansion-program-2026-03-14.md`
- `read-surface-proof-and-showcase-hardening-program-2026-03-22.md`
- `planning-and-analysis-map-2026-03-21.md`

## Purpose

Close the remaining backlog reservoir that still sat outside the earlier proof,
schema, publication, and showcase programs:

1. root machine-consumable CLI failure contracts
2. stronger query-routing and TUI operator proofs
3. explicit local/live/chaos/long-haul validation lanes

This was not a new architecture wave. It was closure work to make the already
landed runtime surfaces easier to trust and easier to operate intentionally.

## Why This Was The Frontier

The repo already had most of the substance:

- machine envelopes and runtime health existed
- chaos/interruption/chronology suites existed
- long-haul benchmark tooling existed
- TUI tests existed
- query-routing unit laws existed

But the frontier was still real because:

- `python -m polylogue` still bypassed the root machine-error adapter
- machine-failure behavior was not proven end-to-end at the real entrypoint
- query-mode subprocess proofs were too soft and under-assertive
- TUI coverage still missed a couple of important operator-facing states
- these surfaces were not exposed as one readable validation control plane

## Executed Outcomes

This slice delivered:

- root module entrypoint now honors the same machine-error adapter as the
  installed `polylogue` script
- direct unit and subprocess proofs for JSON success/failure envelopes
- stronger query-first integration proofs for count, summary-list, stream, and
  stats-by routes
- tighter TUI interaction proofs for theme switching and missing-index search
  failure guidance
- explicit pytest marker taxonomy for the remaining validation surfaces
- a single operator runner for validation lanes:
  - `machine-contract`
  - `query-routing`
  - `tui`
  - `chaos`
  - `scale-fast`
  - `scale-slow`
  - `long-haul-small`
  - `live-exercises`
  - `frontier-local`
  - `frontier-extended`

## Executed Scope

### 1. Root Machine Contract Closure

Fixed the remaining entrypoint mismatch so `python -m polylogue` goes through
the root machine-error adapter instead of calling Click directly.

That means JSON failure behavior is now consistent across:

- installed script entrypoint
- module entrypoint
- command-validation failures
- Click usage errors
- wrapped runtime exceptions

Primary files:

- `polylogue/__main__.py`
- `tests/infra/cli_subprocess.py`
- `tests/integration/test_cli_machine_contract.py`
- `tests/unit/cli/test_click_app_main.py`

### 2. Query Routing Hardening

Replaced the old soft query-mode integration file with a smaller, sharper set
of route proofs that assert actual operator output for:

- count route
- summary-list JSON route
- stream JSON-lines route
- stats-by provider route
- no-args stats surface

Primary file:

- `tests/integration/test_cli_query_mode.py`

### 3. TUI Hardening

Kept the existing Textual coverage and filled two missing seams:

- exact dark/light theme transitions
- explicit rebuild hint when the search index is missing

Primary file:

- `tests/unit/ui/test_tui.py`

### 4. Validation Lane Control Plane

Added a dedicated operator runner that names the remaining validation surfaces
directly and composes them into local frontier bundles.

Primary file:

- `devtools/run_validation_lanes.py`

Companion tests:

- `tests/unit/devtools/test_validation_lanes.py`

### 5. Marker Taxonomy

Promoted the remaining frontier surfaces into explicit pytest markers so the
lane runner and operator docs describe real categories instead of file lists:

- `machine_contract`
- `query_routing`
- `tui`
- `chaos`
- `live`

Primary files:

- `pyproject.toml`
- targeted test modules in `tests/unit/cli/`, `tests/unit/ui/`, and
  `tests/integration/`

## Verification

Executed targeted verification:

- `pytest -q -n 0 tests/unit/cli/test_click_app_main.py tests/unit/devtools/test_validation_lanes.py tests/integration/test_cli_machine_contract.py tests/integration/test_cli_query_mode.py tests/unit/ui/test_tui.py tests/unit/cli/test_machine_contract.py tests/unit/cli/test_json_envelope_contract.py tests/unit/cli/test_check_runtime.py tests/unit/cli/test_query_exec.py tests/unit/cli/test_query_exec_laws.py`

Result:

- `193 passed in 26.27s`

## Outcome

The old “remaining testing/runtime reservoir” is no longer one vague bucket.
Polylogue now has:

- a truly closed root machine-error contract
- stronger operator-grade read-route proofs
- named validation lanes for local, heavy, and live operator workflows

The next frontier is no longer this general runtime/testing closure work.
