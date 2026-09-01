Summary

Add one render-boundary helper and use it for human-facing session dates in CLI lists, streamed headers, rich session headers, archive stats, correlation windows, and Org export. Machine JSON/ISO payloads and stored timestamps remain unchanged.

Problem

Human-facing paths formatted UTC wall-clock values without a zone marker, causing local session history to appear shifted. The timestamp census identified direct render sites in `polylogue/cli/query_output.py`, `polylogue/cli/query_stats.py`, `polylogue/insights/correlation_view.py`, and `polylogue/rendering/formatting.py`.

Solution

`polylogue.core.localtime.format_local_datetime` converts aware datetimes to the host-local timezone and emits a zone marker for datetime displays; date-only displays still receive local date conversion. Legacy naive values are interpreted as UTC at this boundary. Fixed `America/Los_Angeles` tests cover offset conversion and a local-date boundary crossing.

Verification

- `nix develop --accept-flake-config --command python -m devtools test tests/unit/core/test_localtime.py tests/unit/cli/test_query_fmt.py tests/unit/cli/test_correlate_view.py tests/unit/cli/test_archive_query.py`: 187 passed, 1 existing pytest rewrite warning.
- `nix develop --accept-flake-config --command python -m devtools verify --quick`: all quick checks green, including format, lint, mypy, layering, generated surfaces, timestamp doctrine, and schema/privacy checks.
- `git fetch origin && git rebase origin/master`: up to date.
- Post-rebase quick gate and pre-push quick gate: green.

Residual risk: the complete corpus and live daemon were not run; this change does not alter their storage or wire representations.
