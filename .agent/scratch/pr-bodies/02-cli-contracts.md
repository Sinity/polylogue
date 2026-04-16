## Summary

PR 2/9. Stacked on `feature/chore/stack-01-governance`.

First wave of CLI-surface fixes found during live vetting. Every plain-mode `PolylogueError` now renders as a Click user error rather than a raw traceback, and the machine envelope is consistent across every query verb.

## Problem

Live vetting against a real archive exposed a cluster of user-facing regressions on the CLI query surface:

- `polylogue --plain stats --by provider --format json` raised a full traceback ending in `DatabaseError` on stale schema, instead of a structured envelope.
- `polylogue --plain query --help` advertised an option placement that the real parser rejected (`Error: No such option: --provider`).
- `polylogue stats --by provider` silently dropped grouped mode because the positional `stats` verb forced `stats_only=True`.
- `polylogue stats --format json` ignored the requested output format in the SQL-backed stats path.
- `polylogue --plain --similar "debug parser" list` told users to run `polylogue embed` — an old command. Canonical is `polylogue run embed`.
- `polylogue --format json list -n 1` emitted plain early-failure behavior instead of a structured runtime envelope.
- The pre-scan command extractor treated option payloads like `json` / `1` as command words, corrupting the machine envelope's `command` field.
- `polylogue stats --format json` reported `messages_total=137` while `messages_user=0` and `messages_assistant=0` — role counts were hardcoded to zero, and the filtered branch dropped its temp id table before the word-count query.
- `polylogue --format json list -n 10` showed malformed ChatGPT conversation ids like `chatgpt:synthetic-55392 synthetic-55392 syn` because synthetic tree generation left the top-level ChatGPT `id` field as schema-generated text.
- `open --print-path <conversation-id>` failed with `Got unexpected extra argument` when given a direct `provider:id`.
- `polylogue stats --by provider --limit 10` required `--limit` before the verb because the `stats` verb relied on root for `--limit`.
- `polylogue --plain tags --format json` failed with `No such option: --format`.
- Terminal snapshot baselines and showcase CLI boundaries still referenced `python -m polylogue` rather than the public `polylogue` executable.

## Solution

- Harden the root CLI error boundary so plain and JSON machine-main render `PolylogueError` as Click user errors instead of tracebacks.
- Correct the root help example so provider/date filters appear before the query verb where the parser actually accepts them.
- Make `stats_only` false when `--by` is present; pass `output_format` through SQL-backed stats; emit structured JSON/YAML/CSV for archive-wide stats.
- Update the similarity-error branches to point at `polylogue run embed`.
- Harden machine-mode detection for root `--format json` across `list`, `open`, `tags`, `products`, `schema`, and query mode; harden the pre-scan command extractor.
- Fix `aggregate_message_stats` to report real role counts and word counts.
- Assign clean UUIDs and stable timestamps to synthetic ChatGPT conversations.
- Route a single trailing `provider:id` token in `open` through the exact `conv_id` filter.
- Give `stats` a local `--limit/-n` option and forward it into the shared query params.
- Accept `--format json` on `tags` in addition to `--json`.
- Add archive-backed zsh completions for conversation IDs (with provider/title descriptions), `open <target>`, `--tag`/`--exclude-tag`, `--tool`/`--exclude-tool`, comma-aware provider completion.
- Switch showcase CLI verification to the public `polylogue` executable and refresh the full help baseline.
- Refresh terminal snapshot baselines and `invalid option` snapshot.

## Verification

- `pytest -q --ignore=tests/integration`
- `pytest -q tests/unit/cli/test_click_app.py tests/unit/cli/test_machine_main.py tests/unit/cli/test_machine_contract.py tests/unit/cli/test_query_exec_laws.py tests/unit/cli/test_query_exec.py tests/unit/cli/test_products.py tests/unit/cli/test_check.py tests/integration/test_cli_query_mode.py tests/integration/test_cli_tags_surface.py`
- `ruff check polylogue tests devtools`
- Manual live probes: `polylogue --format json list -n 1`, `polylogue --plain stats --by provider --format json`, `polylogue --plain open --print-path claude-code:<id>`, `polylogue --plain tags --format json`.

Commits on this branch: 30 (delta against `feature/chore/stack-01-governance`).

## Stack

Base: `feature/chore/stack-01-governance`. Next: `feature/fix/stack-03-runtime-repair`.
