## Summary

Repair four malformed SQL comment markers in the canonical `source.db` DDL.

## Problem

Fresh archive initialization failed with `sqlite3.OperationalError: near "+": syntax error`, preventing the Sinex obligation and convergence tests from creating their real archive fixtures.

## Solution

Restore the four intended `--` comment prefixes in `polylogue/storage/sqlite/archive_tiers/source.py`.

## Verification

- `./.venv/bin/python devtools/__main__.py test tests/unit/material_protocol/v1 tests/unit/sinex` — `72 passed, 1 warning in 10.99s`.
- `env VIRTUAL_ENV="$PWD/.venv" PATH="$PWD/.venv/bin:$PATH" ./.venv/bin/python devtools/__main__.py verify --quick` — all reported checks passed through schema privacy registry.

## Residual risk

The 303r epic remains open; this lane fixes only the source DDL initialization defect. Cross-repository Sinex authority and live transport acceptance remain outside this change.
