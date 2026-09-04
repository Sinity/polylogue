## Summary

Remove redundant pytest-testmon state from deliberate complete-corpus verification while retaining testmon for affected selection.

## Problem

A complete-corpus run already executes every collected test, so loading testmon adds dependency tracing state without changing the selected corpus.

## Solution

Add a shared managed-plugin profile. Affected runs retain testmon; complete runs omit its plugin and flags. Worker count, xdist distribution, collection roots, and remaining plugins are unchanged.

## Verification

- The command-contract suite passed: 4 tests.
- `devtools verify --quick` passed all static gates before publication.
- PR #4650 quick-gate passed before this rebase.

## Residual risk

The acceptance threshold still needs comparable complete-run PSS and wall-time receipts. This change does not claim that measurement.
