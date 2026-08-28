## Summary

Validate the provisioned native testmon graph after lane dependency setup.

## Problem

Workspace provisioning copied `.cache/testmon/testmondata` and ran `uv sync`,
but did not check whether the copied graph contained the lane's current
environment. Lanes could register successfully and only discover the mismatch
when affected verification refused to select tests.

## Solution

Add `devtools.testmon_provision`, a read-only validator that computes the
current environment fingerprint and reports the graph state. Run it after
`uv sync` in the workspace provision hook. Provisioning now fails with the
expected environment and validator reason when the seed is stale or unusable.

## Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/devtools/test_testmon_provision.py` — 2 passed.
- `nix develop --accept-flake-config --command devtools verify --quick` — all quick stages passed.

## Residual risk

The seed still needs a current full-corpus run on the master checkout; this
change makes an expired seed fail during provisioning instead of at first
affected verification.
