polylogue-02-time-to-proof.md
# Polylogue 02 — clean-machine time-to-proof

## Goal

Select the first public command by measured success and time-to-meaningful-output, not preference.

## Routes

Test every currently supported route that can reasonably be first contact:

- `uvx polylogue demo receipts --compact`;
- installed CLI after `uv tool install`;
- pipx;
- Homebrew on supported macOS/Linux runners;
- Nix one-shot;
- source checkout in the documented environment.

## Protocol

Use fresh disposable environments. Record OS/arch, route, package manager version, network cache state, start/end monotonic timestamps, exit code, stdout/stderr, first meaningful output timestamp, peak disk/network if available, cleanup result, and artifact hash. Run cold and warm repetitions separately. A timeout or interactive prompt is a failure, not missing data.

The meaningful-output marker is the compact verdict plus its failed-action, recovery, and anti-grep fields—not package download progress or `--help`.

## Decision rule

Choose the route with the highest clean success rate. Break ties using median cold time, then p90, then smallest prerequisite burden. Do not choose a faster route that omits the real product path or typed receipt.

## Acceptance criteria

Machine-readable results and a human report exist; every route has at least three cold runs per supported environment; failure logs are retained; README wording matches the measured winner; and no unsupported environment is implied.

