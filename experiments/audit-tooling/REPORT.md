# Audit tooling adoption report

This document records the production-toolchain disposition of the preserved audit lab at `e0f9e1d1d4818688173cd72cce4d5ef7ec63da93`. The lab's probes and experimental rules remain on that preserved branch; they are not part of this production landing.

## Adopted toolchain

- The `[dependency-groups].audit` stanza in [`pyproject.toml`](../../pyproject.toml) provides `ast-grep-cli`. Run `uv sync --group audit` from a fresh checkout before using `ast-grep`; `sg` is a different system command in the devshell.
- The default devshell's `buildInputs` stanza in [`flake.nix`](../../flake.nix) provides `ast-grep`, `scc`, and `codeql`. CodeQL's Nix unfree exception is limited to its package name in the same file.

The lab established distinct production-useful roles for these tools: ast-grep generates structural candidates, scc reports cheap code-size and complexity trends, and CodeQL supports periodic security-focused dataflow analysis. They do not independently enforce any audit policy.

The landing PR also carries a versioned Beads-scope receipt; validate the published boundary with `devtools workspace pr-scope check --pr 3917` before merging.

## Deliberately not adopted

This Bead does not carry forward the lab's probe scripts, query packs, rule files, generated outputs, or exploratory Python dependencies. A future policy change can adopt a specific rule, query, or additional dependency with its own verification contract.
