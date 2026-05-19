# Homebrew Tap Template

This directory is a checked-in template for the separate
[`Sinity/homebrew-tap`](https://github.com/Sinity/homebrew-tap) repository
that distributes Polylogue via Homebrew on macOS and Linux.

The template is the source of truth for the formula shape, the `service`
block, and the test block. The live tap repository is bootstrapped from
this directory and kept in sync by the
[`homebrew-bump.yml`](../../.github/workflows/homebrew-bump.yml) workflow
in this repository, which opens a PR against the tap on every `vX.Y.Z`
tag push.

## Layout

```
nix/homebrew-tap-template/
├── Formula/
│   └── polylogue.rb        # Source of truth for formula shape
└── README.md               # This file (also copied as the tap README)
```

`Formula/polylogue.rb` ships with placeholder `url` / `sha256` / `version`
fields. The bump workflow rewrites them in place using values published to
PyPI by [`release.yml`](../../.github/workflows/release.yml), so the live
formula always tracks the most recently published wheel.

## Installation

Once the tap repo is bootstrapped:

```bash
brew tap sinity/tap
brew install polylogue
```

This installs three binaries into Homebrew's prefix:

| Binary | Role |
| --- | --- |
| `polylogue` | Query CLI for the archive. |
| `polylogued` | Local convergence + HTTP daemon. |
| `polylogue-mcp` | MCP server for AI coding agents. |

## Running the daemon

Polylogue does not install a system service by default. Opt in with
Homebrew's service manager (LaunchAgent on macOS, systemd-user unit on
Linuxbrew):

```bash
brew services start polylogued
brew services info polylogued
```

The first invocation creates the XDG archive at
`~/Library/Application Support/polylogue/` on macOS or
`${XDG_DATA_HOME:-~/.local/share}/polylogue/` on Linux. Override with
`POLYLOGUE_ARCHIVE_ROOT` to relocate the database, FTS indexes, and blob
store.

## Verifying a candidate formula

When iterating on the template locally:

```bash
brew install --build-from-source nix/homebrew-tap-template/Formula/polylogue.rb
brew test polylogue
brew audit --strict --online polylogue
```

The first command compiles the upstream sdist into a venv under
`$(brew --prefix)/Cellar/polylogue/X.Y.Z`. The audit pass mirrors the gate
the bump workflow runs after rewriting the placeholders.

## Out of scope

- No cask is shipped (Polylogue is CLI-only, not a GUI app).
- Submission to `homebrew-core` is intentionally deferred until the
  distribution surface stabilizes; the dedicated tap is the supported
  channel in the meantime.
