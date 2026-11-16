# Polylogue

Polylogue is a local archive and workbench for AI work. It captures every
session you have with AI tools (Claude Code, Codex, Gemini CLI and other coding
agents through their own files and hooks; ChatGPT, claude.ai, AI Studio and
other web chats through their exports and a browser extension), keeps the
original bytes, parses them into one model of sessions and their lineage, and
lets you and your agents search, read, measure and annotate that history from
the command line, over MCP or in a web workbench. Agents are first-class users:
they get the same powers as you, and every session can start with a brief
drawn from the archive.

Polylogue runs alone (standalone, on SQLite) or on
[Sinex](https://github.com/Sinity/sinex) (primary mode), where Sinex stores the
bytes and records and Polylogue keeps a local index. Everything you can do is
the same in both modes.

## Install

With Nix:

```sh
nix build github:Sinity/polylogue      # the package, at ./result/bin/polylogue
nix develop                             # a development shell
```

The flake also provides a Home Manager module (the package and one user
service per named archive) and a NixOS module (the system service). Hooks, MCP
registration and the browser extension's host are installed with
`polylogue install HARNESS` and `polylogue install chrome`
([operations](docs/installing.md#installing)). See
[development](docs/development.md#deployment-modules) for their use and
[operations](docs/installing.md#deployment) for the deployment contract.

With Cargo, from a checkout:

```sh
cargo build --release -p polylogue
```

That builds standalone mode. Primary mode (Sinex) is the Cargo feature
`sinex`: `cargo build --release -p polylogue --features sinex`. The Nix
package includes it.

Polylogue reads each tool's data where the tool writes it, so a running daemon
needs no configuration to start collecting. Exports you download (a ChatGPT or
claude.ai data export) are brought in with `polylogue import`.

## A tour on the Morrow archive

Morrow is a synthetic week of work on a fictional Rust library, in Claude Code,
Codex, ChatGPT and claude.ai, shipped with Polylogue for examples and tests
([`examples/morrow/`](examples/morrow/)). `demo` builds a separate archive
from it in a new directory and prints the first lines to try:

```sh
polylogue demo ./morrow
```

## First commands

```sh
polylogue daemon                          # run the daemon (normally a user service)
polylogue                                 # what is waiting, running and recent, with the lines to type next
polylogue 'repo:morrow since:7d'          # a query: this week's sessions in a repository
polylogue chrono                          # a plain word searches for it
polylogue @1 read --to browser            # open row 1 of the last list in the workbench
polylogue @1 note 'chrono parsing fixed' --as decision
```

The quotes hold which rows; the words after them say what to do. Tab completes
every position, `polylogue help` is one screen, and `--explain` on any line says
how it is read without running it ([interfaces](docs/interfaces.md),
[terminal](docs/terminal.md)). Agents use the same grammar through three MCP
tools ([agents](docs/agents.md#mcp-server)).

## Where to read next

- [`docs/`](docs/README.md): the design, one page per area, starting with
  [architecture](docs/architecture.md).
- [`docs/derived.md`](docs/derived.md): the derived-data reference, rendered
  from the declarations. Use `polylogue --help` and the MCP schemas for the
  commands in the current build.
- [`AGENTS.md`](AGENTS.md): how work is done in this repository, its layout and
  its commands.
- [Python client](clients/python/README.md): the declared commands through CLI
  JSON and the daemon, with unknown values and opaque IDs preserved.

## Licence

See [LICENSE](LICENSE).

Cloud setup and shell startup, including the explicit `native` backend, are described in
[the shared setup guide](https://github.com/Sinity/polylogue-sinex-common/blob/master/dev/cloud/README.md).
