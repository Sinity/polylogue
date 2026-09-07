# Audit tooling adoption report

The audit-tooling lab established this bounded roster for repository analysis.
The production landing is the `[dependency-groups].audit` stanza in
[`pyproject.toml`](../../pyproject.toml), with versions recorded in
[`uv.lock`](../../uv.lock). A fresh checkout can install it with
`uv sync --group audit`.

The group contains `ast-grep-cli`, `grimp`, `import-linter`, `vulture`,
`radon`, `jedi`, `duckdb`, `networkx`, and `libcst`. The default devshell's
`buildInputs` in [`flake.nix`](../../flake.nix) provides `ast-grep`, `scc`, and
`codeql`; CodeQL is enabled through a package-name-scoped unfree allowance.
The tools are audit inputs and do not become runtime product dependencies.

The lab's exclusions remain deliberate: `semgrep` conflicts with the pinned
dependency graph, `pydeps` is redundant with grimp's import graph, and the
tree-sitter Python bindings do not provide the free-threaded CPython 3.14 ABI
used by this project. Probe scripts, query packs, generated outputs, and
experimental policy rules stay in the lab branch rather than this product
checkout.

## Layering adjudication

The lab's six-edge summary does not reproduce as six live edges at the landed
head. The three concrete reverse imports were real and are removed; the other
three claimed slots have no direct import in either graph.

| Claimed edge | Verdict and evidence |
| --- | --- |
| `ed3d03425:polylogue/sources/live/batch.py:208 → polylogue.api` | Real violation. Replaced the type-only surface import with `core.protocols.ArchiveRootOwner`; the AST and grimp graphs are now clean. |
| `ed3d03425:polylogue/sources/live/watcher.py:74 → polylogue.api` | Real violation. Same structural protocol fix; no edge remains. |
| `ed3d03425:polylogue/storage/embeddings/preflight.py:101 → polylogue.api` | Real violation. Calls the substrate-owned `select_pending_session_window` directly; no edge remains. |
| `polylogue/storage → polylogue.daemon` | No direct edge at the landed head (`grimp.build_graph('polylogue')` and the gate's AST inventory both return none); superseded by the current graph. |
| `polylogue/pipeline → polylogue.daemon` | No direct edge at the landed head; superseded by the current graph. |
| `polylogue/sources → polylogue.mcp` | No direct edge at the landed head; superseded by the current graph. |

The layering manifest now includes `polylogue.api` in every substrate
disallow list. `devtools/verify_layering.py` runs grimp as an independent
substrate-to-surface cross-check and fails on extractor disagreement.
