# Architecture

This describes the current Rust module boundaries. The specification and Beads
also contain intended behavior that is not implemented or qualified yet.

## Inputs and stored records

`polylogue-sources` interprets provider formats. An origin declaration owns
recognition, paths, framing, parser version and provider capabilities. Opening
an input selects line or document parsing; callers do not maintain a second
framing flag. Parsers produce model values, record ranges and accounting.

`polylogue-engine` discovers changes, retains input bytes and coordinates
parsing. Its worker pool feeds one bounded writer. Source interpretation stays
in sources; SQLite access stays in store. Rebuilding a derived index must not
replace the retained observations with a claim that a provider file is still
available.

## Storage and queries

`polylogue-store` owns the archive, index, embeddings and blob access. One
writer serializes archive changes; readers use read transactions. Original
observations and authored values differ from derived indexes. Session reads,
derivation and exports share one row mapping, including repository and mode.

`polylogue-query` binds the query language and compiles parameterized SQL.
`daemon/query.rs` executes it through a store reader, computes the final answer
count and prepares pages. Recall uses this path too. Saved-query and kept-result
storage failures propagate as errors; an unavailable collection is not an
empty collection. Cursors retain the query identity and resolved time; they do
not establish a snapshot that survives between requests.

## Application and transports

`daemon/service.rs` owns application state and typed request handling, including
starting the convergence loop. `daemon/server.rs` owns Unix socket framing and
listeners. HTTP and MCP adapters translate requests and present results; they
use the same application operations. Signal evaluation and embedding providers
have their own modules. This is an internal boundary in the application crate,
not another service or crate.

`commands.rs` declares command arguments, capabilities, printers and local
output actions. CLI, MCP and HTTP build their surfaces from that table.
`daemon/measure.rs` declares measurements and their supported metrics once.

## Presentation and backing storage

`polylogue-render` filters model blocks by view, masks credentials and renders
output. HTML exports and the web reader use the same content renderer;
framing, navigation and transport remain with their callers. HTML output writes
to the supplied writer and returns write errors.

Standalone storage keeps Polylogue's archive locally. The optional Sinex
adapter supplies primary backing; its publication and restore qualification
have separate acceptance criteria. Neither mode provides daemonless CLI
access. Shared blob storage, credential masking and development tooling live in
`polylogue-sinex-common`; projects pin immutable releases.

## Changing a boundary

Keep a rule with its owner and update connected callers together. Add a
provider through the origin and parser framework, a command through the
command table, and a measurement through its declaration. Tests exercise the
behavior through these interfaces. Follow [development.md](development.md) for
focused checks and batch regression runs.
