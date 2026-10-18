# Repository Guidelines

This workspace prepares Polylogue’s Rust replacement. The top-level README and CLI help own supported capabilities and their limits; `docs/rewrite/` records the candidate’s specification basis. Preserve explicit unsupported outcomes while completing a path. Predecessor deployment consumers remain separately pinned until authorized cutover.

## Structure

`crates/model` owns pure identity, value and outcome types. `crates/hashed-blobs` is the reviewed shared file layer with a read-only capability. `crates/store` owns every SQLite connection and mutation. `crates/engine` computes provider work outside the writer and installs it through typed store operations. `crates/polylogue` owns configuration, process lifetime, the daemon actor and CLI. `xtask` runs focused developer checks. Retained-source progress and canonical Message readiness are distinct; source EOF alone cannot certify a complete query.

## Tests

Keep substantial unit tests and fixtures in separate source files, using private `#[cfg(test)]` modules for implementation access. Keep production visibility unchanged when moving tests. Small nearby tests may stay inline. Use synthetic inputs and isolated temporary state. CLI and MCP acceptance tests exercise their shared semantics through public routes.

## Commands and style

Use the declared AgentCTL operations for builds and suites. `nix flake check` builds packaged binaries and runs foundation checks; `scripts/check-fast.sh` owns routine checks; `cargo run -p xtask -- check` runs the full local workspace suite. Rust uses four spaces and rustfmt. Stage explicit paths; run the normal hooks. Keep the target crate edges: model never depends on storage or IO; only store uses SQLite; surfaces obtain typed store handles, never raw connections.

## Development flow

Optimize routine development for fast feedback and integration. Use cheap checks and focused behavioral tests for the actual delta; full local packages, full behavioral suites and full-corpus verification are not prerequisites for every development merge. Development bugs can be repaired after integration. Keep pass, fail, pending, unrun and superseded outcomes explicit; integration is not proof of testing. Run extended validation separately on meaningful integrated batches and continue independent development while it runs. Coalesce expensive queued checks to useful latest-main candidates through supported job tooling; do not enqueue a full verification sequence for every small change. Use the normal permitted PR route and existing required checks; do not change branch protection. Real-data custody, secrets, destructive operations and deployment retain their separate safeguards.

## Correctness and delivery

Unsupported operations return a typed non-success result. Unknown time remains unknown. IDs are opaque strings; native IDs may contain colons. Never manufacture a completed empty query from unsupported behavior. Durable schema validation precedes mutation. Preserve existing edits and name remaining unsupported acceptance honestly.

Local foundation commits do not authorize PR publication, source-default replacement, task routing, provider import, archive deletion or host activation. Keep credentials, captures, private reports and actual databases outside Git. The retained LICENSE and historical changelog preserve predecessor attribution/history.
