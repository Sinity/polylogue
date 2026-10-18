# Development checks

There is no hosted CI. Checks run on the development machine, which is much
faster than a hosted runner.

- **While editing:** run the focused tests for what you changed
  (`cargo test -p <crate> <filter>`).
- **On push:** the pre-push hook runs `scripts/check-fast.sh` (formatting,
  crate boundaries, `cargo check` of every target including tests). It takes a
  few seconds warm and skips pushes that touch no Rust or build inputs.
- **On a batch of merged work:** run the full suite, `cargo run --locked -p
  xtask -- check`, or the clean package build, `nix flake check`. Run them as
  AgentCTL jobs (`foundation_check`) and keep developing while they run. Fix
  what they find in follow-up PRs.

A PR merges as soon as it is ready; nothing waits on a hosted check. A bug
found after merge is fixed after merge.
