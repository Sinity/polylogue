# Development checks

Use focused behavioral tests while editing. Run the full `nix flake check --print-build-logs` for a coherent package delivery, not after each patch. The author remains responsible for tests appropriate to the change and for reporting untested scope. `cargo run --locked -p xtask -- check` remains available for a full local workspace test run.

Routine pushes and the required CircleCI `quick-gate` use `scripts/check-fast.sh`: formatting, crate-boundary validation and compilation of all workspace targets, including test code. These checks do not execute behavioral tests. They run through the pinned Nix development environment. Cargo build output and registry downloads are cached by architecture, toolchain lock and dependency lock, with revision-specific saves. Cargo still checks source fingerprints and rebuilds changed code. The first run or a lock change can be cold.

The pre-push hook skips Rust work for documentation, CI and hook-only pushes. Unknown bases, changed Rust/build/acceptance inputs and pushes of a commit other than the current checkout take the fast-check path. Prose-only PRs stop before compilation; changes to CI or hooks exercise the gate.

The full clean Nix package build, workspace tests and crate boundaries run separately on master after integration. Master runs that job instead of repeating both jobs. Failure needs repair before treating the revision as a qualified distributable. It is not a second per-PR merge wait. Run an immutable package check before publication when packaging is part of the delivery. Existing hosted required status names and review rules remain unchanged.
