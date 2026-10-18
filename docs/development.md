# Development checks

Use focused tests while editing. Run the full `nix flake check --print-build-logs` for a coherent package delivery, not after each patch. The author remains responsible for checks appropriate to the change and for reporting untested scope.

The required CircleCI `quick-gate` runs workspace tests, formatting and crate-boundary checks through the pinned Nix development environment. Cargo build output and registry downloads are cached by architecture, toolchain lock and dependency lock, with revision-specific saves. Cargo still checks source fingerprints and rebuilds changed code; the cache does not replace test execution. The first run or a lock change can be cold. Prose-only PRs stop before the build; changes to CI or hooks exercise the gate. Master runs the package job instead of repeating both jobs.

The clean Nix package check runs separately on master after integration. Its failure needs repair before treating that revision as a qualified distributable. It is not a second per-PR merge wait. A local immutable package run can still qualify a release candidate before publication when packaging is part of the change.

The pre-push hook retains the full local check for Rust, build or acceptance-script changes. Documentation, CI and hook-only pushes avoid that duplicate local suite. Unknown bases and pushes of a commit other than the current checkout conservatively run it. Existing required hosted status names and review rules remain unchanged.
