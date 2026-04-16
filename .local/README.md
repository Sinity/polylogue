# Local Working Outputs

This directory is for useful repo-local outputs that should not normally be
committed.

Examples:

- mutation campaign ledgers
- benchmark campaign reports
- canonical Nix build out-link at `.local/result`
- showcase captures and screenshots
- demo workspaces and screencasts
- temporary reports or captures worth inspecting before cleanup

Treat `.local/` as the home for persistent local outputs that are worth keeping
around locally but are not part of the repository history.

Do not reintroduce top-level `artifacts/`, `.benchmarks/`, or similar
ad-hoc output roots. Put new untracked outputs here instead.
