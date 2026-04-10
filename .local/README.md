# Local Working Outputs

This directory is for meaningful repo-local outputs that are useful during
development but should not normally be committed.

Examples:

- mutation campaign ledgers
- benchmark campaign reports
- showcase captures and screenshots
- demo workspaces and screencasts
- temporary proof bundles worth inspecting before cleanup

Treat `.local/` as the home for persistent local outputs that are more valuable
than cache data but still not part of the repository history.

Do not reintroduce top-level `artifacts/`, `.benchmarks/`, or similar
ad-hoc output roots. Put new meaningful untracked outputs here instead.
