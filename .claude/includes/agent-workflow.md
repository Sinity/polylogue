## Agent Workflow

- Keep repository-specific agent guidance in this root `CLAUDE.md` transclusion surface.
- Do not create path-scoped `CLAUDE.md` files for subsystems.
- Pull concise workflow and operator guidance from the public repo docs when it can be transcluded directly.
- Use GitHub issues for upfront planning, design decisions, and durable follow-up chains when they add real value.
- Do not create retrospective bookkeeping issues for self-contained work that is already underway or already finished.
- Confirm with the user before opening an issue unless they explicitly asked for one.
- When an issue is warranted, use the issue templates instead of ad-hoc markdown plans.
- Use `python -m devtools ...` for repo maintenance, generated-surface refresh, validation-lane execution, and operator hygiene.
- Use `python -m devtools --list-commands --json` and `python -m devtools status --json` when an agent needs command discovery or repo-maintenance state.
- PR titles must be written as the final squash-merge commit subject that should land on `master`.
- PR bodies should act as proof receipts: concise summary, problem, solution, verification, and any remaining risk or follow-up.
