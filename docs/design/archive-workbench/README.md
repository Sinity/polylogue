# Polylogue Archive Workbench — standalone design and GitHub issue pack

This pack is self-contained. It does not require any previous Polylogue design milestone or Claude Design output to be useful.

It contains:

- `docs/` — the canonical product/design direction, view model, route/view map, state grammar, and implementation plan.
- `contracts/` — proposed UI contract sketches for object refs, message rendering, topology details, materials, workspaces, context bundles, and actions.
- `screens/` — static MK4 design boards for the major rooms and state surfaces.
- `github/` — issue bodies, issue/comment manifest, milestone plan, labels, and a `gh` helper script.
- `reference/` — the previous working documents preserved only as source/reference material; the canonical docs in `docs/` stand on their own.

Working product thesis: Polylogue should become a local Archive Workbench for AI/chat-agent work: an object graph of conversations, messages, content blocks, paste spans, attachments, raw artifacts, topology edges, insights, user-state objects, workspaces, operations, and context bundles. The core user loops are observe, find, read, relate, compose, continue, and verify.

Grounding snapshot: this pack was prepared from the current attached Polylogue code/material exports and open-issue export dated 2026-05-23. Before bulk-creating issues, check whether referenced issues are still open and whether labels/milestones already exist.

Recommended use:

1. Review `docs/00-canonical-design-program.md` and `docs/01-github-triage-plan.md`.
2. Apply or manually copy `repo-docs.patch` into the repository under `docs/design/archive-workbench/`.
3. Open or comment issues using `github/issue-manifest.json`. The manifest marks which entries are new issues and which are better as comments on existing issues.
4. Start implementation with `POLY-MK4-001` and `POLY-MK4-002`, not the whole design at once.
