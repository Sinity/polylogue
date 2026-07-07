---
created: "2026-06-30T21:30:00+02:00"
purpose: "Record Polylogue /realm/inbox integration into .agent"
status: "active"
project: "polylogue"
---

# Inbox Integration

## What Moved

Polylogue-related historical inbox material has been moved, not copied, into:

- `.agent/archive/inbox-integrated/project-devloops/polylogue/`
- `.agent/archive/inbox-integrated/project-artifacts/polylogue/`
- `.agent/archive/inbox-integrated/project-artifacts/cross-stack/downloads/`
- `.agent/archive/inbox-integrated/project-artifacts/legacy-prompts/polylogue/`
- `.agent/archive/inbox-integrated/project-artifacts/cross-stack/browser-capture-preserve/`
- `.agent/archive/inbox-integrated/project-artifacts/cross-stack/exec-helpers/polylogue_stale_audit.sh`
- `.agent/archive/inbox-integrated/project-devloops/cross-stack/legacy-readable-demos-20260630T0142/`
- `.agent/archive/inbox-integrated/polylogue-conductor-devloop/conductor-polylogue.md`
- `.agent/demos/chatlog-exports/` for the two large Codex devloop session
  export families:
  - `019f12b5-1a85-7b42-858e-44eccf8469dc`
  - `019f12b5-fc19-7110-b069-4f49a78da82d`

The project-devloop/project-artifact inbox README files no longer contain
Polylogue breadcrumbs. Inbox is staging, not a pointer index.

## Why

The inbox is staging. Keeping historical Polylogue downloads and stale conductor
prompts there created a second memory tree and made old counts look current. The
current conductor state already carries the useful doctrine: demonstrable
artifacts, algebraic composition, source/provenance honesty, one canonical
archive, temporal dogfood, projection/render algebra, and no recovery/context
silos.

## Count Correction

The stale larger review tuple came from an old archive view. The current
canonical archive is `/home/sinity/.local/share/polylogue`, index schema v18. At
the time of the repair slice, the active daemon-reported count was 13,116
sessions and 3,947,844 messages, and the run-projection aggregate command
reported:

- sessions: 13,116
- runs: 13,382
- observed events: 1,854,045
- context snapshots: 13,382

Demo 01 now rebuilds through `devtools workspace temporal-archive-aggregates`
instead of direct SQL snippets.

## Chatlog Export Demo Shelf

`.agent/demos/chatlog-exports/` is the current home for the two requested Codex
session export families. `current/` contains legible per-session folders with
operator-readable, dialogue-only, no-output-body, compact-output, full-output,
raw JSONL, and regenerated `product-read/` variants. `archive/` holds moved
superseded packages and raw provenance.

The shelf is intentionally not append-only. Regenerate, consolidate, rename,
replace, or delete weak variants when that makes the externally presentable demo
clearer. The regeneration helper is:

```bash
.agent/demos/chatlog-exports/regenerate.sh
```

While creating this shelf, the product `read --to file` handling for
`--view raw` and `--spec` was fixed so the helper can exercise the real product
file-delivery path instead of shell redirection.

## Active Shelf Exception

`/realm/inbox/demos_polylogue` remains an active operator-facing demo shelf.
Current demo artifacts stay there until the operator changes that convention.
Historical devloop/download/source material should move into `.agent` after
assimilation.
