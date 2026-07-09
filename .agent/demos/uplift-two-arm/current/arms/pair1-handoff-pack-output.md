# Devloop Reconstruction Report — polylogue @ 91eb09549

## (a) Current devloop state

The devloop has been executing a long, disciplined merge train (11+ squash-merged PRs in the last few hours: #2588–#2604), alternating between real correctness fixes and investigation/bookkeeping. Verified via `git log --oneline -15`:

- `91eb09549` close polylogue-4ts.6 (#2604)
- `c06ca601c` feat(lineage): surface a completeness signal on composed reads (#2603)
- `1d7d79ed2`/`6c12e9234` close polylogue-cpf.1 / timestamp-doctrine lint (#2601-2602)
- `9f0e77116`/`7b5a5aa05` actions-view fan-out fix + close xnkf (#2597-2598)
- `aa4a520b4` record 4ts.3 investigation findings, unclaim (#2596)
- `ccd5cfdd5`/`086171701` close 4ts.4 / lineage read-transaction fix (#2594-2595)
- `f7c997386` record 1vpm.1 investigation findings, unclaim (#2593)
- `43d7fdf9b`/`64c079d6e` close 212.8 / honesty anti-demo (#2591-2592)

This matches the pack's shipped-work list exactly and extends it one item further (`c06ca601c`, a lineage completeness signal, landed after the pack was generated).

**Delivery gate board** (`.agent/tools/delivery-gate-status.py --fresh`), the authoritative frontier view:

```
A-trust-floor              23%  closed 14 | ready 43 | blocked 3   <- frontier
B-storage-rebuild-bytes     4%  closed  1 | ready 19 | blocked 3
F-lineage-compaction       17%  closed  2 | ready  3 | blocked 7
I-analytics-experiments     0%  closed  0 | wip 1 (1vpm.1) | ready 9 | blocked 20
L-external-legibility      13%  closed  4 | ready 20 | blocked 6
```

A-trust-floor is the named frontier lane and is the biggest one still open (43 ready items), mostly the `polylogue-9e5.*` read-only audit family (coverage economics, dead-code sweeps, hash-boundary census, etc.) — this is exactly the "trust-floor P1s" the operator's `/goal` names.

## (b) Open threads found

1. **polylogue-1vpm.1 — live/committed state mismatch.** The last committed jsonl record (`f7c997386`, "record findings, unclaim") sets `status:"open"`. But `bd show polylogue-1vpm.1 --json` right now reports `status:"in_progress"`, `assignee:"Sinity"`, `updated_at: 2026-07-09T03:24:19Z` — a timestamp with **no corresponding commit** touching that bead in `.beads/issues.jsonl` history. This means the live embedded-dolt DB has state ahead of the last export/commit (some claim happened that hasn't been exported+committed yet). Not something I should "fix" here, but worth flagging before anyone else claims it — the committed bookkeeping record and the live DB disagree.
2. **F-lineage-compaction ready set is exactly 3 items**: `polylogue-4ts` (epic), `polylogue-4ts.3` (investigated, fix location known), `polylogue-4ts.5` (fresh, "compaction boundary-range columns + effective-context derivation" — not yet investigated).
3. **B-storage-rebuild-bytes ready set** includes `83u.2` (re-investigated/reframed to the "bytes reachable but not fetched" subset, per commit `6796ac1ed`) and `83u.3` ("preserve uploaded attachment bytes in live browser capture" — fresh, untouched).
4. **212.x demo-unlock parallel set**: 212.1/212.3/212.5/212.6 remain open with no assignee; per the pack each has its own blocker (deep interpretive read / missing join primitive / live stagecraft / classifier bug); none looked trivially unblockable from what I can see in their notes (no blocker detail beyond the delivery-upgrade boilerplate), so I did not find a shortcut through any of them.
5. **pj8** depends on `37t.4` (SessionStart preamble rollout) which is itself gated on real MCP-hook-wiring code, not just deployment — it's `open`, unclaimed, priority 3, and the pack says it's now "authorized" for the hook wiring step, but actually landing it still implies a SessionStart hook change that the pack says needs redeploy to prove — consistent with "postpone" guidance.
6. **3tl** — confirmed still fully untouched (17 open children), and independently confirmed P4/docs-marketing in this pass (no evidence contradicting the pack).
7. **cfk** — this very context-pack handoff is explicitly part of the live paired-arm uplift experiment it depends on; I was told not to read `.agent/demos/uplift-two-arm/`, so I can't and shouldn't try to advance or grade that experiment from inside it.

## (c) Recommended next action

**Claim and implement `polylogue-4ts.3`** — "Distinguish subagent auto-compaction from main-session acompact."

Reasoning:
- It is explicitly named as part of the "lineage epic" the operator's goal calls out, and is one of only 3 ready items left in F-lineage-compaction (the epic is 17% closed, actively being worked this session — `4ts.4`/`4ts.6` just shipped).
- Unlike `1vpm.1` (comparable in size to `svfj`, a multi-surface feature — new query unit, new column, new registered `target_kind`) or `4ts.5` (still needs fresh investigation), `4ts.3` already has a completed, verified investigation on the bead itself, with an exact fix location I independently confirmed still exists at HEAD:
  - `polylogue/sources/dispatch.py:354` `_claude_code_grouped_record_specs`
  - `polylogue/sources/dispatch.py:441` `_claude_code_stream_sessions`
  - Both are called from the dispatch layer (lines 541, 831) and are the layer with visibility into sibling session-group content needed to distinguish subagent self-compaction from main-session `acompact`.
- It requires no redeploy/MCP-reboot (unlike pj8/37t.4) and no live paired-arm experiment (unlike cfk).
- It's a real correctness bug (misattributed parent session on ~39/187 affected files per the GH thread cited in the bead), not read-only analysis, so it continues today's demonstrated pattern of shipping concrete lineage-correctness fixes rather than accumulating more audit-lane findings on top of the already-large 43-item A-trust-floor backlog.

Secondary/fallback candidates if `4ts.3` turns out blocked on something I couldn't see (e.g. a fixture gap): `polylogue-83u.3` (fresh, unclaimed, in the storage-rebuild-bytes lane) or one of the trust-floor `9e5.*` P1 audits (e.g. `polylogue-9e5.6`, hash-boundary census — the closest literal match to "storage identity" in the operator's phrasing).

## (d) Confidence and evidence

**Confidence: high** on devloop state and gate-board numbers (directly queried live `bd`/git, not inferred from the pack); **medium-high** on the `4ts.3` recommendation (its investigation note is detailed and the code anchors check out, but I have not read the full GH issue thread it references, nor written/run the fix itself); **medium** on the `1vpm.1` state-mismatch finding (clearly true from the data, but I don't know *why* — could be an uncommitted claim from a concurrent session, consistent with this repo's documented shared-checkout/multi-session risk).

Evidence used: `git log --oneline` / `git log -p -- .beads/issues.jsonl` on the current worktree; `bd show`/`bd ready --json` for `polylogue-pj8`, `37t.4`, `3tl`, `cfk`, `212.9`, `1vpm.1`, `83u.2`, `4ts.3`, `4ts`, `9e5.6`, and the 212.1/212.2/212.3/212.5/212.6 family; `.agent/tools/delivery-gate-status.py --fresh` (whole-board and per-gate); direct `grep` confirmation of the `dispatch.py` fix anchors named in the `4ts.3` bead notes. No files under `.agent/demos/uplift-two-arm/` were read, per instruction.
