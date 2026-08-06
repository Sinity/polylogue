# Tracker authority: GitHub and Beads

Polylogue uses GitHub issues and Beads for different jobs. They are projections of one work system, not duplicate trackers.

## Authority split

- **GitHub issue:** publicly legible outcome, user/reviewer discussion, and external closure claim.
- **Bead:** executable authority, dependency graph, implementation ownership, acceptance detail, incident/proof history, and agent coordination.
- **PR/commit:** delivered code or documentation effect.
- **Live receipt:** evidence that the effect works against the deployed archive or a declared fixture.

A GitHub issue may map to one Bead, an internal Bead subtree, or no Bead when it is only a historical/public discussion. A Bead does not require a GitHub issue when it is an internal repair, incident, proof slice, or coordination task.

## Relation vocabulary

| Relation | Meaning | State rule |
|---|---|---|
| `gh_mirror` | Bead and GitHub issue express the same outcome. | Scope and state must reconcile. |
| `gh_public_parent` | GitHub owns the public aggregate outcome; the Bead owns the internal program. | Children may close independently; GitHub closes only on aggregate proof. |
| `gh_implements` | Bead is an executable child, repair, or proof slice under a public issue. | No direct state lockstep. |
| `gh_supersedes_scope` | Bead is the current implementation authority and replaces stale GitHub solution wording. | Rewrite or replace the GitHub issue before execution. |
| `internal_only` | No public projection is required. | Bead lifecycle is independent of GitHub. |

Beads carry one corresponding label:

```text
tracker:gh-mirror
tracker:gh-public-parent
tracker:gh-implements
tracker:gh-supersedes-scope
tracker:internal-only
```

`external_ref=gh-N` names the public outcome but does not, by itself, imply mirroring. The tracker label and authority note determine the relation.

## Current public map

The machine-readable authority is `devtools/data/tracker-authority.json`.

Current important mappings include:

- #1844 ↔ `polylogue-fnm.4` (`gh_mirror`)
- #2308 ← `polylogue-lkrc`, `polylogue-hjpx`, `polylogue-yla8`, `polylogue-s8q` (`gh_implements`)
- #2309 ← `polylogue-jnj.9` (`gh_implements`)
- #2316 ↔ `polylogue-5hf` (`gh_mirror`)
- #2384 ← `polylogue-rii` (`gh_public_parent`) and `polylogue-rii.1` / `polylogue-rii.2` (`gh_implements`)
- #2391 ↔ `polylogue-20d.6` (`gh_mirror`)
- #2460 ← `polylogue-fs1` (`gh_supersedes_scope`)
- #2478 ↔ `polylogue-4ts.5` (`gh_mirror`)
- #3283 ← `polylogue-z9gh.9` (`gh_public_parent`) and `polylogue-z9gh.9.1` / `.1` / `.2` (`gh_implements`)

GitHub #2459 and #2461 were closed as duplicate public scope. Their Beads remain active internal implementation children under #2384.

## Applying the Beads projection

Run from a repository checkout with the live Beads Dolt database:

```bash
python devtools/reconcile_tracker_authority.py
python devtools/reconcile_tracker_authority.py --apply
```

The first command is a dry run. The apply command:

1. exports current live Beads state;
2. mutates only named rows;
3. applies through `devtools/bd_reimport_guard.py reconcile`;
4. writes a synchronization receipt;
5. re-exports `.beads/issues.jsonl` from the resulting live state.

Do not hand-edit `.beads/issues.jsonl` from a stale branch to perform this reconciliation.

## Hygiene rules

- A new public issue should name its Bead authority in a `Tracker authority` section.
- A new Bead with a GitHub projection should receive exactly one tracker-relation label.
- Closing a mirrored GitHub issue requires inspecting the corresponding Bead.
- Closing a public-parent issue requires aggregate acceptance proof, not merely closed children.
- A non-draft PR carries a versioned `pr-scope` record naming every assigned Bead, its whole-Bead disposition, typed evidence refs, and any open successor for residual work. CI and the merge boundary validate that record against the current head and canonical Bead records; they do not parse acceptance prose.
- A merged PR is evidence for a Bead, not automatic proof that its acceptance criteria are satisfied.
- When a Bead supersedes GitHub solution wording, update GitHub before treating the issue body as an implementation plan.
- Incident Beads should normally remain internal and link upward to a public trust/performance outcome rather than spawning a public issue per incident.
