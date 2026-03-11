# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `filters` | `2026-03-11T06:21:10.783620+00:00` | `147e689d15ca` | 475 | 5 | 117 | 0 | no | 175.18s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-11T06:24:07.341103+00:00` | `147e689d15ca` | 129 | 20 | 3 | 14 | no | 33.30s |
| `pipeline-services` | `2026-03-11T06:31:25.625906+00:00` | `147e689d15ca` | 725 | 687 | 3 | 246 | no | 143.92s |
| `providers-semantics` | `2026-03-11T06:35:38.887231+00:00` | `147e689d15ca` | 162 | 588 | 0 | 432 | no | 35.24s |
| `repository` | `2026-03-11T06:33:49.870267+00:00` | `147e689d15ca` | 343 | 250 | 6 | 81 | no | 88.70s |
| `schema-core` | `2026-03-11T06:25:22.335205+00:00` | `147e689d15ca` | 765 | 900 | 29 | 0 | no | 205.63s |
| `schema-inference` | `2026-03-11T06:28:48.293670+00:00` | `147e689d15ca` | 536 | 759 | 3 | 0 | no | 130.77s |
| `schema-validation` | `2026-03-11T06:30:59.405898+00:00` | `147e689d15ca` | 229 | 167 | 0 | 0 | no | 25.91s |
| `source-detection` | `2026-03-11T06:35:18.888629+00:00` | `147e689d15ca` | 41 | 197 | 0 | 910 | no | 19.66s |
| `sources-parse` | `2026-03-11T06:36:14.451521+00:00` | `147e689d15ca` | 1353 | 2307 | 0 | 2094 | no | 133.77s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
