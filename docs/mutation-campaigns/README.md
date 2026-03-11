# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `filters` | `2026-03-11T06:21:10.783620+00:00` | `147e689d15ca` | 475 | 5 | 117 | 0 | no | 175.18s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-11T06:24:07.341103+00:00` | `147e689d15ca` | 129 | 20 | 3 | 14 | no | 33.30s |
| `pipeline-services` | `2026-03-11T07:46:50.470356+00:00` | `d1e704d7a2ba` | 736 | 595 | 84 | 246 | no | 365.67s |
| `providers-semantics` | `2026-03-11T06:35:38.887231+00:00` | `147e689d15ca` | 162 | 588 | 0 | 432 | no | 35.24s |
| `repository` | `2026-03-11T06:33:49.870267+00:00` | `147e689d15ca` | 343 | 250 | 6 | 81 | no | 88.70s |
| `schema-core` | `2026-03-11T07:42:10.649535+00:00` | `d1e704d7a2ba` | 792 | 895 | 7 | 0 | no | 279.69s |
| `schema-inference` | `2026-03-11T07:33:56.368190+00:00` | `d1e704d7a2ba` | 534 | 317 | 447 | 0 | no | 497.54s |
| `schema-validation` | `2026-03-11T07:32:42.298458+00:00` | `d1e704d7a2ba` | 235 | 161 | 0 | 0 | no | 43.36s |
| `source-detection` | `2026-03-11T06:35:18.888629+00:00` | `147e689d15ca` | 41 | 197 | 0 | 910 | no | 19.66s |
| `sources-parse` | `2026-03-11T06:36:14.451521+00:00` | `147e689d15ca` | 1353 | 2307 | 0 | 2094 | no | 133.77s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
