# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cli-query` | `2026-03-11T19:26:25.486497+00:00` | `e759af23458d` | 963 | 1018 | 16 | 0 | no | 72.88s |
| `cli-run` | `2026-03-11T09:53:20.738093+00:00` | `2bdb267e93b7` | 183 | 92 | 0 | 8 | no | 63.80s |
| `drive-client` | `2026-03-11T17:59:08.012030+00:00` | `7e7c310037f9` | 553 | 327 | 4 | 0 | no | 31.81s |
| `filters` | `2026-03-11T06:21:10.783620+00:00` | `147e689d15ca` | 475 | 5 | 117 | 0 | no | 175.18s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-11T06:24:07.341103+00:00` | `147e689d15ca` | 129 | 20 | 3 | 14 | no | 33.30s |
| `pipeline-services` | `2026-03-11T07:46:50.470356+00:00` | `d1e704d7a2ba` | 736 | 595 | 84 | 246 | no | 365.67s |
| `providers-semantics` | `2026-03-11T19:19:47.469627+00:00` | `e759af23458d` | 847 | 487 | 2 | 0 | no | 49.62s |
| `repository` | `2026-03-11T19:27:44.024440+00:00` | `e759af23458d` | 569 | 104 | 35 | 0 | no | 231.95s |
| `schema-core` | `2026-03-11T07:42:10.649535+00:00` | `d1e704d7a2ba` | 792 | 895 | 7 | 0 | no | 279.69s |
| `schema-inference` | `2026-03-11T07:33:56.368190+00:00` | `d1e704d7a2ba` | 534 | 317 | 447 | 0 | no | 497.54s |
| `schema-validation` | `2026-03-11T07:32:42.298458+00:00` | `d1e704d7a2ba` | 235 | 161 | 0 | 0 | no | 43.36s |
| `site-builder` | `2026-03-11T09:58:59.595335+00:00` | `2bdb267e93b7` | 245 | 228 | 1 | 0 | no | 62.41s |
| `source-detection` | `2026-03-11T19:20:42.417653+00:00` | `e759af23458d` | 713 | 435 | 3 | 0 | no | 70.76s |
| `sources-parse` | `2026-03-11T19:22:10.711397+00:00` | `e759af23458d` | 3494 | 2467 | 11 | 0 | no | 241.31s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
