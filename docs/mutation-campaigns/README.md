# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cli-query` | `2026-03-11T22:54:21.973050+00:00` | `a3440a0f1a4b` | 961 | 989 | 12 | 0 | no | 66.61s |
| `cli-run` | `2026-03-11T09:53:20.738093+00:00` | `2bdb267e93b7` | 183 | 92 | 0 | 8 | no | 63.80s |
| `drive-client` | `2026-03-11T21:40:31.542067+00:00` | `37e26aba2d3d` | 581 | 299 | 3 | 0 | no | 30.06s |
| `filters` | `2026-03-11T06:21:10.783620+00:00` | `147e689d15ca` | 475 | 5 | 117 | 0 | no | 175.18s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-11T06:24:07.341103+00:00` | `147e689d15ca` | 129 | 20 | 3 | 14 | no | 33.30s |
| `pipeline-services` | `2026-03-11T07:46:50.470356+00:00` | `d1e704d7a2ba` | 736 | 595 | 84 | 246 | no | 365.67s |
| `providers-semantics` | `2026-03-11T23:30:52.754765+00:00` | `315beb0f19f1` | 819 | 455 | 2 | 0 | no | 43.17s |
| `repository` | `2026-03-11T20:44:32.418308+00:00` | `b1f1d35bee28` | 568 | 74 | 66 | 0 | no | 197.55s |
| `schema-core` | `2026-03-11T07:42:10.649535+00:00` | `d1e704d7a2ba` | 792 | 895 | 7 | 0 | no | 279.69s |
| `schema-inference` | `2026-03-11T07:33:56.368190+00:00` | `d1e704d7a2ba` | 534 | 317 | 447 | 0 | no | 497.54s |
| `schema-validation` | `2026-03-11T07:32:42.298458+00:00` | `d1e704d7a2ba` | 235 | 161 | 0 | 0 | no | 43.36s |
| `site-builder` | `2026-03-11T09:58:59.595335+00:00` | `2bdb267e93b7` | 245 | 228 | 1 | 0 | no | 62.41s |
| `source-detection` | `2026-03-11T23:19:15.430163+00:00` | `844d52ee925d` | 825 | 324 | 2 | 0 | no | 76.62s |
| `sources-parse` | `2026-03-11T22:24:46.072807+00:00` | `47a9b1cff33f` | 3597 | 2319 | 31 | 0 | no | 265.78s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
