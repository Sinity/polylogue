# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cli-query` | `2026-03-11T17:57:49.901466+00:00` | `7e7c310037f9` | 965 | 1022 | 10 | 0 | no | 64.79s |
| `cli-run` | `2026-03-11T09:53:20.738093+00:00` | `2bdb267e93b7` | 183 | 92 | 0 | 8 | no | 63.80s |
| `drive-client` | `2026-03-11T17:59:08.012030+00:00` | `7e7c310037f9` | 553 | 327 | 4 | 0 | no | 31.81s |
| `filters` | `2026-03-11T06:21:10.783620+00:00` | `147e689d15ca` | 475 | 5 | 117 | 0 | no | 175.18s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-11T06:24:07.341103+00:00` | `147e689d15ca` | 129 | 20 | 3 | 14 | no | 33.30s |
| `pipeline-services` | `2026-03-11T07:46:50.470356+00:00` | `d1e704d7a2ba` | 736 | 595 | 84 | 246 | no | 365.67s |
| `providers-semantics` | `2026-03-11T18:11:49.054310+00:00` | `a27de694650d` | 1009 | 620 | 2 | 0 | no | 62.23s |
| `repository` | `2026-03-11T18:00:19.507984+00:00` | `7e7c310037f9` | 527 | 130 | 51 | 0 | no | 255.62s |
| `schema-core` | `2026-03-11T07:42:10.649535+00:00` | `d1e704d7a2ba` | 792 | 895 | 7 | 0 | no | 279.69s |
| `schema-inference` | `2026-03-11T07:33:56.368190+00:00` | `d1e704d7a2ba` | 534 | 317 | 447 | 0 | no | 497.54s |
| `schema-validation` | `2026-03-11T07:32:42.298458+00:00` | `d1e704d7a2ba` | 235 | 161 | 0 | 0 | no | 43.36s |
| `site-builder` | `2026-03-11T09:58:59.595335+00:00` | `2bdb267e93b7` | 245 | 228 | 1 | 0 | no | 62.41s |
| `source-detection` | `2026-03-11T10:00:02.085699+00:00` | `2bdb267e93b7` | 563 | 455 | 3 | 127 | no | 85.65s |
| `sources-parse` | `2026-03-11T18:11:49.054341+00:00` | `a27de694650d` | 3644 | 2608 | 12 | 0 | no | 280.24s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
