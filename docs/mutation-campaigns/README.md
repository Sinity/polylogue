# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cli-query` | `2026-03-12T04:14:00.022579+00:00` | `58264c2c47be` | 954 | 985 | 23 | 0 | no | 107.32s |
| `cli-run` | `2026-03-12T04:14:00.022530+00:00` | `58264c2c47be` | 167 | 21 | 87 | 8 | no | 106.61s |
| `drive-client` | `2026-03-11T21:40:31.542067+00:00` | `37e26aba2d3d` | 581 | 299 | 3 | 0 | no | 30.06s |
| `filters` | `2026-03-11T06:21:10.783620+00:00` | `147e689d15ca` | 475 | 5 | 117 | 0 | no | 175.18s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-11T06:24:07.341103+00:00` | `147e689d15ca` | 129 | 20 | 3 | 14 | no | 33.30s |
| `pipeline-services` | `2026-03-12T03:40:45.985215+00:00` | `856caf495bab` | 841 | 648 | 84 | 35 | no | 207.35s |
| `providers-semantics` | `2026-03-11T23:30:52.754765+00:00` | `315beb0f19f1` | 819 | 455 | 2 | 0 | no | 43.17s |
| `repository` | `2026-03-12T02:57:49.892990+00:00` | `3bdd3f02dc87` | 538 | 94 | 77 | 0 | no | 187.01s |
| `schema-core` | `2026-03-12T03:44:19.397336+00:00` | `856caf495bab` | 795 | 883 | 16 | 0 | no | 326.14s |
| `schema-inference` | `2026-03-12T03:35:21.908118+00:00` | `856caf495bab` | 561 | 707 | 30 | 0 | no | 309.28s |
| `schema-validation` | `2026-03-12T03:33:53.444194+00:00` | `856caf495bab` | 235 | 161 | 0 | 0 | no | 66.89s |
| `site-builder` | `2026-03-12T04:14:01.278470+00:00` | `58264c2c47be` | 240 | 224 | 10 | 0 | no | 108.46s |
| `source-detection` | `2026-03-11T23:19:15.430163+00:00` | `844d52ee925d` | 825 | 324 | 2 | 0 | no | 76.62s |
| `sources-parse` | `2026-03-11T22:24:46.072807+00:00` | `47a9b1cff33f` | 3597 | 2319 | 31 | 0 | no | 265.78s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
