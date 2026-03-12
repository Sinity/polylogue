# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cli-query` | `2026-03-12T07:02:26.054534+00:00` | `e07c4baebfe6` | 992 | 964 | 6 | 0 | no | 53.71s |
| `cli-run` | `2026-03-12T07:01:48.281647+00:00` | `e07c4baebfe6` | 186 | 89 | 0 | 8 | no | 37.44s |
| `drive-client` | `2026-03-11T21:40:31.542067+00:00` | `37e26aba2d3d` | 581 | 299 | 3 | 0 | no | 30.06s |
| `filters` | `2026-03-12T07:03:20.122895+00:00` | `e07c4baebfe6` | 453 | 55 | 89 | 0 | no | 160.77s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-11T06:24:48.347998+00:00` | `147e689d15ca` | 112 | 21 | 0 | 3 | no | 11.57s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-12T07:00:56.287051+00:00` | `e07c4baebfe6` | 138 | 22 | 3 | 3 | no | 35.52s |
| `pipeline-services` | `2026-03-12T03:40:45.985215+00:00` | `856caf495bab` | 841 | 648 | 84 | 35 | no | 207.35s |
| `providers-semantics` | `2026-03-12T07:06:01.208485+00:00` | `e07c4baebfe6` | 785 | 489 | 2 | 0 | no | 47.61s |
| `repository` | `2026-03-12T02:57:49.892990+00:00` | `3bdd3f02dc87` | 538 | 94 | 77 | 0 | no | 187.01s |
| `schema-core` | `2026-03-12T03:44:19.397336+00:00` | `856caf495bab` | 795 | 883 | 16 | 0 | no | 326.14s |
| `schema-inference` | `2026-03-12T03:35:21.908118+00:00` | `856caf495bab` | 561 | 707 | 30 | 0 | no | 309.28s |
| `schema-validation` | `2026-03-12T03:33:53.444194+00:00` | `856caf495bab` | 235 | 161 | 0 | 0 | no | 66.89s |
| `site-builder` | `2026-03-12T04:14:01.278470+00:00` | `58264c2c47be` | 240 | 224 | 10 | 0 | no | 108.46s |
| `source-detection` | `2026-03-11T23:19:15.430163+00:00` | `844d52ee925d` | 825 | 324 | 2 | 0 | no | 76.62s |
| `sources-parse` | `2026-03-12T07:06:49.146273+00:00` | `e07c4baebfe6` | 3538 | 2365 | 9 | 0 | no | 205.09s |
| `ui-core` | `2026-03-12T07:01:35.777810+00:00` | `e07c4baebfe6` | 11 | 15 | 0 | 0 | no | 12.15s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
