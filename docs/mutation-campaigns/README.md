# Mutation Campaign Index

Latest recorded artifact per campaign.

| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cli-query` | `2026-03-12T14:08:09.840738+00:00` | `c1fd5ce60e82` | 1013 | 935 | 14 | 0 | yes | 71.48s |
| `cli-run` | `2026-03-12T07:01:48.281647+00:00` | `e07c4baebfe6` | 186 | 89 | 0 | 8 | no | 37.44s |
| `drive-client` | `2026-03-13T01:02:52.544157+00:00` | `033d5d6f3130` | 564 | 286 | 0 | 0 | yes | 21.75s |
| `filters` | `2026-03-13T01:02:52.561695+00:00` | `033d5d6f3130` | 457 | 45 | 97 | 0 | yes | 140.28s |
| `fts5` | `2026-03-11T06:25:00.240703+00:00` | `147e689d15ca` | 41 | 7 | 0 | 0 | no | 21.79s |
| `hybrid` | `2026-03-12T09:57:11.806235+00:00` | `eb43cfd48e98` | 113 | 20 | 0 | 3 | yes | 7.52s |
| `json` | `2026-03-11T06:24:41.072016+00:00` | `147e689d15ca` | 24 | 2 | 0 | 0 | no | 6.93s |
| `models` | `2026-03-12T09:56:43.482654+00:00` | `eb43cfd48e98` | 120 | 41 | 3 | 2 | yes | 27.95s |
| `pipeline-services` | `2026-03-12T03:40:45.985215+00:00` | `856caf495bab` | 841 | 648 | 84 | 35 | no | 207.35s |
| `providers-semantics` | `2026-03-12T14:09:59.887622+00:00` | `c1fd5ce60e82` | 784 | 489 | 3 | 0 | yes | 47.01s |
| `repository` | `2026-03-12T09:57:19.645335+00:00` | `eb43cfd48e98` | 534 | 134 | 40 | 1 | yes | 122.63s |
| `schema-core` | `2026-03-12T03:44:19.397336+00:00` | `856caf495bab` | 795 | 883 | 16 | 0 | no | 326.14s |
| `schema-inference` | `2026-03-12T03:35:21.908118+00:00` | `856caf495bab` | 561 | 707 | 30 | 0 | no | 309.28s |
| `schema-validation` | `2026-03-12T03:33:53.444194+00:00` | `856caf495bab` | 235 | 161 | 0 | 0 | no | 66.89s |
| `site-builder` | `2026-03-12T04:14:01.278470+00:00` | `58264c2c47be` | 240 | 224 | 10 | 0 | no | 108.46s |
| `source-detection` | `2026-03-12T14:09:59.882621+00:00` | `c1fd5ce60e82` | 934 | 217 | 0 | 0 | yes | 53.96s |
| `sources-parse` | `2026-03-12T14:09:59.893525+00:00` | `c1fd5ce60e82` | 3886 | 2019 | 7 | 0 | yes | 180.61s |
| `ui-core` | `2026-03-12T14:08:09.833874+00:00` | `c1fd5ce60e82` | 11 | 15 | 0 | 0 | yes | 5.23s |

## Notes

- Artifacts live in this directory as per-campaign JSON and Markdown files.
- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.
- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.
