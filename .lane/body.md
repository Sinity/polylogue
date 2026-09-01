Summary

Completes the reviewed triage fixes and addresses four follow-up findings from Codex review.

Problem

Legacy Claude cursors could mint semantic authority from unverified current bytes. Claude AI and AI Studio/Drive topology declarations omitted parser-carried data. Sinex publication encoded parser tool results before canonical outcomes were derived.

Solution

Legacy cursors retain nonsemantic authority and therefore take the established full-ingest path after a rewrite. Claude AI and AI Studio/Drive now declare their carried message parents and derived branch state. A source-level tool-outcome helper is shared by the archive writer and Sinex adapter, so publication uses the same canonical outcome derivation.

Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/sources/test_live_deferred_append_dedup.py`: 7 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/sinex/test_material_adapter.py`: 5 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/storage/test_tool_outcome.py`: 26 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/sources/test_origin_specs.py -k topology_capability_census_is_complete_and_typed`: 1 passed.
- `nix develop --accept-flake-config --command devtools verify --quick`: exit 0.

Residual risk

The complete corpus and live daemon were not run. The previously filed replay coverage gap remains in `polylogue-rkdej`.

Disposition

| Finding | Result |
| --- | --- |
| 3908111340 | Legacy Claude cursors no longer upgrade unverified current bytes. |
| 3908111349 | Claude AI topology declares carried parents and derived branch state. |
| 3908111353 | Sinex derives canonical tool outcomes before publication encoding. |
| 3908242286 | AI Studio/Drive topology declares carried parents and derived branch state. |

LANE-BRANCH: fix/codex-triage-rest
LANE-COMMIT: 033c8b8f5
LANE-QUICK: green
