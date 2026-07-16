---
title: "Replace actions view"
source_url: "https://chatgpt.com/share/6a4d620b-b324-83ed-83cd-9b73628ccc19"
conversation_id: "6a4d620b-b324-83ed-83cd-9b73628ccc19"
shared_conversation_id: "6a4d620b-b324-83ed-83cd-9b73628ccc19"
model: "gpt-5-5-pro"
message_count: 100
mapping_node_count: 101
create_time: 1783456267.73365
update_time: 1783456268.825209
---

# Replace actions view

## 1. system

[empty message]

## 2. system

[empty message]

## 3. system

[empty message]

## 4. user

Original custom instructions no longer available

## 5. user

_create_time: 1783433274.875_

[empty message]

## 6. tool

_create_time: 1783433277.196325_

[empty message]

## 7. tool

_create_time: 1783433279.7367597_

The output of this plugin was redacted.

## 8. assistant

_create_time: 1783433279.9044847_

[empty message]

## 9. assistant

_create_time: 1783433290.053561_

I’ll inspect the uploaded source and brief, trace the current `actions` view through writes, convergence, readers, and insight aggregation, then draft a concrete design covering DDL, ownership, rollout, tests, and the bead rewrite.

## 10. assistant

_create_time: 1783433279.863935_

bash -lc ls -lah /mnt/data && file /mnt/data/polylogue-all.tar\(87\).gz /mnt/data/01-analysis-actions-relation-materialization.md && sed -n '1,240p' /mnt/data/01-analysis-actions-relation-materialization.md

## 11. tool

_create_time: 1783433293.0241404_

The output of this plugin was redacted.

## 12. assistant

_create_time: 1783433293.023774_

bash -lc mkdir -p /mnt/data/polylogue_src && tar -xzf '/mnt/data/polylogue-all.tar(87).gz' -C /mnt/data/polylogue_src && find /mnt/data/polylogue_src -maxdepth 2 -type d | sed -n '1,80p' && find /mnt/data/polylogue_src -maxdepth 2 -type f | sed -n '1,80p'

## 13. tool

_create_time: 1783433296.4708347_

The output of this plugin was redacted.

## 14. assistant

_create_time: 1783433296.470454_

bash -lc mkdir -p /mnt/data/polylogue_work && tar -xzf /mnt/data/polylogue_src/polylogue/polylogue-working-tree.tar.gz -C /mnt/data/polylogue_work && find /mnt/data/polylogue_work -maxdepth 3 -type d | sed -n '1,80p' && find /mnt/data/polylogue_work -maxdepth 2 -type f | sed -n '1,120p'

## 15. tool

_create_time: 1783433299.7544901_

The output of this plugin was redacted.

## 16. assistant

_create_time: 1783433299.746129_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "CREATE VIEW actions|DROP VIEW IF EXISTS actions|FROM actions|JOIN actions|actions\b" polylogue tests docs .beads 2>/dev/null | sed -n '1,240p'

## 17. tool

_create_time: 1783433303.6729367_

The output of this plugin was redacted.

## 18. assistant

_create_time: 1783433303.672533_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "CREATE VIEW actions|FROM actions|JOIN actions|actions AS|DROP VIEW IF EXISTS actions|CREATE TABLE actions" polylogue tests docs -S

## 19. tool

_create_time: 1783433306.7828922_

The output of this plugin was redacted.

## 20. assistant

_create_time: 1783433306.782414_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '1,450p'

## 21. tool

_create_time: 1783433309.6646426_

The output of this plugin was redacted.

## 22. assistant

_create_time: 1783433309.664243_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/index.py | sed -n '188,258p'

## 23. tool

_create_time: 1783433312.119911_

The output of this plugin was redacted.

## 24. assistant

_create_time: 1783433312.119355_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '1,460p'

## 25. tool

_create_time: 1783433315.436098_

The output of this plugin was redacted.

## 26. assistant

_create_time: 1783433315.435691_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '460,1040p'

## 27. tool

_create_time: 1783433317.9281573_

The output of this plugin was redacted.

## 28. assistant

_create_time: 1783433317.927716_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "def _write_blocks|def _write_messages|def _replace_full_session|def _extract_prefix_tail|def _resolve_session_graph|def _existing_parent_session_id|def _provider_usage_cumulative_baseline" polylogue/storage/sqlite/archive_tiers/write.py

## 29. tool

_create_time: 1783433321.018668_

The output of this plugin was redacted.

## 30. assistant

_create_time: 1783433321.018208_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '1350,1645p' && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '2020,2115p' && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '3860,3950p'

## 31. tool

_create_time: 1783433324.3509667_

The output of this plugin was redacted.

## 32. assistant

_create_time: 1783433324.350413_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '1640,1715p' && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '2110,2195p' && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '2195,2315p'

## 33. tool

_create_time: 1783433328.3991952_

The output of this plugin was redacted.

## 34. assistant

_create_time: 1783433328.398587_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/write.py | sed -n '2315,2460p'

## 35. tool

_create_time: 1783433330.6643784_

The output of this plugin was redacted.

## 36. assistant

_create_time: 1783433330.663828_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/convergence_stages.py | sed -n '1,260p'

## 37. tool

_create_time: 1783433335.029219_

The output of this plugin was redacted.

## 38. assistant

_create_time: 1783433335.028586_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/convergence_stages.py | sed -n '260,620p'

## 39. tool

_create_time: 1783433337.5924625_

The output of this plugin was redacted.

## 40. assistant

_create_time: 1783433337.591599_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/convergence_stages.py | sed -n '620,1120p'

## 41. tool

_create_time: 1783433341.2485607_

The output of this plugin was redacted.

## 42. assistant

_create_time: 1783433341.247958_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/convergence_stages.py | sed -n '1120,1640p'

## 43. tool

_create_time: 1783433343.5980382_

The output of this plugin was redacted.

## 44. assistant

_create_time: 1783433343.597484_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/convergence.py | sed -n '1,260p'

## 45. tool

_create_time: 1783433346.3281665_

The output of this plugin was redacted.

## 46. assistant

_create_time: 1783433346.327658_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/convergence.py | sed -n '260,620p'

## 47. tool

_create_time: 1783433350.8794432_

The output of this plugin was redacted.

## 48. assistant

_create_time: 1783433350.878882_

bash -lc cd /mnt/data/polylogue_work/polylogue && ls -la .beads 2>/dev/null || true && test -f .beads/issues.jsonl && wc -l .beads/issues.jsonl && rg -n "polylogue-xnkf|polylogue-4ts\.4|polylogue-4ts\.6|polylogue-f2qv\.1|xnkf|4ts|f2qv" .beads/issues.jsonl polylogue-beads* docs polylogue 2>/dev/null | sed -n '1,240p'

## 49. tool

_create_time: 1783433354.813116_

The output of this plugin was redacted.

## 50. assistant

_create_time: 1783433354.812436_

bash -lc cd /mnt/data/polylogue_work/polylogue && ls -la | sed -n '1,80p' && find . -maxdepth 2 -type f -name 'issues.jsonl' -o -name '*beads*' | sed -n '1,120p'

## 51. tool

_create_time: 1783433358.2051835_

The output of this plugin was redacted.

## 52. assistant

_create_time: 1783433358.204649_

bash -lc cd /mnt/data/polylogue_src/polylogue && rg -n "polylogue-xnkf|polylogue-4ts\.4|polylogue-4ts\.6|polylogue-f2qv\.1|xnkf|4ts|f2qv" polylogue-beads* | sed -n '1,240p'

## 53. tool

_create_time: 1783433361.5172567_

The output of this plugin was redacted.

## 54. assistant

_create_time: 1783433361.516662_

bash -lc cd /mnt/data/polylogue_src/polylogue && python - <<'PY'
import json
ids={'polylogue-xnkf','polylogue-4ts.4','polylogue-4ts.6','polylogue-f2qv.1','polylogue-4ts','polylogue-f2qv'}
for path in ['polylogue-beads-export.jsonl']:
    for line in open(path):
        rec=json.loads(line)
        if rec.get('_type')=='issue' and rec.get('id') in ids:
            print('\n===', rec['id'], rec.get('title'))
            for k in ['status','issue_type','priority','description','design','acceptance_criteria','notes','labels','dependencies']:
                print(f'{k}:', rec.get(k))
PY

## 55. tool

_create_time: 1783433367.4075074_

The output of this plugin was redacted.

## 56. assistant

_create_time: 1783433367.406767_

bash -lc cd /mnt/data/polylogue_work/polylogue && python - <<'PY'
import subprocess, pathlib, re
root=pathlib.Path('.')
pattern=re.compile(r'(FROM|JOIN) actions')
for p in ['polylogue/storage/sqlite/archive_tiers/archive.py','polylogue/storage/sqlite/queries/tool_usage.py','polylogue/storage/sqlite/queries/filter_builder.py','polylogue/storage/repository/archive/sessions.py','polylogue/sources/import_explain.py','polylogue/api/archive.py','polylogue/cli/commands/status.py']:
    path=root/p
    if not path.exists(): continue
    lines=path.read_text().splitlines()
    locs=[i for i,l in enumerate(lines,1) if pattern.search(l)]
    if not locs: continue
    print('\n###',p)
    for loc in locs:
        print(f'-- line {loc}')
        start=max(1,loc-8); end=min(len(lines),loc+14)
        for i in range(start,end+1):
            print(f'{i:5}: {lines[i-1]}')
        print()
PY

## 57. tool

_create_time: 1783433377.3179927_

The output of this plugin was redacted.

## 58. assistant

_create_time: 1783433377.317393_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "FROM actions|JOIN actions" polylogue -S

## 59. tool

_create_time: 1783433382.4510653_

The output of this plugin was redacted.

## 60. assistant

_create_time: 1783433382.450273_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '450,560p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '2940,3060p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '5320,5525p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '6885,6965p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7180,7250p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7540,7595p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7825,7975p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '8500,8565p'

## 61. tool

_create_time: 1783433387.3819966_

The output of this plugin was redacted.

## 62. assistant

_create_time: 1783433387.360311_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '3020,3065p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7545,7585p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7220,7245p'

## 63. tool

_create_time: 1783433391.5473948_

The output of this plugin was redacted.

## 64. assistant

_create_time: 1783433391.546756_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7240,7355p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7355,7465p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7465,7558p'

## 65. tool

_create_time: 1783433397.5018647_

The output of this plugin was redacted.

## 66. assistant

_create_time: 1783433397.50115_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "def _query_unit_time_expression|_action_sequence_clause|action_sequence" polylogue/storage/sqlite/archive_tiers/archive.py && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '7050,7165p'

## 67. tool

_create_time: 1783433401.1152236_

The output of this plugin was redacted.

## 68. assistant

_create_time: 1783433401.11458_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/archive.py | sed -n '8015,8095p'

## 69. tool

_create_time: 1783433403.9468994_

The output of this plugin was redacted.

## 70. assistant

_create_time: 1783433403.945905_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/queries/tool_usage.py | sed -n '1,230p' && nl -ba polylogue/storage/sqlite/queries/filter_builder.py | sed -n '1,230p'

## 71. tool

_create_time: 1783433407.5108404_

The output of this plugin was redacted.

## 72. assistant

_create_time: 1783433420.716856_

[empty message]

## 73. assistant

_create_time: 1783433407.510107_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba tests/unit/storage/test_archive_tiers_write.py | sed -n '190,260p' && nl -ba tests/unit/storage/test_fts5.py | sed -n '330,520p' && nl -ba tests/unit/storage/test_archive_tiers_ddl.py | sed -n '170,215p' && nl -ba tests/unit/storage/test_archive_tiers_assertions.py | sed -n '210,270p' && nl -ba tests/unit/storage/test_store_ops.py | sed -n '520,565p' && nl -ba tests/unit/pipeline/test_resilience.py | sed -n '760,790p'

## 74. tool

_create_time: 1783433421.586036_

The output of this plugin was redacted.

## 75. assistant

_create_time: 1783433424.251694_

[empty message]

## 76. assistant

_create_time: 1783433421.585301_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "actions view|FROM actions|JOIN actions|PRAGMA table_info\(actions\)|CREATE TABLE actions|CREATE VIEW actions" tests -S

## 77. tool

_create_time: 1783433424.7731156_

The output of this plugin was redacted.

## 78. assistant

_create_time: 1783433430.341277_

[empty message]

## 79. assistant

_create_time: 1783433424.77245_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "INDEX_SCHEMA_VERSION|schema_version|user_version|ArchiveTier|initialize_archive_database|derived" polylogue/storage/sqlite/archive_tiers polylogue/storage/runtime docs/internals.md | sed -n '1,240p'

## 80. tool

_create_time: 1783433428.182503_

The output of this plugin was redacted.

## 81. assistant

_create_time: 1783433428.181716_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/sqlite/archive_tiers/bootstrap.py | sed -n '1,220p' && nl -ba polylogue/storage/sqlite/archive_tiers/archive_plan.py | sed -n '1,150p' && nl -ba docs/internals.md | sed -n '90,130p' && nl -ba docs/internals.md | sed -n '275,305p' && nl -ba docs/internals.md | sed -n '600,630p'

## 82. tool

_create_time: 1783433431.9091587_

The output of this plugin was redacted.

## 83. assistant

_create_time: 1783433435.40617_

[empty message]

## 84. assistant

_create_time: 1783433431.897959_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba docs/internals.md | sed -n '160,185p' && nl -ba docs/internals.md | sed -n '645,660p'

## 85. tool

_create_time: 1783433435.5695064_

The output of this plugin was redacted.

## 86. assistant

_create_time: 1783433435.5688_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "FROM actions|JOIN actions|actions" polylogue/storage/insights polylogue/insights polylogue/archive/semantic polylogue/archive/actions -S | sed -n '1,240p'

## 87. tool

_create_time: 1783433445.2767649_

The output of this plugin was redacted.

## 88. assistant

_create_time: 1783433452.433509_

[empty message]

## 89. assistant

_create_time: 1783433445.276068_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/storage/insights/session/rebuild.py | sed -n '820,895p' && nl -ba polylogue/storage/insights/session/profiles.py | sed -n '120,160p' && nl -ba polylogue/storage/insights/session/profiles.py | sed -n '760,795p' && nl -ba polylogue/insights/tool_usage.py | sed -n '1,120p' && nl -ba polylogue/insights/affordance_usage.py | sed -n '1,240p'

## 90. tool

_create_time: 1783433450.7863224_

The output of this plugin was redacted.

## 91. assistant

_create_time: 1783433458.313473_

[empty message]

## 92. assistant

_create_time: 1783433450.785566_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/archive/actions/actions.py | sed -n '1,230p' && nl -ba polylogue/archive/actions/fields.py | sed -n '1,200p'

## 93. tool

_create_time: 1783433454.013694_

The output of this plugin was redacted.

## 94. assistant

_create_time: 1783433681.680529_

{"queries":["+(materialized actions relation) fan-out NULL-tool_id pairing statuses derived regime deliverables --QDF=1","actions VIEW LEFT JOIN tool_use tool_result tool_id session_id fan-out NULL-tool_id blindness --QDF=1"],"source_filter":["files_uploaded_in_conversation"]}

## 95. tool

_create_time: 1783433683.8189607_

The output of this plugin was redacted.

## 96. tool

_create_time: 1783433684.4564621_

The output of this plugin was redacted.

## 97. assistant

_create_time: 1783433837.759563_

[empty message]

## 98. assistant

_create_time: 1783434377.389602_

[empty message]

## 99. assistant

_create_time: 1783434377.389602_

[empty message]

## 100. assistant

_create_time: 1783434377.389602_

The decision is: **replace the `actions` VIEW with a materialized `actions` table** in `index.db`, refreshed by a daemon convergence stage. A smarter VIEW can fix the simplest exact-id fan-out, but it loses once NULL-id nearest-following pairing, orphan-result visibility, duplicate evidence, status typing, lineage grain, and hot aggregate performance are all requirements. The uploaded brief asks for exactly this decision, with typed pairing semantics, derived-regime refresh, lineage-grain honesty, rollout, tests, and a bead rewrite. fileciteturn1file0 fileciteturn0file0

I inspected the package. The current `actions` relation is a VIEW in `polylogue/storage/sqlite/archive_tiers/index.py:328-347`. It selects every `tool_use` block and left-joins every `tool_result` block with the same `(session_id, tool_id)`. That explains both failures: duplicate `toolu_*` ids produce `N × M` rows, and `NULL` ids never join. The bead export in the package strengthens the brief’s evidence: `polylogue-xnkf` records a real Claude Code sample where the same `toolu_016VnpHdbHMRiRUVcJn5oWyh` appears as two `tool_use` and two `tool_result` blocks, yielding four action rows for one logical action; its later notes record 200 fan-out-affected tool-use rows archive-wide and 17,983 NULL-id tool-use/tool-result blocks, with ChatGPT exports dominating the NULL-result side.

The table should be physical-grain, not logical-composed-grain. `write.py:277-281` and `write.py:3298-3307` show that prefix-sharing children physically store only divergent tails. `session_links` documents the same invariant in `index.py:386-395`. Materializing logical composed actions for every child session would count inherited parent-prefix actions once for the parent and again for every child, recreating the same class of double-counting called out by `polylogue-4ts.4`, `polylogue-4ts.6`, and `polylogue-f2qv.1`. Existing aggregate consumers should therefore count **physical action evidence** by default. Logical transcript readers can compose parent-prefix actions at read time later, but that should be a separate surface with an explicit `grain=logical`, not the canonical aggregate table.

“Following” is still defined against transcript order: `(message.position, message.variant_index, block.position, block_id)`. For prefix-sharing sessions, the materializer resolves the composed ordinal for the child’s own tail rows, but candidate rows remain limited to the physical session’s own blocks. That respects lineage ordering without letting inherited parent-prefix evidence leak into child physical aggregates.

The pairing semantics are:

`exact-pair`: canonical non-empty `tool_id` use group has at least one result with the same non-empty `tool_id` in the same physical session. Duplicate uses with the same `(session_id, tool_id)` collapse into one logical action. Duplicate same-id results are recorded as duplicate evidence, not emitted as more rows.

`nearest-following-result`: after exact-pair claims are reserved, a use with no exact result is paired to the nearest unclaimed `tool_result` whose transcript order is strictly greater than the use order. This handles ChatGPT-style “use has id, result has NULL” and provider formats where both sides lack ids. This is a typed heuristic, never disguised as exact evidence.

`unpaired-use`: a tool-use group remains without any result after exact and nearest passes. Outcome columns stay `NULL`.

`orphan-result`: a tool-result block remains unclaimed after exact and nearest passes. It gets a row so result-only failures and NULL-id blind spots become visible instead of disappearing from the action relation.

I would not store full `output_text` in the table. `index.db` is already multi-GB, and duplicating tool-result text would add storage proportional to result output size. The table should store stable pairing metadata, outcome fields, output presence/byte counts, and block ids. Readers that need output text should join `blocks` through `tool_result_block_id`. This still replaces the `actions` VIEW with a materialized `actions` table; it just avoids copying large evidence text.

DDL:

```sql
-- index.py: bump INDEX_SCHEMA_VERSION from 24 to 25.

CREATE TABLE IF NOT EXISTS actions (
    action_id                       TEXT PRIMARY KEY,

    session_id                      TEXT NOT NULL
                                      REFERENCES sessions(session_id) ON DELETE CASCADE,

    -- Physical-grain order in this session's own stored action evidence.
    action_grain                    TEXT NOT NULL DEFAULT 'physical'
                                      CHECK (action_grain = 'physical'),
    action_index                    INTEGER NOT NULL CHECK (action_index >= 0),

    pairing_version                 INTEGER NOT NULL CHECK (pairing_version > 0),
    pairing_status                  TEXT NOT NULL CHECK (
                                          pairing_status IN (
                                              'exact-pair',
                                              'nearest-following-result',
                                              'unpaired-use',
                                              'orphan-result'
                                          )
                                      ),

    -- Anchor for existing action-row consumers. For use-backed rows this is the
    -- canonical tool-use message. For orphan-result rows it is the result message.
    message_id                      TEXT NOT NULL
                                      REFERENCES messages(message_id) ON DELETE CASCADE,
    message_position                INTEGER NOT NULL CHECK (message_position >= 0),
    message_variant_index           INTEGER NOT NULL CHECK (message_variant_index >= 0),
    block_position                  INTEGER NOT NULL CHECK (block_position >= 0),

    tool_use_block_id               TEXT
                                      REFERENCES blocks(block_id) ON DELETE CASCADE,
    tool_use_message_id             TEXT
                                      REFERENCES messages(message_id) ON DELETE CASCADE,
    tool_use_message_position       INTEGER CHECK (
                                          tool_use_message_position IS NULL
                                          OR tool_use_message_position >= 0
                                      ),
    tool_use_message_variant_index  INTEGER CHECK (
                                          tool_use_message_variant_index IS NULL
                                          OR tool_use_message_variant_index >= 0
                                      ),
    tool_use_block_position         INTEGER CHECK (
                                          tool_use_block_position IS NULL
                                          OR tool_use_block_position >= 0
                                      ),

    tool_result_block_id            TEXT
                                      REFERENCES blocks(block_id) ON DELETE CASCADE,
    tool_result_message_id          TEXT
                                      REFERENCES messages(message_id) ON DELETE CASCADE,
    tool_result_message_position    INTEGER CHECK (
                                          tool_result_message_position IS NULL
                                          OR tool_result_message_position >= 0
                                      ),
    tool_result_message_variant_index INTEGER CHECK (
                                          tool_result_message_variant_index IS NULL
                                          OR tool_result_message_variant_index >= 0
                                      ),
    tool_result_block_position      INTEGER CHECK (
                                          tool_result_block_position IS NULL
                                          OR tool_result_block_position >= 0
                                      ),

    -- Canonical tool-use fields. For orphan-result, tool_id may come from the
    -- result block and tool_name/semantic/tool_input are normally NULL.
    tool_name                       TEXT,
    tool_id                         TEXT,
    semantic_type                   TEXT,
    tool_input                      TEXT CHECK (
                                          tool_input IS NULL OR json_valid(tool_input)
                                      ),
    tool_command                    TEXT GENERATED ALWAYS AS (
                                          json_extract(tool_input, '$.command')
                                      ) VIRTUAL,
    tool_path                       TEXT GENERATED ALWAYS AS (
                                          COALESCE(
                                              json_extract(tool_input, '$.file_path'),
                                              json_extract(tool_input, '$.path')
                                          )
                                      ) VIRTUAL,

    -- Provider-reported result outcome. NULL means unknown.
    tool_result_is_error            INTEGER CHECK (
                                          tool_result_is_error IN (0, 1)
                                          OR tool_result_is_error IS NULL
                                      ),
    tool_result_exit_code           INTEGER,

    -- Output storage stays in blocks.text. These columns support coverage
    -- aggregates without duplicating multi-GB text evidence.
    tool_result_has_text            INTEGER NOT NULL DEFAULT 0
                                      CHECK (tool_result_has_text IN (0, 1)),
    tool_result_text_bytes          INTEGER CHECK (
                                          tool_result_text_bytes IS NULL
                                          OR tool_result_text_bytes >= 0
                                      ),

    exact_tool_id_match             INTEGER NOT NULL DEFAULT 0
                                      CHECK (exact_tool_id_match IN (0, 1)),
    nearest_following_distance      INTEGER CHECK (
                                          nearest_following_distance IS NULL
                                          OR nearest_following_distance > 0
                                      ),

    duplicate_tool_use_count        INTEGER NOT NULL DEFAULT 0
                                      CHECK (duplicate_tool_use_count >= 0),
    duplicate_tool_result_count     INTEGER NOT NULL DEFAULT 0
                                      CHECK (duplicate_tool_result_count >= 0),

    source_tool_use_block_ids_json  TEXT NOT NULL DEFAULT '[]'
                                      CHECK (json_valid(source_tool_use_block_ids_json)),
    source_tool_result_block_ids_json TEXT NOT NULL DEFAULT '[]'
                                      CHECK (json_valid(source_tool_result_block_ids_json)),

    materialized_at_ms              INTEGER NOT NULL CHECK (materialized_at_ms >= 0),

    CHECK ((pairing_status = 'orphan-result') = (tool_use_block_id IS NULL)),
    CHECK ((pairing_status = 'unpaired-use') = (tool_result_block_id IS NULL)),
    CHECK (
        pairing_status NOT IN ('exact-pair', 'nearest-following-result')
        OR tool_result_block_id IS NOT NULL
    ),
    CHECK (
        pairing_status = 'orphan-result'
        OR tool_use_block_id IS NOT NULL
    ),
    CHECK (
        pairing_status != 'exact-pair'
        OR (
            tool_id IS NOT NULL
            AND tool_id != ''
            AND exact_tool_id_match = 1
            AND nearest_following_distance IS NULL
        )
    ),
    CHECK (
        pairing_status != 'nearest-following-result'
        OR (
            exact_tool_id_match = 0
            AND nearest_following_distance IS NOT NULL
            AND nearest_following_distance > 0
        )
    ),
    CHECK (
        tool_result_block_id IS NOT NULL
        OR (
            tool_result_is_error IS NULL
            AND tool_result_exit_code IS NULL
            AND tool_result_has_text = 0
            AND tool_result_text_bytes IS NULL
        )
    )
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_session_action_index
ON actions(session_id, action_index);

CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_tool_use_block
ON actions(tool_use_block_id)
WHERE tool_use_block_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_tool_result_block
ON actions(tool_result_block_id)
WHERE tool_result_block_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_session_tool_id_use
ON actions(session_id, tool_id)
WHERE tool_id IS NOT NULL
  AND tool_id != ''
  AND tool_use_block_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_actions_session_status_order
ON actions(session_id, pairing_status, action_index);

CREATE INDEX IF NOT EXISTS idx_actions_pairing_status
ON actions(pairing_status, session_id);

CREATE INDEX IF NOT EXISTS idx_actions_message
ON actions(message_id, action_index);

CREATE INDEX IF NOT EXISTS idx_actions_tool_use_message
ON actions(tool_use_message_id, tool_use_block_position)
WHERE tool_use_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_actions_tool_result_message
ON actions(tool_result_message_id, tool_result_block_position)
WHERE tool_result_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_actions_tool_id
ON actions(session_id, tool_id)
WHERE tool_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_actions_tool_name
ON actions(COALESCE(NULLIF(LOWER(tool_name), ''), 'unknown'), session_id)
WHERE tool_use_block_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_actions_semantic_type
ON actions(COALESCE(NULLIF(semantic_type, ''), 'tool_use'), session_id)
WHERE tool_use_block_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_actions_tool_path
ON actions(session_id, tool_path)
WHERE tool_path IS NOT NULL AND tool_path != '';

CREATE INDEX IF NOT EXISTS idx_actions_result_outcome
ON actions(tool_result_is_error, tool_result_exit_code, session_id)
WHERE tool_result_block_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS actions_materialization (
    session_id                       TEXT PRIMARY KEY
                                       REFERENCES sessions(session_id) ON DELETE CASCADE,
    pairing_version                  INTEGER NOT NULL CHECK (pairing_version > 0),
    source_hash                      BLOB NOT NULL CHECK (length(source_hash) = 32),
    session_content_hash             BLOB NOT NULL CHECK (length(session_content_hash) = 32),
    source_tool_use_block_count      INTEGER NOT NULL CHECK (source_tool_use_block_count >= 0),
    source_tool_result_block_count   INTEGER NOT NULL CHECK (source_tool_result_block_count >= 0),
    action_count                     INTEGER NOT NULL CHECK (action_count >= 0),
    tool_action_count                INTEGER NOT NULL CHECK (tool_action_count >= 0),
    exact_pair_count                 INTEGER NOT NULL CHECK (exact_pair_count >= 0),
    nearest_following_result_count   INTEGER NOT NULL CHECK (nearest_following_result_count >= 0),
    unpaired_use_count               INTEGER NOT NULL CHECK (unpaired_use_count >= 0),
    orphan_result_count              INTEGER NOT NULL CHECK (orphan_result_count >= 0),
    materialized_at_ms               INTEGER NOT NULL CHECK (materialized_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_actions_materialization_version
ON actions_materialization(pairing_version, session_id);
```

The pairing implementation should live as a shared pure module, for example `polylogue/storage/actions/materialization.py`, so SQL refresh, public API action extraction, and insight builders do not retain divergent pairing logic.

Sketch:

```python
ACTION_PAIRING_VERSION = 1
_DAEMON_ACTION_SESSION_PAGE_SIZE = 250

@dataclass(frozen=True, slots=True)
class ActionSourceBlock:
    session_id: str
    message_id: str
    message_position: int
    message_variant_index: int
    block_id: str
    block_position: int
    block_type: Literal["tool_use", "tool_result"]
    text_len: int | None
    has_text: bool
    tool_name: str | None
    tool_id: str | None
    tool_input: str | None
    semantic_type: str | None
    tool_result_is_error: int | None
    tool_result_exit_code: int | None

def pair_action_blocks(
    session_id: str,
    rows: Sequence[ActionSourceBlock],
    *,
    now_ms: int,
    pairing_version: int = ACTION_PAIRING_VERSION,
) -> tuple[list[ActionInsertRow], ActionMaterializationState]:
    """
    Input rows are ordered by:
        message_position, message_variant_index, block_position, block_id

    Pass 1: group non-empty tool_use.tool_id by (session_id, tool_id).
            Keep first use as canonical; record duplicate use block ids.

    Pass 2: exact-pair canonical non-empty ids to same-id result groups.
            Claim all same-id result blocks as evidence for that one action.
            Canonical result is the first same-id result following the canonical
            use, or the first same-id result if all are before the use. If all
            are before the use, record it as exact evidence but not nearest
            evidence; do not silently mark success.

    Pass 3: for remaining use groups, greedily claim the nearest unclaimed
            result whose order key is strictly greater than the use order key.

    Pass 4: emit unpaired-use rows for remaining use groups and orphan-result
            rows for remaining result blocks.

    Finally sort emitted rows by anchor order and assign action_index 0..N-1.
    """
```

The daemon convergence stage should follow the existing `fts` and `insights` pattern in `polylogue/daemon/convergence_stages.py`. `ConvergenceStage` already supports `check`, `execute`, `check_many`, `execute_many`, `check_sessions`, `execute_sessions`, and `false_means_pending` in `convergence.py:61-83`; `false_means_pending` is handled in `convergence.py:122-139` and rechecked for session debt in `convergence.py:467-470`.

Sketch:

```python
def make_actions_stage(db_path: Path) -> ConvergenceStage:
    def check(path: Path) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is None:
            return False
        conn = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path)
            return bool(_archive_actions_check_sessions(conn, session_ids, limit=1))
        finally:
            conn.close()

    def check_many(paths: Sequence[Path]) -> set[Path]:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is None or not paths:
            return set()
        conn = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            all_ids = tuple(dict.fromkeys(
                session_id for ids in by_path.values() for session_id in ids
            ))
            stale = set(_archive_actions_check_sessions(conn, all_ids))
            return {path for path, ids in by_path.items() if stale.intersection(ids)}
        finally:
            conn.close()

    def execute(path: Path) -> StageExecuteReturn:
        return execute_many((path,))

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is None or not paths:
            return True
        conn = _open_archive_insight_write_connection(archive_db)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            session_ids = tuple(dict.fromkeys(
                session_id for ids in by_path.values() for session_id in ids
            ))
            return _archive_actions_execute_sessions(conn, session_ids)
        finally:
            conn.close()

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is None or not session_ids:
            return set()
        conn = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True, timeout=5.0)
        try:
            return set(_archive_actions_check_sessions(conn, session_ids))
        finally:
            conn.close()

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is None or not session_ids:
            return True
        conn = _open_archive_insight_write_connection(archive_db)
        try:
            return _archive_actions_execute_sessions(conn, session_ids)
        finally:
            conn.close()

    return ConvergenceStage(
        name="actions",
        description="Materialize typed tool-use/tool-result action pairings",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        cpu_bound=False,
        false_means_pending=True,
    )
```

`make_default_convergence_stages` should become:

```python
def make_default_convergence_stages(db_path: Path) -> tuple[ConvergenceStage, ...]:
    return (
        make_fts_stage(db_path),
        make_actions_stage(db_path),
        make_embed_stage(db_path),
        make_insights_stage(db_path),
    )
```

Actions should run before insights because the current insight path still has old pairing behavior. `archive/actions/parsing.py:56-64` only records outputs for non-empty result ids, and `archive/actions/parsing.py:76-113` emits one call per `tool_use`; `archive/semantic/facts.py:269-312` aggregates those message-level actions; `insights/transforms.py:1573-1616` and `1709-1766` also use id maps. The fix should either make those paths call the same pure pairer or make insight rebuild read from materialized `actions` rows after the actions stage is ready. Do not leave SQL actions fixed while insight facts retain the old construct.

The refresh function should be deterministic and bounded:

```python
def _archive_actions_execute_sessions(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> StageExecutionResult:
    ids = _archive_existing_session_ids(conn, session_ids)
    stale = _archive_actions_check_sessions(conn, ids)
    page = tuple(stale[:_DAEMON_ACTION_SESSION_PAGE_SIZE])

    if not page:
        return StageExecutionResult(success=True)

    counts = refresh_actions_for_sessions(conn, page)
    conn.commit()

    remaining = _archive_actions_check_sessions(conn, stale)
    return StageExecutionResult(
        success=not remaining,
        timings_s={
            "actions.sessions": float(len(page)),
            "actions.rows": float(counts.action_count),
            "actions.remaining": float(len(remaining)),
        },
    )
```

Staleness should be based on `actions_materialization`, not by trusting timestamps alone. For a targeted session, read ordered `tool_use`/`tool_result` rows joined to `messages`, compute a SHA-256 over the pairing-relevant source fields, and compare it to `actions_materialization.source_hash` and `pairing_version`. The hash should include order coordinates, block ids, block type, normalized `tool_id`, `tool_name`, `tool_input`, `semantic_type`, outcome fields, and `length(text)`/has-text for result blocks. Full result text need not be pulled into Python because the table does not store it.

For a global debt scan, use a cheap first pass over counts and version:

```sql
WITH source_counts AS (
    SELECT
        session_id,
        SUM(CASE WHEN block_type = 'tool_use' THEN 1 ELSE 0 END) AS use_count,
        SUM(CASE WHEN block_type = 'tool_result' THEN 1 ELSE 0 END) AS result_count
    FROM blocks
    WHERE block_type IN ('tool_use', 'tool_result')
    GROUP BY session_id
)
SELECT s.session_id
FROM sessions s
LEFT JOIN source_counts sc ON sc.session_id = s.session_id
LEFT JOIN actions_materialization am ON am.session_id = s.session_id
WHERE am.session_id IS NULL
   OR am.pairing_version != ?
   OR am.session_content_hash != s.content_hash
   OR am.source_tool_use_block_count != COALESCE(sc.use_count, 0)
   OR am.source_tool_result_block_count != COALESCE(sc.result_count, 0)
ORDER BY s.session_id
LIMIT ?;
```

That count probe can over-select but must not under-select. The exact hash check runs before refresh.

Rollout is a derived-regime schema bump, not a migration. Change `INDEX_SCHEMA_VERSION = 24` to `25` in `polylogue/storage/sqlite/archive_tiers/index.py`. Replace the VIEW DDL at `index.py:328-347` with the table/state DDL above. No source, user, ops, or embeddings tier bump is required. `archive_tiers/bootstrap.py:36-48` marks `index.db` rebuildable, `bootstrap.py:149-180` rejects mismatched nonzero tier versions unless a durable migration applies, and `schema.py:1-8` says the runtime does not patch out-of-band layouts. The existing mismatch message in `schema_bootstrap.py:55-58` already points operators to move aside/reset the index and rebuild from source.

At ~16K sessions and a multi-GB `index.db`, the rebuild cost remains dominated by the existing source-to-index pass, FTS, embeddings if enabled, and session insights. The new actions stage adds a linear pass over `tool_use`/`tool_result` blocks plus one write per logical action or orphan result. Because output text is not duplicated, storage growth is proportional to action metadata, ids, JSON evidence arrays, and state rows, not to tool-output byte volume. The daemon should page refreshes, e.g. 250 sessions per transaction, and return `False` with `false_means_pending=True` whenever more stale sessions remain.

Reader switches found in the package:

`devtools/affordance_usage.py:1112` and `1157`: switch joins from `a.message_id` to the table’s anchor `message_id`; select `a.tool_result_is_error AS is_error`, `a.tool_result_exit_code AS exit_code`; join `blocks rb ON rb.block_id = a.tool_result_block_id` only when output text is needed. For tool-call counts, filter `a.tool_use_block_id IS NOT NULL`.

`devtools/daemon_workload_probe.py:1521`: replace bare `COUNT(*) FROM actions` with `COUNT(*) WHERE tool_use_block_id IS NOT NULL` for existing `action_count`; add total row count and `orphan_result_count` separately.

`polylogue/api/archive.py:775-776`: session action-type facets should count use-backed rows only. Add a separate pairing-status facet from `pairing_status` so orphan/nearest/unpaired coverage becomes visible.

`polylogue/api/archive.py:1668-1685` and `3132-3155`: `_actions_for_session`, `get_actions`, and `get_actions_batch` should stop rebuilding old semantics from content blocks. Use repository materialized rows or the shared pure pairer. Update the docstring that currently says actions are derived on read.

`polylogue/cli/commands/status.py:802`: report `tool_action_count`, `action_row_count`, `orphan_result_count`, and missing/stale action materialization debt from `actions_materialization`.

`polylogue/sources/import_explain.py:323`: use use-backed count for “actions produced” and optionally include `orphan_results`.

`polylogue/storage/repository/archive/sessions.py:342-347`: group action kinds over `WHERE tool_use_block_id IS NOT NULL`; add pairing status counts if returned payload supports it.

`polylogue/storage/sqlite/queries/tool_usage.py:110-121`: filter out orphan-result rows. `COUNT(*)` becomes the logical invocation count because duplicate ids are collapsed. Rename or fix `distinct_tool_ids`: it currently counts distinct `tool_use_block_id`, not ids. Use `COUNT(DISTINCT a.tool_id) FILTER WHERE a.tool_id IS NOT NULL` if the field really means ids, or rename it to distinct use blocks.

`polylogue/storage/sqlite/queries/tool_usage.py:160-178`: provider coverage should distinguish sessions, use-backed actions, exact pairs, nearest pairs, unpaired uses, and orphan results. `has_tool_id_signal` should be `COUNT(*) WHERE a.tool_id IS NOT NULL`, not `COUNT(a.tool_use_block_id)`.

`polylogue/storage/sqlite/queries/filter_builder.py:131-190`: tool/action/path filters should normally add `a.tool_use_block_id IS NOT NULL`, because “tool:none” and “action:none” should refer to no observed tool-use action, not no orphan results. Output-text filters should join through `tool_result_block_id` and may include orphan results.

`polylogue/storage/sqlite/archive_tiers/archive.py:475-508`: `_ACTION_FOLLOWUP_RELATION_SQL` should read `tool_result_is_error`/`tool_result_exit_code`, not legacy `is_error`/`exit_code`. The failure condition must be `tool_result_is_error = 1 OR (tool_result_exit_code IS NOT NULL AND tool_result_exit_code != 0)`. Do not `COALESCE(NULL, 0)`.

`archive.py:609-615`: group-by fields can still render unknown via `COALESCE(..., 'unknown')`; grouping unknown separately is honest. Do not use that expression for truth predicates.

`archive.py:2973-2990` and `3022-3038`: tool usage and provider coverage need the same use-backed filters and status coverage described above.

`archive.py:5168-5252`: `query_actions` and `query_session_actions` should select from materialized `actions`, alias canonical outcome fields for the existing row constructor, and join `blocks` for output text. The row type should gain `pairing_status`; `tool_use_block_id` must become optional if orphan-result rows are returned.

`archive.py:5255-5313`: `query_session_action_occurrences` currently reimplements the old block join directly. Replace it with the materialized table.

`archive.py:5315-5380` and `5475-5510`: file queries should count use-backed rows with non-empty `tool_path`; orphan rows have no use-side path.

`archive.py:6916-6938`: stats by tool/action should filter use-backed rows and expose pairing status as a separate groupable field if needed.

`archive.py:7175-7238`: message field predicates that use actions should join via `message_id` anchor, but use-side fields should require `tool_use_block_id IS NOT NULL`.

`archive.py:7256-7270`: rewrite `is_error` and `exit_code` predicates to avoid `COALESCE`. `is_error:false` means `tool_result_is_error = 0`, not unknown. `exit_code = 0` means provider reported zero, not missing.

`archive.py:7516-7584`: EXISTS predicates over files/actions should select the correct grain: use-backed for tool/action/path, result-backed for output.

`archive.py:7838-7965`: session filters for `tool:none`, `action:none`, tool names, semantic action types, action text, referenced paths, and cwd should use use-backed filters unless explicitly operating on result output.

`archive.py:8028-8092`: action sequence should order by stored `action_index` or the stored order columns instead of rejoining blocks to recover order. For orphan rows, sequence matching should require a use-backed action unless the query unit explicitly supports `pairing_status:orphan-result`.

`archive.py:8532-8542`: latency tool category counts should filter `tool_use_block_id IS NOT NULL`.

`polylogue/archive/actions/parsing.py:50-114`, `polylogue/archive/semantic/facts.py:269-312`, and `polylogue/insights/transforms.py:1573-1616`, `1709-1766`, `2191-2202`: replace old id-only maps with the shared pairer or table-backed rows, while preserving the existing “NULL outcome = unknown” status rule.

Docs examples such as `docs/design/archive-storytelling.md:174`, `191`, and `205` should be updated after the runtime switch.

The test plan should cover five layers.

First, DDL tests. Assert `sqlite_master` reports `actions` as a table, not a view. Assert `STRICT` constraints reject invalid `pairing_status`, exact-pair rows without a non-empty `tool_id`, nearest rows without a positive distance, unpaired-use rows with result outcome columns, and orphan-result rows with a use block. Assert `actions_materialization` state has one row per refreshed session.

Second, pairing fixtures. Use tiny sessions with ordered messages and blocks:

```text
exact-pair:
  m0 assistant block0 tool_use id=t1
  m1 tool      block0 tool_result id=t1 is_error=0 exit_code=0
  => one row, pairing_status=exact-pair, exact_tool_id_match=1

nearest-following-result:
  m0 assistant block0 tool_use id=NULL
  m1 tool      block0 tool_result id=NULL is_error=1 exit_code=2
  => one row, pairing_status=nearest-following-result, distance>0

unpaired-use:
  m0 assistant block0 tool_use id=NULL
  => one row, pairing_status=unpaired-use, outcome NULL

orphan-result:
  m0 tool block0 tool_result id=NULL is_error=1 exit_code=1
  => one row, pairing_status=orphan-result, tool_use_block_id NULL
```

Third, reproduce `polylogue-xnkf`. Build a Claude Code fixture with distinct native message ids, `variant_index=0`, two or four re-emitted `tool_use` blocks sharing the same `toolu_*` id, and matching duplicate result blocks. The old view would emit `N × M`; the table emits exactly one exact-pair row for that id. Assert `duplicate_tool_use_count` and `duplicate_tool_result_count` record the suppressed evidence. Re-ingest the same source twice and assert `COUNT(*)`, status counts, and action ids are unchanged.

Fourth, lineage fixtures. Parent P has one action. Prefix-sharing child C repeats P’s prefix physically in the raw input plus one tail action; after lineage normalization, C stores only the tail. Assert `actions WHERE session_id=P` counts one physical action and `actions WHERE session_id=C` counts one physical tail action, not two. Add a child-first/parent-later fixture so `_reextract_prefix_tail_db` deletes inherited child rows; assert the child action materialization becomes stale and the converger refresh leaves only tail actions. Add a spawned-fresh edge and assert no prefix composition occurs.

Fifth, metamorphic assertions. For every refreshed session:

```sql
-- No use-backed action fan-out.
SELECT COUNT(*)
FROM actions
WHERE session_id = ?
  AND tool_use_block_id IS NOT NULL;
```

must equal:

```text
distinct non-empty tool_id use groups
+ count(NULL/empty-id tool_use blocks)
```

not the number of matching results. Every canonical `tool_use_block_id` appears in at most one row. Every canonical `tool_result_block_id` appears in at most one row. Result outcome columns are byte-for-byte copied from the result block and remain NULL when the block reports unknown. Nearest-following tests must include same-message ordering with block positions and same-message-position variants ordered by `variant_index`. Output predicates must find paired results and orphan results through the `blocks` join without requiring copied output text in `actions`.

Proposed replacement text for `polylogue-xnkf`:

```text
design: |
  Replace the index-tier actions VIEW with a daemon-refreshed materialized
  actions table in index.db schema v25. The table is derived from tool_use and
  tool_result blocks plus their owning message order columns; it is never written
  by the ingest write path and never refreshed by an operator command.

  Materialization grain is physical session rows. Prefix-sharing lineage children
  store only divergent tails, so actions for existing aggregate/stat/filter
  consumers count only the child’s own stored tail evidence. Logical composed
  transcript action reads may be added as a separate grain later, but the canonical
  actions table must not count inherited parent-prefix actions once per child.

  Pairing is typed:
  - exact-pair: a canonical non-empty tool_id tool_use group pairs to result
    evidence with the same non-empty tool_id in the same physical session.
    Duplicate re-emitted uses/results with the same (session_id, tool_id) collapse
    to one action row and are recorded in duplicate/evidence columns.
  - nearest-following-result: after exact claims are reserved, an unmatched use
    pairs to the nearest unclaimed tool_result whose composed transcript order is
    strictly greater than the use order. Order is message.position,
    message.variant_index, block.position, block_id.
  - unpaired-use: a use has no result after exact and nearest passes.
  - orphan-result: a result remains unclaimed and is emitted as visible result
    evidence instead of disappearing.

  Outcome columns are provider-reported only:
  tool_result_is_error and tool_result_exit_code are copied from the paired or
  orphan result block. NULL means unknown and is never coerced to success.
  Readers must stop using COALESCE(is_error, 0) / COALESCE(exit_code, 0) as a
  truth predicate.

  The materialized table stores pairing metadata, ids, use-side tool fields,
  canonical result outcome, output presence/byte counts, duplicate evidence
  arrays, and action order. Full output text remains in blocks.text and is joined
  through tool_result_block_id by readers that need output search/display. This
  avoids duplicating multi-GB result text in index.db.

  Add an actions convergence stage before insights. It implements check,
  check_many, check_sessions, execute, execute_many, and execute_sessions with
  false_means_pending=True and bounded session pages. Staleness is tracked in an
  actions_materialization table keyed by session_id, pairing_version, source hash,
  source tool_use/tool_result block counts, and status counts.

acceptance_criteria: |
  1. index.py bumps INDEX_SCHEMA_VERSION from 24 to 25, removes the actions VIEW,
     and creates STRICT actions and actions_materialization tables with CHECKed
     pairing_status values: exact-pair, nearest-following-result, unpaired-use,
     orphan-result.
  2. Fresh index bootstrap creates actions as a table; old v24 index.db is rejected
     and rebuilt under the existing derived-regime reset/reingest flow.
  3. A daemon convergence stage named actions refreshes materialization from blocks
     and messages, supports path-scoped and session-scoped checks/executes, pages
     work, and uses false_means_pending for remaining debt.
  4. A Claude Code re-emission fixture with repeated native messages carrying the
     same toolu_* id emits one exact-pair action row, not N×M rows. Duplicate use
     and result counts/evidence ids record the suppressed rows.
  5. NULL-id fixtures cover nearest-following-result, unpaired-use, and
     orphan-result. NULL-id result outcomes become visible through orphan or
     nearest rows instead of disappearing.
  6. Provider outcome fields remain honest: NULL is rendered/grouped as unknown,
     is_error:false matches only provider-reported 0, and exit_code:0 matches only
     provider-reported zero.
  7. Lineage fixtures prove physical grain: parent prefix actions are not counted
     again for prefix-sharing child sessions, including child-first parent-later
     tail re-extraction.
  8. All runtime consumers of FROM actions/JOIN actions switch to the materialized
     schema, use use-backed filters for tool invocation aggregates, join blocks for
     output text when needed, and expose pairing_status coverage where relevant.
  9. Insight/action builders reuse the same pairer or table-backed rows; no
     remaining id-only action builder can preserve the old fan-out/NULL blindness.
  10. Metamorphic tests assert action counts and action ids are invariant under
      repeated ingest of the same source and never multiply by duplicate result
      count.
```
