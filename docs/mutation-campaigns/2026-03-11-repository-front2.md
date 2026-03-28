# Mutmut Campaign: `repository`

- Recorded on `2026-03-11T20:44:32.418308+00:00`
- Commit: `b1f1d35bee28cb78714a7ba5e10a86fa06c176f2`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.q2QMWP/mutmut-repository-1mtm744e/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_async.py`, `tests/unit/storage/test_repository.py`, `tests/unit/storage/test_repository_laws.py`, `tests/unit/storage/test_crud.py`, `tests/unit/storage/test_crud_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 568 |
| Survived | 74 |
| Timeout | 66 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `197.55s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 19 |
| `_conversation_to_record` | 15 |
| `_save_via_backend` | 9 |
| `search_similar` | 8 |
| `similarity_search` | 6 |
| `search` | 4 |
| `save_conversation` | 3 |
| `search_summaries` | 2 |
| `get_stats_by` | 2 |
| `_get_message_conversation_mapping` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 12 |
| `iter_messages` | 10 |
| `get_render_projection` | 9 |
| `_hydrate_conversations` | 6 |
| `_get_root_record` | 6 |
| `get_session_tree` | 6 |
| `get_parent` | 5 |
| `iter_summary_pages` | 3 |
| `list` | 2 |
| `get_conversation_stats` | 2 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_summaries__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_17`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_32`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_35`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_37`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_message_conversation_mapping__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_message_conversation_mapping__mutmut_17`
- `polylogue.storage.repository.xǁConversationRepositoryǁsave_conversation__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁsave_conversation__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁsave_conversation__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_conversation_to_record__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_conversation_to_record__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_conversation_to_record__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_conversation_to_record__mutmut_6`
- ... 49 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_8`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_24`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_25`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_28`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_44`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_46`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁaggregate_message_stats__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_2`
- ... 41 more

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
