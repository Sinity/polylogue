# Mutmut Campaign: `repository`

- Recorded on `2026-03-11T09:54:24.616381+00:00`
- Commit: `2bdb267e93b79f1f0dc863f86b5ed859e4e0dbdd`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.rzHew1/mutmut-repository-y46ey2p_/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_repository.py`, `tests/unit/storage/test_repository_laws.py`, `tests/unit/storage/test_crud.py`, `tests/unit/storage/test_crud_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 370 |
| Survived | 181 |
| Timeout | 124 |
| Not checked | 5 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `274.90s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 42 |
| `_save_via_backend` | 39 |
| `iter_summary_pages` | 33 |
| `_conversation_to_record` | 20 |
| `search_similar` | 10 |
| `iter_messages` | 7 |
| `similarity_search` | 6 |
| `search_summaries` | 4 |
| `search` | 4 |
| `get_stats_by` | 3 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `list_summaries` | 31 |
| `list` | 28 |
| `count` | 24 |
| `_hydrate_conversations` | 8 |
| `iter_summary_pages` | 8 |
| `get_parent` | 4 |
| `get_conversation_stats` | 4 |
| `get_render_projection` | 3 |
| `get_session_tree` | 3 |
| `get_archive_stats` | 3 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `get_conversation` | 1 |
| `conversation_exists` | 1 |
| `aggregate_message_stats` | 1 |
| `get_source_conversations` | 1 |
| `get_message_counts_batch` | 1 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_16`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_17`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_18`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_19`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_20`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_23`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_24`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_25`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_27`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_29`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_30`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_31`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_32`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_33`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_34`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_35`
- ... 156 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁ__init____mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁ__init____mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁresolve_id__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_24`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_25`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_28`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_16`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_17`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_18`
- ... 99 more

## Not-Checked Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget_conversation__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁconversation_exists__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁaggregate_message_stats__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_source_conversations__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_message_counts_batch__mutmut_1`

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
