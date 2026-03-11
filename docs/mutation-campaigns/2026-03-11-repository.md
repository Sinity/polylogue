# Mutmut Campaign: `repository`

- Recorded on `2026-03-11T06:33:49.870267+00:00`
- Commit: `147e689d15caf23fc4036c3af6211af4f71bbaad`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.jtFedr/mutmut-repository-z35w2ufi/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_repository.py`, `tests/unit/storage/test_crud.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 343 |
| Survived | 250 |
| Timeout | 6 |
| Not checked | 81 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `88.70s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 42 |
| `_save_via_backend` | 39 |
| `iter_summary_pages` | 32 |
| `list` | 28 |
| `list_summaries` | 27 |
| `count` | 24 |
| `search_similar` | 10 |
| `_hydrate_conversations` | 8 |
| `iter_messages` | 8 |
| `similarity_search` | 6 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `__init__` | 2 |
| `iter_summary_pages` | 2 |
| `list_summaries` | 1 |
| `get_session_tree` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `_conversation_to_record` | 57 |
| `get_render_projection` | 16 |
| `get_stats_by` | 3 |
| `get_conversation` | 1 |
| `conversation_exists` | 1 |
| `aggregate_message_stats` | 1 |
| `get_source_conversations` | 1 |
| `get_message_counts_batch` | 1 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁview__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁview__mutmut_3`
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
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_16`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_17`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_18`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_19`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_20`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_23`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_24`
- ... 225 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁ__init____mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁ__init____mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_29`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_28`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_48`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_10`

## Not-Checked Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_8`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_16`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_conversation__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁconversation_exists__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁaggregate_message_stats__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_source_conversations__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_message_counts_batch__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_conversation_to_record__mutmut_1`
- ... 56 more

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
