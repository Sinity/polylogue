# Mutmut Campaign: `repository`

- Recorded on `2026-03-11T19:27:44.024440+00:00`
- Commit: `e759af23458dfdc67e1a820513f09f3828460458`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.HZq1Dj/mutmut-repository-tm3goviv/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_async.py`, `tests/unit/storage/test_repository.py`, `tests/unit/storage/test_repository_laws.py`, `tests/unit/storage/test_crud.py`, `tests/unit/storage/test_crud_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 569 |
| Survived | 104 |
| Timeout | 35 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `231.95s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 28 |
| `_conversation_to_record` | 17 |
| `search_similar` | 10 |
| `_save_via_backend` | 9 |
| `iter_messages` | 7 |
| `similarity_search` | 6 |
| `search` | 4 |
| `save_conversation` | 3 |
| `iter_summary_pages` | 2 |
| `get_session_tree` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_hydrate_conversations` | 8 |
| `get_session_tree` | 5 |
| `get_render_projection` | 4 |
| `get_parent` | 4 |
| `get_conversation_stats` | 4 |
| `iter_messages` | 4 |
| `get` | 2 |
| `list_summaries` | 1 |
| `iter_summary_pages` | 1 |
| `list` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_44`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_46`
- `polylogue.storage.repository.xǁConversationRepositoryǁaggregate_message_stats__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_root__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_summaries__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_many__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_many__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_16`
- ... 79 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_24`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_25`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_28`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_4`
- ... 10 more

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
