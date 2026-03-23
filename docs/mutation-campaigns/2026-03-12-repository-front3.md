# Mutmut Campaign: `repository`

- Recorded on `2026-03-11T23:55:58.017610+00:00`
- Commit: `027519a111180c3844a06800bb79432ea33469d4`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.JVI1AS/mutmut-repository-xxdcqo15/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_async.py`, `tests/unit/storage/test_repository_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 535 |
| Survived | 102 |
| Timeout | 72 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `149.94s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 28 |
| `_conversation_to_record` | 17 |
| `search_similar` | 14 |
| `_save_via_backend` | 9 |
| `similarity_search` | 8 |
| `save_conversation` | 5 |
| `search` | 4 |
| `embed_conversation` | 3 |
| `get_session_tree` | 2 |
| `search_summaries` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_messages` | 13 |
| `iter_summary_pages` | 8 |
| `list_summaries` | 7 |
| `list` | 7 |
| `_hydrate_conversations` | 6 |
| `count` | 5 |
| `_get_root_record` | 5 |
| `get_render_projection` | 4 |
| `get_parent` | 4 |
| `get_conversation_stats` | 4 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget_conversation__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁconversation_exists_by_hash__mutmut_1`
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
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_stats_by__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_messages__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_8`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_13`
- ... 77 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_24`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_25`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_28`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_44`
- ... 47 more

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
