# Mutmut Campaign: `repository`

- Recorded on `2026-03-12T02:57:49.892990+00:00`
- Commit: `3bdd3f02dc87b2a84b23b523ed0483f5fcb63c1c`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.FNTTci/mutmut-repository-89hj8pty/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_repository_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 538 |
| Survived | 94 |
| Timeout | 77 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `187.01s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 28 |
| `_conversation_to_record` | 17 |
| `search_similar` | 14 |
| `_save_via_backend` | 8 |
| `similarity_search` | 8 |
| `search` | 4 |
| `save_conversation` | 3 |
| `embed_conversation` | 3 |
| `search_summaries` | 2 |
| `get_stats_by` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `get_session_tree` | 13 |
| `get_render_projection` | 10 |
| `iter_messages` | 10 |
| `list` | 7 |
| `_hydrate_conversations` | 6 |
| `_get_root_record` | 6 |
| `get_parent` | 4 |
| `get_conversation_stats` | 4 |
| `resolve_id` | 3 |
| `iter_summary_pages` | 3 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget_conversation__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁconversation_exists_by_hash__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁaggregate_message_stats__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_3`
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
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_8`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_17`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_32`
- `polylogue.storage.repository.xǁConversationRepositoryǁsearch_similar__mutmut_35`
- ... 69 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁresolve_id__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁresolve_id__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁresolve_id__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
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
- ... 52 more

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
