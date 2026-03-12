# Mutmut Campaign: `repository`

- Recorded on `2026-03-12T09:57:19.645335+00:00`
- Commit: `eb43cfd48e989a58780604e9cbf9d3ba93700ff8`
- Worktree dirty: `yes`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.SPqtjJ/mutmut-repository-34wbdldv/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_repository_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 534 |
| Survived | 134 |
| Timeout | 40 |
| Not checked | 1 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `122.63s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `search_similar` | 17 |
| `_conversation_to_record` | 16 |
| `get_archive_stats` | 13 |
| `_save_via_backend` | 8 |
| `similarity_search` | 8 |
| `iter_messages` | 7 |
| `_hydrate_conversations` | 6 |
| `update_metadata` | 6 |
| `search` | 4 |
| `get_many` | 4 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `get_archive_stats` | 15 |
| `get_session_tree` | 13 |
| `_get_root_record` | 6 |
| `get_parent` | 4 |
| `iter_summary_pages` | 1 |
| `get_provider_conversation_ids` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `conversation_exists_by_hash` | 1 |

## Survivor Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_conversation__mutmut_1`
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
- ... 109 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_47`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_provider_conversation_ids__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_8`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_get_root_record__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_4`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_5`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_6`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_7`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_8`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_11`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_session_tree__mutmut_14`
- ... 15 more

## Not-Checked Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁconversation_exists_by_hash__mutmut_1`

## Source Worktree Status

- ` M tests/unit/cli/test_helpers.py`
- ` M tests/unit/cli/test_query_exec.py`
- ` M tests/unit/core/test_conversation_semantics.py`
- ` M tests/unit/core/test_dates.py`
- ` M tests/unit/core/test_filters_adv.py`
- ` M tests/unit/core/test_filters_props.py`
- ` M tests/unit/core/test_hashing.py`
- ` M tests/unit/core/test_message_laws.py`
- ` M tests/unit/core/test_models.py`
- ` D tests/unit/core/test_properties.py`
- ` M tests/unit/sources/test_claude.py`
- ` M tests/unit/sources/test_content_extraction.py`
- ` M tests/unit/sources/test_extraction.py`
- ` M tests/unit/sources/test_parsers.py`
- ` M tests/unit/sources/test_seeded_parser_contracts.py`
- ` M tests/unit/storage/test_backend.py`
- ` M tests/unit/storage/test_hybrid_laws.py`
- ` M tests/unit/storage/test_repository_laws.py`
- ` M tests/unit/storage/test_store_ops.py`
- ` M tests/unit/storage/test_vec.py`

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
