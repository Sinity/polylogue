# Mutmut Campaign: `repository`

- Recorded on `2026-03-11T18:00:19.507984+00:00`
- Commit: `7e7c310037f9f8cf89ba9c016d8eb4713d1b2f3d`
- Worktree dirty: `no`
- Description: Repository query, projection, and CRUD contracts
- Workspace: `/tmp/nix-shell.oYHkDe/mutmut-repository-f_gf_cdl/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/storage/repository.py`
- Selected tests: `tests/unit/storage/test_async.py`, `tests/unit/storage/test_repository.py`, `tests/unit/storage/test_repository_laws.py`, `tests/unit/storage/test_crud.py`, `tests/unit/storage/test_crud_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 527 |
| Survived | 130 |
| Timeout | 51 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `255.62s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `iter_summary_pages` | 33 |
| `get_archive_stats` | 28 |
| `_conversation_to_record` | 17 |
| `search_similar` | 10 |
| `_save_via_backend` | 9 |
| `iter_messages` | 6 |
| `similarity_search` | 6 |
| `search` | 4 |
| `save_conversation` | 3 |
| `search_summaries` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `get_session_tree` | 10 |
| `get_render_projection` | 9 |
| `_hydrate_conversations` | 8 |
| `iter_messages` | 6 |
| `get_parent` | 4 |
| `get_archive_stats` | 4 |
| `get` | 2 |
| `get_eager` | 1 |
| `list_summaries` | 1 |
| `iter_summary_pages` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

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
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_36`
- ... 105 more

## Timeout Keys

- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_9`
- `polylogue.storage.repository.xǁConversationRepositoryǁget__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_2`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_3`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_10`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_12`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_15`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_render_projection__mutmut_16`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_eager__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_13`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_14`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_21`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_22`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_24`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_25`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_26`
- `polylogue.storage.repository.xǁConversationRepositoryǁ_hydrate_conversations__mutmut_28`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist_summaries__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁiter_summary_pages__mutmut_47`
- `polylogue.storage.repository.xǁConversationRepositoryǁlist__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_1`
- `polylogue.storage.repository.xǁConversationRepositoryǁget_parent__mutmut_2`
- ... 26 more

## Notes

- Large surface; use to gauge storage law readiness before repository-law work.
