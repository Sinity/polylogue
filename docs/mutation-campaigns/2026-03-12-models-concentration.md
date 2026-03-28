# Mutmut Campaign: `models`

- Recorded on `2026-03-12T07:00:56.287051+00:00`
- Commit: `e07c4baebfe68af194fd423cd8fa0ecab515ca01`
- Worktree dirty: `no`
- Description: Message/Conversation semantic helpers and pairing logic
- Workspace: `/tmp/nix-shell.WFGSwJ/mutmut-models-bxor5xie/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/lib/models.py`
- Selected tests: `tests/unit/core/test_models.py`, `tests/unit/core/test_message_laws.py`, `tests/unit/core/test_conversation_semantics.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 138 |
| Survived | 22 |
| Timeout | 3 |
| Not checked | 3 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `35.52s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_is_chatgpt_thinking` | 10 |
| `extract_thinking` | 7 |
| `to_text` | 3 |
| `iter_pairs` | 2 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `iter_pairs` | 3 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| `mainline_messages` | 2 |
| `project` | 1 |

## Survivor Keys

- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_5`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_7`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_14`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_16`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_21`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_23`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_26`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_36`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_38`
- `polylogue.lib.models.xǁMessageǁ_is_chatgpt_thinking__mutmut_41`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_14`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_16`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_17`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_18`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_19`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_22`
- `polylogue.lib.models.xǁMessageǁextract_thinking__mutmut_24`
- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_7`
- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_19`
- `polylogue.lib.models.xǁConversationǁto_text__mutmut_1`
- `polylogue.lib.models.xǁConversationǁto_text__mutmut_4`
- `polylogue.lib.models.xǁConversationǁto_text__mutmut_8`

## Timeout Keys

- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_16`
- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_17`
- `polylogue.lib.models.xǁConversationǁiter_pairs__mutmut_20`

## Not-Checked Keys

- `polylogue.lib.models.xǁConversationǁmainline_messages__mutmut_1`
- `polylogue.lib.models.xǁConversationǁmainline_messages__mutmut_2`
- `polylogue.lib.models.xǁConversationǁproject__mutmut_1`
