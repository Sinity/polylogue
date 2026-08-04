# Unpaired Tool Results Classification

## Scope and method

This audit classifies the corrected 2026-08-03 baseline of 49,886 unpaired, identity-bearing tool calls. The old 26,188 figure compared total block counts and therefore could not establish per-call pairing. The corrected query pairs within `(session_id, tool_id, transcript rank)` and counts a tool use only when its correlation id is non-empty and the corresponding result rank is absent.

The audit intentionally excludes 2,096 tool uses with no usable correlation id. They cannot be paired by identity and are separately represented by the same `no_result` action state. They are not part of the corrected 49,886 baseline.

## Classification

| Origin | Cause | Calls | Classification |
| --- | --- | ---: | --- |
| `codex-session` | Identity-bearing invocation has no emitted result record. Spot checks show a command followed by a later user intervention, so an interior position is not evidence that a result existed. | 33,994 | Honest no-result evidence, including interrupted or superseded work. |
| `chatgpt-export` | Recipient-addressed provider invocation has no child/result record. A representative `dalle.text2im` call is the only tool block in its exported turn. | 15,423 | Honest provider no-result evidence. |
| `claude-code-session` | Invocation ends without a result event, commonly `ExitPlanMode` awaiting an operator decision. | 466 | Honest pending or interrupted evidence. |
| `claude-ai-export` | Sparse provider invocation without a result record. Observed examples include terminal `Gmail:create_label` and `create_file` calls. | 3 | Honest provider no-result evidence. |
| **Total** |  | **49,886** | **No parser pairing defect was established.** |

## Pairing-defect decision

No category is reclassified as a parser defect. The Corpus Lens note already ruled out tail truncation as the dominant explanation for Codex and ChatGPT. Live, read-only samples strengthen the more precise conclusion: a result row is sometimes absent because the transcript records interruption, operator disposition, or provider-side omission. Reconstructing a result from later prose, a neighboring event, or a guessed positional match would fabricate evidence and risk false pairings.

The resulting action read model names the expected state `no_result`. It is distinct from `outcome_unknown`, which requires an existing paired `tool_result` row whose structured `is_error` and `exit_code` signals are both absent. The index change is view-only, so it does not mutate source data, blobs, or existing materialized action rows.

## Evidence

Read-only live archive checks used `index.db` in SQLite `mode=ro&immutable=1`:

```sql
SELECT COUNT(*)
FROM action_pairs
WHERE tool_result_block_id IS NULL
  AND tool_id IS NOT NULL
  AND tool_id != '';
```

The 2026-08-03 baseline returned 49,886. Origin counts were produced from the same predicate, grouped by the session origin. Current archive growth is deliberately not substituted for this historical baseline.
