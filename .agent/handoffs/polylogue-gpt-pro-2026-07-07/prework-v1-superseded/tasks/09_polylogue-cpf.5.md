# 09. polylogue-cpf.5 — Propagate weakest temporal provenance through aggregates

Priority: **P1**  
Lane: **temporal-honesty**  
Readiness: **ready-now / code-local with schema propagation**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

Time-source laundering makes fallback or synthetic dates look like provider timestamps once they enter an aggregate. Timeline claims then look more grounded than they are.

## Static diagnosis / likely mechanism

Root cause: `classify_aggregate_hwm_source(source_updates: list[str])` returns `provider_ts` whenever the list is non-empty, explicitly assuming every input is provider-sourced (`polylogue/insights/temporal_source.py:97-108`). Leaf classifiers also treat any non-null timestamp as `provider_ts` (`:66-81`, `:84-94`). Aggregators pass timestamp strings instead of source classes.

## Implementation plan

Implementation shape:
1. Define a temporal-source rank lattice where weaker evidence wins. Suggested weakest-to-strongest: `fallback_date`, `materialization_ts`, `file_mtime`, `sort_key`, `hook_event_ts`, `provider_ts`.
2. Add `weakest_temporal_source(sources: Iterable[TemporalSource]) -> TemporalSource`.
3. Replace aggregate classifier with a source-aware helper. The old `source_updates: list[str]` signature should be deprecated or made to return the weakest safe value, not provider.
4. Ensure profile/work/thread rows store or expose their `input_high_water_mark_source`; aggregates must collect that field.
5. Add audit output for rows where a timestamp exists but its source class is unknown, so future materializers can be corrected.
6. Update renderers/docs to describe “weakest source wins.”

## Test plan

Tests:
- provider + fallback => aggregate fallback.
- provider + file_mtime => aggregate file_mtime.
- provider-only => provider.
- empty => fallback.
- materialized day/tag rollup fixture with mixed input sources stores the weakest source.
- old timestamp-only path does not silently return provider for mixed/unknown inputs.

## Verification command / proof

`devtools test tests/unit/insights/test_temporal_source_taxonomy.py tests/unit/insights/test_archive_summaries.py -k 'temporal or high_water or weakest'`

## Pitfalls

Do not use lexicographic enum order; define an explicit rank. Do not fix only labels in renderer; the stored aggregate provenance must be correct.

## Files/functions to inspect or touch

- `polylogue/insights/temporal_source.py:66-108`
- `polylogue/insights/archive_summaries.py`
- `polylogue/insights/archive_rollups.py`
- `profile/thread/tag/day summary models`
