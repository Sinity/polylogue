# NeMo Relay ATIF fixture provenance

`nemo_relay_atif_v1.7_real_redacted.json` is shape-preserving, privacy-redacted
evidence from the bundled Hermes NeMo Relay producer's ATIF-v1.7 export.

Steps 1-5 are a real, minimal live session (2026-07-18, isolated temporary
Hermes home, message-only steps: no `tool_calls`/`observation` in that run).
Step 6 was appended 2026-07-18 from a separate real, live
`~/.hermes/observability/nemo-relay/atif/trajectory-*.json` trajectory (an
ordinary local coding session) specifically to prove the `tool_calls`/
`observation` step shape against real bytes: the producer emits multiple
`tool_calls` entries per step (each with `tool_call_id`/`function_name`/
`arguments`/`extra.provider_data.call_id`) plus a correlated
`observation.results` list keyed by `source_call_id`. Tool names
(`process`/`terminal`/`search_files`) are preserved verbatim -- they are
producer-defined tool identifiers, not private content. All argument values,
observation content, timestamps, and identifiers are replaced with
`"<redacted>"` while preserving key names, nesting, and per-field type/
presence, per this repo's payload-hygiene discipline (parsers record only
bounded presence evidence, never real argument/observation text).

`final_metrics.total_steps` was updated to 6 to stay internally consistent
with `len(steps)`; `total_completion_tokens`/`total_prompt_tokens` still
reflect only the original 5-step run (step 6's real token counts were not
retained, since the parser does not read `metrics`/token fields from tool
steps).
