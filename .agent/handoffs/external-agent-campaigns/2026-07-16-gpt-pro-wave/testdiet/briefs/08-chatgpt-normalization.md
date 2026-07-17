Title: "[testdiet 08] ChatGPT capture and export normalization laws"

Job ID: `testdiet-08`
Result ZIP: `testdiet-08-chatgpt-normalization-r01.zip`

## Mission

Implement provider-family normalization survivors for ChatGPT exports and
native browser-capture payloads. Use real, privacy-safe wire fixtures and the
actual detect/lower/parse routes. Prove tight detector selection, active-path
and variant identity, role/material-origin, content blocks including reasoning,
attachments and assistant-produced assets, timestamps/durations, model effort,
status/end-turn, and full-fidelity-versus-fallback replacement semantics.

Pay particular attention to native metadata spellings actually present in
current fixtures (`reasoning_start_time`, `reasoning_end_time`,
`finished_duration_sec`, and any legacy duration field) and to provider tree
nodes that duplicate metadata. Normalize one semantic duration once without
double counting. Preserve raw fields even when their semantic meaning remains
explicitly provider-reported rather than model-compute time.

Name detector-order, authoredness, active-branch, duration-loss, or missing
asset mutations the survivors must kill. Do not make browser capture a special
campaign protocol or infer semantic facts from prose/UI labels when structured
provider data exists.
