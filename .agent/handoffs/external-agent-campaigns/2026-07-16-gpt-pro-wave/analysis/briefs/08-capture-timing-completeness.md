Title: "[analysis 08] Browser-capture timing and completion evidence audit"

Job ID: `analysis-08`
Result ZIP: `analysis-08-capture-timing-completeness-r01.zip`

Audit the full ChatGPT browser-capture path—from page bridge/network payload,
DOM/lifecycle observation, receiver spool/acquisition, raw source evidence,
parser normalization, index columns, semantic timing/profile, and public
queries—for generation start/progress/terminal state, reasoning duration,
message duration, model/effort, attachments, and assistant-produced assets.
Use actual captured fixture shapes, including `reasoning_start_time`,
`reasoning_end_time`, and `finished_duration_sec`. Distinguish already captured
raw evidence, projection loss, genuinely transient UI/stream state, and
unsupported claims such as relabeling wall duration as model compute. Return a
field-by-field completeness matrix and implementation/test plan tied to current
Beads.
