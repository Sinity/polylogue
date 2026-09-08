# Provider schema inference

Infer provider schemas from explicitly selected ordinary source files:

```bash
devtools schema commit --provider claude-code \
  --source claude-code=/path/to/projects \
  --source-cache /path/to/private/evidence.sqlite3 \
  --source-workers 2 --privacy strict --progress
```

Repeat `--source provider=path` for additional roots or individual files. The
provider token selects the observation contract; select roots belonging to that
provider. `--output-dir` selects the package destination. `--dry-run` produces a
preview and still updates the private evidence cache. `devtools schema generate`
provides the non-persisting generation surface.

Source inputs use whole members; `--max-samples` and `--no-full-corpus` are
incompatible with source-backed commits. Without `--source`, generation uses the
archive-backed sampler. Selecting ordinary sources requires no archive rebuild.

## Inputs and reuse

JSONL and NDJSON records are streamed. JSON export arrays and supported ZIP
members are observed by logical session. Export evidence is spooled to private
SQLite files with bounded active contributions. Single-object JSON documents
above 32 MiB require a streaming adapter and are reported as unsupported. Provider-native identifiers distinguish
sessions, including Claude Code subagents. Gemini CLI checkpoint streams retain
raw headers, turns, and update records; complete checkpoints retain document
granularity. Current revisions contribute value
statistics; historical revisions retain structural evidence. Exact duplicates
and repeated export captures do not add current source-record weight. ZIP
members remain separate until native-identity and revision deduplication.

Loose JSONL revisions use strict byte-prefix selection before declared update
times. ZIP revisions use declared update times. Missing or tied times use a
deterministic hash order; they do not establish chronology.

Subagents retain their own native source identity. Schema inference does not
reconstruct parent links or subtract inherited transcript prefixes. Counts
describe records in the selected raw source revisions.

Incremental updates require the private SQLite cache, which retains reduced
evidence per source revision. Public schemas carry aggregate statistics and
provenance. Completed members survive an interrupted run.

Structure and statistics have separate semantic contracts. Cache addresses bind
source context, content hash, the phase contract, and key normalization. Changes
to absent paths preserve a source's statistics. A higher key limit reuses
structural summaries that retained their names; erased names require another
source read. Changed inputs are read again. The cache contains no transcript
bodies, but field names and reduced evidence still require private storage.

Change the affected revision in `polylogue/schemas/source_recipe.py` when
the meaning of admitted identities, structures, or statistics changes. Newly
supported inputs can reuse existing admitted contributions. Performance and
presentation edits preserve these revisions. The implementation fingerprint is
recorded separately for provenance; normalization parameters are part of each
contract. Identity and ZIP-member changes have separate revisions. Old
contributions are retained only when their identity and record counts prove
they are compatible. Missing Codex ordering metadata is recovered separately
from field statistics.

The run reports aggregate input bytes, record counts, terminal outcomes and
reason codes, cache hits/misses by phase, computation contracts, and phase timings. A successful package write
means evidence was emitted; inspect terminal outcomes to determine input
coverage. Unsupported sources, sidecars, malformed documents, changing files,
and incomplete trailing records are reported separately. Browser envelopes
require a native-payload adapter; SQLite state and opaque protobuf files require
their own observation routes.

## Versions and statistics

Package `vN` identifies an observed structural family. It is allocated from the
anchor element's structure; changing only statistics preserves that version.
Producer release numbers are separate metadata, read only from declared provider
fields. Missing and unrecognized releases are counted. The conflicting-release
count denotes files containing multiple declared releases.

Counts, extrema, string lengths, newlines, array lengths, and object widths are
measured before value compaction. Quantiles use bounded histograms; cardinality
and categorical summaries use bounded sketches. Their annotations identify
estimates and saturation. Historical shape counts and current source-record counts
have separate denominators.

Publication uses the configured privacy rules for values and dynamic keys.
Review generated packages before committing them; input paths and transcript
text do not belong in public schema metadata.
