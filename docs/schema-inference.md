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
provides the generation preview: it never writes committed package files.

To promote a single evidence cluster instead of committing a full regeneration,
generate with `--cluster --retain-clusters`. `--retain-clusters` saves the
cluster manifest to the schema registry; without it the clusters are only
reported, and `devtools schema promote` then refuses with `No cluster manifest
found`:

```bash
devtools schema generate --provider chatgpt --cluster --retain-clusters
devtools schema promote --provider chatgpt --cluster <cluster-id>
```

Source inputs use whole members; `--max-samples` and `--no-full-corpus` are
incompatible with source-backed commits. Without `--source`, generation uses the
archive-backed sampler. Selecting ordinary sources requires no archive rebuild.

`--frontier` takes the inputs from the declared source frontier
(`devtools schema frontier`) instead of repeating `--source`, and refuses when
the live roots no longer match the recorded baseline. Restricted directory roots
supply their recorded members as individual inputs. A restricted single-file root
supplies that declared file, including when its recorded member names a resolved
target. A subject whose declared
roots admit no member reports `zero_eligible_material` with the declared reason
rather than falling back to the archive-backed sampler over a different
population.

After a pass, `devtools schema reconcile --receipts <dir>` accounts for every
declared subject exactly once. It derives the subject denominator from the
schema-subject kernel and the executable `OriginSpec` registry rather than from
the pass's own receipts or the committed packages, so a subject the pass never
reached is recorded as `not_run` instead of disappearing, and it binds the
matrix to the baseline digest, the provider declaration digest, the code
revision, the generator semantics revision and the resolved inference
configuration. Generation results record the full resolved privacy policy, including
custom thresholds and ordered rules. JSON `field_overrides` uses an ordered list
of `{pattern, action}` objects so sorted JSON object keys cannot change the
first matching rule. TOML configuration retains its ordered field table. Reconciliation verifies every receipt
against the policy selected by its own `--privacy` and `--privacy-config`
options. Omitted options mean the same standard default as generation. Missing,
malformed, mixed, or mismatched policies refuse reconciliation before matrix
publication; pass the matching explicit policy for a nonstandard generation.
Receipt argv alone does not certify the effective policy. Dry runs and
`zero_eligible_material` results carry the same policy evidence.

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
coverage. Every declared artifact family has an `OriginSpec` observation
contract: session and structured sidecars contribute privacy-safe shape
evidence even when they are not session-admitted, while opaque/binary families
have an explicit non-applicability outcome and remain raw acquisition
evidence. That declaration holds inside ZIP exports too: a declared opaque
member, such as a ChatGPT `file-<id>.json` attachment, is never read as
evidence. SQLite members use the logical table/column route and preserve
member/table retention dispositions without row values. A live SQLite member's
revision is the digest of its whole-database logical export, so a commit still
in the WAL moves it, and its structure is read from that export's header in the
same read transaction. Unsupported sources,
malformed documents, changing files, and incomplete trailing records are
reported separately. Browser envelopes require a native-payload adapter;
opaque protobuf files retain their typed non-applicability outcome.

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

A committed package publishes *observed member values* in exactly one place,
`x-polylogue-values`, and that is an allowlist: a field publishes its members
only when its declared semantic role is in
`polylogue.schemas.privacy.PUBLISHABLE_VOCABULARY_ROLES`. Every other field
publishes its type, frequency and distribution with no member list, because
value shape cannot separate a provider protocol constant from a recurring
private token. The rule is enforced at `SchemaRegistry.write_package`, the one
physical element writer, so a version that is only carried forward is sanitized
on re-serialization rather than keeping members admitted under an older rule.
`devtools gate schema-privacy` re-checks every committed element independently,
including property names, for an unjustified vocabulary and for any filesystem
path, mail address or URL.

Review generated packages before committing them; input paths and transcript
text do not belong in public schema metadata.

Generated package updates retain numeric observation counts and structural size
distributions. Observed numeric magnitudes, timestamp ranges, and empirical time
deltas remain in private evidence.
