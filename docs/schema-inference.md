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
above 32 MiB use the existing disk-backed JSON reader; the threshold selects a
reader rather than refusing evidence. ZIP objects use private temporary byte
spools. The source collector keeps borrowed document views alive through the
final evidence flush and contribution serialization. Provider-native identifiers distinguish
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

Structure revisions apply to the affected provider: generic reduction is5,
Codex tool grammar is6 and Claude named tool forms and public Todo tokens are8.
A batch uses the highest revision applicable to its actual inventoried providers; an empty
batch uses5. A ChatGPT-only batch therefore keeps its unchanged revision5
contract. Mixed batches conservatively bind one combined contract rather than
claiming separate reuse for each provider. Stored provenance is never relabeled.

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

Synthetic schema coverage uses a finite plan for every union branch and type
choice, with its ancestor choices fixed. Each plan generates once; there is no
witness count cap or random retry tail. The receipt verifies coverage from
schema-valid artifacts whose actual parser route retains their messages,
identity, semantics and content. An incidental `raw_provider_payload` field in
a direct export does not select a capture route. Native and fallback browser
captures use the same selection predicates and lowering as production parsing.

Witness bytes stream with corpus state restored before each yield. The receipt
retains per-artifact evidence and aggregate construct keys and counts. The test
memo retains complete entries within its heap budget; oversized or abandoned
streams remain uncached without dropping their output. Package-backed witnesses retain their original cross-record grammar and
measured field presence. A construct assignment applies only to a compatible
original record; other records retain their selected wire forms. Missing
construct evidence remains visible in the receipt.

Optional field sampling overrides its marginal frequency only with an explicit
co-occurrence count. Missing public maps or keys are unknown; an explicit zero
means measured absence. Raw mutual-exclusion inference can use absence from a
complete counter, but never absence from a counter whose keys were truncated.
Categorical value pools retain one entry per generation path, bound to that
branch's categorical, string-length and newline annotations, so another branch cannot
reuse its values or weights.

Co-occurrence ratios use normalized occurrence counts: `type_counts` already
includes nulls from real collection, while a separately declared manual null
count supplies it only if the type map omits null. Input-document counts do not
count repeated array records. A manual measurement with only `encountered_documents` declares one
occurrence per document. A count above its matching denominator is refused,
not clamped into a probability.

Nullable sampling uses the explicit null count in `type_counts` when present.
A separately declared `null_observations` count supplies it only when the
manual type-count map omits null; those two counts are never added.

Actual session generation selects a union branch and scalar type before
semantic defaults. The scalar sampler owns observed numeric values, lengths,
newlines and categories; themed body or title prose is a proposal fitted by
that sampler. Explicit labels and recognized wire formats remain authoritative.
Known labels use the producer's matching hash-bucket weights when available;
labels sharing one bucket divide its unidentified mass. Unmatched evidence
never invents a role or tool name. Valid UUID/date spellings are never truncated
to imitate an incompatible length annotation; such conflicts are visible.

Provider adapters preserve generated observed fields and use the selected
record schema for missing wire fields. Selected schema bindings last only
through one session's shaping and are cleared on completion or cancellation.
Witness generation restores its independent state before yielding bytes.
Finite schema generation uses an explicit traversal stack, without the former
ordinary/witness depth cutoffs. Selected null values remain present. Cyclic
in-memory schema objects have no finite JSON representation and are refused
before generation or witness planning.

Categorical values use stable ordinal codes at their left edge, without a
property-name prefix. A finite codebook allocates distinct values over the
supported length/newline combinations, preserving category weights. Empty
strings have capacity one; an impossible vocabulary is refused explicitly.
The ordinary producer's 256 hash buckets fit a neutral printable alphabet;
larger manual vocabularies use a lazy Unicode enumeration without an alphabet
cap. Newline placements are enumerated without retaining the Cartesian space.
Length, newline and category marginals do not identify their joint
distribution. Allocation conditions on representable combinations and remaining
capacity; it does not claim to recover exact joint frequencies. Hash collisions
remain an approximation in the original evidence.

Field collection, annotation, generation, coverage and conservation walk finite
structures iteratively. Deep observations retain their depth telemetry and
reach generated content instead of disappearing at a producer depth cutoff.
Categorical enumeration retains output-sized odometer state, not powers or
sets of every previously emitted string. Flat Codex records retain generated
payload fields; competing top-level and payload content owners are refused
because the production parser can conserve only the selected payload owner.

Missing wire fields, including identity, role, discriminator and metadata
carriers, sample the selected schema through the same field owner. Existing
selected nulls or labels are never replaced by an adapter default; a carrier
that cannot represent the required wire object is refused. Missing content
aliases reuse their existing owner deterministically only when its value
satisfies the same supported type, label, format and observed support policy.
Unsupported constraints or unavailable relational custody make that proof
explicitly unavailable; no constraint is ignored to admit an alias.

The existing graph builder owns parent, children and session links. It
conditions those structural fields on its declared topology and checks their
schema types and vocabulary, rather than claiming independent per-field or
exact topology marginals. Declared node identity values remain authoritative;
nonstring, empty or duplicate identities are refused visibly.

String shape evidence is conditioned on the format measured for each string.
The collector keeps separate length, newline and anonymous categorical sketches
for each detected format and for unformatted strings. Replay and incremental
merge retain those partitions. Synthetic generation chooses a partition with
its measured frequency before sampling its shape; the marginal sketches remain
unchanged. A dominant UUID annotation therefore does not turn shorter or longer
unformatted identifiers into UUIDs. Within each format, the existing anonymous
categorical buckets retain conditional length and newline sketches; codebooks
use those measured shapes without retaining or recovering observed spellings.
Persisted-output pointers additionally retain a fixed-log numeric joint of total
length, newline count, token start, prefix newlines, token width and basename
width. Each occupied bin retains actual coordinate-extreme tuples, preserving
its coarse frequency and tails without filenames or body text. The existing
scalar sampler chooses that joint before neutral spelling; a directory is not
replaced by a fixed-width basename. Replay and merge retain the same witnesses.
The marginal distributions and their tails remain available. Existing packages without conditional
measurements need reinference before mixed-format generation can be qualified;
the generator does not reconstruct joint evidence from marginal histograms.

Package workload profiles own cross-record structure. The original record pass
measures public kind starts, endings, transitions, scope counts and conditional
content block paths. Each public block kind also carries a finite presence
joint for `text`, `thinking`, `content` and `input`; unrelated body fields are
not drawn independently from merged block marginals. This retains no body
values or observed private field names. Claude Code role carriers preserve their measured joint
presence/null/value combinations and parser precedence; empty ChatGPT nodes
retain the measured absence or null form. Missing conditional evidence requires
reinference, rather than guessing a joint distribution from field marginals.
Native Codex call and output kinds retain the presence of their fixed public
value fields together with the selected output carrier and shape. An output
field therefore cannot disappear because another measured carrier was drawn
independently. Coverage witnesses explicitly select optional field coordinates
while retaining the same measured body forms and mutual exclusions.

The same native field form carries known public Codex tool names and anonymous
argument syntax for patch inputs and static `exec` programs. Fixed patch headers,
call punctuation and public tool tokens remain grammar; paths, commands and
other text become ASCII or non-ASCII run widths. Publication validates every
literal against that fixed vocabulary. Generation streams neutral operands
through the existing field sampler and checks measured length, newline and
character support. Unknown tool names remain anonymous. This does not execute
programs or qualify arbitrary JavaScript or every tool's argument semantics.

Claude Code uses the same measured field-form owner for the public Read, Edit,
Write, TodoWrite, Bash and Agent labels. Input field presence and representation
stay coupled to the sampled label. Result envelope forms follow the actual
generated call identity and its measured public class, including absence of an
envelope. Dispatcher class and result-content mode use the same eligible
measured form and actual call inventory before either draw. Argument bodies,
paths and arbitrary tool names remain anonymous;
the reducer publishes only fixed public keys, labels, modes and counts.
TodoWrite status and priority fields retain only their declared public token
counts aggregated by actual items within the existing native form; unknown
values stay anonymous, and descriptions remain content. Tokens
are sampled per actual string field through the same measured text-shape checks.
This does not establish numeric outcome fidelity, foreground/background subtype
correlation or asynchronous completion and progress semantics.

Original Claude Code child depths remain exact. Each depth's conditional record
grammar is a row in the existing `record_structures_by_depth` profile object.
Inference retains its serialized public rows before retiring the private journal;
publication, profile hashing and captured-byte loading stream those rows.
The driver reads only its selected depth grammar from the indexed carrier, so a
terminal child cannot inherit a shallower dispatcher continuation. Registry and
fixture handoffs retain the same carrier until its last borrower retires.
The publication audit checks the bounded profile fields and each retained depth
grammar through the same privacy predicates. Public JSON re-emission uses the
profile serializer, which includes the original retained rows.

Codex first-header profiles retain public wire form and parent-field presence
classes together with the original parent/prefix count joint. Record grammars
condition on both the parent carrier and first-header wire form, so a selected
flat header retains flat messages and an envelope header retains envelope
records. Profiles lacking that joint need fresh inference. The planner
selects an already completed generated parent on the original native cohort,
then binds that identity before drawing the header fields. Existing element
schemas remain the sole authority for identity width, format and field branches;
a parent binding cannot add an undeclared field or replace a selected null.
Incomplete or ambiguous original parent evidence is an explicit refusal.
Declared required properties are drawn even when their package-wide occurrence
frequency is optional. A required property without a declared field schema, or
one that conflicts with the selected wire form, refuses. Publication provenance
counters and dynamic-key markers remain metadata; message-container hints apply
to object and array carriers without substituting material values.
Inherited message bytes and tool identity schemas/history remain on that same
native owner across session boundaries. Reported message counts include the
actual inherited prefix and exclude the session header.
An authored replay canary may declare session order and prefix counts from its
measured header classes while retaining the measured record grammar. A declared
root count absent from that evidence refuses before generation.

The same lazy session/chunk driver serves collected fixtures, file writers and
`devtools bench fresh-build corpus synthetic`. A borrowed session must be
consumed before advancing the driver, and closing it releases its original
sampler owner. Fresh-build generation requires a published package workload
profile; a newly inferred neutral package can qualify mechanics but does not
stand in for the operator aggregate workload.

Curated `tool-heavy`, `demo-tool-heavy` and Gemini `demo-attachments`
fixtures select an explicitly authored neutral population before any field
draws. The same inference journal produces its schemas and conditional record
profiles, and the same session driver traverses its declared record order.
Tool pairs, failure follow-up and attachment bytes therefore belong to the
selected original field and relation operands. Their profile provenance is
`authored-fixture`; they do not claim to measure an operator archive. The
caller's original schema/profile selection is restored after physical driver
settlement.

Per-field distributions
remain exclusively in each element schema; profile publication, loading and
synthetic construction refuse the retired `field_profiles` copy. Re-inference
produces the current representation instead of retaining two sampling owners.
Tool pairing uses the original private observation journal: all identities are
counted, noncontiguous fragments of the same observed scope are replayed in
original ingestion order, and no ID or structural-variant capacity truncates
these observations. Empty observed scopes retain a zero record count. Private
tool identifiers stay in run-scoped scratch and never enter the published
profile. Public event classes, identity field presence and reuse classes bind
the original relation owner before field draws. Default profile-driven tool records come
from the measured public record path; no later template appends calls or
results on that route. For proved all-tool Claude call arrays, the same path
feasibility calculation supplies the minimum call count needed by its remaining
result suffix. Only that selected native content array conditions its existing
count histogram on the minimum; no count is invented or clamped. Mixed
text/tool arrays and repeated-result histories retain their separate joint
constraints. Tool-result content shape and pointer format are selected before
sampling, so a later null or absent field retains its measured representation.
Original prior result modes and count buckets constrain reuse of the same
identity, rather than independently mixing duplicate-result histories. Opaque
file counts are conditioned on original history count and reference presence;
filename widths and byte lengths remain measured by their original scopes.
The original ID/basename width sketch retains actual paired extrema, including
pairs in the same histogram bin, so width marginals cannot invent a pair.
Field annotations still own the selected field's length, newlines and format.
Neutral filenames are finalized on the same relation owner before any bytes
are yielded, retaining direct-name precedence and prior persisted references.
Opaque metadata and exact byte streams are borrowed with the session: all must
be consumed before the next session. Byte targets count both streams and stop
only at a complete session boundary. Written session paths correspond to
session facts; the complete file list also includes their opaque adjuncts.

Curated fixture styles select explicitly authored neutral operands before field
draws through the same inference/profile producer. Their provenance is distinct
from measured package profiles. Native session IDs bind through the selected
original document or first-header address; a missing address refuses rather
than adding a header or rewriting a sampled identity afterwards. Raw Codex
usage records carry declared cache/input and reasoning/output subset laws. The
relation owner conditions the scalar owner's original support, including its
neutral public range when magnitude evidence is private; it neither publishes
private numeric observations nor clamps generated values.

Per-field lengths, type/presence mixes and categorical frequencies come from
the element schema annotations. Cross-record order, tool relationships, child
streams, sidecars and session prefixes come only from the same package's
workload-profile artifact. `devtools schema commit` measures and publishes
both through the source inference owner. Fixtures and benchmark corpus writers
consume that package through `SyntheticCorpus.iter_sessions`; the separate
per-origin workload extractor, loader, profiles and hand renderer are retired.
Packages missing measured evidence require reinference and publication before
they can supply the driver. Neutral publication controls qualify this path;
they do not stand in for operator package publication.

`SyntheticComposition` selects providers by exact published main-session
revision counts, using the same seeded session driver for each selection. The
producer excludes separately typed Claude child transcripts and Codex headers
that declare a parent, including missing and ambiguous parent associations.
The population counts retained original revisions or neutral scopes, not unique
conversations. The fresh-build corpus writer records populations, normalized
weights, profile identities and actual selection counts. Composition refuses
packages without this measured denominator; it never substitutes byte sizes or
bundle totals. Session artifact writes refuse existing targets.
