# Offline schema tools

`polylogue schema infer` reads explicit stored input revisions and writes a
JSON artifact containing field paths, types, presence and counts. It runs
with or without the daemon running, without ingestion. Archive and stored-byte
reads use read-only handles and do not take the daemon's archive lock. It
never opens the original source files or tool databases. `schema diff`
compares the field paths in two such artifacts.

```sh
polylogue schema infer claude-code --input INPUT_ID --revision 1 \
  --output /path/to/schema-before.json
polylogue schema infer claude-code --input INPUT_ID --revision 2 \
  --output /path/to/schema-after.json
polylogue schema diff /path/to/schema-before.json /path/to/schema-after.json
polylogue --json schema diff /path/to/schema-before.json /path/to/schema-after.json
```

Repeat `--input` to select several inputs of the same origin. The default
revision is the latest of each selected input. A missing input, missing
revision, mismatched origin, unsupported framing or unavailable stored bytes
is listed under `selection` with a fixed reason and counted under
`unreadable_inputs`. Malformed JSON lines count under `unreadable_records`;
complete lines still contribute structure. A complete last JSON line needs
no newline. Malformed documents, exports and capture logs contribute no
partial structure. Selection never substitutes other inputs.

The artifact always says `sample: true`: explicit inputs are a sample of
that origin, without a claim of full coverage. Each selection lists the
actual revision and its readable/unreadable record counts. JSON lines and
top-level export-array elements count as records; a wrapped JSON document
counts as one record. The observer streams JSON lines, export-array elements
and capture bodies, validating scalar values without copying their bodies. It keeps counts and
at most 51 discovery names per object path, rather than per-record key sets.
Dynamic names are grouped while reading; memory follows the observed schema,
forced names and nesting depth, rather than record count or scalar size.
Parsing, merging, output-path traversal and cleanup use explicit stacks or
flat storage, with no fixed JSON depth cutoff. Artifact paths themselves
grow with nesting depth. Capture logs use the shared length and digest
format and `Header`'s streaming verifier. Arrays use the observer's reader because
`export::scan` buffers each complete element as a Serde `RawValue` and uses
recursive validation. Observation must discard scalar bodies and accept deep
JSON without a call-stack limit; both array readers require a closing bracket
and end of input before accepting the document. Tool databases are read only
as their already-stored JSON row dumps; row fields contribute column shape,
and the schema map contributes table structure. SQL text stays out of the
artifact.

## Field names and counts

Values never appear as examples, lists or enums. Object member names appear
by default. Identifier-shaped keys (UUIDs, hexadecimal hashes of at least
16 characters, numerals of at least six digits, dates/timestamps and paths)
are grouped under `*`. When a map has more than 50 distinct sibling names
across the selected records, its members are grouped under `*` regardless
of their spelling. Grouping uses the whole selected sample, including
members seen in separate inputs. These rules change paths, not which
records are read or counted.

Use `--field NAME` to force a name into paths at every depth even when it
would otherwise be grouped. The artifact's `named_fields` lists these
overrides. Use the same overrides for artifacts you intend to compare.

Paths start at the empty root path and use `/` between components. `[]`
means an array element; `*` means an unnamed object member. In named fields,
`~` and `/` become `~0` and `~1`; literal `*` and `[]` become `~2` and `~3`.
These escapes keep literal names separate from structural markers.

For each path, `types` counts values of each JSON type, `occurrences` counts
all values, and `present` counts parent objects or arrays with at least one
value at that path. `parents` counts eligible parents: object occurrences
for fields, array occurrences for elements. `missing` is `parents - present`.
`required` means every eligible parent in this sample had the field or an
array element. These are observed counts, not a guarantee about future
inputs. A nullable field can be required; a missing field has no null value.
Mixed object/scalar parents count only their object occurrences when
computing their fields' presence. Several unnamed members in one object
increase `occurrences`, while increasing `present` once.

Diff reports added paths, removed paths, changed type sets and changed
requiredness as separate findings, ordered by path. Different origin names,
selected revisions, failed reads or empty samples label the samples as
incomparable; their observed schema differences remain visible. Counts alone do
not cause structural changes. Unsupported rule versions and different
field-name declarations are errors. Plain reports use backslash escapes for
control characters in names.

## Destinations and primary mode

Inference writes JSON to stdout unless `--output FILE` names a destination.
Diff writes plain text by default and JSON with `--json`, also supporting
`--output FILE`. Neither command chooses a repository file or updates a
reference schema automatically.

`--archive DIRECTORY` selects an archive instead of Polylogue's default
data directory. In primary mode, add `--sinex-store DIRECTORY` to read source
bytes through Store's Sinex content-store handle. This uses the current
primary metadata cache in the selected index directory, without a
standalone `archive.db` or local blob copy. Missing metadata or inaccessible
Sinex bytes remain errors; the commands do not rebuild or ingest anything.

Developers can run the same command through
`xtask schema --binary /path/to/polylogue infer ...` or `diff ...`.
The wrapper delegates to the binary's command table, so argument definitions
and behavior stay in one place. Build and test through AgentCTL as described
in [development.md](development.md).
