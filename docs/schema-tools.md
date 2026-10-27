# Offline schema tools

`polylogue schema infer` reads explicit stored input revisions and writes a
JSON artifact containing field paths, types, presence and counts. It runs
without the daemon or ingestion. It never opens the original source files
or tool databases. `schema diff` compares two such artifacts recursively.

```sh
polylogue schema infer claude-code --input INPUT_ID --revision 1 \
  --field type --field message --field content --field text \
  --output /path/to/schema-before.json
polylogue schema infer claude-code --input INPUT_ID --revision 2 \
  --field type --field message --field content --field text \
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
counts as one record. Export arrays reuse Sources' streaming reader. JSON
observation keeps counts rather than a corpus of decoded values. Capture
logs use the shared length and digest format. Tool databases are read only
as their already-stored JSON row dumps; row fields contribute column shape,
and the schema map contributes table structure. SQL text stays out of the
artifact.

## Field names and counts

There are no examples, observed value lists, enums or source paths in an
artifact. `--field` declares each field name permitted in output paths, at
any depth. Every other object member is grouped under `*`, including names
that look like ordinary words. With no `--field`, all object names are
hidden. This makes dynamic maps usable without disclosing their member
names. Use the same declarations for artifacts you intend to compare.

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
selected revisions, failed reads or empty samples label the populations as
incomparable; their observed differences remain visible. Counts alone do
not cause structural changes. Unsupported rule versions and different
field-name declarations are errors.

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
