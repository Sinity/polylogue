# Provider Package Completeness

Polylogue import support is larger than a parser function. A provider/capture
mode is product-ready only when the detector, parser, fixtures, schema/docs,
query/read coverage, ImportExplain, and caveat surfaces are all visible.

`devtools lab provider completeness` renders that readiness map from the current
tree:

```bash
devtools lab provider completeness
devtools lab provider completeness --json
devtools lab provider completeness --origin codex-session --json
devtools lab provider completeness --check
```

Rows are keyed by public `origin` plus `capture_mode`. `provider_wire` is
reported as evidence when the parser boundary still needs a provider token, but
it is not the grouping authority. See
[`docs/provider-origin-identity.md`](provider-origin-identity.md) for the
vocabulary split.

## Status Model

Each row carries item-level evidence:

| Item | Meaning |
| --- | --- |
| `detector` | Shape/path evidence that selects the importer. |
| `raw_model` | Typed or structured raw-record model for the input mode. |
| `parser` | Parser implementation that emits normalized sessions/messages. |
| `normalizer` | Shared conversion helpers that map provider quirks into archive contracts. |
| `fixtures` | Tests or sample data that exercise the mode. |
| `schema_package` | Generated provider schema package or catalog, when applicable. |
| `query_units` | Query-row/read substrate coverage after import. |
| `read_views` | Reader/recovery/profile surfaces that can consume imported sessions. |
| `import_explain` | `polylogue import --explain` coverage. |
| `privacy_caveats` | Documentation of source/capture privacy boundaries. |
| `generated_docs` | User-facing provider documentation. |
| `debt_rows` | Archive debt/readiness rows when applicable. |

Item statuses are:

```text
complete | partial | missing | not_applicable
```

Row statuses are:

```text
complete | partial | missing | proposed
```

`--check` fails only for accepted rows with missing or partial required items.
Proposed rows are allowed to be incomplete as long as the incompleteness is
visible in the report.

## Adding a Provider or Capture Mode

When adding a provider/importer:

1. Add or update the parser/detector and tests first.
2. Add schema package evidence when the raw format is stable enough.
3. Ensure `polylogue import PATH --explain --format json` reports detector,
   parser, produced-row, skipped-row, and caveat evidence for the mode.
4. Add provider docs and privacy/capture caveats.
5. Add a row to `polylogue/sources/provider_completeness.py`.
6. Run:

```bash
devtools lab provider completeness --json
devtools lab provider completeness --check
devtools render all --check
```

Do not call a mode complete merely because a parser exists.
