# Query-Action Workflows

Polylogue's query-first surface should feel like one executable workflow across
CLI, web, MCP, HTTP, and JSON clients:

```bash
polylogue find QUERY then ACTION
```

The grammar is not only command syntax. It is the product contract for how a
user moves from recall to inspection, handoff, mutation, or analysis without
guessing what the current result set means.

This document defines the target contract for #2305. It uses the current CLI
verbs as the starting point, but treats current behavior as implementation
evidence rather than as the final design where the two differ.

## Workflow Shape

Every query-action flow has the same stages:

1. `Find`: compile the query into a typed selection envelope.
2. `Inspect`: show enough result metadata for the user or client to understand
   cardinality, exactness, ranking, freshness, and available actions.
3. `Act`: run one verb against the selection, one selected item, or a nested
   unit such as a message, action row, assertion, file, or browser capture.
4. `Render`: return a human-readable view and a machine envelope with the same
   refs, warnings, and next actions.
5. `Continue`: expose stable follow-up affordances so the user can narrow,
   read, export, mutate, or package the result without restarting the search.

The same selection envelope must be visible in every surface. A web result list
and a shell `find` response may render differently, but they should not disagree
about the selected refs or what actions are valid.

## Selection Envelope

A query response represents a materialized selection, not just rows. The shared
envelope should carry:

| Field | Meaning |
| --- | --- |
| `query` | Original query string plus normalized typed AST/explain metadata when requested. |
| `unit` | Selected unit: `session`, `message`, `action`, `block`, `file`, `assertion`, `run`, `observed_event`, or `context_snapshot`. |
| `cardinality` | `empty`, `one`, or `many`, computed before the action decides what to do. |
| `exactness` | `exact_ref`, `narrow`, or `broad`; exact refs should not be treated like fuzzy search hits. |
| `result_window` | `limit`, `offset`, `returned`, `total` when known, and whether the action sees the window or the complete match set. |
| `items` | Stable refs in ranked/query order, with enough display metadata to choose without another call. |
| `affordances` | Actions that can run against the selection or selected item, including disabled reasons. |
| `warnings` | Staleness, partial ingest, degraded browser capture, unembedded semantic lane, route failures, or capped result windows. |
| `next_actions` | CLI/MCP/HTTP/web route equivalents for the most likely follow-up actions. |

The envelope should make the destructive blast radius explicit. Mutating verbs
must act on either the complete match set or a deliberately selected subset,
never on an incidental display page unless the command says so.

## Cardinality Rules

Cardinality is a first-class state. Different verbs consume it differently:

| Verb | Empty | One | Many |
| --- | --- | --- | --- |
| `select` | Return no selectable rows and an empty-state hint. | Print/return the single ref. | Prompt, fzf, or print a pipeable ranked list. |
| `read` | Error with no-match guidance. | Render the requested read view. | Require `--all`, `--first`, or an explicit ref/selection for singleton views; bulk/export views may accept many. |
| `continue` | Error with no-match guidance. | Emit successor handoff/recovery packet. | Require explicit selection unless running `continue --candidates`. |
| `analyze` | Return zero-count/facet envelope, not an error. | Analyze the singleton as a set of one. | Analyze the full result set by default. |
| `mark` | Error. | Mutate the singleton. | Require `--all` or `--first`; report affected refs. |
| `delete` | Error. | Require confirmation; support `--dry-run`. | Require `--all --yes`; `--dry-run` must preview the same complete set that deletion would remove. |
| `export` | Return empty export metadata. | Export one item. | Export the complete requested set; streamable formats must preserve order. |

Current CLI code already has shared cardinality guards for mutating verbs in
`polylogue/cli/verb_cardinality.py`. The remaining convergence work is to make
read/continue, web routes, MCP tools, shell completions, and JSON envelopes
observe the same policy.

## Action Affordances

Surfaces should not hard-code action buttons independently. They should consume
a shared action inventory with this shape:

| Field | Meaning |
| --- | --- |
| `id` | Stable verb/action id such as `read.summary`, `read.messages`, `continue`, `mark.tag_add`, or `delete.preview`. |
| `label` | Short human label for buttons, menus, command palette rows, and shell help. |
| `applies_to` | `selection`, `session`, `message`, `action`, `assertion`, `file`, or `browser_capture`. |
| `state` | `enabled`, `disabled`, `loading`, `dangerous`, `requires_confirmation`, or `unsupported`. |
| `disabled_reason` | Machine-readable reason plus short text when state is not directly runnable. |
| `cardinality` | Required cardinality or explicit support for full result sets. |
| `formats` | Supported render/output formats. |
| `routes` | CLI, MCP, HTTP, and web route equivalents where they exist. |
| `next_actions` | Follow-up actions after completion. |

Existing `ReaderActionAvailabilityPayload` covers part of this for reader
actions. The target is one inventory that drives shell completion, web action
strips, MCP discovery, `/api` affordance metadata, and JSON output schemas.

## Verb Contracts

### `select`

`select` is the bridge between broad recall and exact action. It should be the
default answer to "I found several plausible things; which one did you mean?"

- Terminal interactive mode may use fzf or a prompt when available.
- Non-interactive terminal mode prints stable refs, one per line by default.
- JSON mode returns selected item metadata, not only the id string.
- Web mode exposes the same selection as a focused result list with action
  buttons on each row and a bulk action bar for result-set verbs.

### `read`

`read` renders evidence. It should never hide whether the requested view is a
singleton view or a set view.

- `summary`, `messages`, `raw`, `context`, `recovery`, `neighbors`, and
  `correlation` are singleton read views unless explicitly documented as bulk
  views.
- `--all` is a deliberate bulk/export mode. It must not silently fall back to
  "first result" behavior.
- `read --views --format json` is the discovery surface for view metadata and
  should align with the action inventory.
- Direct refs such as `session:<id>` should bypass fuzzy search and report
  `exactness=exact_ref`.

### `continue`

`continue` emits successor-agent handoff material. The output must be useful
without hidden chat context.

- It requires one seed session except for `continue --candidates`.
- JSON output should include seed refs, recovery/read views used, omitted
  evidence, caveats, and token/size estimates.
- File and clipboard delivery should use the same content as terminal output.
- Web mode should expose "copy handoff", "download", and "open seed session"
  actions from the same affordance inventory.

### `analyze`

`analyze` operates on a result set. It should be comfortable for broad queries:

- Empty sets are valid and should return zero counts/facets.
- `--count`, `--facets`, `--by`, and cost views should report the query scope.
- JSON and CSV outputs must preserve the grouping dimension and query metadata
  needed to reproduce the analysis.

### `mark`

`mark` records user intent and must be reversible or inspectable.

- Singleton is the default. Many requires `--all` or `--first`.
- JSON output should include affected refs, before/after state where cheap, and
  the inverse action when an undo-like operation exists.
- Web mode should distinguish lightweight marks from durable assertions and
  show candidate assertion review as a separate flow.

### `delete`

`delete` is destructive and should have the strictest contract:

- `--dry-run` previews the exact refs that `--yes` would delete.
- Many requires `--all --yes`.
- The JSON result should include matched count, deleted count, skipped refs, and
  durable blockers.
- Web mode must use preview, confirmation, and a final mutation result envelope;
  it should never execute deletion from a row button without confirmation.

### `export`

`export` is the missing explicit result-set verb. Current `read --all` covers
some export use cases, but an explicit verb would make automation clearer:

- `polylogue find QUERY then export --format ndjson`
- `polylogue find QUERY then export --view messages --format markdown`
- `polylogue find QUERY then export --schema`

The verb should be tracked if `read --all` keeps accumulating behavior that is
not really "reading one matched thing".

## Rendering Rules

Terminal human output should bias toward fast continuation:

- Empty: show the query and one suggested narrowing/broadening action.
- One: show the match and the most likely next actions.
- Many: show a compact ranked table with id, title, origin, time, repo/cwd, and
  signal chips such as paste/tool/topology/browser-capture.
- Dangerous: show preview counts and exact confirmation syntax.

Machine output should be boring and stable:

```json
{
  "status": "ok",
  "query": {"raw": "repo:polylogue browser capture", "unit": "session"},
  "selection": {
    "cardinality": "many",
    "exactness": "broad",
    "returned": 20,
    "total": 143,
    "items": [{"ref": "session:...", "title": "..."}]
  },
  "affordances": [
    {
      "id": "read.messages",
      "state": "disabled",
      "disabled_reason": "requires_selection",
      "cardinality": "one",
      "cli": "polylogue find ... then read --view messages"
    }
  ],
  "warnings": [],
  "next_actions": []
}
```

`ndjson` should stream one item per line with a small header/footer envelope
only when the selected consumer asks for it. Automation should not need to parse
Rich tables or terminal prose.

## Browser Capture And Live Sessions

Browser-captured sessions need two extra bits in the same flow:

- `capture_fidelity`: whether Polylogue got a provider-native conversation
  payload, a structured app payload, DOM text, or a degraded placeholder.
- `capture_state`: whether the source tab/session is live, stale, closed, or
  already materialized in the archive.

If a full provider-native ChatGPT or Claude payload is available, it is the
preferred source. DOM extraction is useful for live progress and degraded
visibility, but it should not overwrite cleaner captured data. Result rows and
read views should expose fidelity so the user can trust or distrust the
transcript without opening raw artifacts.

## Shell Completion Contract

Shell completion should be generated from the same action/read-view inventory:

- after `polylogue find QUERY` -> suggest `then`;
- after `then` -> suggest `select`, `read`, `continue`, `analyze`, `mark`,
  `delete`, and future `export`;
- after `then read --view` -> suggest live read-view ids;
- after `then read --format` -> suggest formats allowed by the selected view;
- after `then analyze --by` -> suggest supported dimensions;
- after dangerous verbs -> suggest `--dry-run`, `--yes`, `--all`, and `--first`
  only where the verb supports them.

Completions should work in a non-interactive test harness by invoking the shell
completion entrypoint directly. They should not require a live archive unless
the candidate list genuinely depends on archive contents.

## Golden Paths

These paths should be covered by CLI snapshots, JSON-schema fixtures, and web
route/DOM smoke tests:

1. Exact ref read:
   `polylogue find id:<session> then read --view messages --format json`
   reports `exact_ref`, one selected session, and message rows.
2. Broad query select:
   `polylogue find 'repo:polylogue browser capture' then select --json`
   returns a selected ref that can be fed to `read`.
3. Successor handoff:
   `polylogue find id:<session> then continue --format json` includes seed
   refs, recovery evidence, omissions, and next actions.
4. Result-set analysis:
   `polylogue find 'repo:polylogue since:7d' then analyze --facets --format json`
   reports the scoped query and facet counts.
5. Destructive preview:
   `polylogue find 'repo:polylogue tag:stale' then delete --dry-run` and
   `--yes --all` resolve the same complete ref set.
6. Browser capture read:
   the latest captured ChatGPT session can be found, reports capture fidelity,
   and renders transcript text from the cleanest available source.
7. Web route parity:
   opening the same selection in the daemon reader shows the same refs,
   disabled reasons, and action availability as the CLI JSON envelope.

## Implementation Slices

The coherent implementation order is:

1. Define a shared query-action affordance DTO and registry from current CLI
   verbs/read-view profiles.
2. Make CLI JSON output include selection/cardinality/exactness/action metadata
   for `find` and `find ... then <verb>`.
3. Route shell completion through the registry.
4. Make web result rows and detail views consume the registry, including
   disabled reasons and dangerous-action state.
5. Align MCP/HTTP action discovery with the same DTO.
6. Add golden-path fixtures for terminal text, JSON, and web route parity.
7. Add browser-capture fidelity fields to result/read payloads and smoke tests.

This should be a convergence effort, not a parallel surface. Existing payloads
such as `SearchEnvelope`, `ReaderActionAvailabilityPayload`, read-view profile
metadata, and `MutationResultPayload` are the raw material to fold together.
