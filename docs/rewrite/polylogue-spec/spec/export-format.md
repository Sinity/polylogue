# Export format

One documented JSON shape for a normalized session. It is what
`polylogue export` writes, what `read --format json` returns, what the parser
example tests compare against (`plan/testing.md`), what the Python client
receives, what Sinex receives (`purpose.md`), and an input format Polylogue
itself reads back ("Reading an export back", below). One shape, so these
never drift apart. (`--json` on a query is different: it prints the query
result fields of `query.md`, "Results"; only `read --format json` and
`export` write this format.)

The worked example in `model.md` is a `session` and its `messages` in this
format, with the document header, the other arrays and optional fields left
out.

## Files

- `--format json`: one JSON document per session,
  `<dir>/<origin>/<native id with each byte outside [A-Za-z0-9._~-] written as %XX>.json`;
  `origin` has the validated namespace grammar from `model.md`, Origins;
  the document's `id` field, not the file name, is the identity.
- `--format jsonl`: one line per session in one file, same objects.
- Each document starts with `"format": "polylogue-session"` and
  `"version": 1`. A reader refuses a higher major version than it knows,
  with a message naming both versions; it never guesses.
- **The manifest.** Every export directory (or the first line of a `jsonl`
  file) carries a manifest saying what was selected and how
: `{"format": "polylogue-export-manifest", "version": 1,
  "query": "<the query text>", "ran_at": …, "view": "<index generation:counter>",
  "up_to_date": true|false, "sessions": N, "with_notes": …, "with_inherited":
  …, "with_blobs": …, "exported_by": …}` in `manifest.json`. An export
  without a query (one session named directly) records that reference as
  its query. Reading an export back stores the manifest with the input; it
  never changes what is parsed.
- **Profiles**. `export --profile NAME`
  chooses a named selection and shape for a consumer, recorded in the
  manifest as `"profile"`: `full` (the default, this document), `markdown`
  (readable transcripts), and `wiki` (one Markdown file per session with
  frontmatter for origin, models, time, repo and tags, and wiki-links to
  related sessions, for Obsidian-style knowledge bases). A profile only
  selects and formats; it never changes what a field means.
- **Citations back to the archive**. Every
  exported session, message and note carries its stable reference
  (`model.md`, Identity), so a report, a PR description or a git note can
  cite exactly what it rests on and anyone with the archive can resolve it.
  `polylogue cite REF` prints a short citation line (reference, title, time)
  for pasting. A published claim cites references, never a rendered page
  that could change; a later change to the archive never rewrites what an
  earlier export or citation said.

## Rules for every object

- Field names are `snake_case`; the order shown here is the order written,
  so exports diff cleanly.
- **Absent means unknown; `null` is never written.** A field the provider
  did not record is left out. A field that is known to be empty is written
  empty (`"blocks": []`, `"title": ""` only if the provider set an empty
  title).
  (Query results are different: they are tables with a fixed set of
  columns, so an unknown value there is `null`, `query.md`. A session
  document has no fixed columns, so it simply omits what is unknown.)
- Times are RFC 3339 in UTC with `Z`, always with milliseconds (the index
  stores integer milliseconds; a provider that gave seconds gets `.000`).
  This holds for every JSON Polylogue writes, query results included;
  only terminal output shows local time.
- Token counts are integers as reported. Polylogue's computed cost is not
  exported (it depends on the price table in use, `analysis.md`); the
  provider's own cost figure is, when it gave one (`usage.provider_cost_usd`).
- IDs are the stable IDs from `model.md`, written in full.
- Text is exactly as stored; nothing is trimmed or re-wrapped.
- Blob hashes are written `blake3:<hex>` everywhere in the format. Input
  IDs are written as stored: the hex BLAKE3 of an input stored complete, or
  a UUIDv7 for a location whose content can change (a growing log, a file
  rewritten in place, a database, a Drive file) (`model.md`, Identity); they carry no
  prefix, because an input ID is not always a hash.
- Binary content (images, files) is never inlined. A block refers to it by
  `blob` (`blake3:<hex>`); `export --with-blobs` also writes
  `<dir>/blobs/<hex>` files.

## The document

```json
{
  "format": "polylogue-session",
  "version": 1,
  "exported_at": "2026-10-08T10:00:00.000Z",
  "exported_by": "polylogue 0.1.0",
  "session": {... },
  "messages": [... ],
  "links": [... ],
  "events": [... ],
  "notes": [... ],
  "runs": [... ],
  "deliveries": [... ],
  "inputs": [... ]
}
```

### `session`

| Field | Type | When present |
|---|---|---|
| `id`, `origin`, `native_id`, `kind` | string | always; `kind` is `main`, `subagent` or `chat` (`model.md`, Session) |
| `title` | string | provider gave one |
| `display_title`, `title_source` | string | always: the title every surface shows and where it came from (`note`, `provider`, `index`, `job`, `instruction`, `first_prompt`, `id`; rule in `model.md`, Session) |
| `started_at`, `last_activity_at` | time | an own message or event has a time |
| `cwd`, `repo`, `branch`, `commit` | string | from git evidence (`model.md`, Session) |
| `repo_root` | string | the repository's root commit hash, its identity |
| `checkout` | string | the worktree path, when separate |
| `collection` | object `{"id", "name"}` | the provider's grouping (ChatGPT project or custom GPT, claude.ai project) |
| `harness`, `harness_version` | string | agent sessions |
| `team`, `agent_name` | string | a named member of an agent team: the team and the member's name |
| `machine` | object `{"id", "label"}` | the machine the session ran on (machine ID and host name), when known |
| `models` | object model → message count | any model message |
| `temporary` | `true` | only when true |

### `messages[]`

| Field | Type | When present |
|---|---|---|
| `id`, `role`, `written_by` | string | always (`written_by` is `human`, `model`, `harness` or `unknown`) |
| `prompt_kind` | string | `written_by: human`: `prose` or `command` |
| `model` | string | model messages that name it |
| `at` | time | provider gave a time |
| `completed_at` | time | a message assembled from several timed records: the last record's time |
| `first_seen` | time | once an observation covers the message's bytes (within about a minute of their arrival): when Polylogue first had them. It depends on when collection ran, so the law tests leave it out of what they compare (`plan/testing/laws.md`) |
| `parent` | message ID | provider records a tree |
| `on_active_path` | `true` or `false` | always, except when unknown (then absent) |
| `id_kind` | `"provisional"` | only when the ID is Polylogue-computed (`model.md`, provider and provisional IDs) |
| `inherited_from` | message ID | a copy: the message it was copied from, as the evidence names it |
| `copy_origin` | `"unknown"` | a copy whose source the provider does not name |
| `counted_as` | message ID | a copy: the message counted in its place, when it differs from `inherited_from` |
| `from_session`, `from_name` | session ID / string | a message one agent sent to another (`written_by: model`): the sending session when resolved, and the sender's name as written |
| `usage` | object: `input`, `output`, `cache_read`, `cache_write` (all cache writes), `cache_write_5m` and `cache_write_1h` (the parts by lifetime, when reported), `reasoning`, `web_requests`, `speed`, `service_tier`, `provider_cost_usd` | reported; each key only when reported |
| `stop_reason` | string | recorded |
| `in_progress` | bool | present and true only when the provider marked this version as still being produced (`model.md`) |
| `text_origin` | object | human messages, when evaluated: the exact evidence object from `model.md`, Text-origin evidence; candidate clipboard matches and provider-reported pastes stay separate, with revision pins and proxy metric names |
| `duration_ms` | integer | recorded |
| `sources` | array of `{"input": "<input id>", "chain": N, "start": N, "end": N}` | every stored version of the message: the byte range of its records in a revision chain of an input (offsets are stable within a chain, which only appends; a rewrite starts a new chain). The first entry is the version shown (the preferred version, `model.md`); a message seen through several routes or chains has one entry for each. An entry into bytes excised since reads as excised |
| `attrs` | object | provider fields kept as given that have no field of their own (attribution, effort, Codex `phase`, citations, speech or recap markers); keys as the provider wrote them |
| `blocks` | array | always (may be empty) |

Inherited messages are included with `inherited_from` set, so one file is a
complete session. `export --no-inherited` leaves them out.

### `blocks[]`

Every block has `id` and `kind`. A block's `id` has one of three forms (`model.md`, Identity): `<message id>:<index>`; `<message id>:r<index>` for a reasoning item in a format whose rewrites drop reasoning (Codex); `<message id>:a:<attachment id>` for an attachment the provider lists outside the content list. Further fields by kind:

| `kind` | Fields |
|---|---|
| `text` | `text` |
| `thinking` | `text` (may be absent), `signature` (when present), `encrypted: true` with `data` (as stored) when the provider encrypted it |
| `tool_call` | `tool`, `call_id` (when the provider gives one), `input` (JSON as given), `command` (shell-like tools), `data` (provider fields kept as given, e.g. Claude Code `caller`), and `outcome: "unknown"` with its `outcome_reason` (`no_result`, `background_never_reported`) only when no result block answers it |
| `tool_result` | `call_id`, `outcome` (`ok`/`error`/`unknown`, the stored fact; an export never carries the read-time `pending`), `outcome_reason` (with `unknown`), `exit_code`, `duration_ms`, `text` or `blob` (large side-file output), `data` (structured tool output kept as given, e.g. Claude Code `toolUseResult` fields) |
| `image`, `file` | `blob` (when the bytes were obtained), `name`, `media_type`, `size`, `direction` (`input`, `output`, `unknown`), `produced_by` (block ID, for outputs); `url` for a linked file that was downloaded |
| `other` | `type` (the provider's type name), `data` (as given) |

### `links[]`, `events[]`

- Link (`model.md`, Link): `kind` (`subagent`, `continuation`,
  `reference`), `how` (always; subagent `fresh`, `copied`, `summarized`;
  continuation `resume`, `fork`, `after_compaction`; reference `briefing`,
  `ping`, `mention`), `parent`, `child`, `branch_point` (message ID),
  `spawned_by` (block ID of the tool call), `retry_of` (the spawning call
  it retries, when recorded), `result_use` (`returned`, `referenced`;
  absent when unknown), `evidence` (the input field that
  established it, e.g. `"meta.toolUseId"`), `attrs` (the provider's facts
  about the relationship as given: agent type, role, depth, worktree). A
  link whose parent is not in the archive is still written, with the
  parent's ID. A document carries every link whose `child` is this session
  (its incoming links). Outgoing links appear in the children's documents;
  importing a set of documents restores every link once.
- Event: `id`, `kind`, `at`, `after_message` (the message it sits after, when known),
  `model_saw` (`true` when the text was given to the model as context),
  `data` (the provider's payload as given).

### `notes[]`

Notes from `archive.db` targeting this session or its messages, so an
export carries the operator's annotations with it: `id`, `root`, `kind`,
`key`, `value`, `check` (a claim's check, as stored), `author`, `machine`,
`created_at`, `valid_until`, `source` (for inline markers), `targets`
(references, each with `seen_hash` when recorded) and `relations` (`kind`
and `other`, e.g. `revises`, `withdraws`). Every note of each family that
has a note on the session is included, older revisions and withdrawals
with it, so no relation in the export points outside it, and a conflict, a
withdrawal or an edit history travels with the export.
`export --no-notes` leaves them out.

### `runs[]`, `deliveries[]`

- Run (`model.md`, Run): `id`, `kind`, `task` (for an attempt),
  `instruction`, `state`, `started_at`, `ended_at`, `evidence`, for every
  run the session carried out.
- Delivery (`model.md`, Delivery): `id`, `event`, `at`, `content_hash`
  (`blake3:<hex>`), `bytes`, `refs`, `left_out`, `signal_rules`, `machine_rules`, `outcome`, `reason`, for
  every delivery the session received (and `rule`, `rows`, `author` on a
  signal's delivery). A signal notice to the operator belongs to no session
  and is not in a session export. A delivery the session sent (a
  ping from a page it acted for) appears in the recipient's export.

### `inputs[]`

Which stored inputs produced this session (from `parsed`): `id`, `format`,
`path` (as first seen), `parser_version`. This is how an exported session can be traced
back to the exact bytes.

## Reading an export back

Importing a Polylogue export (`polylogue import DIR`) is an input of format
`polylogue-session` (the same name the documents carry), parsed like any
provider. Because every ID is written in
full, the parsed sessions have the same IDs as the originals; importing an
export of sessions already in the archive changes nothing except recording
the new input (`plan/testing/laws.md`, law "export round trip"). Notes in an
import are written by the import request itself, at import time, in the
same transaction that records the input, and only those whose `id` is not
already present and is not named by a `note:` excision; parsing never
writes them, so a rebuild does not add them again.
`display_title`, `title_source` and any other field this format marks as
derived are recomputed by the importing archive, never read.

## OpenTelemetry export (slice 13)

Sessions can also be written as OpenTelemetry GenAI traces, for tools that
read them. Trace and span IDs are derived from the session and message
references with BLAKE3 (the first 32 and 16 hex digits). Paths are
redacted unless `--keep-paths` is set (traces usually leave the machine); tool output is never
copied, only whether it exists and its length; message text only with
`--with-text`. There is no default limit on the number of sessions.

## Versioning

- Adding an optional field does not change `version`.
- Renaming or removing a field, or changing a field's meaning, increments
  `version`, and the reader for the previous version stays (an old export is
  an input, and inputs are never refused for being old).
- The JSON Schema for each version is generated from the Rust types
  (`schemars`) and published in the repository as
  `docs/export-format.v1.schema.json`; a test checks it is current.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.
