## Turn 110 (role=assistant) — heredoc writing 5 proof-package doc files: docs/proof/semantic-renderer/{README,CARD-CONTRACT,COMPARISON,NON-CLAIMS,REVIEW-GUIDE}.md

bash -lc set -euo pipefail
cd /mnt/data/semantic-renderer-work/git
mkdir -p docs/proof/semantic-renderer/follow-ons
cat > docs/proof/semantic-renderer/README.md <<'EOF'
# Semantic transcript renderer proof packet

This directory is the review and reproduction packet for the launch slice of
Bead `polylogue-ap7`.

The slice changes the existing CLI `read --view messages` path from flat
message text into evidence-oriented semantic cards for shell commands, file
edits, and session-lineage boundaries. It also lands the provider-neutral card
contract, task and attachment model shapes, an explicit provider/tool registry,
raw fallback behavior, and a hostile golden corpus that the web reader can
consume in a bounded follow-on.

## Result

The implementation establishes five properties:

1. **One pure semantic model.** Card construction accepts hydrated messages and
   an optional already-read lineage descriptor. It performs no archive, SQLite,
   network, or daemon access.
2. **Structural outcomes only.** `tool_result_is_error` and
   `tool_result_exit_code` determine success, failure, or unknown. Prose cannot
   promote a `NULL` outcome to success or failure.
3. **Open-world provider handling.** Persisted `semantic_type` has precedence,
   followed by a repository-grounded exact provider/tool alias. Every unlisted
   name becomes a fallback card containing raw input and result evidence.
4. **Exact source coordinates.** Cards retain the session, message, block,
   paired result message, and paired result block coordinates used to build
   them.
5. **Bounded but disclosed previews.** Large outputs retain first and last
   evidence, count exact omitted lines or characters, and disclose invalid
   UTF-8 replacements.

## What is implemented

| Surface | State in this slice |
|---|---|
| Provider-neutral `semantic-card.v1` model | Implemented |
| Exact provider/tool registry and generated gap table | Implemented |
| Shell cards | Implemented in the existing CLI messages view |
| File-edit/write cards and reconstructed diffs | Implemented in the existing CLI messages view |
| Lineage-boundary cards | Implemented in the existing CLI messages view |
| Unknown-tool raw fallback | Implemented in the existing CLI messages view |
| Task/subagent and attachment card data models | Implemented and golden-tested; deeper links/actions are follow-ons |
| Structural outcome hydration from SQLite | Implemented and regression-tested |
| Provider origin hydration | Implemented and regression-tested |
| Web visual card backend | Deliberately not claimed; bounded packet `follow-ons/01-web-reader-wiring.md` |
| Cross-page streaming result pairing | Deliberately not claimed; bounded packet `follow-ons/05-pagination-streaming.md` |

## Review order

1. Read [CARD-CONTRACT.md](CARD-CONTRACT.md).
2. Inspect the generated [tool map](../../generated/semantic-card-tool-map.md).
3. Compare [before](before/) and [after](after/) CLI receipts.
4. Inspect the hostile [golden corpus](golden/index.json) and source fixtures
   under `tests/data/semantic_cards/`.
5. Read [INTEGRATION-POINTS.md](INTEGRATION-POINTS.md) to verify that the
   existing read and hydration paths were extended rather than bypassed.
6. Run the commands below, including the expected-failure anti-vacuity control.
7. Read [NON-CLAIMS.md](NON-CLAIMS.md) before reusing any wording externally.

## Reproduce the proof

```bash
# Pure generated contracts and hostile fixtures.
PYTHONPATH=. python -m devtools.render_semantic_card_registry --check
PYTHONPATH=. python -m devtools.render_semantic_card_fixtures --check
PYTHONPATH=. python -m devtools.render_semantic_renderer_proof --check

# Seed the same private-data-free demonstration world used by the receipts.
archive="$(mktemp -d)/polylogue-semantic-renderer"
PYTHONPATH=. python -m polylogue demo seed \
  --root "$archive" --force --with-overlays --format json
PYTHONPATH=. python -m polylogue demo verify \
  --root "$archive" --require-overlays --format json
PYTHONPATH=. python -m devtools.render_semantic_renderer_proof \
  --check --archive-root "$archive"

# Launch-slice tests.
PYTHONPATH=. pytest -q \
  tests/unit/rendering/test_rendering.py \
  tests/unit/rendering/test_semantic_cards.py \
  tests/unit/devtools/test_render_semantic_renderer_proof.py \
  tests/unit/cli/test_messages.py \
  tests/unit/cli/test_streaming_markdown_read_view.py \
  tests/unit/api/test_facade_contracts.py::test_message_hydration_preserves_origin_provider_and_structural_tool_outcome \
  tests/unit/core/test_models.py \
  tests/unit/storage/test_archive_tiers_write.py::test_archive_tiers_writer_materializes_codex_session
```

The external campaign packet also includes an executable corrupted-fixture run.
It changes one frozen card kind, invokes the same fixture checker, and requires a
non-zero exit. This complements the in-suite anti-vacuity test; it is not a
special test-only checker.

## Proof artifacts

| Artifact | Purpose |
|---|---|
| `proof-manifest.json` | Machine-readable claims, fixture index, demo verification summary, commands, sizes, and SHA-256 receipts |
| `before/*.md` | Flat baseline output generated at the stated base revision |
| `after/*.md` | Output from the real CLI route over the seeded demo archive |
| `golden/*.cards.json` | Frozen provider-neutral card contracts |
| `golden/*.md` | Backend-neutral Markdown render of each hostile fixture |
| `../../generated/semantic-card-tool-map.json` | Machine-readable provider/tool mapping and open-namespace policies |
| `../../schemas/semantic-card-v1.schema.json` | Public card JSON schema used by tests |
| `COMPARISON.md` | Human explanation of the before/after construct |
| `follow-ons/` | One-PR packets for the remainder of the epic |

## Why the proof is not vacuous

The implementation is not judged by whether output “looks nicer.” The oracles
are independent structural facts planted in the fixture inputs:

- `is_error=false` and `exit_code=0` remain success even when output contains
  the word `ERROR`;
- success-like prose remains unknown when both structural fields are `NULL`;
- conflicting structural fields resolve conservatively to failure and disclose
  the disagreement;
- a 10,000-line result must report exactly 9,936 omitted lines under the
  configured 48-head/16-tail policy;
- malformed JSON remains visible as raw input;
- invalid UTF-8 bytes become visible replacement characters and a counted
  caveat;
- interleaved subagents pair by exact tool ID rather than adjacency;
- an intentionally corrupted frozen expectation fails the real fixture check.

## Architecture in one line

```text
existing archive reads → hydrated evidence → pure semantic-card builder
                       → semantic-card.v1 → CLI Markdown now / web backend next
```

The archive and API remain outside card construction. The CLI reads session and
lineage data at its existing orchestration seam, then hands only values to the
pure renderer.
EOF

cat > docs/proof/semantic-renderer/CARD-CONTRACT.md <<'EOF'
# `semantic-card.v1` contract

`semantic-card.v1` is a provider-neutral, evidence-preserving intermediate
representation for transcript presentation. It is not an analytics result and
is not an alternative archive model.

## Classification precedence

Classification is deliberately closed and reviewable:

1. A persisted, trusted `semantic_type` maps to a card kind.
2. Otherwise, an exact `(provider family, tool name)` row in the generated
   registry maps to a card kind.
3. Otherwise, the tool becomes `fallback` and carries raw evidence.

There is no substring, regular-expression, language-model, or prose-based
classification in the renderer. Provider tool namespaces are treated as open.
The finite registry records only names supported by parser fixtures, parser
record vocabulary, or the repository's existing classifier contract.

## Card vocabulary

| Kind | Meaning in v1 | Launch state |
|---|---|---|
| `shell` | A command-like tool use with structurally paired output | CLI launch slice |
| `file_edit` | An edit or write with an exact supplied or reconstructed diff | CLI launch slice |
| `task` | A task/subagent dispatch and its exact result evidence | Model and fixtures; deep link follow-on |
| `lineage` | A session parent/root boundary supplied by the topology reader | CLI launch slice |
| `attachment` | An image/document/file reference without embedded bytes | Model and fixtures; actions follow-on |
| `fallback` | Unknown tool use or unpaired result, preserving raw evidence | CLI launch slice |

## Source coordinates

Every card identifies the evidence from which it was built:

- `session_id`;
- provider family;
- tool-use message and block ID or block index;
- tool name and tool ID;
- paired result message and block ID or block index, when present.

The source object is not a claim that every coordinate already has a public HTTP
resolver. It ensures backends do not discard the coordinates while rendering.

## Structural outcome truth table

Outcome is a three-state value. Missing evidence is not success.

| `is_error` | `exit_code` | State | Caveat |
|---|---:|---|---|
| `true` | any/`NULL` | `failed` | Conflict disclosed when exit is `0` |
| `false` | non-zero | `failed` | Conflict disclosed; non-zero exit wins conservatively |
| `false` | `0`/`NULL` | `succeeded` | None |
| `NULL` | `0` | `succeeded` | None |
| `NULL` | non-zero | `failed` | None |
| `NULL` | `NULL` with result | `unknown` | Result exists but has no structural outcome |
| no paired result | n/a | `unknown` | Missing result disclosed |

Text such as “SUCCESS,” “failed,” or “ERROR” is output evidence, not outcome
metadata.

## Tool-result pairing

Tool uses and results pair by exact `tool_id`, FIFO within a duplicated ID. The
pairing supports interleaving and does not infer a relation from adjacency.
Unpaired results become fallback cards and retain their own outcome and text.

The launch implementation groups a paired result into the card at the tool-use
position. It retains both original coordinates so a future forensic profile can
present the exact temporal positions independently.

## Preview contract

Default bounds are:

- first 48 lines;
- last 16 lines;
- 16,000 characters when line bounds still exceed the character ceiling.

A preview declares:

- original line count;
- exact omitted line count or omitted character count;
- truncation state;
- selection strategy;
- number of UTF-8 replacement characters introduced while decoding.

The raw archive remains authoritative. A preview is a bounded presentation of
that evidence, not a replacement for it.

## File-edit contract

The card prefers a supplied `patch` or `diff`. Otherwise it constructs a unified
diff only from explicit old/new or content fields. If those fields are absent,
it does not invent an edit from result prose; it emits the card with raw input
and the caveat that no exact diff could be constructed.

## Purity boundary

The following are permitted inputs:

- hydrated messages or message-shaped mappings;
- hydrated content blocks;
- an already-read lineage descriptor;
- an optional session provider family.

The builder must not open the archive, query SQLite, call the daemon, read
attachments, resolve refs, or fetch URLs. That boundary allows the CLI, web
reader, tests, exports, and future presentation profiles to share the same card
structure.

## Serialization

`SemanticCard.to_document()` emits the JSON contract validated by
`docs/schemas/semantic-card-v1.schema.json`. `SemanticTranscript.to_document()`
wraps ordered prose and card entries for a future web payload. Backends may vary
in typography or interaction, but structure, refs, outcome, previews, and
caveats must remain equal.
EOF

cat > docs/proof/semantic-renderer/COMPARISON.md <<'EOF'
# Before and after: what changed, and what did not

The comparison uses the existing `read --view messages` command over the
private-data-free deterministic demo archive. The baseline was captured from
base revision `f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51`; the after files are
regenerated by `devtools.render_semantic_renderer_proof` through the real CLI
route.

## Structural shell failure

Before, the command result was an undifferentiated JSON paragraph:

```text
{"metadata": {"exit_code": 4}, "output": "ERROR: file or directory not found: tests/missing_test.py"}
```

After, the same evidence is grouped as a shell card:

```text
### Shell command · FAILED
- command: pytest tests/missing_test.py
- structural outcome: failed, is_error=true, exit_code=4
- exact tool-use and result refs
```

The improvement is not the red badge by itself. The card's failure state comes
from stored outcome fields, while the raw result remains visible.

## Unknown is not success

The demo contains a Bash result whose text says `1 passed` but whose stored
outcome fields are `NULL`. The after output deliberately says:

```text
### Shell command · outcome unknown
> Caveat: tool result exists but carries no structural success/failure outcome
```

This is a stricter and more useful result than a heuristic green badge.

## File edit

The seeded live demo's first `Write` tool does not contain sufficient edit input
to reconstruct a diff. The card therefore says so. The independent hostile
fixture `golden/claude-edit-diff.md` supplies explicit old and new strings and
proves the exact unified-diff path.

This two-part proof avoids a common demo mistake: manufacturing a pretty diff
from a live record that did not actually contain enough evidence.

## Lineage

Before, a composed fork read as four prose messages. Nothing in the rendering
explained why messages from two physical sessions appeared together.

After, a lineage-boundary card precedes the content and names the child, root,
parent, relation, and relation uncertainty. The edge remains `unknown` because
that is what the seeded topology establishes; the renderer does not upgrade it
to a fork merely because the title looks fork-like.

## Unknown tools

Before, an unknown tool could collapse into generic message text. After, the
fallback card shows:

- exact tool name and ID;
- raw JSON input or malformed raw input;
- paired result evidence when present;
- structural outcome when present;
- a caveat that classification is unknown.

Fallback is therefore a first-class safe behavior, not a visual error path.

## What stayed unchanged

- JSON and NDJSON message contracts are unchanged.
- Archive storage remains authoritative.
- The existing CLI messages view remains the entry point.
- Provider parsers remain responsible for normalization.
- The renderer does not query the archive.
- Prose remains in transcript order.
- Raw evidence remains available and exact refs are retained.
EOF

cat > docs/proof/semantic-renderer/NON-CLAIMS.md <<'EOF'
# Non-claims and release boundaries

This packet is intentionally explicit about what it does **not** establish.

## Surface completion

- It does not claim that the web reader visually renders semantic cards yet.
- It does not claim CLI/web structure-parity snapshots yet; the shared JSON
  contract and web packet are the prerequisite.
- It does not claim interactive expand/collapse in terminal output. The CLI
  emits bounded, omission-disclosing Markdown.
- It does not claim a linked Task/subagent card. Task cards exist in the pure
  model and fixtures, but child-session resolution and navigation are follow-on
  work.
- It does not claim inline web images, attachment downloads, or attachment-byte
  access from cards.
- It does not claim cross-page or streaming-window tool-result pairing.
- It does not claim operator/forensic/presentation layout profiles.

## Provider completeness

- The registry is not a universal catalog of every tool any provider may emit.
- Provider namespaces are open. Only repository-grounded exact aliases are in
  the table; all other names fall back unless upstream persisted a semantic
  type.
- The renderer does not improve parser correctness for records that were never
  normalized into tool-use/result blocks.
- The launch proof covers five provider families through fixtures, not every
  historical provider variant or future schema.

## Evidence and outcome

- A green or red card is not an independent verification that an external side
  effect happened; it reports the structural result captured by the provider.
- A `NULL` outcome remains unknown even when prose appears conclusive.
- A reconstructed edit diff is limited to fields actually present in tool
  input. It is not a filesystem diff and is not proof that the write landed.
- Result pairing by tool ID is not proof that a provider's duplicated IDs are
  globally unique; pairing is FIFO within the supplied transcript.
- The lineage card reports stored topology. It does not infer a narrower edge
  kind from text, timing, or title similarity.

## Performance and scale

- The 10,000-line hostile fixture proves bounded preview behavior; it is not a
  full large-archive latency benchmark.
- The deterministic demo has 11 sessions and 43 messages. It proves construct
  behavior, not private-archive scale.
- This packet does not claim browser rendering or streaming performance at the
  multi-million-message live-archive scale.

## Compatibility and release status

- The implementation does not add a public daemon endpoint in this slice.
- It does not promise `semantic-card.v1` stability beyond the reviewed v1
  contract without normal project compatibility review.
- It does not complete the full original `polylogue-ap7` acceptance criteria;
  the remaining web, links, card families, profiles, and pagination work is
  explicitly packetized under `follow-ons/`.
- It does not by itself make Polylogue release-ready.
EOF

cat > docs/proof/semantic-renderer/REVIEW-GUIDE.md <<'EOF'
# Reviewer guide

The patch series is organized so a reviewer can reject presentation work
without losing the evidence-hydration repair, or review the pure model without
first reading CLI orchestration.

## Commit 1: evidence-preserving pure contract

Review for:

- exact provider and structural-outcome hydration;
- absence of archive access in semantic-card construction;
- classification precedence and open-world fallback;
- stable JSON shape and schema;
- bounded preview arithmetic;
- hostile fixture independence and anti-vacuity.

A useful hostile review is to add a new unknown tool whose input resembles a
shell command. It must remain fallback until either upstream semantic type or an
explicit evidence-backed registry row exists.

## Commit 2: existing CLI view integration

Review for:

- continued routing through `run_read_messages` and `run_messages`;
- unchanged JSON/NDJSON behavior;
- destination capture via the existing `click.echo` adapter;
- topology read outside the pure renderer;
- exact before/after receipt generation through the public CLI.

## Commit 3: proof and follow-on packets

Review for:

- no claims stronger than the manifest and tests;
- exact base revision on baseline files;
- generated files reproducible in `--check` mode;
- bounded one-PR follow-on scopes;
- no hidden implementation claim for web parity or task links.

## Fast review commands

```bash
PYTHONPATH=. python -m devtools.render_semantic_card_registry --check
PYTHONPATH=. python -m devtools.render_semantic_card_fixtures --check
PYTHONPATH=. python -m devtools.render_semantic_renderer_proof --check
PYTHONPATH=. pytest -q tests/unit/rendering/test_semantic_cards.py
ruff check \
  polylogue/rendering/semantic_card_models.py \
  polylogue/rendering/semantic_card_registry.py \
  polylogue/rendering/semantic_cards.py \
  polylogue/rendering/semantic_markdown.py
```
EOF


---

## Turn 113 (role=assistant) — heredoc writing 7 more proof-package files: INTEGRATION-POINTS.md + 6 follow-on task docs (docs/proof/semantic-renderer/follow-ons/*.md)

bash -lc set -euo pipefail
cd /mnt/data/semantic-renderer-work/git
mkdir -p docs/proof/semantic-renderer/follow-ons
cat > docs/proof/semantic-renderer/INTEGRATION-POINTS.md <<'EOF'
# Existing-path integration points

The launch slice extends Polylogue's current archive hydration and CLI read
path. It does not introduce an archive reader, duplicate query path, or card
constructor that performs storage access.

Line numbers below refer to the implementation branch for this packet.

## Storage and hydration

| Integration point | File and lines | Change and reason |
|---|---|---|
| Archive block query | `polylogue/storage/sqlite/queries/attachment_blocks.py:19-50` | Selects existing block rows and now retains `semantic_type`, `tool_result_is_error`, and `tool_result_exit_code`. These are source-structural fields; dropping them would force the renderer to guess an outcome. |
| Block hydration | `polylogue/storage/hydrators.py:33-57` | Carries the persisted block ID and structural result fields into `BlockRecord`. Exact block coordinates are required for card evidence refs. |
| Session/message repository | `polylogue/storage/repository/archive/sessions.py:154-215` and `:284-349` | Both paginated and iterator paths use the existing block query and translate origin to provider family through `provider_from_origin`; no card-specific query is added. |
| Rendering adapter | `polylogue/rendering/block_models.py:13-38` and `:150-223` | Extends the existing `RenderableBlock` adapter with raw input, semantic type, block ID, outcome fields, and UTF-8 replacement counts. `NULL` is preserved as `None`. |

## Pure card construction

| Integration point | File and lines | Contract |
|---|---|---|
| Card data contract | `polylogue/rendering/semantic_card_models.py:13-247` | Defines `semantic-card.v1`, exact source coordinates, closed outcome states, bounded previews, lineage input, and ordered transcript entries. It imports no archive or daemon service. |
| Registry | `polylogue/rendering/semantic_card_registry.py:40-374` | Resolves persisted semantic type first, then only repository-grounded exact aliases. Unlisted names return fallback. Provider namespaces are explicitly open. |
| Transcript builder | `polylogue/rendering/semantic_cards.py:152-257` | Accepts already-hydrated messages and optional lineage; exact tool IDs are paired FIFO within the supplied transcript. No storage access occurs here. |
| Structural outcome | `polylogue/rendering/semantic_cards.py:289-333` | Computes success/failure/unknown only from `is_error` and exit code and emits a caveat when fields conflict. Text never participates. |
| Bounded evidence | `polylogue/rendering/semantic_cards.py:548-626` | Preserves head and tail evidence, exact omission counts, and decoding-replacement counts for large or invalid output. |
| Markdown backend | `polylogue/rendering/semantic_markdown.py:17-143` | Renders the same pure semantic transcript contract consumed by tests and intended for the web follow-on. |

## Existing CLI read surface

| Integration point | File and lines | Change and compatibility |
|---|---|---|
| Human messages view | `polylogue/cli/messages.py:39-123` | Reuses `Polylogue.get_messages_paginated`, `get_session`, and `get_session_topology` at the existing orchestration seam, then passes hydrated values to the pure builder. |
| Machine formats | `polylogue/cli/messages.py:62-102` | JSON and NDJSON contracts are intentionally unchanged. Only the human Markdown branch uses cards. |
| Existing destination adapter | `polylogue/cli/messages.py:118-123` | Uses `click.echo`, so existing `--to stdout|file|clipboard` capture continues to work without Rich interpreting evidence as markup. |
| Streaming adapter | `polylogue/cli/read_views/streaming_markdown.py:55-92` | Preserves IDs, semantic type, and outcome fields while retaining the current streaming rendering behavior. Cross-page tool/result pairing is deferred rather than guessed. |

## Existing web path reserved for the next packet

The current daemon session envelope flattens blocks to text at
`polylogue/daemon/http.py:2492-2512`. The browser then calls
`messageBlocksHtml(c.messages)` at `polylogue/daemon/web_shell.py:1364-1377`,
whose renderer is supplied by `polylogue/daemon/web_shell_reader.py:119-355`.
The web follow-on changes those exact seams to serialize and render
`semantic-transcript.v1`; it must not add a second session endpoint or a
JavaScript-only tool classifier.

## Verification that these are the real paths

- `tests/unit/api/test_facade_contracts.py::test_message_hydration_preserves_origin_provider_and_structural_tool_outcome`
  writes archive-tier rows and reads them through the public facade.
- `tests/unit/cli/test_messages.py` invokes `run_messages` and inspects the
  existing human and machine output contracts.
- `tests/unit/cli/test_streaming_markdown_read_view.py` checks the existing
  stream adapter rather than a new test-only path.
- The demo receipts invoke `python -m polylogue --id ... read --view messages`,
  the public CLI command used before this patch.
EOF

cat > docs/proof/semantic-renderer/follow-ons/README.md <<'EOF'
# Bounded follow-on packets for `polylogue-ap7`

The launch slice deliberately completes the pure contract and the existing CLI
messages view first. The packets here finish the original epic without turning
one review into a web rewrite, pagination redesign, and every-tool taxonomy at
once.

Each packet is sized for one pull request, names its owned files, and has a
public-path acceptance test. Packets may run in parallel only where their file
ownership does not overlap.

Recommended order:

1. `01-web-reader-wiring.md`
2. `02-task-attachment-links.md` and `03-read-search-web-mcp-cards.md`
3. `04-prose-thinking-layout-profiles.md`
4. `05-pagination-streaming.md`
5. `06-parity-permalinks-visual-proof.md`

The machine-readable mirror is `index.json`.
EOF

cat > docs/proof/semantic-renderer/follow-ons/01-web-reader-wiring.md <<'EOF'
# Packet 1 — web reader wiring over `semantic-transcript.v1`

## Purpose

Make the daemon reader serialize the same pure card contract already used by
the CLI. This packet changes data transport and a minimal visual backend; it
does not add new card classifications.

## Owned files

- `polylogue/daemon/http.py`
- `polylogue/daemon/web_shell.py`
- `polylogue/daemon/web_shell_reader.py`
- focused daemon envelope and browser snapshot tests

## Design

1. Read session topology at the existing session-envelope orchestration seam.
2. Call `build_semantic_transcript` in Python; JavaScript must not classify
   tool names or infer outcomes.
3. Add a versioned `semantic_transcript` field while retaining legacy
   `messages` for one compatibility window.
4. Render shell, file-edit, lineage, and fallback cards from card JSON.
5. Escape every field as untrusted evidence; card payloads may contain HTML,
   terminal control sequences, or Markdown fences.
6. Unknown schema versions must refuse to specialize and fall back to the
   existing block renderer.

## Acceptance criteria

- The seeded Bash failure, Edit, lineage fork, and unknown-tool fixtures have
  the same ordered `kind`, source coordinates, outcome, caveats, and omission
  counts in CLI and web structure snapshots.
- A `NULL` result displays `unknown`, never a green success state.
- An unknown tool displays raw input/result evidence and does not receive a
  specialized icon or title.
- The existing session endpoint remains bounded and reports a version marker.
- No provider/tool registry is duplicated in JavaScript.

## Verification

```bash
pytest -q \
  tests/unit/daemon/test_http_session*.py \
  tests/unit/daemon/test_web_shell*.py \
  tests/unit/rendering/test_semantic_cards.py
PYTHONPATH=. python -m devtools.render_semantic_renderer_proof --check
ruff check polylogue/daemon/http.py polylogue/daemon/web_shell.py \
  polylogue/daemon/web_shell_reader.py tests/unit/daemon
```
EOF

cat > docs/proof/semantic-renderer/follow-ons/02-task-attachment-links.md <<'EOF'
# Packet 2 — task/subagent and attachment links

## Purpose

Promote the already-golden-tested task and attachment card shapes into useful
CLI and web interactions without inferring missing child or blob identity.

## Owned files

- `polylogue/rendering/semantic_cards.py`
- `polylogue/rendering/semantic_markdown.py`
- daemon reader card renderer after packet 1
- typed-ref and attachment-envelope helpers
- focused topology, attachment, and renderer tests

## Design

- Resolve a task card to a child session only from stored topology/run refs or
  an exact provider-native relation; prompt similarity is not evidence.
- Represent unresolved dispatches as unresolved, preserving prompt and tool
  coordinates.
- Attachment cards carry media type, size, filename, acquisition status, and a
  typed content/ref action when available.
- Missing attachment bytes remain explicit and cannot render as an acquired
  image.

## Acceptance criteria

- Seeded Task dispatch links to the exact child session in CLI and web.
- An interleaved two-subagent fixture pairs each result and child correctly.
- An unresolved Task remains useful but has no fabricated child link.
- Acquired and metadata-only attachment controls are visibly distinct.
- Every action resolves through the public typed-ref resolver.

## Verification

```bash
pytest -q \
  tests/unit/rendering/test_semantic_cards.py \
  tests/unit/insights/test_topology*.py \
  tests/unit/api/test_ref*.py \
  tests/unit/daemon/test_*attachment*.py
```
EOF

cat > docs/proof/semantic-renderer/follow-ons/03-read-search-web-mcp-cards.md <<'EOF'
# Packet 3 — read, search, web, and MCP card families

## Purpose

Replace high-volume generic dumps with compact evidence cards for remaining
common tool families, while keeping the registry evidence-backed and open-world.

## Owned files

- `polylogue/rendering/semantic_card_models.py`
- `polylogue/rendering/semantic_card_registry.py`
- `polylogue/rendering/semantic_cards.py`
- both presentation backends
- generated mapping and hostile golden fixtures

## Card additions

- file-read: path, requested range, returned range/size, bounded excerpt;
- search: query, scope, match count when structural, bounded matches;
- web: URL/domain/title only when structurally present, bounded response;
- MCP: exact server/tool badge, folded input and result, structural outcome;
- provider protocol/unknown: raw fallback remains the floor.

## Acceptance criteria

- Every new exact alias cites a committed parser fixture, parser record type, or
  existing classifier contract in the generated mapping.
- A search prose sentence cannot become a match count.
- A URL-like string cannot become a web card without structural classification.
- Unknown MCP servers/tools preserve exact names and raw evidence.
- Every new card kind has at least one positive, one missing-field, and one
  hostile-large-output golden case.

## Verification

```bash
PYTHONPATH=. python -m devtools.render_semantic_card_registry --check
PYTHONPATH=. python -m devtools.render_semantic_card_fixtures --check
pytest -q tests/unit/rendering/test_semantic_cards.py tests/unit/sources
```
EOF

cat > docs/proof/semantic-renderer/follow-ons/04-prose-thinking-layout-profiles.md <<'EOF'
# Packet 4 — prose, thinking, and layout profiles

## Purpose

Finish transcript readability outside tool cards using declared render
profiles rather than a growing set of output flags.

## Owned files

- semantic transcript/card profile models
- Markdown and web presentation backends
- read-view profile registration
- snapshot and accessibility tests

## Profiles

- `operator`: compact cards, bounded evidence, thinking folded;
- `forensic`: exact coordinates, all caveats, larger limits, no hidden fields;
- `presentation`: stable demo layout with the same evidence semantics.

Profiles may change folding and presentation, not classification, outcome, or
source identity.

## Acceptance criteria

- Markdown prose follows the existing safe Markdown policy.
- Thinking blocks are distinct from assistant-authored conclusions and expose
  only token counts actually present in source evidence.
- All profiles serialize the same card kinds, source coordinates, and outcomes.
- Web controls are keyboard accessible and survive copy/paste without dropping
  refs or caveats.
- Presentation mode cannot hide an error or unknown outcome.

## Verification

```bash
pytest -q tests/unit/rendering tests/unit/daemon/test_web_shell* \
  tests/unit/cli/test_messages.py
```
EOF

cat > docs/proof/semantic-renderer/follow-ons/05-pagination-streaming.md <<'EOF'
# Packet 5 — pagination and streaming interaction

## Purpose

Make semantic rendering correct when a tool use and its result land in separate
pages or streaming chunks. The launch slice intentionally does not guess across
an incomplete window.

## Owned files

- messages read-view pagination orchestration
- streaming Markdown adapter
- daemon session-envelope pagination
- pure incremental pairing state and tests

## Design

- Introduce an explicit bounded pairing state keyed by session and exact tool
  ID, with duplicate-ID FIFO order and a declared retention limit.
- A page carries completeness metadata: complete, leading-context-missing, or
  trailing-result-pending.
- Pending cards remain unknown and disclose why. A later page may replace the
  projection, but must not rewrite the source event.
- `--full` continues to produce the same transcript as one-shot construction.

## Acceptance criteria

- Use/result split across two pages pairs exactly once after both arrive.
- Duplicate IDs, interleaved subagents, missing leading context, and permanent
  missing results have explicit controls.
- Buffer exhaustion produces a caveat and raw evidence, not a wrong pairing.
- Finite JSON and NDJSON machine contracts remain unchanged unless a separately
  versioned card format is requested.
- Streaming memory is bounded under a generated million-block sequence test.

## Verification

```bash
pytest -q \
  tests/unit/cli/test_streaming_markdown_read_view.py \
  tests/unit/cli/test_messages.py \
  tests/unit/rendering/test_semantic_cards.py
```
EOF

cat > docs/proof/semantic-renderer/follow-ons/06-parity-permalinks-visual-proof.md <<'EOF'
# Packet 6 — parity gate, permalinks, and visual proof

## Purpose

Close the original epic after all card families are present by proving
structural parity, stable navigation, and a public-safe before/after recording.

## Owned files

- cross-backend structure snapshot harness
- typed block/message permalink routes
- deterministic demo-tour recording assets
- `docs/proof/semantic-renderer/` final proof packet
- Bead closure evidence

## Acceptance criteria

- CLI and web emit the same ordered semantic transcript structure for every
  golden fixture; presentation markup may differ.
- Message and block anchors resolve after refresh and copied links select the
  exact evidence object.
- A browser recording uses only the private-data-free corpus and includes one
  shell failure, edit, linked Task, lineage boundary, unknown fallback, and
  attachment.
- The recording packet contains commands, environment identity, structured
  receipt, and explicit non-claims.
- A cold reviewer can regenerate the packet without an existing private
  archive or daemon state.
- The final Bead closure links tests and artifacts for both web and CLI ACs.

## Verification

```bash
polylogue demo tour --out-dir /tmp/polylogue-semantic-tour --force
PYTHONPATH=. python -m devtools.render_semantic_renderer_proof --check \
  --archive-root /tmp/polylogue-semantic-tour/archive
pytest -q tests/unit/rendering tests/unit/cli tests/unit/daemon
```
EOF

python - <<'PY'
from pathlib import Path
import json
root=Path('docs/proof/semantic-renderer/follow-ons')
packets=[]
for order, name, title, depends in [
 (1,'01-web-reader-wiring.md','Web reader wiring over semantic-transcript.v1',[]),
 (2,'02-task-attachment-links.md','Task/subagent and attachment links',['01-web-reader-wiring']),
 (3,'03-read-search-web-mcp-cards.md','Read, search, web, and MCP card families',['01-web-reader-wiring']),
 (4,'04-prose-thinking-layout-profiles.md','Prose, thinking, and layout profiles',['01-web-reader-wiring']),
 (5,'05-pagination-streaming.md','Pagination and streaming interaction',['01-web-reader-wiring']),
 (6,'06-parity-permalinks-visual-proof.md','Parity gate, permalinks, and visual proof',['02-task-attachment-links','03-read-search-web-mcp-cards','04-prose-thinking-layout-profiles','05-pagination-streaming']),
]:
 packets.append({'order':order,'packet_id':name[:-3],'title':title,'path':name,'depends_on':depends,'delivery_unit':'one_pull_request'})
(root/'index.json').write_text(json.dumps({'schema_version':'semantic-renderer-follow-ons.v1','bead':'polylogue-ap7','packets':packets},indent=2)+'\n')
PY

python - <<'PY'
from pathlib import Path
p=Path('docs/proof-artifacts.md')
text=p.read_text()
needle='| Polylogue reads tool outcomes from structure, not assistant prose. | Fact |'
row='| Polylogue renders shell, file-edit, lineage, and unknown-tool evidence as provider-neutral semantic cards in the CLI messages view. | Fact | [Semantic renderer proof packet](proof/semantic-renderer/README.md) | The public `read --view messages` path emits `semantic-card.v1` cards with exact source coordinates, structural outcomes, bounded previews, a repository-grounded registry, hostile golden fixtures, and deterministic before/after receipts. | Web visual wiring, linked Task cards, the remaining tool families, and cross-page pairing are bounded follow-ons; the full `polylogue-ap7` epic is not claimed complete. |\n'
if row not in text:
 idx=text.index(needle)
 text=text[:idx]+row+text[idx:]
p.write_text(text)
PY


---

## Turn 122 (role=assistant) — heredoc writing 5 follow-on-agent kit files: 00-START-HERE.md, 01-CAMPAIGN-REPORT.md, 02-VALIDATION-REPORT.md, 03-INTEGRATION-AND-REVIEW.md, 04-ARCHITECTURE-AND-CONTRACT.md

bash -lc set -euo pipefail
kit=/mnt/data/polylogue-ap7-semantic-renderer-kit
repo=/mnt/data/semantic-renderer-work/git
mkdir -p "$kit/follow-on-agent-prompts"
cat > "$kit/00-START-HERE.md" <<'EOF'
# Polylogue semantic transcript renderer — implementation and proof kit

This kit implements and proves the launch slice of Bead `polylogue-ap7` against
base commit `f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51`.

## Start here

1. Read [01-CAMPAIGN-REPORT.md](01-CAMPAIGN-REPORT.md) for what changed and why.
2. Open [semantic-renderer-proof.html](semantic-renderer-proof.html) for the
   self-contained evidence explorer.
3. Review the three commits listed in [review/commit-log.txt](review/commit-log.txt).
4. Apply [patches/0001…0003](patches/) with
   [`scripts/apply-series.sh`](scripts/apply-series.sh).
5. Run [`scripts/verify-applied.sh`](scripts/verify-applied.sh).
6. Read [proof/NON-CLAIMS.md](proof/NON-CLAIMS.md) before reusing public wording.
7. Use [05-FRONTIER-AGENT-COMPLETION-RUNBOOK.md](05-FRONTIER-AGENT-COMPLETION-RUNBOOK.md)
   to finish web parity and the rest of the epic.

## What is actually implemented

- a pure, provider-neutral `semantic-card.v1` model;
- a repository-grounded exact tool registry plus exhaustive persisted semantic-family policy;
- shell, file-edit/write, lineage, attachment, task, and raw-fallback card shapes;
- structural success/failure/unknown semantics with `NULL` preserved as unknown;
- exact message/block/result coordinates;
- bounded large-output previews with disclosed omission counts;
- invalid UTF-8 replacement accounting;
- existing CLI `read --view messages` integration for shell, edit, lineage, and fallback cards;
- storage hydration repairs needed to preserve provider family and structural outcomes;
- 15 hostile golden cases spanning Claude Code, Codex, Gemini CLI, ChatGPT, and Hermes;
- deterministic before/after receipts from the exact base and patched public CLI;
- a six-packet plan for the unimplemented web and remaining epic scope.

## Headline validation

- 199 launch-slice and existing-path tests passed;
- 87.99% focused coverage over the four new semantic-renderer modules;
- Ruff format and lint passed for all 19 changed Python files;
- generated registry, fixture, and proof checks passed;
- a deliberately corrupted expected card failed with exit code 1;
- all 30 deterministic demo constructs verified on a fresh archive;
- all three patches applied warning-free to a clean exact-base worktree;
- the patch-applied worktree passed 59 focused checks and remained clean;
- MyPy reports only the same four unrelated failures present at the base commit.

The full original epic is intentionally still open: the web backend, linked
Task cards, remaining card families, layout profiles, pagination-aware pairing,
and final cross-backend parity/recording gate are follow-ons, not hidden claims.
EOF

cat > "$kit/01-CAMPAIGN-REPORT.md" <<'EOF'
# Campaign report: `polylogue-ap7` semantic transcript renderer

## Outcome

The implementation replaces flat human CLI transcript output with an
evidence-oriented semantic presentation while retaining the existing archive,
query, read-view, and destination paths. The work is split into three commits:

1. evidence-preserving hydration and a pure semantic-card contract;
2. integration into the existing CLI messages view;
3. generated proof receipts and bounded completion packets.

It does not create a parallel archive reader or let a renderer query storage.

## The two upstream defects that had to be repaired first

A cosmetic renderer would have been misleading because the existing adapters
lost two pieces of evidence before presentation.

First, session origin values such as `codex-session` were passed through a
legacy provider parser that reduced them to unknown. The repository now uses
`provider_from_origin` in both paginated and iterator message paths.

Second, the archive block query and hydrator dropped
`tool_result_is_error`, `tool_result_exit_code`, and stable block identity.
Those fields are now preserved through the public facade. A regression test
writes archive-tier rows and reads them through the normal API path.

These repairs are semantically prior to cards: without them, a renderer could
only guess provider family and outcome from names or prose.

## The pure card contract

`semantic-card.v1` has six kinds:

- `shell`;
- `file_edit`;
- `task`;
- `lineage`;
- `attachment`;
- `fallback`.

A card retains exact source coordinates, optional structural outcome, compact
fields, bounded previews, raw evidence, and caveats. Cards are assembled into
an ordered `semantic-transcript.v1` containing prose and card entries.

The builder accepts hydrated messages and an optional already-read lineage
descriptor. An AST-enforced test rejects imports from `polylogue.storage`,
`polylogue.api`, or `polylogue.daemon` in the four semantic renderer modules.

## Outcome honesty

Outcome is closed over three values: `succeeded`, `failed`, and `unknown`.
Only stored `is_error` and exit-code fields participate. Text containing
`ERROR`, `success`, or `1 passed` is evidence text, never a badge signal.

Conflicting structural fields fail conservatively and carry a caveat. A paired
result with both fields absent remains unknown. A missing result remains
unknown. The renderer never paints missing evidence green.

## Registry and provider coverage

Classification precedence is:

1. persisted semantic type;
2. exact provider/tool alias grounded in a parser fixture, parser record type,
   or existing classifier contract;
3. raw fallback.

Every provider namespace is explicitly open. The generated table contains 28
exact aliases and an exhaustive policy row for all 11 persisted
`SemanticBlockType` values. It covers Claude Code, Codex, Gemini CLI, ChatGPT,
and Hermes without pretending their tool namespaces are finite.

An unlisted tool whose input looks exactly like a shell command remains
fallback. This is tested.

## Bounded evidence rather than hidden truncation

The default preview retains the first 48 and last 16 lines, then applies a
16,000-character ceiling if needed. It records original line count, exact
omitted lines or characters, selection strategy, and UTF-8 replacement count.

The hostile 10,000-line fixture produces a 64-line preview and reports exactly
9,936 omitted lines. A non-UTF-8 Hermes result remains visible with replacement
accounting rather than being dropped.

## Existing CLI path

The human branch of `run_messages` still calls the public facade's paginated
message read. It reads session and topology at that existing orchestration
boundary, passes fully hydrated values to the pure builder, and emits Markdown
through `click.echo`, preserving existing stdout/file/clipboard capture.

JSON and NDJSON machine contracts are unchanged.

The direct streaming path now preserves semantic and outcome fields but does
not yet perform cross-page card pairing. That incompleteness is explicit in the
pagination packet rather than guessed around.

## Construct-valid proof corpus

Fifteen hand-authored input cases freeze both a compact independent oracle and
complete card JSON. Together they cover every card kind and requested provider
family, including:

- an explicit success whose output says `ERROR`;
- an explicit file edit;
- interleaved subagents;
- 10,000-line failed shell output;
- a missing patch result;
- conflicting structural outcome fields;
- success-like prose with a `NULL` outcome;
- malformed/truncated raw input;
- invalid UTF-8;
- unknown ChatGPT web tooling;
- lineage uncertainty;
- attachment evidence;
- an orphan result.

The same fixture checker is used in normal generation and in the anti-vacuity
control. Changing one expected kind from `file_edit` to `shell` causes a
non-zero exit and identifies the exact mismatch.

## Baseline and after receipts

Three baseline files were reproduced byte-for-byte by running the public CLI
from the exact unpatched base commit against the seeded demo archive. Three
after files are regenerated by the patched public CLI and hashed in the proof
manifest.

The comparison demonstrates structural shell failure, honest unknown outcome,
file-edit behavior, lineage context, and raw fallback. It does not claim
private-archive scale or browser parity.

## Why the epic remains open

The original Bead also requires web cards, linked Task/subagent cards, remaining
tool families, richer prose/thinking treatment, profiles, permalinks,
pagination-aware pairing, backend structural parity, and a final recording.
Those concerns are separated into six one-PR packets under `follow-ons/`.

Keeping the parent open is part of the claims discipline. The launch slice is
real and useful; the full epic is not complete.
EOF

cat > "$kit/02-VALIDATION-REPORT.md" <<'EOF'
# Validation report

## Revisions

- Base: `f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51`
- Head: recorded in `review/head.txt` and `validation-manifest.json`
- Series: three commits in `review/commit-log.txt`

## Static checks

All 19 changed Python files pass `ruff format --check` and `ruff check`.
`git diff --check` passes across the complete base-to-head patch.

The project MyPy configuration reports four errors, all in unrelated files:

- `polylogue/cli/shell_words.py:7`;
- `polylogue/cli/machine_main.py:125`;
- `polylogue/insights/registry.py:62`;
- `polylogue/cli/click_option_groups.py:47`.

Running MyPy against the base commit produces the same four failures. No error
is reported in a changed file.

## Executed tests

The final launch-slice command passed 199 tests covering:

- the existing shared renderer;
- all semantic-card golden and behavior tests;
- proof generation and stale-artifact rejection;
- CLI human, JSON, NDJSON, and destination behavior;
- streaming field preservation;
- core message/block models;
- public-facade provider/outcome hydration;
- archive-tier materialization.

The same test selection, with coverage focused on the four new semantic modules,
reported 87.99% total coverage and passed the repository's 82% gate.

A clean patch-applied worktree separately passed 59 focused tests.

## Generated-contract checks

The following passed in both the implementation tree and clean patch-applied
tree:

```text
python -m devtools.render_semantic_card_registry --check
python -m devtools.render_semantic_card_fixtures --check
python -m devtools.render_semantic_renderer_proof --check
```

With a fresh seeded archive, the full proof check validated 35 artifacts,
including live CLI receipts and the proof manifest.

## Fresh demo proof

The archive was deleted and reseeded. Verification reported:

- 11 sessions;
- 43 messages;
- five origins;
- 30 of 30 declared constructs present;
- five failed tool results;
- attachment acquisition;
- lineage, subagents, compaction, context snapshots, observed events, and
  synthetic embeddings;
- zero reported problems;
- zero absolute-path leaks.

## Baseline authenticity

The unpatched base worktree ran the same public CLI against the same seeded
archive for three sessions. Each output was byte-identical to the committed
`proof/before/*.md` receipt.

## Anti-vacuity

A copied fixture corpus was corrupted so the Claude Edit case expected a shell
card. The normal checker exited 1 with:

```text
kinds expected ['shell'], got ['file_edit']
```

The suite also contains a test that corrupts a frozen generated artifact and
requires the proof checker to reject it.

## Patch replay

The three `git format-patch` files were applied with `git am` to a detached
worktree at the exact base. Application produced no warnings. Generated checks
and 59 focused tests passed, `git diff --check` passed, and the worktree was
clean afterward.

## Performance smoke

The environment-specific in-process receipt under `logs/performance-smoke.json`
renders the hostile 10,000-line case 50 times. It is included to catch gross
algorithmic regressions, not as a public latency or scale claim.

## Not executed

The entire repository test matrix was not run. The web backend is not wired in
this slice, so browser visual, accessibility, and cross-backend parity tests are
necessarily follow-on work. See `proof/NON-CLAIMS.md`.
EOF

cat > "$kit/03-INTEGRATION-AND-REVIEW.md" <<'EOF'
# Integration and review guide

## Apply the series

```bash
./scripts/apply-series.sh /path/to/polylogue
```

The script requires the exact base by default, refuses a dirty tree, and uses
`git am` so the three review boundaries are retained.

A binary-safe combined patch and a Git bundle are also supplied. The format
patch series is the reviewed path.

## Verify after application

```bash
./scripts/verify-applied.sh /path/to/polylogue
```

To include live demo receipts:

```bash
./scripts/verify-applied.sh /path/to/polylogue /path/to/seeded-demo-archive
```

## Review sequence

### Commit 1 — pure contract and evidence hydration

Review source-field preservation before presentation. Confirm that missing
outcome fields survive as `None`, provider origin is not collapsed to unknown,
and card construction has no storage access. Audit the exact registry evidence
paths and hostile fixtures.

### Commit 2 — existing CLI integration

Confirm that JSON/NDJSON are unchanged and the human branch still uses the
public facade, existing topology API, and destination adapter. Review the
before/after receipts only after the pure contract is accepted.

### Commit 3 — proof and completion packets

Review claims, reproduction commands, baseline identity, generated manifests,
and the six follow-ons. Reject any wording that implies web parity or epic
closure.

## Conflict policy

The storage/hydration changes are correctness prerequisites and should be
preserved even if presentation conflicts require reworking commit 2. The pure
semantic modules are intentionally isolated. If the daemon has moved, adapt
packet 1 to the new session-envelope seam rather than adding a second endpoint.

## Suggested Beads update

Leave `polylogue-ap7` open. Add a note linking the launch-slice commits and proof
manifest. Create six child tasks from `follow-ons/proposed-child-beads.json`.
Close the parent only after packet 6 proves both CLI and web acceptance criteria.
EOF

cat > "$kit/04-ARCHITECTURE-AND-CONTRACT.md" <<'EOF'
# Architecture and contract summary

```text
SQLite archive rows
  └─ message/block hydration
       ├─ stable block ID
       ├─ provider family from Origin
       ├─ persisted semantic type
       ├─ is_error / exit_code (nullable)
       └─ raw input and result evidence
             │
             ▼
    pure build_semantic_transcript(...)
       ├─ exact tool-ID FIFO pairing
       ├─ persisted semantics → exact alias → fallback
       ├─ structural outcome only
       ├─ bounded disclosed preview
       └─ exact source coordinates
             │
             ▼
      semantic-transcript.v1
       ├─ CLI Markdown backend — implemented
       └─ web card backend — bounded follow-on
```

The intermediate representation is not a new archive model. It is a pure,
versioned presentation contract over hydrated evidence.

The source of truth remains the archive. Previews disclose loss. Unknown tools
show raw evidence. Missing results and `NULL` outcome fields remain unknown.
Classification never reads assistant prose.

Full details are in:

- `proof/CARD-CONTRACT.md`;
- `proof/INTEGRATION-POINTS.md`;
- `mapping/semantic-card-tool-map.md`;
- `source-contracts/semantic-card-v1.schema.json`.
EOF


---

