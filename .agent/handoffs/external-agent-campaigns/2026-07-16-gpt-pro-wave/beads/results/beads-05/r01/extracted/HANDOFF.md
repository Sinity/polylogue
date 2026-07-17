# HANDOFF — beads-05 semantic transcript rendering

## Result

This package implements the semantic transcript slice requested by `beads-05` against the attached Polylogue snapshot. The patch creates one provider-neutral, ordered transcript model containing typed content, compact typed absence evidence, and `semantic-card.v1` cards. CLI Markdown and daemon/web session reads consume that same model, with presentation-specific rendering only at the leaf.

The implementation deliberately extends the semantic-card work already present at the snapshot rather than creating another renderer. It closes the main gaps in `polylogue-ap7.1` and `polylogue-395j`: normalized file-read/search/web/MCP coverage, exhaustive Origin policy, ordered content/card/absence output, honest empty-thinking compaction, inherited physical ownership, attachment propagation through composed reads, and equivalent semantic placement for canonical database and archive-envelope routes.

## Snapshot identity

- Project: `polylogue`
- Snapshot generated: `2026-07-17T043202Z`
- Snapshot branch: `master`
- Snapshot commit: `f654480cadb7cc4c194704e24dfd483199547b35`
- Commit timestamp: `2026-07-17T03:45:52+02:00`
- Commit subject: `chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm`
- Snapshot manifest dirty flag: `true`
- Recoverable tracked branch delta: none. `polylogue-branch-delta.patch`, `polylogue-branch-delta-files.txt`, and `polylogue-branch-delta-log.txt` are all zero bytes, and the branch-delta report names the same commit as merge base. The patch therefore targets the named commit exactly rather than inventing an unavailable dirty delta.

The repository was reconstructed from `polylogue-all-refs.bundle` and `polylogue-working-tree.tar.gz` in the supplied archive. No operator worktree, live daemon, secrets, browser session, or NixOS deployment was accessed.

## Evidence inspected

Repository instructions and test policy:

- `CLAUDE.md`
- `CONTRIBUTING.md`
- `TESTING.md`
- `pyproject.toml`

Primary and adjacent Beads:

- `polylogue-ap7.1` — complete semantic-card family coverage and bounded lineage parity
- `polylogue-395j` — compact empty-thinking treatment in web and CLI
- `polylogue-e2yk` — ChatGPT recipient-addressed JSON tool-call canary
- `polylogue-4p1` — longer-term Query × Projection × Render authority
- `polylogue-37km`, `polylogue-1ilk`, and `polylogue-1lm` — later canonical visual/browser/projection work

Relevant history:

- `0f5059068 feat(rendering): add semantic transcript evidence cards (#2700)`
- `0c251b600 feat(rendering): wire semantic transcript cards into the web reader (#2736)`

Production paths followed:

- semantic models, registry, pure renderer, Markdown renderer, and web placement adapter
- CLI `messages` read route
- daemon normal session-detail and archive-envelope session-detail routes
- canonical SQLite message query, row mapper, hydrator, and composed repository reads
- archive-tier envelope composition and archive-to-domain conversion
- attachment storage/hydration and lineage topology projection
- ChatGPT parser recipient-addressed tool-use path
- generated card registry, public JSON schema, frozen fixtures, visual shell tests, and public facade tests

## Mechanism

### One ordered semantic stream

`SemanticTranscript` now serializes as `semantic-transcript.v1` and contains ordered `entry_type` values:

- `content`: typed `text`, `thinking`, or `code`, retaining message/block identity, physical source session, authoredness, timing, and code language;
- `absence`: a compact `thinking_unavailable` record with every contributing message/block coordinate;
- `card`: the existing `semantic-card.v1` document for tools, attachments, or lineage.

`semantic_cards` remains as a compatibility projection for older web payload consumers, but new readers prefer `semantic_entries` and do not reclassify raw blocks when those entries are present.

### Structural semantics only

Tool classification uses, in order:

1. persisted normalized `semantic_type`;
2. the exact structural `mcp__server__method` grammar;
3. an explicit provider-family/tool alias registry grounded in parser fixtures or classifier contracts;
4. a generic fallback card retaining raw input/result evidence.

No provider-name prose heuristic, command-text heuristic, or result-text regex establishes tool family or outcome. Outcome comes only from `tool_result_is_error` and `tool_result_exit_code`.

### Tool pairing and source ownership

Tool uses and results pair exactly by `tool_id`, FIFO for duplicate IDs. Invocation and result preserve separate physical source-session IDs, messages, blocks, roles, message types, authoredness, timestamps, and durations. Orphan results remain explicit fallback cards.

### Empty thinking

Typed `THINKING`/`REASONING` records with empty or whitespace-only content become one compact absence marker per contiguous run. Every source coordinate remains in the marker. Ordinary prose such as “Thinking unavailable” has no semantic effect. Non-empty thinking remains foldable typed content.

### Lineage and attachments

Canonical database hydration and archive-envelope conversion retain the physical `source_session_id` of each composed message. Full, paginated, and bulk canonical reads fetch attachments only from the physical owner sessions represented by the composed message rows, preventing inherited-prefix attachments from disappearing without hydrating an entire family.

Archive envelopes expose lineage completeness and truncation reason. Normal topology reads do not currently carry composition completeness; the lineage card says that completeness was not supplied rather than asserting completeness.

### Leaf rendering

- CLI Markdown renders the shared transcript directly and discloses page-bounded pairing/absence compaction.
- Daemon session detail emits the full transcript and per-message placed entries.
- Web JavaScript dispatches by `entry_type`, rendering code and non-empty thinking as folds, compact absence markers, and cards from the shared document. Raw-block/card-only logic remains solely as backward compatibility for older payloads.

## Decisions

1. Preserve absence as evidence rather than silently omitting it.
2. Keep `semantic-card.v1` stable for cards and add `semantic-transcript.v1` as the ordered wrapper needed by `polylogue-395j`.
3. Retain historical `TranscriptProse` as a source-compatible Python class name while serializing it as typed `content`.
4. Keep the provider namespace open. Exhaustive Origin policy means every current Origin has an explicit provider/fallback policy, not that unknown future tools are guessed.
5. Treat pagination as a semantic boundary. Pairing and absence compaction are page-bounded and disclosed instead of looking beyond the requested page.
6. Suppress a result-only message row only when all of its evidence is represented on the invocation card. A result row that also owns content or an attachment remains visible.
7. Fetch ancestor attachments from represented physical owners rather than traversing/hydrating complete families.

## Changed files

Patch summary:

```text
 devtools/render_semantic_card_fixtures.py          |  49 ++
 devtools/render_semantic_card_registry.py          |  51 +-
 docs/generated/semantic-card-tool-map.json         | Bin 12592 -> 16991 bytes
 docs/generated/semantic-card-tool-map.md           | Bin 5902 -> 7048 bytes
 docs/schemas/semantic-card-v1.schema.json          | 231 +++++--
 polylogue/api/archive.py                           |   2 +
 polylogue/archive/message/models.py                |   2 +
 polylogue/cli/messages.py                          |   1 +
 polylogue/daemon/http.py                           |  52 +-
 polylogue/daemon/web_shell_reader.py               |  17 +-
 polylogue/daemon/web_shell_semantic_cards.py       |  81 ++-
 polylogue/rendering/semantic_card_models.py        | 154 ++++-
 polylogue/rendering/semantic_card_placement.py     | 100 +--
 polylogue/rendering/semantic_card_registry.py      | 197 +++---
 polylogue/rendering/semantic_cards.py              | 750 +++++++++++++++++----
 polylogue/rendering/semantic_markdown.py           |  78 ++-
 polylogue/storage/hydrators.py                     |   3 +
 polylogue/storage/repository/archive/sessions.py   |  69 +-
 polylogue/storage/sqlite/archive_tiers/write.py    |   2 +
 .../storage/sqlite/queries/mappers_archive.py      |   1 +
 .../storage/sqlite/queries/message_query_reads.py  |   3 +-
 .../data/semantic_cards/cases/attachment-card.json |  46 +-
 .../semantic_cards/cases/chatgpt-canmore-edit.json |  95 ++-
 .../semantic_cards/cases/chatgpt-web-fallback.json | 127 ----
 tests/data/semantic_cards/cases/chatgpt-web.json   | 195 ++++++
 .../cases/claude-bash-explicit-success.json        |  84 ++-
 .../semantic_cards/cases/claude-edit-diff.json     |  95 ++-
 .../semantic_cards/cases/claude-file-read.json     | 214 ++++++
 .../cases/claude-interleaved-subagents.json        | 178 ++++-
 tests/data/semantic_cards/cases/claude-search.json | 199 ++++++
 .../cases/codex-apply-patch-missing-result.json    |  62 ++
 .../cases/codex-conflicting-outcome.json           |  86 ++-
 .../codex-exec-failure-ten-thousand-lines.json     |  86 ++-
 tests/data/semantic_cards/cases/codex-mcp.json     | 216 ++++++
 .../cases/gemini-shell-null-outcome.json           |  84 ++-
 .../semantic_cards/cases/gemini-write-file.json    |  94 ++-
 .../cases/hermes-shell-non-utf8.json               |  86 ++-
 .../cases/hermes-truncated-unknown-input.json      |  56 ++
 .../cases/lineage-unknown-boundary.json            |  56 +-
 .../semantic_cards/cases/message-attachment.json   | 139 ++++
 .../cases/mixed-content-thinking-code.json         | 279 ++++++++
 tests/data/semantic_cards/cases/orphan-result.json |  61 +-
 tests/unit/api/test_facade_contracts.py            |  94 ++-
 tests/unit/core/test_models.py                     |  32 +
 tests/unit/daemon/test_web_reader.py               |  13 +
 tests/unit/rendering/test_semantic_cards.py        | 101 ++-
 tests/unit/rendering/test_semantic_transcript.py   | 243 +++++++
 tests/unit/storage/test_lineage_normalization.py   |   6 +
 tests/visual/test_reader_semantic_cards.py         |  35 +-
 49 files changed, 4384 insertions(+), 521 deletions(-)
```

Path inventory (`M` modified, `D` deleted, `A`/new paths appear as additions when applied):

- `M	devtools/render_semantic_card_fixtures.py`
- `M	devtools/render_semantic_card_registry.py`
- `M	docs/generated/semantic-card-tool-map.json`
- `M	docs/generated/semantic-card-tool-map.md`
- `M	docs/schemas/semantic-card-v1.schema.json`
- `M	polylogue/api/archive.py`
- `M	polylogue/archive/message/models.py`
- `M	polylogue/cli/messages.py`
- `M	polylogue/daemon/http.py`
- `M	polylogue/daemon/web_shell_reader.py`
- `M	polylogue/daemon/web_shell_semantic_cards.py`
- `M	polylogue/rendering/semantic_card_models.py`
- `M	polylogue/rendering/semantic_card_placement.py`
- `M	polylogue/rendering/semantic_card_registry.py`
- `M	polylogue/rendering/semantic_cards.py`
- `M	polylogue/rendering/semantic_markdown.py`
- `M	polylogue/storage/hydrators.py`
- `M	polylogue/storage/repository/archive/sessions.py`
- `M	polylogue/storage/sqlite/archive_tiers/write.py`
- `M	polylogue/storage/sqlite/queries/mappers_archive.py`
- `M	polylogue/storage/sqlite/queries/message_query_reads.py`
- `M	tests/data/semantic_cards/cases/attachment-card.json`
- `M	tests/data/semantic_cards/cases/chatgpt-canmore-edit.json`
- `D	tests/data/semantic_cards/cases/chatgpt-web-fallback.json`
- `A	tests/data/semantic_cards/cases/chatgpt-web.json`
- `M	tests/data/semantic_cards/cases/claude-bash-explicit-success.json`
- `M	tests/data/semantic_cards/cases/claude-edit-diff.json`
- `A	tests/data/semantic_cards/cases/claude-file-read.json`
- `M	tests/data/semantic_cards/cases/claude-interleaved-subagents.json`
- `A	tests/data/semantic_cards/cases/claude-search.json`
- `M	tests/data/semantic_cards/cases/codex-apply-patch-missing-result.json`
- `M	tests/data/semantic_cards/cases/codex-conflicting-outcome.json`
- `M	tests/data/semantic_cards/cases/codex-exec-failure-ten-thousand-lines.json`
- `A	tests/data/semantic_cards/cases/codex-mcp.json`
- `M	tests/data/semantic_cards/cases/gemini-shell-null-outcome.json`
- `M	tests/data/semantic_cards/cases/gemini-write-file.json`
- `M	tests/data/semantic_cards/cases/hermes-shell-non-utf8.json`
- `M	tests/data/semantic_cards/cases/hermes-truncated-unknown-input.json`
- `M	tests/data/semantic_cards/cases/lineage-unknown-boundary.json`
- `A	tests/data/semantic_cards/cases/message-attachment.json`
- `A	tests/data/semantic_cards/cases/mixed-content-thinking-code.json`
- `M	tests/data/semantic_cards/cases/orphan-result.json`
- `M	tests/unit/api/test_facade_contracts.py`
- `M	tests/unit/core/test_models.py`
- `M	tests/unit/daemon/test_web_reader.py`
- `M	tests/unit/rendering/test_semantic_cards.py`
- `A	tests/unit/rendering/test_semantic_transcript.py`
- `M	tests/unit/storage/test_lineage_normalization.py`
- `M	tests/visual/test_reader_semantic_cards.py`

`chatgpt-web-fallback.json` is superseded by `chatgpt-web.json` because the landed ChatGPT parser canary and the completed registry now classify the exact recipient-addressed `web` invocation structurally as a web card rather than fallback.

## Acceptance matrix

| Requirement | Result | Evidence |
|---|---|---|
| Every normalized semantic family and executable Origin covered or explicitly fallback | Met for the current enums/registry | Generated Origin and semantic-type policies; completeness tests fail when an enum mapping is removed. |
| Shell, edit/write, file read/search, task/delegation, web, MCP, attachment, lineage, and unknown fixtures | Met in automated scope | 20 frozen cases validate complete ordered entries; card documents validate against the public schema. |
| CLI and web share structure, with no backend classifier fork | Met | Both consume `SemanticTranscript`/`SemanticCard.to_document()`; web route and DOM contract tests assert typed dispatch. Legacy raw heuristics are compatibility-only when typed entries are absent. |
| Database and archive readers emit bounded lineage/delegation semantics without family hydration | Mostly met | Both routes use the same placement renderer; archive route, lineage ownership, hydration, and composed attachment tests pass. No single byte-for-byte DB-versus-archive delegation-heavy parity fixture was added. |
| ChatGPT recipient-addressed JSON remains `TOOL_USE`, without raw JSON leak | Met for synthetic canary | Positive JSON and negative non-JSON parser canaries pass. The operator’s named live export was not available and remains unverified. |
| Empty typed thinking is compact in CLI and web; non-empty thinking remains foldable | Met in automated scope | Mixed frozen fixture, renderer tests, Markdown assertions, and typed web dispatch pass. Manual browser verification against a real ChatGPT export remains unverified. |

## Apply order

From a clean checkout at `f654480cadb7cc4c194704e24dfd483199547b35`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
python devtools/render_semantic_card_fixtures.py --check
python devtools/render_semantic_card_registry.py --check
pytest -q tests/unit/rendering/test_semantic_cards.py tests/unit/rendering/test_semantic_transcript.py
```

The patch contains generated registry documents and frozen expected fixture output, so there is no separate generation step required before application. The `--check` commands certify that committed generated output matches its source declarations.

## Verification performed

- Current patch applied with `git apply --check` and `git apply` to a detached clean worktree at `f654480cadb7cc4c194704e24dfd483199547b35`.
- `git diff --check` passed in both implementation and apply-check worktrees.
- Apply-check worktree regenerated and verified 20 frozen cases and two registry surfaces.
- Apply-check worktree renderer suite: 84 passed.
- Consolidated implementation regression command: 317 passed across renderer, transcript, visual, core hydration, message-query, lineage, archive-tier, public-facade attachment, daemon session/workspace, CLI, and ChatGPT canary tests.
- Post-change route/docstring checks: 3 passed.
- Ruff: all changed Python files passed.
- Mypy: no issues in 18 changed production/dev modules.
- Combined semantic-card and reader JavaScript: `node --check` passed.
- Patch contains 49 paths and is 268,567 bytes before ZIP compression.

Detailed commands, anti-vacuity mutations, and the non-aggregate interpretation of test counts are in `TESTS.md`.

## Risks and limitations

- The complete repository test suite was not run. The focused scope is broad, but unrelated integration failures remain possible.
- A prior attempt to run all of `tests/unit/daemon/test_web_reader.py` hit a 300-second execution limit after progressing without a reported failure. The relevant session and workspace classes pass independently.
- The default project environment initially lacked the declared test extras, including Hypothesis. Verification used the repository’s dev/test dependency group; this environment event is recorded in `TESTS.md`.
- No Playwright or real-browser interaction was run. DOM/source contracts and JavaScript syntax pass, but visual weight, focus, keyboard behavior, and responsive presentation are unverified.
- No live ChatGPT export, live daemon, operator archive, credentials, or deployment was accessed.
- Normal topology projection does not expose lineage-composition completeness. The renderer explicitly marks that state unavailable; adding it would require a topology contract extension.
- Pairing and compaction cannot cross a paginated response boundary. The transcript exposes `pairing_scope=page` and `absence_compaction_scope=page` so consumers cannot mistake a page for a complete session.
- Exact MCP grammar intentionally sends malformed or provider-variant lookalikes to fallback. Future normalized aliases must be added with evidence rather than inferred.

## Value of another iteration

A small repair pass could add roughly 5–10% confidence by running the full suite, adding one byte-for-byte database/archive delegation parity fixture, and performing Playwright plus a sanitized real-export check. Those are verification additions, not a redesign.

A substantial second pass would be a different scope: moving all read surfaces under the broader `polylogue-4p1` Query × Projection × Render algebra and adopting the canonical HTML/Playwright work in `polylogue-37km` and `polylogue-1ilk`. That could add materially more architectural and visual value, but it should not be folded into this patch as incidental cleanup.
