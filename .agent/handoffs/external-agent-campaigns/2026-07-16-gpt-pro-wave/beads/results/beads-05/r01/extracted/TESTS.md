# TESTS — beads-05 semantic transcript rendering

## Test strategy

The tests combine three independent forms of evidence:

1. Frozen truth: 20 committed fixtures contain both expected `semantic-card.v1` cards and the complete ordered `semantic-transcript.v1` entry sequence.
2. Real production routes: canonical archive writer/query/hydrator/facade reads, archive-envelope reads, daemon HTTP detail, CLI Markdown, and the served web shell are exercised directly.
3. Anti-vacuity mutations: each new behavioral test names a representative production mutation/removal that would make it fail.

The fixture generator does not derive expected output during verification. It renders current production output and compares it to frozen `expected_cards` and `expected_entries` already stored in each case.

## Test-to-production dependency matrix

| Test or corpus | Production dependency | Representative mutation/removal that fails it |
|---|---|---|
| `test_semantic_transcript_golden_entries` over 20 cases | `build_semantic_transcript`, card registry, model serialization | Drop content/absence/attachment entries, change ordering, pairing, source fields, or family classification. |
| `test_typed_empty_thinking_compacts_only_contiguous_runs` | Typed absence compaction | Emit one marker per record, merge across intervening content, or infer absence from prose. |
| `test_nonempty_thinking_and_code_language_remain_typed_and_foldable` | Typed block projection and Markdown leaf | Flatten blocks to text, drop code language, source ownership, or thinking fold. |
| `test_fifo_pairing_preserves_invocation_and_result_ownership` | FIFO `tool_id` pairing and separate source projection | Pair by adjacency/last-result, or collapse result metadata into invocation metadata. |
| `test_placement_keeps_mixed_result_row_that_owns_visible_content` | Message placement/suppression adapter | Suppress every paired result row, including rows with independent content. |
| `test_page_bounded_scope_is_machine_readable_and_visible_in_markdown` | Transcript scope contract and Markdown warning | Treat a partial page as a complete session or remove scope disclosure. |
| Origin policy completeness | `Origin` → provider policy | Remove any current Origin mapping. |
| Exact MCP grammar and malformed canaries | Structural MCP classifier | Replace exact parsing with prefix/substring matching. |
| Hydrator source/duration/language test | SQLite row mapper and hydrator | Drop physical owner, duration, or block metadata language. |
| Prefix-sharing lineage test | Archive-envelope composition | Re-identify inherited parent messages as child-owned. |
| Composed attachment facade test | Full/paginated/bulk repository reads | Query attachments only by requested child session. |
| Archive daemon detail test | Archive-to-domain conversion and shared placement | Bypass semantic placement or drop archive source ownership. |
| Web JSON contract | Normal daemon detail and `SemanticCard.to_document()` | Add a web-only classifier, omit transcript envelope, or stop result suppression. |
| Web DOM/source contract | Typed `semantic_entries` dispatcher | Remove typed dispatch or revert to raw-block heuristics only. |
| ChatGPT recipient JSON/non-JSON canaries | ChatGPT parser recipient handling | Parse JSON tool payload as text, or overclassify ordinary recipient prose. |

## Commands and results

### Environment

The first broad collection attempt under the default environment stopped because the test extras were not installed (`hypothesis` was missing). No production assertion from that attempt was counted. The repository’s declared dev/test dependency group was then installed with `uv sync --extra dev`, and all reported results below use that environment.

### Generated truth

```bash
.venv/bin/python devtools/render_semantic_card_fixtures.py --check
.venv/bin/python devtools/render_semantic_card_registry.py --check
```

Result:

```text
semantic-card fixtures: verified 20 case(s)
semantic-card registry: verified 2 surface(s)
```

The same commands also pass from the clean apply-check worktree after applying `PATCH.diff`.

### Consolidated production regression set

```bash
.venv/bin/pytest -q   tests/unit/rendering/test_semantic_cards.py   tests/unit/rendering/test_semantic_transcript.py   tests/visual/test_reader_semantic_cards.py   tests/unit/core/test_models.py   tests/unit/storage/test_message_query_reads.py   tests/unit/storage/test_lineage_normalization.py   tests/unit/storage/test_archive_tiers_write.py   tests/unit/api/test_facade_contracts.py::test_composed_message_reads_retain_ancestor_owned_attachments   tests/unit/daemon/test_web_reader.py::TestReaderSessionState   tests/unit/daemon/test_web_reader.py::TestReaderWorkspaceRoutes   tests/unit/cli/test_messages.py   tests/unit/cli/test_streaming_markdown_read_view.py   tests/unit/sources/test_parsers_chatgpt.py::test_chatgpt_recipient_addressed_json_payload_parses_as_tool_use   tests/unit/sources/test_parsers_chatgpt.py::test_chatgpt_recipient_addressed_non_json_text_stays_text
```

Result: `317 passed in 27.66s`.

This count includes overlapping files later rerun after small typing/docstring changes. It must not be added to the rerun counts as a unique-test total.

### Renderer rerun after final behavior/type changes

```bash
.venv/bin/pytest -q   tests/unit/rendering/test_semantic_cards.py   tests/unit/rendering/test_semantic_transcript.py
```

Result: `84 passed in 1.11s` in the implementation worktree.

The same 84 tests pass in `1.15s` from a detached clean worktree after applying the final `PATCH.diff`.

### Final real-route/docstring check

```bash
.venv/bin/pytest -q   tests/visual/test_reader_semantic_cards.py   tests/unit/daemon/test_web_reader.py::TestReaderSessionState::test_archive_file_set_session_detail_and_messages_from_archive_tiers
```

Result: `3 passed in 2.89s`.

### Targeted checks run during implementation

- Inherited attachment through full, paginated, and bulk facade reads: `1 passed`.
- Archive-backed daemon semantic transcript route: `1 passed` after correcting the assertion to account for the intentionally first lineage card.
- ChatGPT JSON tool-use and non-JSON counterexample canaries: `2 passed`.
- CLI messages and streaming Markdown: `14 passed`.
- Canonical message hydration/query route: `7 passed`.
- Archive-tier writer and lineage normalization set: `89 passed`.
- Daemon session-state/workspace focused set: `20 passed` in an earlier focused invocation.

These are diagnostic reruns and overlap the 317-test command.

### Static and generated quality gates

```bash
.venv/bin/ruff check <all changed Python paths>
```

Result: `All checks passed!`

```bash
.venv/bin/mypy   devtools/render_semantic_card_fixtures.py   devtools/render_semantic_card_registry.py   polylogue/api/archive.py   polylogue/archive/message/models.py   polylogue/cli/messages.py   polylogue/daemon/http.py   polylogue/daemon/web_shell_reader.py   polylogue/daemon/web_shell_semantic_cards.py   polylogue/rendering/semantic_card_models.py   polylogue/rendering/semantic_card_placement.py   polylogue/rendering/semantic_card_registry.py   polylogue/rendering/semantic_cards.py   polylogue/rendering/semantic_markdown.py   polylogue/storage/hydrators.py   polylogue/storage/repository/archive/sessions.py   polylogue/storage/sqlite/archive_tiers/write.py   polylogue/storage/sqlite/queries/mappers_archive.py   polylogue/storage/sqlite/queries/message_query_reads.py
```

Result: `Success: no issues found in 18 source files`.

The first Mypy invocation identified unsafe generic `int(object)` coercions and an untyped scope literal in `semantic_cards.py`. Those were repaired with explicit supported scalar types and a `Literal` annotation; the passing result above is after repair.

```bash
node --check /tmp/beads05-reader.js
```

`/tmp/beads05-reader.js` is the concatenation of `SEMANTIC_CARD_JS` and `READER_JS`. Result: exit status 0.

```bash
git diff --check
```

Result: exit status 0 in both implementation and apply-check worktrees.

### Patch application certification

```bash
git worktree add --detach /mnt/data/beads05_applycheck f654480cadb7cc4c194704e24dfd483199547b35
git -C /mnt/data/beads05_applycheck apply --check PATCH.diff
git -C /mnt/data/beads05_applycheck apply PATCH.diff
git -C /mnt/data/beads05_applycheck diff --check
```

Result: all commands passed; the applied worktree reports the same 49 changed paths.

## Failed or incomplete verification

A full invocation of `tests/unit/daemon/test_web_reader.py` was attempted with a 300-second execution limit. It progressed through roughly 60 tests without reporting a failure, then did not complete within that limit in a later query-completion area outside the semantic transcript routes. The relevant `TestReaderSessionState` and `TestReaderWorkspaceRoutes` classes pass in the consolidated command.

The complete repository suite was not run. Playwright/browser visual checks, a sanitized real ChatGPT export, the operator’s live archive/daemon, and deployment checks remain unverified.
