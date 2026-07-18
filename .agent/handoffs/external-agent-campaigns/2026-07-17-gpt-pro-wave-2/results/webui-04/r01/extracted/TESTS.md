# WebUI v2 transcript renderer — test design and execution record

## Test doctrine

The tests are designed to fail when the production dependency named by the mission is removed or replaced with a vacuous test double. The central rule is that the semantic registry and ordered transcript own meaning; daemon, SSR, terminal, and Preact only project the resulting document.

The strongest route test does not monkeypatch the card document/detail loader. It writes sanitized parsed messages through `ArchiveStore`, starts the real loopback daemon HTTP server, resolves a native-id alias, reads the split archive, builds the shared semantic document, requests exact evidence, and verifies JSON, HTML, SSR, auth, headers, and local assets.

Random-order plugins were disabled for deterministic final lanes with `-p no:randomly -p no:random-order`. An earlier combined invocation hit a `pytest-randomly` seed-range failure before tests ran; disabling that unrelated plugin produced repeatable green results.

## Rendering contract and parity

### Canonical registry parity

Test: `test_card_document_and_terminal_projection_match_canonical_registry`.

Production dependencies:

- `build_semantic_transcript(...)` as the sole classifier/structure owner;
- `semantic_card_placement_from_transcript(...)` for source-row ownership and result suppression;
- web document serialization;
- terminal `render_semantic_transcript_markdown(...)` family/outcome facts.

Representative fixtures:

- failed Codex shell result with 10,000-line output;
- fork/compaction lineage family;
- Hermes unknown tool with truncated/malformed input.

Assertions: canonical card count, order, family, and structural outcome exactly match web and terminal projections.

Anti-vacuity mutation: make the web serializer classify from `tool_name`, omit fallback cards, change FIFO pairing, drop the lineage session entry, or remove terminal family/outcome facts. At least one signature differs and the test fails.

### Every canonical fixture is schema-valid

Test: `test_every_canonical_fixture_serializes_to_the_strict_document_schema`.

Production dependencies: fixture materialization, canonical transcript, card-document serializer, checked-in Draft 2020-12 schema.

Anti-vacuity mutation: add an unknown property, emit an unbounded field/preview, omit a required source state, permit an incoherent present-result state without a result ref, or serialize raw unbounded prose. Schema validation fails.

### Exact identity and source-complete entries

Tests:

- `test_document_schema_preserves_exact_identities_and_complete_message_entries`;
- `test_document_row_ownership_and_suppression_match_shared_placement`.

Production dependencies: exact session/message/block identities, complete ordered entries for each page row, placement seam, pure paired-result suppression.

Anti-vacuity mutation: truncate an identity string, collapse entries into a summary, attach a card to the result row instead of the invocation row, or suppress a non-result row. Equality and ownership assertions fail.

### Bounded fallback evidence and explicit omissions

Tests:

- `test_document_is_bounded_and_unknown_tool_keeps_exact_raw_evidence`;
- `test_preview_marker_is_inside_declared_character_budget`;
- `test_document_bounds_provider_metadata_and_only_links_expandable_fields`.

Production dependencies: field/title/summary/source bounds, preview builder, raw fallback evidence, exact disclosure capability, typed omission counts.

Anti-vacuity mutation: serialize the original 40,000-character malformed input, clip without omission metadata, place the marker outside the limit, emit detail links for values that cannot be reconstructed, or drop unknown evidence. Bound, schema, and evidence assertions fail.

### Pagination and authoredness

Tests:

- `test_document_paginates_rows_clamps_limits_and_emits_session_entries_once`;
- `test_role_and_material_origin_remain_independent_in_document`;
- `test_provider_override_normalizes_transcript_and_document_rows_consistently`.

Production dependencies: server clamp, page metadata, session-entry placement, provider normalization, independent role/material authoredness.

Anti-vacuity mutation: emit lineage on every page, return more than 200 messages, conflate `role=user` with `human_authored`, or apply provider override only to one projection. Assertions fail.

### Exact-ref detail, no widening, and Unicode continuity

Test: `test_detail_is_exact_typed_chunked_and_unicode_code_point_coherent`.

Production dependencies: `EvidenceRef.parse`, exact session/message/block coordinate selection, detail-part enum, server clamp, Unicode code-point slicing, typed unavailable states.

Assertions include:

- exact structured tool input chunks;
- non-BMP character offset continuity;
- cross-session ref rejection;
- missing block coordinate;
- unsupported part as `unknown`;
- block ref requesting `message_text` as `missing` rather than widened parent-message text.

Anti-vacuity mutation: search by tool id, widen a block ref to the parent message, count UTF-16 code units, coerce missing to empty available text, or ignore the requested part. Assertions fail.

### Restored result-before-use ordering case

Existing tests in `tests/unit/rendering/test_semantic_cards.py`:

- `test_result_serialized_before_use_is_paired_once`;
- `test_pure_paired_result_row_is_suppressed`.

Production dependency: whole-document FIFO structural pairing by `tool_id`, independent of serialization order.

Anti-vacuity mutation: pair only forward in source order, duplicate a result across cards, or render the pure result row after pairing. The restored fixture makes the pre-existing tests fail.

## Daemon and SSR tests

### Real archive-backed HTTP route

Test: `test_archive_backed_document_detail_ssr_and_local_assets`.

Production path exercised:

1. `ParsedSession` and `ParsedMessage` with structural tool-use/result blocks;
2. `ArchiveStore.write_parsed(...)` into a temporary split archive;
3. `DaemonAPIHTTPServer` and `DaemonAPIHandler` on loopback;
4. native-id alias resolution to canonical `codex-session:<id>` identity;
5. archive envelope hydration and shared semantic document builder;
6. authenticated card-document JSON;
7. exact tool-input detail JSON;
8. authenticated `/v2/s` SSR;
9. unauthenticated data-free bootstrap;
10. committed local JavaScript/CSS serving.

Assertions include structural shell failure, `is_error=true`, exit code 4, exact result evidence ref, page metadata, auth denial, no-store/no-referrer/nosniff headers, script-safe `</script>` escaping, canonical identity adoption, and asset presence.

Anti-vacuity mutation: replace `ArchiveStore` with a hand-built document, pair from prose, widen the detail ref, embed transcript data before authorization, route SSR through the legacy classifier, fail alias resolution, or delete a local asset. A distinct assertion fails.

### Pure SSR projection

Tests in `test_webui_v2_ssr.py` cover:

- every closed family rendered from a serialized document;
- prominent failed shell command/error/exit state, file target, and task child link;
- role and `material_origin` as independent authoredness axes;
- bounded omissions and exact disclosure controls;
- data-free page without authority;
- canonical document identity and script-safe state;
- escaping of untrusted message/card values.

Production dependency: `render_semantic_card_document_html(...)` and `render_semantic_card_page(...)` accept the serialized contract and do not reclassify provider data.

Anti-vacuity mutation: add a raw-message classifier, omit a family component, label every `role=user` row human-authored, inject unescaped card values, include state in the bootstrap shell, or use the route alias instead of the canonical document identity. Tests fail.

### Route contract pins

`tests/unit/daemon/test_route_contracts.py` explicitly pins:

- `/v2/s/:session_id`;
- both local asset routes;
- card-document JSON/HTML route;
- card-detail JSON route;
- auth and mutation posture.

Anti-vacuity mutation: remove a route from the daemon dispatcher or contract catalog, change method/auth/mutation metadata, or silently rename the endpoint. Contract tests fail.

## Preact/browser-contract tests

`webui/tests/unit/transcript.test.tsx` contains 24 tests.

### Strict document validation

Coverage:

- empty page exactly at transcript end;
- cross-session exact refs rejected;
- present result state requires exact result ref;
- bounded metadata and explicit truncation records;
- exact identities and source-complete entry arrays retained;
- prose represented by one bounded preview rather than unbounded text.

Anti-vacuity mutation: replace validators with type assertions, ignore unknown properties/bounds, accept stale schema versions, or omit cross-session ref checks. Invalid fixtures are accepted and tests fail.

### Family and authoredness rendering

Coverage:

- all ten families;
- runtime protocol in provider `user` envelope labeled non-human;
- failed shell command, error flag, and exit code;
- task child links and fallback raw evidence;
- visible title/field/preview/source omission counts;
- runtime-context authoredness distinct from provider role.

Anti-vacuity mutation: map all cards to one generic component, hide fallback evidence, infer authoredness from role, or omit structural result facts. DOM assertions fail.

### Page accumulation

Coverage:

- immutable server page contract separated from accumulated state;
- canonical session, total, offset, continuation, duplicate id, and later-session-entry drift rejected;
- rapid repeated clicks issue one request;
- nonzero initial page rejected;
- credential bootstrap adopts canonical session identity returned for an alias.

Anti-vacuity mutation: concatenate messages into a mutated `CardDocument`, accept gaps/overlaps, fetch against the original alias forever, or remove the in-flight guard. Tests fail.

### Exact detail accumulation

Coverage:

- mismatched ref/part/offset rejected;
- inconsistent completion metadata rejected;
- typed `missing`/`unknown` labeled unavailable rather than complete;
- Unicode chunks appended at the server continuation offset;
- changing total across chunks rejected;
- in-flight and client-cap behavior.

Anti-vacuity mutation: append arbitrary responses, use JavaScript string length for offsets, trust `complete` without totals, or treat unavailable as empty success. Tests fail.

### Measured virtualization and SSR boot

Coverage:

- `ResizeObserver`-driven variable-height window mounts fewer rows than a 100-message transcript;
- embedded SSR document boots without refetching page zero;
- malformed, stale, oversized, nested-invalid, and incoherent embedded documents return null.

Anti-vacuity mutation: render every row, revert to a fixed estimate, always fetch page zero, or use unchecked embedded JSON. Tests fail.

## Commands and exact results

All commands were run from `/mnt/data/work-webui04/repo` unless a different working directory is shown.

### Final Python static gate

```text
mapfile -t pyfiles < <(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.py$')
ruff check "${pyfiles[@]}"
ruff format --check "${pyfiles[@]}"
python -m py_compile "${pyfiles[@]}"
git diff --cached --check
```

Result: PASS.

- Ruff: `All checks passed!`
- format: `11 files already formatted`
- byte compilation: exit 0
- diff whitespace check: exit 0

### Targeted strict Mypy with normal imports

```text
mypy \
  polylogue/rendering/semantic_card_models.py \
  polylogue/rendering/semantic_card_placement.py \
  polylogue/rendering/semantic_cards.py \
  polylogue/rendering/semantic_markdown.py \
  polylogue/daemon/route_contracts.py \
  polylogue/daemon/web_shell_semantic_cards.py \
  tests/unit/rendering/test_semantic_card_document.py \
  tests/unit/daemon/test_semantic_card_http.py \
  tests/unit/daemon/test_webui_v2_ssr.py \
  tests/unit/daemon/test_route_contracts.py
```

Result: PASS, `Success: no issues found in 10 source files`.

No `--follow-imports=skip` shortcut was used.

### Focused production vertical

```text
pytest -q -p no:randomly -p no:random-order \
  tests/unit/rendering/test_semantic_card_document.py \
  tests/unit/daemon/test_semantic_card_http.py \
  tests/unit/daemon/test_webui_v2_ssr.py \
  tests/unit/daemon/test_route_contracts.py \
  tests/unit/daemon/test_metrics_endpoint.py
```

Result: PASS, `179 passed in 9.05s` on the final explicit-exit run.

### Canonical semantic-card baseline

```text
pytest -q -p no:randomly -p no:random-order \
  tests/unit/rendering/test_semantic_cards.py
```

Result: PASS, `99 passed in 2.47s`.

This includes the two previously failing ordering/suppression tests after restoring `result-before-use.json`.

### Daemon HTTP security matrix

```text
pytest -q -p no:randomly -p no:random-order \
  tests/unit/daemon/test_daemon_http_security.py
```

Result: PASS, `524 passed in 4.17s` on the final run.

### Generated semantic-card corpus

```text
python -m devtools.render_semantic_card_fixtures --check
```

Result: PASS, `semantic-card fixtures: verified 24 case(s)`.

### Broader rendering/visual lane

An earlier completed run of the affected rendering/visual lane reported 233 passing tests and 17 snapshots. During final repetition, the broader command over the entire `tests/unit/rendering` directory exceeded the 300-second tool ceiling before producing results. No product assertion failed, and no Python production source changed after the earlier green run. The focused semantic/document/SSR/HTTP lanes above are the final reproducible evidence used for release certification.

### Frontend clean offline install, typecheck, unit tests, and audit

From `webui/`:

```text
rm -rf node_modules
npm ci --offline --ignore-scripts
npm run typecheck
npm run test:unit -- --reporter=dot
npm audit --offline --omit=optional --json
```

Results:

- clean install: PASS, 89 packages added;
- strict TypeScript: PASS;
- Vitest: PASS, 1 file and 24 tests;
- audit: PASS, zero info/low/moderate/high/critical vulnerabilities;
- lockfile scan: no private/internal registry references.

### Consecutive Vite reproducibility

```text
npm run build
cp ../polylogue/daemon/static/webui-v2/transcript.{js,css} <first-proof>/
npm run build
cp ../polylogue/daemon/static/webui-v2/transcript.{js,css} <second-proof>/
cmp <first-proof>/transcript.js <second-proof>/transcript.js
cmp <first-proof>/transcript.css <second-proof>/transcript.css
```

Result: PASS, both files byte-identical across consecutive builds.

Final committed assets:

| Asset | SHA-256 | Bytes |
|---|---|---:|
| `transcript.js` | `6cc8fdfbf38abd9ccd66de5b7015ac6b9c0fafd61815d32c7536bc44d39006db` | 58,928 |
| `transcript.css` | `dc39c243ba661da3fa859bbbc366ed146542908341949fb34c8e430dbba4ebdb` | 5,740 |

Vite transformed 10 modules. No source maps were emitted.

### Offline wheel build and inspection

```text
uv build --offline --wheel --out-dir /mnt/data/work-webui04/wheel-final
```

Result: PASS.

| Property | Value |
|---|---|
| Wheel | `polylogue-0.2.0-py3-none-any.whl` |
| SHA-256 | `19483aa51be3d16372879be9e47a3a595b27529b74a7e26891dd931ffef0b54a` |
| Size | 4,200,885 bytes |
| ZIP CRC | valid |
| JS member | present, 58,928 bytes |
| CSS member | present, 5,740 bytes |
| `uv.lock` change | none |

### Topology

```text
python -m devtools verify topology
```

Result: PASS/non-blocking:

```text
realized=1023 declared=1023 blocking=False
```

Findings:

- missing: 0
- orphaned: 0
- conflicts: 0
- kernel-rule: 0
- `tbd`: 9 pre-existing `polylogue/storage/*.py` entries

### Fresh-worktree patch proof

```text
git worktree add --detach /mnt/data/work-webui04/apply-check \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
git -C /mnt/data/work-webui04/apply-check apply --check PATCH.diff
git -C /mnt/data/work-webui04/apply-check apply PATCH.diff
```

Additional proof:

- SHA-256 comparison: 30 of 30 patched files byte-identical with the implementation worktree;
- `git diff --check`: pass;
- changed Python byte compilation: pass;
- both public schemas: valid Draft 2020-12;
- applied-copy tests:

```text
pytest -q -p no:randomly -p no:random-order \
  tests/unit/rendering/test_semantic_card_document.py \
  tests/unit/daemon/test_semantic_card_http.py \
  tests/unit/daemon/test_webui_v2_ssr.py
```

Result: PASS, `44 passed in 7.10s`.

### Patch hygiene scan

`PATCH.diff` was scanned over added lines and file names for:

- explicit `TODO`, `FIXME`, `XXX`, `HACK`, `WIP`, `TBD`, placeholder, or not-implemented markers;
- bare Python `pass` and ellipsis sentinels;
- pseudocode phrases;
- private keys, access tokens, password/secret literals;
- `/mnt/data`, `/realm`, internal host names, and private package registries;
- supplied archive/report filenames;
- runtime remote URLs outside schemas and standard DOM namespaces.

Result: PASS. Broad ellipsis matching found only legitimate TypeScript spread syntax and tuple annotations; explicit marker scans found zero placeholders. Schema namespace URLs, W3C DOM namespace strings from the Vite bundle, and the loopback test URL were reviewed as legitimate.

## Browser journey attempt

A real external Playwright/Chromium navigation to the loopback fixture server was attempted. Chromium returned:

```text
net::ERR_BLOCKED_BY_ADMINISTRATOR
```

The environment's managed policy at `/etc/chromium/policies/managed/000_policy_merge.json` contains:

```json
{"URLBlocklist": ["*"]}
```

No bundled Playwright browser cache was available to bypass that managed system browser. Therefore this package does not claim a real browser journey, screenshot, frame-rate trace, or token-bootstrap interaction in Chromium. HTTP/SSR behavior is proven through the real loopback server; DOM/state behavior is proven in Vitest/jsdom.

## Unverified work

- operator's private archive and live daemon;
- unrestricted real-browser journey and visual screenshot;
- measured 60-fps scrolling on the operator machine;
- NixOS/Nix deployment and service upgrade;
- private provider fixtures or credentials;
- full repository test suite;
- future `4p1` stable keyset read/continuation transaction;
- independent deletion certification for legacy render paths.
