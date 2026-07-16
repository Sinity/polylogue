#!/usr/bin/env python3
"""File the verified over-abstraction audit findings as a7xr children. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:120] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:250])


COMMON = ["--parent", "polylogue-a7xr", "-l", "area:substrate,horizon:frontier,tech-tree"]

bd(
    "create",
    "Delete the dead search-provider class lane (FTS5Provider/HybridSearchProvider/factories)",
    *COMMON,
    "--type",
    "chore",
    "-p",
    "2",
    "-d",
    "VERIFIED 2026-07-06 (abstraction audit + spot-check): create_search_provider and "
    "create_hybrid_provider have ZERO production call sites — only their own tests import them. "
    "Production FTS is inline SQL (messages_fts MATCH in archive_tiers/archive.py:4545/4661/7668) "
    "and the live --retrieval-lane hybrid path re-implements fusion inline at "
    "cli/archive_query.py:830-852 via reciprocal_rank_fusion, never the provider classes. "
    "create_search_provider's own docstring admits 'currently always FTS5Provider'. Dead weight: "
    "~360 prod LOC (search_providers/fts5.py, hybrid.py, hybrid_factory.py, the SearchProvider "
    "Protocol at protocols.py:43) + ~1,200 LOC of tests testing the dead code "
    "(test_hybrid.py, test_hybrid_laws.py, part of test_fts5.py).",
    "--design",
    "Delete FTS5Provider, HybridSearchProvider, both factories, the SearchProvider Protocol, and "
    "their dedicated tests. KEEP: reciprocal_rank_fusion + hybrid_sessions.py helpers (real "
    "consumers) and the entire vector half — SqliteVecProvider is live via create_vector_provider "
    "(8 prod call sites). Check test_protocol_conformance.py:68 and prune the dead entries. "
    "Protected-files check: test_hybrid* are not in the protected list (verify before delete).",
    "--acceptance",
    "rg shows zero references to the deleted names; hybrid retrieval lane behavior unchanged "
    "(goldens on --retrieval-lane hybrid); SqliteVecProvider paths untouched; devtools verify "
    "green. One PR, surgical-renewal shaped.",
)

bd(
    "create",
    "Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug",
    *COMMON,
    "--type",
    "chore",
    "-p",
    "2",
    "-d",
    "VERIFIED 2026-07-06: 6 of 14 protocols in protocols.py have zero consumers anywhere "
    "(SessionReader, SearchStore, ArchiveMessageQueryStore, SemanticArchiveQueryStore, "
    "SessionSemanticStatsStore, SessionArchiveReadStore) — violating the module's own docstring "
    "rule ('only protocols with 2+ implementations earn their existence'). The 18-filter-kwarg "
    "signature is spelled out 3x in SessionReader alone. The repo kwarg methods are equally dead: "
    "RepositoryArchiveQueryMixin.list (docstring-example-only), .count (zero callers), "
    ".list_summaries (sole caller iter_summary_pages, itself zero callers). All real traffic "
    "goes SessionRecordQuery -> list_by_query/count_by_query. The "
    "SessionListQueryKwargs/SessionCountQueryKwargs TypedDicts are a pure 1:1 re-expansion "
    "consumed once. LATENT BUG (verified): archive/query/fields.py:797 maps record_attr='cursor' "
    "but SessionRecordQuery has no cursor field — dataclasses.replace would TypeError if a plan "
    "ever carried a cursor; unreachable today, proving the mapping is dead.",
    "--design",
    "Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + "
    "iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at "
    "query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor "
    "field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive "
    "with result-set pagination; if so, wire it properly instead of deleting). KEEP protocols "
    "with real consumers: SessionQueryRuntimeStore, SessionOutputStore, "
    "SessionArchiveStatsStore, TagStore, RawPersistenceStore, RawValidationStore (genuine test "
    "double). mypy --strict is the net.",
    "--acceptance",
    "protocols.py contains only consumed protocols (each with a named consumer in a comment); "
    "dead kwarg surface gone; cursor mapping resolved (deleted or actually wired); mypy strict "
    "green; ~600 LOC removed. Verify: devtools verify.",
)

bd(
    "create",
    "neighbor_candidates needs a 4-method protocol, not the 20-method SessionQueryRuntimeStore",
    *COMMON,
    "--type",
    "chore",
    "-p",
    "3",
    "-d",
    "VERIFIED counts: archive/session/neighbor_candidates.py calls exactly 4 store methods "
    "(resolve_id, get, list_summaries_by_query, search_summary_hits) but is typed against the "
    "~20-method SessionQueryRuntimeStore — forcing api/archive.py:1341-1747 "
    "_ArchiveNeighborRuntime to stub ~15 unneeded methods (~400 lines), re-implement the "
    "18-kwarg trio a FOURTH time, and still need a cast at :4216 because it does not truly "
    "conform.",
    "--design",
    "Define a 4-method NeighborStore protocol next to neighbor_candidates.py; retype the "
    "consumer; _ArchiveNeighborRuntime shrinks to ~60 lines; the cast disappears. Sequence "
    "AFTER the protocols.py prune (previous bead) so the kwarg trio is already gone.",
    "--acceptance",
    "Adapter under 80 lines; no cast; neighbor_candidates behavior unchanged (existing tests); mypy strict green.",
)

bd(
    "create",
    "api/contracts write-surface shadow adapters verify copies, not surfaces — delete or re-anchor",
    *COMMON,
    "--type",
    "task",
    "-p",
    "3",
    "-d",
    "Abstraction audit: api/contracts/ (8 files, 977 LOC) write-surface protocols "
    "(IngestSurface, MaintenanceSurface, TagMutationSurface, SessionDeleteSurface) have zero "
    "consumers outside the package; the adapters are constructed only in the two contract test "
    "files. Worse, the adapters are hand-written MIRRORS of the CLI handlers that openly "
    "diverge (CLIWriteSurface.ingest_path returns a synthetic failed envelope instead of the "
    "real stage-and-POST flow; tag/delete conformance passes on method presence, not "
    "execution) — so the parity guarantee attaches to a shadow object free to drift from "
    "cli/commands/, which is exactly the drift (#859) the layer was built to catch. "
    "TUIReadSurface is genuinely consumed by ui/tui screens TODAY, but f94 (decided KILL) "
    "removes that consumer — coordinate: execute f94 first, then nothing in read_surface "
    "needs keeping either.",
    "--design",
    "Preferred: re-anchor assert_implements on the ACTUAL facade/handler objects (the real "
    "CLI write path and MCP tool functions) so conformance means execution-path conformance — "
    "if that is not cheaply possible, delete the shadow layer and record the parity intent on "
    "the owning issue (#859 successor = t46 golden equivalence, which tests real surfaces). "
    "Sequence with f94 (TUI kill) to sweep read_surface in the same pass.",
    "--acceptance",
    "Either assert_implements binds to real handler objects (test proves a signature drift in "
    "cli/commands/ fails the contract) or api/contracts/ is deleted with intent recorded; no "
    "shadow adapter remains that reimplements handler logic. Verify: devtools verify.",
)

bd(
    "create",
    "Collapse the one-operation operations-contract framework to concrete Import models",
    *COMMON,
    "--type",
    "chore",
    "-p",
    "3",
    "-d",
    "Abstraction audit: operations/operation_contract.py (277 LOC of "
    "OperationRequest/Ack/FollowUp/Status generics) serves exactly ONE operation — "
    "OperationKind.IMPORT (9 prod uses); the other 9 of 10 enum members have zero prod uses "
    "and nothing dispatches on .kind (rendered once as a markdown label). "
    "OperationStatus.RUNNING/COMPLETED/FAILED are documented 'reserved'. ImportAck exists "
    "solely to re-wrap the generic ack pinning kind=IMPORT ('future import-specific fields "
    "land here' — speculative generality). _require_operation_kind guards subclasses that "
    "do not exist.",
    "--design",
    "Collapse to concrete ImportRequest/ImportAck; reintroduce a base when a SECOND operation "
    "actually lands (rxdo query-runs or fs1.5 export may become that — check before deleting "
    "whether either is imminent; if yes, keep the base and delete only the unused enum "
    "members/statuses). WIRE-STABILITY: ImportAck's field names/JSON are on the daemon HTTP "
    "wire (daemon/http.py) — the collapse removes the abstract layer, never the payload shape "
    "(golden on the wire envelope). specs.py catalog stays (feeds artifact graph); "
    "OperationKind there can be a plain string label.",
    "--acceptance",
    "Wire envelope byte-identical (golden); one concrete model pair; unused enum members gone "
    "or each carries a consumer; devtools verify green.",
)

bd(
    "create",
    "payloads.py: generic from_row for the 74 identical-name copy lines (keeps typed wire contract)",
    *COMMON,
    "--type",
    "chore",
    "-p",
    "3",
    "-d",
    "Abstraction audit (rg-verified): 30 hand-rolled from_row/from_* classmethods in "
    "surfaces/payloads.py where 74 of 74 x=row.y copy lines are identical-name — pure "
    "mechanical transcription across 2,921 LOC / 85 classes. The model DECLARATIONS are the "
    "typed wire contract (drive render openapi / cli-output-schemas) and stay untouched.",
    "--design",
    "One generic from_row on the shared base (cls(**{f: getattr(row, f) for f in "
    "cls.model_fields if hasattr(row, f)})) with explicit overrides only where renames/"
    "defaults exist (title=row.session_title, material_origin='unknown' at :1275). PRESERVE "
    "THE NET the explicit bodies provided: a test comparing model_fields against the source "
    "row dataclass fields per class (missing/extra fields fail loudly instead of silently "
    "defaulting). Bonus: fewer import-time bytes on the 20d.2 CLI-startup path (payloads is "
    "~2,915 lines of the heavy import).",
    "--acceptance",
    "Identical wire output (goldens across list/search/read payloads); per-class field-parity "
    "test in place; ~400-500 LOC removed; render openapi/cli-output-schemas unchanged. "
    "Verify: devtools verify + render all --check.",
)

bd(
    "create",
    "Table-drive the hand-aligned column triplicates in archive_tiers write/read hot core",
    *COMMON,
    "--type",
    "task",
    "-p",
    "3",
    "-d",
    "Abstraction audit: archive_tiers/write.py:1420-1430 hand-aligns a 30-column messages "
    "INSERT to 30 placeholders, blocks tuple-yield at :1498-1504 must stay positionally "
    "synced by hand; archive.py:4898-5780 has 14 query_* methods with 388 hand-written "
    "row[col] accessors. Zero column-spec constants in the tier — every column list exists "
    "at least 3x (DDL, INSERT, tuple order) and must be edited in lockstep. This is MISSING "
    "table-driving in the STRICT-schema hot core; the drift hazard is real (a mis-ordered "
    "tuple silently writes wrong columns of the same affinity).",
    "--design",
    "Derive column list + placeholder string + tuple order from the row dataclasses "
    "(dataclasses.fields()) with an escape hatch for expression columns (NULL literals, "
    "_sqlite_text coercions, JSON decoders). Mechanical refactor gated on the existing "
    "crud/property tests (test_crud.py is protected — it is the net). Generated-column trap "
    "applies: session_id/message_id are GENERATED STORED — never in INSERT lists (the "
    "derivation must exclude them by marker, not by name-list).",
    "--acceptance",
    "One source of truth per table's column order; INSERT/SELECT built from it; crud + "
    "property tests green; a deliberate column reorder in the dataclass produces correct SQL "
    "(test). Verify: devtools test tests/unit/storage/test_crud.py + property suite.",
)

print("--- abstraction beads filed")
