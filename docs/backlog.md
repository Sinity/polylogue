# Polylogue Codebase Backlog

> Generated from comprehensive multi-agent analysis (2026-01-19)
> Updated: 2026-01-20 (hardening sprint + agent sweep completed)

## Executive Summary

Analysis identified **113 issues** across 6 dimensions. Two sprints addressed critical issues and code quality.

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Type Safety | ~~1~~ 0 | ~~2~~ 0 | ~~11~~ 0 | 9 | ~~23~~ 9 |
| API Contracts | ~~5~~ 3 | ~~4~~ 3 | 7 | 3 | ~~19~~ 16 |
| Data Flow | ~~5~~ 2 | ~~4~~ 3 | 5 | 4 | ~~18~~ 14 |
| Performance | ~~2~~ 1 | ~~4~~ 3 | 4 | 4 | ~~14~~ 12 |
| Test Coverage | ~~5~~ 4 | ~~3~~ 1 | 4 | 1 | ~~13~~ 10 |
| Dependencies | ~~1~~ 0 | ~~2~~ 0 | ~~2~~ 0 | ~~3~~ 0 | ~~8~~ 0 |
| Static Analysis | - | - | ~~11~~ 0 | ~~56~~ 0 | ~~67~~ 0 |

---

## Completed

### Hardening Sprint (2026-01-20)

**P0 Critical Fixes:**
- [x] Race conditions in runner.py - Added `_counts_lock` around mutations
- [x] Thread-unsafe mutation in ids.py - Returns `(id, meta, path)` tuple
- [x] Transaction boundary in store.py - `conn.commit()` inside `_WRITE_LOCK`
- [x] Pydantic version constraint - Pinned `pydantic>=2.0,<3.0`

**Dependency Cleanup:**
- [x] Removed unused deps: `pathvalidate`, `aiofiles`, `python-frontmatter`, `watchfiles`, `tiktoken`, `pypdf`, `alembic`
- [x] Added version upper bounds: `httpx<1.0`, `fastapi<1.0`, `uvicorn<1.0`, `rich<14.0`, `click<9.0`
- [x] Deleted orphaned `alembic.ini`

**API Formalization:**
- [x] Exported public API in `polylogue/lib/__init__.py`
- [x] Added `view()` method to repository with partial ID resolution
- [x] Added docstrings to `lib/models.py` explaining semantic projections

### Agent Sweep (2026-01-20)

**Type Safety:**
- [x] Created `DatabaseError` exception in `db.py`
- [x] Created `UIError` exception in `ui/facade.py`
- [x] Narrowed 15+ bare `except Exception:` blocks to specific types
- [x] Added logging for silently swallowed errors in `search.py`, `repository.py`

**Static Analysis:**
- [x] Ran `ruff --fix` across codebase
- [x] Fixed SIM simplifications, E501 line-length issues
- [x] Import sorting fixes

**Performance:**
- [x] Fixed N+1 query in `store.py:_prune_attachment_refs()` - single UPDATE with IN clause

**Test Coverage:**
- [x] Created `tests/test_pipeline_concurrent.py` (6 tests for thread safety)
- [x] Enhanced `tests/test_search_health.py` (+278 lines)
- [x] Enhanced `tests/test_source_ingest.py` (+326 lines, encoding fallback tests)
- [x] Enhanced `tests/test_cli.py` (+167 lines)
- [x] Enhanced `tests/test_ingest_render.py` (+132 lines)
- [x] Enhanced `tests/test_lib.py` (+126 lines)

---

## Critical Issues (P0) - Remaining

### 1. Reference Counting Without Transactional Guarantees
**File:** `polylogue/store.py:70-102`

`_prune_attachment_refs()` performs multi-step operations without explicit transaction control.

**Impact:** ref_count could become incorrect, orphaned attachments, premature deletion.

**Fix:** Wrap in SAVEPOINT or add trigger-based ref_count maintenance.

---

## High Priority Issues (P1)

### Performance

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `runner.py` | 214-222 | Sequential rendering | Parallelize with ThreadPoolExecutor |
| `runner.py` | 91-111 | Redundant JSON parsing when no filter | Skip when `source_names` is None |
| `lib/repository.py` | 103-179 | 3 queries per bulk fetch | Single JOIN query |
| `pipeline/ids.py` | 42-62 | Redundant file hashing | Check DB for existing hash first |
| `search.py` | 86-87 | `json_extract()` prevents index | Add computed column with index |

### API Contracts

| File | Line | Issue |
|------|------|-------|
| `db.py` | 306 | `connection_context()` ambiguous transaction semantics |
| `runner.py` | 310 | `latest_run()` returns raw dict instead of `RunRecord` |
| `config.py` | 214 | `update_config()` mutates instead of returning copy |

### Test Coverage Gaps

| Module | Gap |
|--------|-----|
| `importers/codex.py` | No test file exists |

---

## Low Priority Issues (P3)

### Missing Type Annotations

mypy reports 67 errors, primarily:
- Missing `conn: sqlite3.Connection` annotations
- Union type access without None checks
- Generator return type mismatches in `db.py`

### Documentation Gaps

Functions needing docstrings for mutation semantics:
- `update_config()` - mutates config in place
- `update_source()` - mutates source in list
- `connection_context()` - transaction ownership rules

---

## View Command Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core `view` command with basic filters/formats | ✅ Done |
| 2 | Add transforms (strip-tools, strip-thinking) | Pending |
| 3 | Add pattern annotations | Pending |
| 4 | Add `turns` entity and semantic grouping | Pending |
| 5 | Interactive mode and editor integration | Pending |

### Future Extensions (v2)
- Query language: `polylogue query "messages where provider='claude' | strip_tools | limit 100"`
- Saved views: `polylogue view --save my-analysis` / `--use my-analysis`
- Agent piping: `polylogue view messages --format jsonl | polylogue agent analyze`

---

## Lynchpin Integration Path

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Projection API (`polylogue/lib/`) | ✅ Done |
| 2 | `view` command using projections | ✅ Done |
| 3 | Lynchpin imports polylogue library, syncs to warehouse | Pending |
| 4 | Deprecate rendered file generation (make optional) | Pending |
| 5 | Dashboard reads from warehouse, not polylogue | Pending |

---

## Architecture Notes

### Thread Safety Model (Fixed)
```
ThreadPoolExecutor(4 workers)
    ├─ Worker 1: prepare_ingest() → returns (id, meta, path) immutably
    └─ ...

_WRITE_LOCK protects writes AND commits
    Thread A: acquire lock → write → commit → release lock
    (atomic transaction boundaries)

_counts_lock protects shared counters
    All workers: acquire → update counts → release
```

### Content Hashing Model (Sound)
```
ParsedConversation → SHA256(serialized) → content_hash
    ├─ Same content → same hash → no update (idempotent)
    └─ Different content → different hash → update
```

Thread safety fixes ensure this model works correctly under concurrent load.

---

## Recommended Next Steps

### Next Sprint - Performance & Stability
1. Add SAVEPOINT for ref counting operations (P0)
2. Parallelize rendering
3. Single JOIN query for bulk fetch

### Following Sprint - Polish
4. Fix mypy errors (type annotations)
5. Add remaining docstrings
6. Create `tests/test_importers_codex.py`
