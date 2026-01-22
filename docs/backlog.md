# Polylogue Backlog

> Last updated: 2026-01-22
> Status: All tests passing (859 tests), critical issues resolved

## Remaining Work Summary

| Category | P0 (Critical) | P1 (High) | P2 (Medium) | P3 (Low) |
|----------|---------------|-----------|-------------|----------|
| API Contracts | 0 | 3 | 7 | 3 |
| Data Flow | 0 | 3 | 5 | 4 |
| Performance | 1 | 3 | 4 | 4 |
| Test Coverage | 0 | 1 | 4 | 1 |
| Type Safety | 0 | 0 | 0 | 9 |

**Recent work:** Hardening Sprint, Agent Sweep, and Bug Hunt (2026-01-20 to 2026-01-22) resolved all critical type safety, dependency, and static analysis issues. See git history for details.

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

The `view` command provides semantic projections over conversations for analysis and export. See `docs/design-view-command.md` for detailed design.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core `view` command with basic filters/formats | ✅ Done |
| 2 | Add transforms (strip-tools, strip-thinking) | Pending |
| 3 | Add pattern annotations | Pending |
| 4 | Add `turns` entity and semantic grouping | Pending |
| 5 | Interactive mode and editor integration | Pending |

**Current capabilities:**
- Entity types: `conversations`, `messages`
- Filters: `--provider`, `--since`, `--limit`, `--id`
- Output formats: `table`, `jsonl`, `json`, `markdown`
- Repository API: `ConversationRepository.view()` for programmatic access

**Planned features:**
- Content transforms: `--transform strip-tools`, `--transform strip-thinking`
- Pattern annotations: `--annotate frustration`, `--annotate verbosity`
- Field projections: `--fields id,role,text` or `--fields minimal`
- Additional entities: `turns` (dialogue pairs), `stats` (aggregations)

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
