# Polylogue Backlog

> Last updated: 2026-01-23
> Status: ✅ Production-ready (951 tests passing, 0 mypy errors, Priorities 1-4 complete)

## Major Improvements Completed (Jan 2026)

**Priorities 1-4 Complete:**
- ✅ Architecture refactoring: Layered modules, protocols, DI framework
- ✅ Type safety: 211 → 0 mypy errors (100% strict mode compliance)
- ✅ Test coverage: 83% core business logic (1.93:1 test-to-code ratio)
- ✅ Performance: 21,343x search cache speedup, parallel rendering, incremental indexing
- ✅ Renderer abstraction: OutputRenderer protocol with Markdown/HTML implementations

**Recent work:** Comprehensive refactoring via 12-agent swarm (2026-01-22 to 2026-01-23) resolved all architectural debt, type safety issues, and performance bottlenecks. See git history and docs/refactoring-plan.md for details.

## Remaining Work Summary

| Category | P0 (Critical) | P1 (High) | P2 (Medium) | P3 (Low) |
|----------|---------------|-----------|-------------|----------|
| API Contracts | 0 | 1 | 4 | 2 |
| Data Flow | 0 | 1 | 2 | 3 |
| Performance | 0 | 0 | 1 | 2 |
| Test Coverage | 0 | 0 | 2 | 1 |
| Type Safety | 0 | 0 | 0 | 0 |

---

## High Priority Issues (P1)

### API Contracts

| File | Line | Issue | Status |
|------|------|-------|--------|
| `config.py` | - | `update_config()` mutates instead of returning copy | Open |

### Data Flow

| Issue | Status |
|-------|--------|
| ConversationRepository partial ID resolution could be more efficient | Open (low impact, works correctly) |

---

## Low Priority Issues (P3)

### Documentation Gaps

Functions that could benefit from more detailed docstrings:
- `update_config()` - clarify mutation vs copy semantics
- `update_source()` - clarify mutation behavior
- `connection_context()` - document transaction ownership rules

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

### Future Enhancements
1. Expand view command transforms (strip-tools, strip-thinking, annotations)
2. Lynchpin integration (library import, warehouse sync, deprecate file rendering)
3. Additional renderer implementations (PDF, EPUB)
4. Additional storage backends (PostgreSQL, DuckDB)
5. Performance monitoring dashboard (expose benchmark metrics)
6. Improve docstring coverage for mutation semantics
7. Create `tests/test_importers_codex.py` for complete coverage
