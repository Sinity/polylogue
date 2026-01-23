# AGENTS.meta.md - Agent Memory Design for Polylogue

> Design philosophy and maintenance guide for CLAUDE.md. Use this as a style guide when updating the agent memory file.

**Canonical example**: See `CLAUDE.md` in this repository.

---

## Core Purpose

**Agent memory files prevent repeated exploration of the codebase.**

Every session, an AI agent starts with zero knowledge. Without CLAUDE.md, each session must explore:
- Storage backend architecture (protocols, SQLiteBackend, repository pattern)
- Search provider abstraction (FTS5, Qdrant, caching)
- Pipeline services and dependency injection
- Thread safety patterns (_WRITE_LOCK, thread-local connections)
- Content hashing and deduplication logic

**Exploration cost per subsystem**: ~15-30K tokens

**CLAUDE.md loading cost**: ~7-10K tokens once

**ROI**: After 2-3 sessions, CLAUDE.md pays for itself. Over 50 sessions, it saves 500K+ tokens.

---

## Polylogue-Specific Patterns

### Knowledge Tiers for Polylogue

**Tier 1: Architecture (Stable, high leverage)**
- Storage layer: backend protocol, SQLiteBackend, repository
- Pipeline: services (ingestion, indexing, rendering)
- Protocols: StorageBackend, SearchProvider, OutputRenderer
- Thread safety: _WRITE_LOCK, thread-local connections

**Tier 2: Behavioral Patterns (Semi-stable, high leverage)**
- Content hash deduplication (NFC normalization, SHA-256)
- Bounded submission (max 16 in-flight futures)
- Search caching (LRU, version-based invalidation)
- Renderer selection (--format flag, create_renderer())
- DI container (singleton vs factory providers)

**Tier 3: Runtime Semantics (Volatile, medium leverage)**
- Parallel ingestion workers (4 threads)
- Search cache size (128 entries)
- FTS5 batch sizes (chunked updates of 200)
- Retry logic (Voyage 5×, Qdrant 3×)

**Tier 4: Operational (Volatile, medium leverage)**
- Mypy strict mode settings
- Test coverage metrics (951 tests, 69% overall, 83% core)
- Performance baselines (search speedup 21,343x)

**Tier 5: Implementation Details (DON'T DOCUMENT)**
- Specific SQL queries
- Internal variable names
- Line-by-line function implementations

---

## Polylogue Documentation Principles

### 1. Behaviors Over Architecture Diagrams

**Bad**: "The storage layer uses protocols for abstraction"
**Good**:
```
StorageBackend protocol → SQLiteBackend (750 lines)
  - Thread-local connections via connection_context()
  - _WRITE_LOCK in repository (not backend)
  - Schema v4: source_name GENERATED column with index
```

### 2. Tables for Multi-Dimensional Data

**Protocols Overview**:
| Protocol | Purpose | Key Methods | Implementations |
|----------|---------|-------------|-----------------|
| StorageBackend | DB abstraction | save_conversation, get_conversation, iter_conversations | SQLiteBackend (750 lines) |
| SearchProvider | Search abstraction | index, search, delete | FTS5Provider, QdrantProvider |
| OutputRenderer | Format abstraction | render, supports_format | MarkdownRenderer, HTMLRenderer |

**Thread Safety**:
| Lock/Pattern | Location | Protects | Usage |
|--------------|----------|----------|-------|
| _WRITE_LOCK | repository.py:48 | All DB writes | `with _WRITE_LOCK: save(...)` |
| Thread-local conn | db.py | SQLite connections | `connection_context()` context mgr |
| Bounded futures | runner.py | Memory explosion | max 16 in-flight, FIRST_COMPLETED |

### 3. Inline Performance Numbers

**Bad**: "Search caching provides significant speedup"
**Good**: "Search cache: 69ms → 0.003ms (21,343x speedup), LRU size=128, version-based invalidation"

### 4. Code Examples: Minimal, Realistic

**Bad**: Three variations of the same pattern
**Good**: One complete example showing the actual usage:
```python
# Container usage in CLI
container = get_container()
repo = container.storage()  # Singleton - same instance
svc1 = container.ingestion_service()  # Factory - new instance
svc2 = container.ingestion_service()  # Factory - different instance
```

### 5. Troubleshooting as Diagnostic Trees

**Bad**: Prose description of debugging steps
**Good**:
```
Database locked errors:
  ☐ Check: All writes via StorageRepository? (grep _WRITE_LOCK)
  ☐ Check: No direct sqlite3.connect()? (grep "sqlite3.connect")
  ☐ Verify: connection_context() used? (storage/repository.py:48)
```

---

## Maintenance Triggers for Polylogue

Update CLAUDE.md immediately after:

1. **Architecture changes**:
   - New protocol (e.g., VectorProvider)
   - New backend implementation (e.g., PostgresBackend)
   - Service layer changes (new service, different lifecycle)

2. **Behavioral changes**:
   - Thread safety patterns modified
   - Cache invalidation logic changed
   - Batch sizes or worker counts adjusted
   - Content hash algorithm updated

3. **Performance changes**:
   - Benchmark results change significantly
   - New optimization added (update baseline numbers)

4. **Provider changes**:
   - New importer (ChatGPT, Claude, etc.)
   - Importer behavior changed (content_blocks extraction)
   - Search provider added/modified

5. **Testing/type changes**:
   - Test count changes by >10%
   - Mypy errors increase/decrease significantly
   - Coverage metrics shift

---

## Polylogue-Specific Sections

### Critical Information to Maintain

**Storage Layer** (prevents 20K tokens of exploration):
- Backend protocol methods and signatures
- SQLiteBackend features (schema v4, migrations, CRUD)
- Repository pattern (_WRITE_LOCK ownership)
- Thread-local connection pattern

**Pipeline Services** (prevents 15K tokens):
- Service lifecycles (singleton vs factory)
- Dependency injection container
- Service responsibilities (ingestion, indexing, rendering)

**Content Hashing** (prevents 10K tokens):
- NFC normalization requirement
- SHA-256 algorithm
- Idempotency guarantee
- Hash composition (what's included/excluded)

**Thread Safety** (prevents 15K tokens):
- _WRITE_LOCK location and usage
- Thread-local connections
- Bounded submission pattern (max 16 in-flight)
- Parallel rendering (4 workers)

**Search Abstraction** (prevents 10K tokens):
- Provider protocol
- FTS5 vs Qdrant differences
- Cache mechanics (LRU, version invalidation)
- Performance characteristics

---

## Anti-Patterns Specific to Polylogue

### 1. Duplicating docs/architecture.md

**Problem**: CLAUDE.md and architecture.md contain same information
**Prevention**:
- CLAUDE.md: Facts, behaviors, patterns (dense tables)
- architecture.md: Diagrams, explanations, design rationale (prose)

### 2. Missing Performance Numbers

**Problem**: Says "fast" instead of "266K msg/sec"
**Prevention**: Always include baseline numbers from benchmarks

### 3. Stale Type Safety Info

**Problem**: Says "211 errors" when it's now "0 errors"
**Prevention**: Update mypy stats after type-related changes

### 4. Missing Protocol Evolutions

**Problem**: New protocol added but not documented
**Prevention**: Protocol changes trigger CLAUDE.md update in same commit

### 5. Vague Thread Safety Claims

**Problem**: "Thread-safe" without explaining how
**Prevention**: Always specify lock/pattern and location

---

## Document Structure for Polylogue

### Current Sections (Optimized Order)

1. **Core Architecture** (~500 tokens)
   - Design principles
   - Key invariants
   - Data flow

2. **Critical Files** (~1500 tokens)
   - Organized by layer
   - File paths + purpose + key details (lines, features)

3. **Thread Safety Model** (~800 tokens)
   - Architecture diagram
   - Lock inventory (table)
   - Rules (DO/DON'T)

4. **Content Hashing** (~400 tokens)
   - NFC normalization code
   - Idempotency mechanics

5. **Storage Backend Abstraction** (~600 tokens)
   - Protocol definition
   - SQLiteBackend features
   - Extension points

6. **Search Provider Abstraction** (~600 tokens)
   - Protocol comparison table
   - FTS5 vs Qdrant features
   - Cache mechanics

7. **Service Layer** (~500 tokens)
   - Service responsibility table
   - Orchestration pattern

8. **Dependency Injection** (~400 tokens)
   - Container structure
   - Singleton vs Factory table

9. **Renderer Abstraction** (~300 tokens)
   - Protocol + implementations table

10. **Performance Optimizations** (~500 tokens)
    - Baseline table
    - Cache impact numbers

11. **Configuration** (~400 tokens)
    - Config objects table
    - Env var precedence

12. **Testing** (~500 tokens)
    - Coverage metrics
    - Key test suites

13. **Type Safety** (~300 tokens)
    - Mypy metrics
    - Annotation patterns

14. **Development Patterns** (~1000 tokens)
    - Adding backend/provider/service
    - Anti-patterns

15. **Build Commands** (~200 tokens)
    - Common commands with actual flags

16. **Common Debugging** (~400 tokens)
    - Error → diagnostic checklist

17. **Architecture Evolution** (~300 tokens)
    - Recent commits with impact

**Target**: ~8000-10000 tokens for comprehensive coverage

---

## Verification Procedure

### After Each Update

1. **Density check**: Is every sentence a unique fact?
2. **Table scan**: Could prose be a table?
3. **Value check**: Are numbers inline or referenced?
4. **Example quality**: Realistic, minimal, correct?

### Monthly Review

```bash
# Check file paths still exist
rg 'storage/backends/sqlite.py' CLAUDE.md && ls storage/backends/sqlite.py

# Verify mypy claims
uv run mypy polylogue/ 2>&1 | tail -1  # Should match "0 errors"

# Check test count
uv run pytest --co -q | tail -1  # Should match "951 tests"

# Verify performance numbers (if benchmarks exist)
uv run python tests/benchmarks/run_all.py | grep speedup
```

---

## Success Metrics

Track across sessions:

| Metric | Target | Signal |
|--------|--------|--------|
| Orientation questions | 0 | No "where is X?" questions |
| Pattern adherence | >90% | First implementation attempts correct |
| Exploration loops | <3 per session | Agent knows where to look |
| Doc updates | >50% of sessions | Self-maintenance happening |
| Mypy accuracy | Exact match | Stats stay current |

---

## Summary

For Polylogue specifically:

✅ **DO**:
- Table-ify protocol comparisons, thread safety, configuration
- Inline all performance numbers (21,343x, 951 tests, 69% coverage)
- Update mypy/test stats in same commit as code changes
- Document behaviors (NFC normalization, _WRITE_LOCK, bounded futures)
- Use exact file paths (storage/repository.py:48)

❌ **DON'T**:
- Duplicate docs/architecture.md explanations
- Document implementation details (SQL queries, internal vars)
- Omit numbers ("fast" vs "266K msg/sec")
- Write prose where tables work
- Let stats go stale

**Goal**: Maximum exploration prevention per token loaded.
