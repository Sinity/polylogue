# Storage Twins: Documented Divergences

**Status**: Committed artifact for regression testing  
**Updated**: 2026-07-14  
**Author**: Sinity (agent)  
**Bead**: polylogue-pf1  

---

## Overview

The Polylogue archive has two SQLite storage backends that serve different roles:

- **Async backend** (`polylogue/storage/sqlite/async_sqlite*.py`): 
  - Async/await API for daemon and MCP surfaces
  - Delegation-heavy, stateless per-operation
  - Mixin-based architecture for composition
  - Reads primarily via `self.queries` (SQLiteQueryStore)

- **Sync backend** (`polylogue/storage/sqlite/archive_tiers/`):
  - Synchronous write-capable full DB owner
  - Organizes logic by tier (source, index, user, embeddings, ops)
  - Schema ownership and connection lifecycle management
  - Monolithic archive.py (11K lines) plus tier-specific modules

## The 10 Documented Divergences

All divergences are **architectural**, not bugs. They reflect the different roles and compositional boundaries. Each is listed with its file:line references and rationale.

### 1. Method Naming: `_session_id_query` vs `session_id_query`

**Async location**: `async_sqlite_archive.py` — delegates via `self.queries.session_id_query()`  
**Sync location**: `query_store_archive.py` — public `session_id_query` API  
**Rationale**: Private delegate naming convention vs public query-store API. Same underlying function; called through different architectural layers.  
**Status**: Architectural—no action needed.

### 2. Search Sessions Delegation Path

**Async location**: `async_sqlite_archive.py:search_sessions()` → delegates to `self.queries`  
**Sync location**: `query_store_archive.py:search_sessions()` → delegates to `search_session_hits()` → chains internal logic  
**Rationale**: Backend delegates to its composed query store; query store delegates to its own internal search implementation. Both reach identical result.  
**Status**: Architectural—no action needed.

### 3. Get Messages: Content Blocks Attachment Strategy

**Async location**: `async_sqlite_archive.py:get_messages()` — blocks pre-attached by query store  
**Sync location**: `query_store_archive.py:get_messages()` — canonical two-step load+merge pattern  
**Rationale**: Query store is the canonical read-only implementation; async backend inherits its behavior via composition.  
**Status**: Architectural—no action needed.

### 4. Connection Management Strategy

**Async location**: `async_sqlite.py:_get_connection()` — ensures schema before every use (backend responsibility)  
**Sync location**: `archive_tiers/archive.py:_connection_factory` — provides pre-configured read-only connections  
**Rationale**: Backend owns the DB and ensures schema; query store provides composable read-only connections that assume schema is ready.  
**Status**: Architectural—no action needed.

### 5. Write Methods Exist Only on Backend

**Async location**: `async_sqlite_archive.py` — `save_session_record()`, `save_messages()`, etc.  
**Sync location**: `archive_tiers/write.py`, `user_write.py`, etc. — write tier methods  
**Rationale**: Query store is deliberately read-only; write capability is backend-only by design. This enforces the read/write split.  
**Status**: Architectural—no action needed.

### 6. Query API Methods Exist Only on Query Store

**Async location**: `async_sqlite_archive.py` — accesses via `self.queries` (e.g., `queries.list_sessions()`)  
**Sync location**: `query_store_archive.py` — `list_sessions()`, `count_sessions()`, `search_action_*()` defined here  
**Rationale**: Read-only query operations belong only to the query-store layer. Backend doesn't reimplement these.  
**Status**: Architectural—no action needed.

### 7. get_session_insight_status Implementation Location

**Async location**: `async_sqlite_archive.py` — method on `SQLiteArchiveMixin`  
**Sync location**: `query_store.py` — separate independent implementation  
**Rationale**: Both backends provide this method; query store has its own version to maintain independence.  
**Status**: Architectural—no action needed.

### 8. get_messages_batch: Early-Exit Clarity

**Async location**: `async_sqlite_archive.py:get_messages_batch()` — delegates to `self.queries`  
**Sync location**: `query_store_archive.py:get_messages_batch()` — explicit empty-session_ids early exit with comment  
**Rationale**: Equivalent behavior; sync version adds explicit clarity on the empty-list edge case.  
**Status**: Architectural—no action needed.

### 9. iter_messages: Chunk-Size Fast Path

**Async location**: `async_sqlite.py:iter_messages()` — `chunk_size=100` optimization, delegates to query store  
**Sync location**: `query_store_archive.py:iter_messages()` — calls `messages_q.iter_messages()` directly  
**Rationale**: Both reach the same destination (message query module); async adds an optimization layer around the chunking.  
**Status**: Architectural—no action needed.

### 10. search_session_hits: Access Pattern

**Async location**: `async_sqlite_archive.py:search_session_hits()` — backend delegates to `self.queries`  
**Sync location**: `query_store_archive.py:search_session_hits()` — opens a direct connection to the query module  
**Rationale**: Same destination via different architectural layers (backend delegation vs direct query-store composition).  
**Status**: Architectural—no action needed.

---

## Testing Strategy

The regression test `tests/unit/storage/test_storage_twins.py` verifies:

1. **Divergence count is stable**: All 10 are accounted for; new divergences fail the test.
2. **Divergence documentation exists**: The source comment in `async_sqlite_archive.py:14-54` is present.
3. **Architectural roles are clear**: Async uses mixins, sync uses tier modules.
4. **Backend files exist**: References in the divergence table point to real files with expected content.

### Running Tests

```bash
devtools test -k twin -v
```

Expected output: All tests pass, confirming the divergences are as documented.

---

## Why Divergences Exist

The async and sync backends serve different compositional goals:

- **Async** (daemon/MCP): Needs `async def` signatures and delegation to external query stores for testability.
- **Sync** (archive operations): Owns the full DB lifecycle, schema, and all write paths; monolithic but self-contained.

The hiu epic (collapse storage twins onto sync core) plans to retire the async duplication by introducing an async adapter that wraps the sync core, eliminating the divergence maintenance burden while preserving the async API.

---

## Future: hiu Epic Resolution

**Bead**: polylogue-hiu  
**Decision**: Direction B — sync core, async adapter  
**Plan**:
1. **Prerequisite (this bead, polylogue-pf1)**: Reconcile the 10 divergences INTO the sync store as the canonical implementation.
2. **Adapter (hiu Step 1)**: Build an async executor wrapping the sync core.
3. **Migration (hiu Steps 2+)**: Retire async_sqlite*.py mixin lane by mixin, keeping async API.
4. **Cleanup (hiu Final)**: Delete async_sqlite.py, async_sqlite_archive.py, async_sqlite_raw.py.

When hiu is complete, this divergence document will be retired in the same PR that deletes the async backends.
