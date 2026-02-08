# Storage Backend Abstraction Layer

**Implemented**: Section 3.1 of Polylogue refactoring plan

## Overview

This implementation creates a storage abstraction layer that decouples database-specific operations from the core application logic, enabling future support for multiple database backends (PostgreSQL, DuckDB, etc.) while maintaining full backward compatibility with existing SQLite-based code.

## Architecture

### Protocol Definition

**File**: `polylogue/protocols.py`

Extended the `StorageBackend` protocol with the following methods:

- `get_conversation(id: str) -> ConversationRecord | None`
- `list_conversations(source: str | None = None) -> list[ConversationRecord]`
- `save_conversation(record: ConversationRecord) -> None`
- `get_messages(conversation_id: str) -> list[MessageRecord]`
- `save_messages(records: list[MessageRecord]) -> None`
- `get_attachments(conversation_id: str) -> list[AttachmentRecord]`
- `save_attachments(records: list[AttachmentRecord]) -> None`
- `begin() -> None`
- `commit() -> None`
- `rollback() -> None`

### Backend Implementation

**Package**: `polylogue/storage/backends/`

#### SQLiteBackend (`sqlite.py`)

A complete SQLite backend implementation that:

- Implements the `StorageBackend` protocol
- Includes full schema creation (from `storage/db.py`)
- Includes all migrations (v1 → v2, v2 → v3, v3 → v4)
- Provides transaction management (begin/commit/rollback)
- Supports nested transactions via SAVEPOINTs
- Maintains thread-local connection state
- Includes context manager for convenient transaction handling

**Key Features**:

```python
backend = SQLiteBackend(db_path=Path("data.db"))

# Manual transaction control
backend.begin()
backend.save_conversation(conv)
backend.save_messages(messages)
backend.commit()

# Context manager (automatic commit/rollback)
with backend.transaction():
    backend.save_conversation(conv)
    backend.save_messages(messages)
```

#### Backend Factory (`__init__.py`)

Provides `create_backend()` function for backend instantiation:

```python
from polylogue.storage.backends import create_backend

# Use default SQLite backend
backend = create_backend()

# Use specific database path
backend = create_backend(db_path=Path("/path/to/db.sqlite"))

# Future: Use configuration to select backend type
# backend = create_backend(config)  # Might return PostgreSQLBackend based on config
```

### Repository Integration

**File**: `polylogue/storage/repository.py`

Updated `StorageRepository` to support both backend abstraction and legacy modes:

```python
# New: Use backend abstraction
backend = SQLiteBackend(db_path=Path("data.db"))
repository = StorageRepository(backend=backend)

# Legacy: Direct SQLite operations (backward compatible)
repository = StorageRepository()  # No backend parameter
```

**Dual Mode Operation**:

- **Backend Mode**: When `backend` parameter is provided, all operations delegate to the backend
- **Legacy Mode**: When `backend=None`, uses existing `store.py` functions and `connection_context`

This design ensures:
- Zero breaking changes to existing code
- Gradual migration path for callers
- Full test coverage for both modes

## Backward Compatibility

**Guaranteed Compatibility**:

1. **Existing `store_records()` function**: Still works unchanged
2. **Existing `StorageRepository()` initialization**: Works without backend parameter
3. **All existing tests pass**: 38 storage-related tests verified
4. **Module exports**: All existing exports in `storage/__init__.py` preserved

**Migration Path**:

```python
# Phase 1: Current code (no changes needed)
from polylogue.storage import StorageRepository
repository = StorageRepository()

# Phase 2: Opt-in backend usage
from polylogue.storage.backends import create_backend
backend = create_backend()
repository = StorageRepository(backend=backend)

# Phase 3: Switch backend type (future)
from polylogue.storage.backends import create_backend
backend = create_backend(config)  # Returns PostgreSQLBackend if configured
repository = StorageRepository(backend=backend)
```

## Testing

**Test Coverage**: 13 new tests across 3 test files

### Test Files

1. **`tests/test_backend_sqlite.py`** (7 tests)
   - Basic CRUD operations (conversations, messages, attachments)
   - List operations with filtering
   - Transaction management (begin/commit/rollback)
   - Context manager usage
   - Exception handling with rollback

2. **`tests/test_repository_backend.py`** (4 tests)
   - Repository integration with backend
   - Deduplication behavior
   - Attachment handling
   - Backend vs legacy mode compatibility verification

3. **`tests/test_backend_protocol.py`** (2 tests)
   - Protocol conformance verification
   - Method availability checks

### Verification Results

All tests pass:
- 13 new backend tests ✅
- 25 existing db_store tests ✅
- 31 existing db tests ✅
- Type checking passes (mypy) ✅

## File Structure

```
polylogue/
├── protocols.py                    # Extended StorageBackend protocol
├── storage/
│   ├── __init__.py                # Added backend exports
│   ├── repository.py              # Updated with backend support
│   ├── backends/
│   │   ├── __init__.py           # Backend factory
│   │   └── sqlite.py             # SQLiteBackend implementation
│   ├── db.py                     # Unchanged (legacy support)
│   └── store.py                  # Unchanged (legacy support)
└── tests/
    ├── test_backend_sqlite.py    # SQLiteBackend tests
    ├── test_repository_backend.py # Integration tests
    └── test_backend_protocol.py  # Protocol conformance tests
```

## Implementation Details

### Schema and Migrations

The `SQLiteBackend` includes complete schema management:

- **Schema Version**: 4 (matching existing db.py)
- **Tables**: conversations, messages, attachments, attachment_refs, runs
- **Migrations**: All existing migrations (v1→v2, v2→v3, v3→v4) included
- **Indexes**: All performance indexes maintained

### Transaction Semantics

**Nested Transactions**:
- First `begin()` → SQLite `BEGIN`
- Subsequent `begin()` → SQLite `SAVEPOINT sp_N`
- Inner `commit()` → `RELEASE SAVEPOINT`
- Outer `commit()` → `COMMIT`
- Any `rollback()` → Rolls back to matching savepoint/transaction

**Thread Safety**:
- Each backend instance maintains a single connection
- Not thread-safe across instances (use one backend per thread)
- Repository maintains write lock for thread-safe operations

### Type Safety

All backend methods are fully typed:
- Uses proper NewTypes (`ConversationId`, `MessageId`, `AttachmentId`)
- Protocol is `@runtime_checkable`
- Passes mypy strict type checking

## Future Extensions

This abstraction layer enables:

1. **PostgreSQL Backend** (`backends/postgresql.py`)
   - Same protocol interface
   - Different schema DDL
   - Connection pooling support

2. **DuckDB Backend** (`backends/duckdb.py`)
   - Columnar storage for analytics
   - Same protocol interface
   - Optimized for read-heavy workloads

3. **Configuration-Based Selection**
   ```python
   # In config.py
   backend_type: Literal["sqlite", "postgresql", "duckdb"] = "sqlite"

   # Factory automatically selects
   backend = create_backend(config)
   ```

4. **Testing with In-Memory Backends**
   - Fast test execution
   - Isolated test environments
   - Mock backends for integration testing

## Design Principles

1. **Protocol-Oriented Design**: Use Python protocols for interface contracts
2. **Backward Compatibility**: Never break existing code
3. **Gradual Migration**: Support both old and new patterns simultaneously
4. **Type Safety**: Full mypy coverage with strict checking
5. **Test Coverage**: Comprehensive tests for all code paths
6. **Transaction Safety**: Proper ACID semantics with rollback support

## Summary

This implementation successfully abstracts the storage layer while maintaining full backward compatibility. The design allows for:

- **Immediate value**: Better separation of concerns, clearer architecture
- **Future flexibility**: Easy to add new database backends
- **Zero disruption**: Existing code continues to work unchanged
- **Gradual adoption**: New code can opt-in to backend abstraction

All tests pass and type checking is clean, confirming the implementation is production-ready.
