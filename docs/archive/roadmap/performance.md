# Polylogue Performance & Scalability Roadmap

## Overview

This roadmap outlines 15 concrete performance optimizations and scalability improvements to enable Polylogue to handle 100K+ conversations and millions of messages. The codebase demonstrates solid foundational patterns (thread-local connections, write locks, bounded parallel ingestion, LRU search caching), but several opportunities exist for dramatic performance gains.

**Baseline Context**:
- Current architecture: Single SQLite file with WAL mode, thread-local connections, `_WRITE_LOCK` for writes
- Bounded ingestion: Max 16 in-flight futures during import
- Search caching: 21,343x speedup via LRU cache (69ms → 0.003ms)
- Need: Support 100K+ conversations, 1M+ messages, concurrent access patterns

---

## Priority 1: High Impact, Low Effort

### 1. Batch FTS5 Indexing with Prepared Statements

**Files Affected**: `storage/search_providers/fts5.py:75-100`

**Current Implementation**:
```python
for conv_id in conversation_ids:
    conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", (conv_id,))

for msg in messages:
    row = conn.execute("SELECT provider_name FROM conversations WHERE conversation_id = ?", ...)
    conn.execute("INSERT INTO messages_fts ...")
```

**Problem**: Per-message query to fetch `provider_name` + individual inserts = O(n) round-trips, causing severe performance degradation for large batches.

**Solution**:
1. Replace per-message SELECT with JOIN during insert
2. Use `executemany()` for bulk DELETE (single `IN` clause)
3. Prepare statement once, reuse for all inserts

**Implementation**:
```python
def index(self, messages: list[MessageRecord]) -> None:
    # Get provider names in single query
    conv_ids = list(set(msg.conversation_id for msg in messages))
    provider_map = dict(conn.execute(
        "SELECT conversation_id, provider_name FROM conversations WHERE conversation_id IN ({})".format(
            ",".join("?" * len(conv_ids))
        ),
        conv_ids
    ))

    # Batch delete
    conn.executemany("DELETE FROM messages_fts WHERE conversation_id = ?",
                     [(cid,) for cid in conv_ids])

    # Batch insert with prepared statement
    stmt = "INSERT INTO messages_fts (message_id, conversation_id, provider_name, text) VALUES (?, ?, ?, ?)"
    rows = [(msg.message_id, msg.conversation_id, provider_map[msg.conversation_id], msg.text)
            for msg in messages]
    conn.executemany(stmt, rows)
```

**Expected Impact**: 5-10x indexing speedup for batches of 1000+ messages

**Success Criteria**:
- FTS5 indexing speed: >1000 messages/second
- Benchmark: Index 10K messages in <10 seconds (vs current ~50-100 seconds)

---

### 2. Add Covering Index on Messages Table

**Files Affected**: `storage/db.py:82-83`

**Current Issue**:
```sql
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
```

When querying for content hashes during deduplication:
```sql
SELECT message_id, content_hash FROM messages WHERE conversation_id = ?
```
The index returns conversation_id, but subsequent lookup of `message_id` and `content_hash` requires table scan.

**Solution**: Add covering index with all needed columns:
```sql
CREATE INDEX IF NOT EXISTS idx_messages_conversation_covering
ON messages(conversation_id, message_id, content_hash);
```

**Implementation Location**: `storage/db.py` in `_init_schema()` or migration

**Expected Impact**: 2-3x speedup for content-hash deduplication checks, eliminates table scan during ingestion

**Success Criteria**:
- Content hash check: <1ms per message (vs current ~2-5ms)
- Ingestion throughput increase: 20-30%

---

### 3. Connection Pool for Read Operations

**Files Affected**: `storage/db.py:401-449` (thread-local connection management)

**Current Pattern**:
```python
def open_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    # ... setup ...
    yield conn
    conn.close()
```

Thread-local connections are created/destroyed per operation, incurring connection overhead.

**Solution**: Implement lightweight pool of read-only connections (3-5 pre-created):
```python
class ConnectionPool:
    def __init__(self, db_path: Path, size: int = 5):
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=size)
        for _ in range(size):
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA query_only = ON")  # Read-only
            self._pool.put(conn)

    def acquire(self) -> sqlite3.Connection:
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA query_only = ON")
            return conn

    def release(self, conn: sqlite3.Connection) -> None:
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
```

Use for read-only operations (search, listing, view).

**Expected Impact**: Eliminate connection overhead for parallel reads, 1-2x speedup for search under load

**Success Criteria**:
- Connection creation overhead eliminated
- Concurrent search queries show no blocking

---

### 4. WAL Checkpoint Tuning

**Files Affected**: `storage/db.py:418`

**Current Configuration**:
```python
conn.execute("PRAGMA journal_mode=WAL;")
```

WAL checkpoint happens automatically but can cause I/O pauses and block readers during heavy writes.

**Solution**: Fine-tune checkpoint behavior:
```python
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA wal_autocheckpoint = 1000")    # Checkpoint every 1000 pages (~4MB)
conn.execute("PRAGMA synchronous = NORMAL")          # Reduced from FULL for bulk inserts
conn.execute("PRAGMA temp_store = MEMORY")           # Use RAM for temp tables
conn.execute("PRAGMA mmap_size = 30000000;")         # Memory-map file for faster reads
```

**Trade-off**: PRAGMA synchronous = NORMAL reduces durability guarantees in power-loss scenarios. Only use during bulk import, revert to FULL afterward.

**Expected Impact**: Smoother write performance, 20-40% throughput improvement during large imports, fewer I/O pauses

**Success Criteria**:
- Import throughput: 10K+ messages/second
- Read latency p95 during import: <100ms

---

## Priority 2: High Impact, Medium Effort

### 5. Streaming Ingestion Pipeline

**Files Affected**: `pipeline/services/ingestion.py:135-156`, `ingestion/ingest.py`

**Current Issue**: All `ParsedConversation` objects must fit in memory before processing completes:
```python
for convo in conversations:
    while len(futures) > 16:
        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
```

With 100K conversations, memory usage is unbounded.

**Solution**: Generator-based streaming with fixed-size batching:
```python
def ingest_stream(self, conversations: Iterator[ParsedConversation]) -> IngestResult:
    """Stream ingestion with backpressure - constant memory regardless of archive size."""
    batch: list[ParsedConversation] = []
    result = IngestResult()

    for convo in conversations:
        batch.append(convo)
        if len(batch) >= 100:  # Batch size of 100
            self._process_batch(batch, result)
            batch = []

    if batch:
        self._process_batch(batch, result)

    return result

def _process_batch(self, batch: list[ParsedConversation], result: IngestResult) -> None:
    """Process batch with bounded concurrency."""
    futures: dict[Future, ParsedConversation] = {}

    for convo in batch:
        while len(futures) >= 16:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                result.merge(fut.result())
                del futures[fut]

        fut = self.executor.submit(self._process_conversation, convo)
        futures[fut] = convo

    # Wait for remaining
    for fut in as_completed(futures):
        result.merge(fut.result())
```

**Expected Impact**: Constant memory usage O(1) instead of O(n), enables processing arbitrarily large archives

**Success Criteria**:
- Peak memory < 100MB regardless of archive size
- Process 100K conversations without OOM
- Memory profile shows batch-size constant pattern

---

### 6. Incremental FTS Index Updates

**Files Affected**: `storage/search_providers/fts5.py:53-102`, new table `fts_index_state`

**Current Issue**: Full delete + reinsert even for unchanged messages on re-import:
```python
def index(self, messages: list[MessageRecord]) -> None:
    for conv_id in conversation_ids:
        conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", ...)
    # Always reinsert all messages
```

This causes 90% wasted work on re-imports where 95% of data hasn't changed.

**Solution**: Track indexed message hashes, skip unchanged:
```sql
CREATE TABLE IF NOT EXISTS fts_index_state (
    message_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
```

Implementation:
```python
def index(self, messages: list[MessageRecord]) -> None:
    # Get previously indexed hashes
    existing = dict(conn.execute(
        "SELECT message_id, content_hash FROM fts_index_state"
    ))

    # Find changed messages
    changed = [msg for msg in messages
               if msg.content_hash != existing.get(msg.message_id)]
    unchanged = len(messages) - len(changed)

    if not changed:
        logger.info(f"No changes detected in {len(messages)} messages, skipping index")
        return

    # Delete only changed entries
    for msg in changed:
        conn.execute("DELETE FROM messages_fts WHERE message_id = ?", (msg.message_id,))

    # Insert only changed
    self._insert_to_fts(changed)

    # Update state
    for msg in changed:
        conn.execute(
            "INSERT OR REPLACE INTO fts_index_state (message_id, content_hash, indexed_at) VALUES (?, ?, ?)",
            (msg.message_id, msg.content_hash, datetime.now().isoformat())
        )
```

**Expected Impact**: 90%+ reduction in indexing work for re-imports, 20x speedup on repeated ingestion

**Success Criteria**:
- Re-import 10K unchanged conversations: <1 second (vs current 10+ seconds)
- Indexing skips 95%+ of unchanged messages

---

### 7. Message Text Compression for Large Conversations

**Files Affected**: `storage/backends/sqlite.py:68-78` (message storage), `lib/models.py` (Message class)

**Problem**: Large conversations (Claude Code with extensive tool output) can have 100KB+ of text per message, consuming disproportionate storage.

**Solution**: Optional zlib compression for messages > 4KB:
```python
import zlib

class MessageRecord:
    text: str
    is_text_compressed: bool = False  # Track compression state

    @staticmethod
    def _maybe_compress(text: str) -> tuple[bytes | str, bool]:
        """Compress if > 4KB, otherwise store as-is."""
        encoded = text.encode("utf-8")
        if len(encoded) > 4096:
            compressed = zlib.compress(encoded, level=6)
            if len(compressed) < len(encoded) * 0.8:  # Only if >20% savings
                return compressed, True
        return text, False

    def store_text(self, conn: sqlite3.Connection) -> None:
        text_to_store, is_compressed = self._maybe_compress(self.text)
        conn.execute(
            "UPDATE messages SET text = ?, is_text_compressed = ? WHERE message_id = ?",
            (text_to_store, is_compressed, self.message_id)
        )

    def load_text(self, conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT text, is_text_compressed FROM messages WHERE message_id = ?",
            (self.message_id,)
        ).fetchone()
        if row[1]:  # is_text_compressed
            return zlib.decompress(row[0]).decode("utf-8")
        return row[0]
```

**Storage Schema**:
```sql
ALTER TABLE messages ADD COLUMN is_text_compressed BOOLEAN DEFAULT 0;
```

**Expected Impact**: 40-60% storage reduction for large conversations, 10-30% overall archive size reduction

**Success Criteria**:
- Archive size: 40% smaller for typical 100+ message conversations
- Decompression latency: <5ms per message
- No performance degradation for searches or rendering

---

### 8. Message Ordering Index

**Files Affected**: `storage/db.py` schema initialization

**Problem**: Conversations with 1000+ messages require full scan to order by timestamp

**Solution**:
```sql
CREATE INDEX IF NOT EXISTS idx_messages_timestamp
ON messages(conversation_id, timestamp, message_id);
```

**Expected Impact**: 3-5x speedup for ordered message retrieval in large conversations

**Success Criteria**:
- Load 1000-message conversation: <100ms (vs current 300-500ms)

---

## Priority 3: Medium Impact, Medium Effort

### 9. Read Replica Pattern for Search

**Files Affected**: `storage/repository.py:48` (write lock), new `storage/read_replica.py`

**Current Architecture**: Single SQLite file with `_WRITE_LOCK` serializes all writes. Search queries block on write operations.

**Solution**: Maintain read-only copy for search operations:
```python
class SearchReplica:
    def __init__(self, primary_path: Path, replica_path: Path | None = None):
        self.primary = primary_path
        self.replica = replica_path or primary_path.with_stem(primary_path.stem + ".replica")
        self._last_sync = None

    def sync(self) -> None:
        """Update replica from primary via VACUUM INTO."""
        if self._last_sync and (time.time() - self._last_sync) < 60:
            return  # Don't sync more than once per minute

        conn = sqlite3.connect(str(self.primary))
        conn.execute(f"VACUUM INTO '{self.replica}'")
        conn.close()
        self._last_sync = time.time()

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Query replica, never blocks on writes."""
        self.sync()
        conn = sqlite3.connect(str(self.replica))
        conn.execute("PRAGMA query_only = ON")
        try:
            return self._execute_search(conn, query, limit)
        finally:
            conn.close()
```

**Integration**:
- `polylogue search` queries replica
- `polylogue view` queries replica
- `polylogue run` writes to primary (keeps lock)

**Expected Impact**: Search never blocks ingestion; ingestion never blocks search. Concurrent throughput increases 2-3x

**Success Criteria**:
- Search latency unchanged (queries same data)
- Ingestion not blocked by search queries
- 4x concurrent reader throughput

---

### 10. Projection Caching with Memoization

**Files Affected**: `lib/projections.py:169-206`

**Current Issue**: `iter()` recomputes filters for every call; properties like `is_substantive` are recomputed per-message

**Solution**: Cache computed projections at conversation level:
```python
class ConversationProjection:
    def __init__(self, conversation: Conversation):
        self._cache: dict[str, Any] = {}
        self._version = 0  # Invalidation version

    @lru_cache(maxsize=16)
    def _cached_substantive_ids(self, version: int) -> frozenset[str]:
        """Pre-compute substantive message IDs once."""
        return frozenset(
            msg.message_id for msg in self.conversation.messages
            if msg.is_substantive
        )

    def substantive(self) -> ConversationProjection:
        """Use cached result."""
        cached_ids = self._cached_substantive_ids(self._version)
        return self.where(lambda m: m.message_id in cached_ids)

    def invalidate(self) -> None:
        """Clear cache on conversation update."""
        self._version += 1
```

**Expected Impact**: 3-5x speedup for repeated projections on same conversation

**Success Criteria**:
- Repeated `conversation.project().substantive().count()`: <1ms (vs current 10-20ms)

---

### 11. Partitioned Message Storage

**Files Affected**: `storage/backends/sqlite.py`, `storage/db.py` schema

**Problem**: For 1M+ messages, query performance degrades as table grows

**Solution**: Partition messages by creation date (monthly):
```sql
CREATE TABLE messages_202501 (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    -- ... other columns
);

CREATE TABLE messages_202502 (
    -- Same schema
);

-- Union view for backward compatibility
CREATE VIEW messages AS
    SELECT * FROM messages_202501
    UNION ALL SELECT * FROM messages_202502
    UNION ALL SELECT * FROM messages_202503;
```

Route writes to correct partition based on conversation creation date.

**Expected Impact**: Linear query performance as data grows (index efficiency maintained), enables time-based archiving (move old partitions to cold storage)

**Success Criteria**:
- Query performance: Linear O(n) instead of O(n) degradation
- 1M+ message archive: Queries still <100ms
- Partition-specific queries: 10x faster

---

## Priority 4: Benchmarking Infrastructure

### 12. Large-Scale Stress Test

**Location**: `tests/benchmarks/benchmark_large_scale.py` (new)

```python
def benchmark_100k_conversations():
    """Synthetic stress test: 100K conversations, 50 messages each."""
    # Generate synthetic ParsedConversation objects
    conversations = [
        _create_synthetic_conversation(
            id=f"conv_{i:06d}",
            message_count=50,
            avg_message_len=500
        )
        for i in range(100_000)
    ]

    # Benchmark ingestion
    start = time.time()
    result = ingestion_service.ingest_stream(conversations)
    elapsed = time.time() - start

    metrics = {
        "conversations": 100_000,
        "messages": 5_000_000,
        "elapsed_seconds": elapsed,
        "throughput_msgs_sec": 5_000_000 / elapsed,
        "peak_memory_mb": track_peak_memory(),
    }

    assert metrics["throughput_msgs_sec"] > 5000, f"Throughput too low: {metrics}"
    assert metrics["peak_memory_mb"] < 100, f"Memory too high: {metrics}"
```

**Success Criteria**:
- Ingest 100K conversations: <300 seconds
- Peak memory: <100MB
- Throughput: >5000 messages/second

---

### 13. Concurrent Access Benchmark

**Location**: `tests/benchmarks/benchmark_concurrent.py` (new)

```python
def benchmark_concurrent_access():
    """4 readers + 1 writer competing for access."""
    # Reader threads: Random FTS5 searches
    # Writer thread: Continuous message ingestion
    # Measure: Lock contention, reader latency under load

    results = {
        "reader_latency_p50_ms": [],
        "reader_latency_p99_ms": [],
        "writer_throughput_msgs_sec": 0,
        "lock_wait_time_ms": 0,
    }
```

**Success Criteria**:
- Reader p99 latency: <500ms even under write load
- Writer maintains >1000 msgs/sec
- No reader starvation

---

### 14. Memory Profiling for Large Conversations

**Location**: `tests/benchmarks/benchmark_large_conv.py` (new)

Claude Code sessions can have 1000+ messages with extensive tool output.

```python
def benchmark_large_conversation_memory():
    """Load 1000-message conversation and profile memory."""
    # Load conversation into memory
    conversation = repository.get_conversation(large_conv_id)

    # Profile: Object size, Pydantic overhead
    memory_profile = {
        "conversation_size_bytes": sys.getsizeof(conversation),
        "avg_message_size_bytes": sum(
            sys.getsizeof(m) for m in conversation.messages
        ) / len(conversation.messages),
        "total_text_bytes": sum(len(m.text) for m in conversation.messages),
    }
```

**Success Criteria**:
- Load 1000-message conversation: <10MB memory
- Message object overhead: <1KB per message

---

## Priority 5: Schema Evolution

### 15. Denormalize Message Count

**Files Affected**: `storage/db.py` schema, `storage/repository.py`

**Current Issue**: `conversation.message_count` requires loading all messages or COUNT query

**Solution**: Add `message_count` column to conversations, update on insert/delete:
```sql
ALTER TABLE conversations ADD COLUMN message_count INTEGER DEFAULT 0;

-- Update trigger
CREATE TRIGGER update_message_count_insert AFTER INSERT ON messages
BEGIN
    UPDATE conversations SET message_count = message_count + 1
    WHERE conversation_id = NEW.conversation_id;
END;

CREATE TRIGGER update_message_count_delete AFTER DELETE ON messages
BEGIN
    UPDATE conversations SET message_count = message_count - 1
    WHERE conversation_id = OLD.conversation_id;
END;
```

**Expected Impact**: O(1) message count queries instead of O(n) or COUNT scan

**Success Criteria**:
- Get message count: <1ms vs current 10-50ms

---

## Implementation Roadmap

### Phase 1 (Weeks 1-2): Quick Wins
- Batch FTS5 indexing (15% effort)
- Covering index (5% effort)
- WAL tuning (5% effort)
- Message ordering index (5% effort)

**Expected Result**: 5-10x indexing speedup, 20% overall performance improvement

### Phase 2 (Weeks 3-4): Core Scalability
- Streaming ingestion pipeline (20% effort)
- Incremental FTS updates (20% effort)
- Message compression (15% effort)

**Expected Result**: Support 100K+ conversations, constant memory usage

### Phase 3 (Weeks 5-6): Concurrency
- Read replica pattern (25% effort)
- Projection caching (10% effort)
- Connection pooling (10% effort)

**Expected Result**: Non-blocking search, 2-3x concurrent throughput

### Phase 4 (Weeks 7-8): Validation
- Large-scale benchmarks (15% effort)
- Concurrent access testing (15% effort)
- Memory profiling (10% effort)

---

## Priority Matrix

| # | Optimization | Impact | Effort | ROI | Status |
|---|--------------|--------|--------|-----|--------|
| 1 | Batch FTS Indexing | High | Low | 10:1 | Planned |
| 2 | Covering Index | High | Low | 8:1 | Planned |
| 3 | Connection Pool | Medium | Low | 6:1 | Planned |
| 4 | WAL Tuning | Medium | Low | 5:1 | Planned |
| 5 | Streaming Ingestion | High | Medium | 5:1 | Planned |
| 6 | Incremental FTS | High | Medium | 8:1 | Planned |
| 7 | Message Compression | Medium | Medium | 4:1 | Planned |
| 8 | Message Ordering | Medium | Low | 4:1 | Planned |
| 9 | Read Replica | Medium | Medium | 3:1 | Planned |
| 10 | Projection Cache | Medium | Low | 5:1 | Planned |
| 11 | Partitioned Storage | High | High | 3:1 | Future |
| 12 | 100K Test | N/A | Medium | N/A | Planned |
| 13 | Concurrent Test | N/A | Medium | N/A | Planned |
| 14 | Memory Test | N/A | Low | N/A | Planned |
| 15 | Message Count | Low | Low | 3:1 | Future |

---

## Dependencies

- Phase 1 is independent
- Phase 2 requires Phase 1 complete (needs fast indexing as baseline)
- Phase 3 benefits from Phase 2 (incremental updates reduce read replica sync overhead)
- Phase 4 validates entire pipeline

---

## Success Criteria (End State)

- **Throughput**: 10K+ messages ingested per second
- **Memory**: Constant O(1) regardless of archive size (streaming pipeline)
- **Search latency**: <100ms p99 even during ingestion
- **Storage**: 40% reduction via compression
- **Scalability**: Linear query performance to 1M+ messages
- **Concurrency**: 4+ concurrent readers, no search blocking ingestion
