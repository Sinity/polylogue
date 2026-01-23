# Performance Characteristics

This document describes the performance characteristics of Polylogue, benchmark results, and optimization strategies.

## Benchmark Results (Baseline)

### System Information

- **Date**: 2026-01-23
- **Python**: 3.13.11
- **Platform**: NixOS 26.05 (Linux 6.12.63)
- **CPU**: Intel i7-13700K (16 cores, 24 threads)

### Hashing Operations

Content hashing is a core operation used for deduplication and idempotency.

| Operation | Throughput | Latency (per op) |
|-----------|------------|------------------|
| Small text (50 chars, 10k ops) | 3.0M ops/sec | 0.3 µs |
| Medium text (5KB, 1k ops) | 422K ops/sec | 2.4 µs |
| Large text (50KB, 100 ops) | 48K ops/sec | 21 µs |
| Conversation hash (50 msgs, 100 ops) | 22K ops/sec | 45 µs |

**Key Finding**: Unicode normalization (NFC) adds ~811% overhead to hashing operations. This is acceptable because:
1. It prevents duplicates from visually identical text with different Unicode compositions
2. Normalization is only applied once during ingestion
3. The absolute overhead is small (< 1µs for typical messages)

### Search Operations

#### FTS5 Query Escaping

Query escaping is fast and not a bottleneck (>1M ops/sec for all patterns).

| Pattern | Throughput |
|---------|------------|
| Simple query | 1.97M ops/sec |
| Query with operators (AND/OR) | 1.40M ops/sec |
| Special characters | 2.46M ops/sec |
| Quoted query | 2.72M ops/sec |

#### FTS5 Index Building

| Messages | Create Time | Build Time | Total | Throughput |
|----------|-------------|------------|-------|------------|
| 100 | 3.13ms | 1.36ms | 4.49ms | 22K msg/sec |
| 1,000 | 3.33ms | 3.45ms | 6.78ms | 148K msg/sec |
| 5,000 | 3.31ms | 15.46ms | 18.78ms | 266K msg/sec |

**Key Finding**: Index building scales well. Throughput *increases* with larger batches due to amortized overhead.

#### FTS5 Search Performance

| Query Type | 100 Iterations | Throughput |
|------------|----------------|------------|
| Common keyword (many results) | 549ms | 182 ops/sec |
| Multi-word query | 662ms | 151 ops/sec |
| Rare keyword (few results) | 30ms | 3,301 ops/sec |

**Key Finding**: Search performance is heavily dependent on result set size. Queries returning fewer results are ~18x faster.

#### Incremental Index Updates

| Operation | Time |
|-----------|------|
| Single conversation | 1.98ms |
| 5 conversations | 3.64ms |

### Pipeline Operations

#### Ingestion

| Operation | Time | Notes |
|-----------|------|-------|
| Prepare ingest (cold) | 4.49ms | First insert with hashing |
| Prepare ingest (warm) | 0.73ms | Skipped (content hash match) |

**Key Finding**: Content hash checking provides 6.1x speedup when re-importing unchanged conversations.

#### Parallel Processing

| Configuration | Time | Throughput | Speedup |
|---------------|------|------------|---------|
| Sequential (20 convs) | 137.81ms | 145 convs/sec | 1.0x |
| Parallel 4 workers (20 convs) | 47.89ms | 418 convs/sec | 2.88x |

**Key Finding**: Parallel ingestion with 4 workers achieves 2.88x speedup on a 16-core system. This is reasonable given:
1. SQLite write serialization under `_WRITE_LOCK`
2. I/O-bound operations (file reading, parsing)
3. Thread pool overhead

#### Bounded Submission

| Configuration | Time | Overhead |
|---------------|------|----------|
| Unbounded (50 convs) | 108.88ms | baseline |
| Bounded (16 max in-flight) | 107.86ms | -0.94% |

**Key Finding**: Bounded submission (max 16 in-flight futures) has negligible overhead while preventing memory explosion.

## Performance Characteristics by Operation

### What's Fast ✅

1. **Content Hash Checking** (0.73ms warm path)
   - Prevents redundant work during re-imports
   - SQLite index lookups are very efficient

2. **Query Escaping** (>1M ops/sec)
   - Not a bottleneck even with complex patterns
   - Hardened against injection attacks

3. **Parallel Ingestion** (2.88x speedup)
   - Effective use of multi-core systems
   - Hashing and parsing parallelized
   - Only writes are serialized

4. **FTS5 Index Building** (266K msg/sec for large batches)
   - Scales well with dataset size
   - Incremental updates are fast (2ms per conversation)

### What's Slow ⚠️

1. **FTS5 Search with Large Result Sets** (182 ops/sec)
   - 18x slower than queries returning few results
   - Root cause: Deduplication and snippet generation in Python
   - Workaround: Limit result set size, use more specific queries

2. **Unicode Normalization** (811% overhead)
   - Required for correctness (prevents duplicates)
   - Applied once during ingestion
   - Not a bottleneck in practice

## Optimization Strategies

### Implemented Optimizations

1. **Content-Hash Deduplication**
   - SHA-256 with NFC normalization
   - Prevents re-ingesting unchanged conversations
   - 6.1x speedup on warm path

2. **Parallel Ingestion**
   - ThreadPoolExecutor with max_workers=4
   - Bounded submission (max 16 in-flight)
   - 2.88x speedup over sequential

3. **Incremental FTS5 Updates**
   - Only updates changed conversations
   - Chunked batches of 200 to avoid SQLite limits
   - 2ms per conversation vs full rebuild

4. **Generated Column Index (v4 schema)**
   - `source_name` generated from JSON
   - Enables indexed filtering without JSON extraction
   - Significant speedup for provider/source filters

### Potential Future Optimizations

#### 1. Search Result Caching

**Observation**: Common queries (e.g., "python", "error") are likely repeated.

**Strategy**: LRU cache for search results keyed by (query, source, since, limit).

**Expected Impact**:
- Cache hit: ~1ms (vs 549ms for large result sets)
- 500x+ speedup for cached queries
- Minimal memory overhead (cache ~100 recent queries)

**Implementation**:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def search_messages_cached(query, source, since, limit):
    return search_messages(query, source=source, since=since, limit=limit)
```

**Trade-offs**:
- Stale results until cache invalidation
- Cache invalidation needed on re-ingest
- Memory usage (small, ~10MB for 100 entries)

#### 2. Projection Result Caching

**Observation**: Conversation projections (`.substantive()`, `.dialogue()`) are deterministic.

**Strategy**: Cache projected message lists per conversation.

**Expected Impact**:
- Cache hit: <1ms (vs 10-50ms for complex projections)
- 10-50x speedup for repeated projections
- Moderate memory overhead

**Implementation**:
```python
@property
def _projection_cache_key(self):
    return self.content_hash

@lru_cache(maxsize=1000)
def _get_projected_messages(cache_key, projection_type):
    # Compute projection
    pass
```

**Trade-offs**:
- Memory usage (1000 conversations × ~50 messages × metadata)
- Cache invalidation on content changes

#### 3. Parallel Rendering

**Observation**: Current rendering is parallel (4 workers) but could be optimized further.

**Strategy**:
- Increase worker pool to match CPU cores (16 on test system)
- Use process pool instead of thread pool (avoid GIL)

**Expected Impact**:
- Potential 2-4x additional speedup (up to ~8x total vs sequential)
- Better CPU utilization

**Trade-offs**:
- Higher memory usage (multiple Python processes)
- Increased context switching overhead
- Diminishing returns beyond CPU count

**Status**: Not implemented. Current 4-worker configuration is reasonable for most use cases.

#### 4. Prepared Statements for Repeated Queries

**Observation**: Same SQL queries executed repeatedly during ingestion.

**Strategy**: Use SQLite prepared statements for hot paths.

**Expected Impact**:
- 5-10% speedup on ingestion
- Reduced SQL parsing overhead

**Trade-offs**:
- More complex code
- Marginal benefit (SQLite query cache is already effective)

**Status**: Not implemented. Current performance is acceptable.

#### 5. Batch FTS5 Insertions

**Observation**: FTS5 updates are already batched (chunks of 200).

**Strategy**: Optimize chunk size based on profiling.

**Expected Impact**:
- Potentially 10-20% faster indexing
- Needs empirical testing

**Trade-offs**:
- SQLite has query size limits (must stay < 1MB)
- Diminishing returns beyond certain batch size

**Status**: Current batch size (200) is a reasonable default.

## Performance Best Practices

### For Users

1. **Re-import Strategy**
   - Polylogue's content-hash deduplication makes re-imports very cheap (6.1x faster)
   - Safe to re-run `polylogue run` frequently
   - Unchanged conversations are skipped automatically

2. **Search Strategy**
   - Use specific queries to reduce result set size
   - Combine keywords to narrow results
   - Use `--source` filter to limit scope

3. **Incremental Indexing**
   - Run `polylogue run` (ingests + indexes changed conversations)
   - Faster than `polylogue index --rebuild` (full rebuild)
   - Full rebuild only needed after schema changes

4. **Source Filtering**
   - Use `--source` filter to process specific providers
   - Reduces ingestion time proportionally
   - Leverages indexed `source_name` column (v4)

### For Developers

1. **Hashing**
   - Use `hash_text()` for content hashing (includes NFC normalization)
   - Don't bypass normalization (causes duplicates)
   - Hashing is fast enough for hot paths (~3M ops/sec for small text)

2. **Database Access**
   - Always use `_WRITE_LOCK` for writes
   - Use thread-local connections via `connection_context()`
   - Batch operations when possible (e.g., FTS5 updates)

3. **Parallelization**
   - Parallelize pure operations (hashing, parsing)
   - Serialize writes under `_WRITE_LOCK`
   - Use bounded submission (max 16 in-flight) to prevent memory explosion

4. **FTS5**
   - Use `escape_fts5_query()` for all user input (prevents injection)
   - Prefer incremental updates (`update_index_for_conversations()`)
   - Batch updates in chunks of 200

5. **Memory Management**
   - Clear large objects after use
   - Use iterators instead of lists for large datasets
   - Bounded submission prevents unbounded future accumulation

## Benchmark Maintenance

### Running Benchmarks

```bash
# Run all benchmarks
QDRANT_URL="" uv run python tests/benchmarks/run_all.py

# Run specific benchmark
QDRANT_URL="" uv run python tests/benchmarks/benchmark_hashing.py
QDRANT_URL="" uv run python tests/benchmarks/benchmark_search.py
QDRANT_URL="" uv run python tests/benchmarks/benchmark_pipeline.py
```

**Note**: Set `QDRANT_URL=""` to disable Qdrant vector search during benchmarks (avoids numpy dependency issues).

### Benchmark Structure

- `tests/benchmarks/benchmark_hashing.py` - Content hashing operations
- `tests/benchmarks/benchmark_search.py` - FTS5 indexing and search
- `tests/benchmarks/benchmark_pipeline.py` - Ingestion and parallel processing
- `tests/benchmarks/run_all.py` - Master runner, generates JSON report

### Adding New Benchmarks

1. Create a new function in the appropriate benchmark file
2. Return a dictionary with results (time_ms, ops_per_sec, etc.)
3. Add to the `run_all()` function
4. Document in this file

## Profiling

For detailed profiling, use Python's built-in profilers:

```bash
# cProfile (deterministic profiling)
python -m cProfile -o profile.out -m polylogue.cli run --preview

# View results
python -m pstats profile.out
>>> sort cumtime
>>> stats 20

# Or use snakeviz for visual profiling
uv add --dev snakeviz
snakeviz profile.out
```

## Version History

### Baseline (Current)

- Benchmarking infrastructure established
- Baseline measurements documented
- Optimization opportunities identified

### Future

- Search result caching (pending)
- Projection result caching (pending)
- Process-based parallel rendering (deferred)
