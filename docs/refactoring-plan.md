# Polylogue Architecture Refactoring Plan

**Status**: Draft
**Created**: 2026-01-22
**Estimated Total Effort**: 6-8 weeks

This document tracks architectural improvements to enhance modularity, testability, and extensibility of Polylogue.

---

## Quick Reference

- 🟢 **Priority 1 (High Impact, Low Effort)**: ~1 week total
- 🟡 **Priority 2 (Medium Impact, Medium Effort)**: ~2-3 weeks total
- 🔴 **Priority 3 (High Impact, High Effort)**: ~3-4 weeks total

**Progress**: 55/200+ tasks completed (Priority 1: Sections 1.1, 1.2, 1.3, 1.4 complete - PRIORITY 1 COMPLETE ✅)

---

## Priority 1: High Impact, Low Effort (1 week)

### 1.1 Module Organization (1 day)

**Goal**: Reorganize scattered root modules into coherent packages

#### Storage Package
- [x] Create `polylogue/storage/` package directory
- [x] Move `db.py` → `storage/db.py`
- [x] Move `store.py` → `storage/store.py`
- [x] Move `index.py` → `storage/index.py`
- [x] Move `search.py` → `storage/search.py`
- [x] Move `index_qdrant.py` → `storage/index_qdrant.py`
- [x] Update all imports across codebase (use IDE refactor)
- [x] Update `__init__.py` to re-export public APIs
- [x] Run test suite to verify

#### Rendering Package
- [x] Create `polylogue/rendering/` package directory
- [x] Move `render.py` → `rendering/render.py`
- [x] Move `render_paths.py` → `rendering/paths.py`
- [x] Move `assets.py` → `rendering/assets.py`
- [x] Update all imports across codebase
- [x] Update `__init__.py` to re-export public APIs
- [x] Run test suite to verify

#### Ingestion Package
- [x] Create `polylogue/ingestion/` package directory
- [x] Move `ingest.py` → `ingestion/ingest.py`
- [x] Move `source_ingest.py` → `ingestion/source.py`
- [x] Move `drive_ingest.py` → `ingestion/drive.py`
- [x] Move `drive_client.py` → `ingestion/drive_client.py`
- [x] Update all imports across codebase
- [x] Update `__init__.py` to re-export public APIs
- [x] Run test suite to verify

### 1.2 Storage Repository Encapsulation (2 days) ✅ COMPLETE

**Goal**: Eliminate global `_WRITE_LOCK` by encapsulating storage operations

**Dependencies**: 1.1 (Storage Package)

- [x] Create `storage/repository.py` with `StorageRepository` class
- [x] Move `_WRITE_LOCK` from module-level to `StorageRepository.__init__`
- [x] Add `StorageRepository.save_conversation()` method
- [x] Add `StorageRepository.record_run()` method (combined save_messages/attachments into save_conversation)
- [x] Update `ingestion/ingest.py` to accept repository parameter
- [x] Update `pipeline/runner.py` to instantiate repository once
- [x] Add repository fixture to test suite (conftest.py)
- [x] Verified tests pass with repository pattern (9/9 ingest_render tests, concurrent tests)

### 1.3 Protocol Definitions (1 day) ✅ COMPLETE

**Goal**: Define interfaces for external dependencies to enable mocking and swapping

- [x] Create `polylogue/protocols.py` file
- [x] Define `SearchProvider` protocol with `index()` and `search()` methods
- [x] Define `VectorProvider` protocol with `upsert()` and `query()` methods
- [x] Define `StorageBackend` protocol with CRUD operations
- [x] Define `Renderer` protocol with `render_markdown()` and `render_html()`
- [ ] Add protocol type hints to function signatures in `storage/index.py` (deferred to 2.3)
- [ ] Add protocol type hints to `rendering/render.py` (deferred to 3.3)
- [x] Run mypy to verify protocol compliance

### 1.4 Type Safety Improvements (1 day) ✅ COMPLETE

**Goal**: Replace string IDs with NewType for type safety

**Dependencies**: None

- [x] Create `polylogue/types.py` if not exists
- [x] Define `ConversationId = NewType('ConversationId', str)`
- [x] Define `MessageId = NewType('MessageId', str)`
- [x] Define `AttachmentId = NewType('AttachmentId', str)`
- [x] Define `ContentHash = NewType('ContentHash', str)`
- [x] Update `store.py` to use NewType IDs
- [x] Update `lib/models.py` to use NewType IDs
- [x] Update `pipeline/ids.py` return types
- [x] Update `lib/repository.py` method signatures
- [x] Run mypy with strict mode
- [x] Fix all type errors
- [x] Run test suite

---

## Priority 2: Medium Impact, Medium Effort (2-3 weeks)

### 2.1 Configuration Object Refactoring (3 days)

**Goal**: Replace environment variable checks with config objects

**Dependencies**: 1.1 (Module Organization)

#### Index Configuration
- [ ] Add `IndexConfig` dataclass to `config.py`
- [ ] Add fields: `fts_enabled`, `qdrant_url`, `qdrant_api_key`, `voyage_api_key`
- [ ] Update `Config` class to include `index_config: IndexConfig`
- [ ] Update `storage/index.py` to accept `IndexConfig` parameter
- [ ] Remove `os.environ.get("QDRANT_URL")` check in `index.py:37`
- [ ] Update `pipeline/runner.py` to pass config to index functions
- [ ] Update CLI commands to pass config
- [ ] Update tests to use config fixtures

#### Drive Configuration
- [ ] Add `DriveConfig` dataclass to `config.py`
- [ ] Add fields: `credentials_path`, `token_path`, `retry_count`, `timeout`
- [ ] Update `ingestion/drive_client.py` to accept `DriveConfig`
- [ ] Remove all `os.environ.get()` calls in drive modules
- [ ] Update `pipeline/runner.py` to pass drive config
- [ ] Update tests with drive config fixtures

#### Configuration Migration
- [ ] Add migration logic in `config.py` to read from env vars for backward compatibility
- [ ] Update documentation with new config schema
- [ ] Add deprecation warnings for env var usage
- [ ] Update example config files

### 2.2 Pipeline Service Decomposition (1 week)

**Goal**: Break pipeline god object into focused services with dependency injection

**Dependencies**: 1.2 (Storage Repository), 2.1 (Config Objects)

#### Service Extraction
- [ ] Create `pipeline/services/` package directory
- [ ] Create `pipeline/services/ingestion.py` with `IngestionService` class
- [ ] Create `pipeline/services/indexing.py` with `IndexService` class
- [ ] Create `pipeline/services/rendering.py` with `RenderService` class
- [ ] Move ingestion logic from `runner.py` to `IngestionService`
- [ ] Move indexing logic from `runner.py` to `IndexService`
- [ ] Move rendering logic from `runner.py` to `RenderService`

#### Service Interfaces
- [ ] Define `IngestionService.__init__(storage_repo, drive_client, config)`
- [ ] Define `IndexService.__init__(fts_provider, vector_provider, config)`
- [ ] Define `RenderService.__init__(template_path, output_dir, config)`
- [ ] Add dependency injection to services (accept dependencies in constructor)
- [ ] Update `PipelineRunner` to orchestrate services instead of doing work
- [ ] Reduce `PipelineRunner` imports from 18 to ~5

#### Testing Improvements
- [ ] Create service test fixtures with mock dependencies
- [ ] Add unit tests for `IngestionService` with mock storage
- [ ] Add unit tests for `IndexService` with mock providers
- [ ] Add unit tests for `RenderService` with mock templates
- [ ] Update integration tests to use real services
- [ ] Verify test coverage remains >85%

### 2.3 Search Provider Abstraction (3 days)

**Goal**: Implement strategy pattern for search backends

**Dependencies**: 1.3 (Protocols), 2.1 (Config)

#### Provider Implementations
- [ ] Create `storage/search_providers/` package
- [ ] Create `storage/search_providers/fts5.py` with `FTS5Provider` class
- [ ] Create `storage/search_providers/qdrant.py` with `QdrantProvider` class
- [ ] Implement `SearchProvider` protocol in both providers
- [ ] Move FTS5 logic from `index.py` to `FTS5Provider`
- [ ] Move Qdrant logic from `index_qdrant.py` to `QdrantProvider`

#### Integration
- [ ] Create provider factory in `storage/search_providers/__init__.py`
- [ ] Add `create_search_provider(config: IndexConfig)` factory function
- [ ] Update `IndexService` to use provider factory
- [ ] Remove conditional imports from `index.py`
- [ ] Update tests to use provider mocks
- [ ] Add provider-specific test suites

### 2.4 CLI Dependency Injection (3 days)

**Goal**: Refactor CLI to use factory pattern for testability

**Dependencies**: 2.2 (Service Decomposition)

- [ ] Create `cli/container.py` for dependency container
- [ ] Define `create_config()` factory function
- [ ] Define `create_storage_repository()` factory function
- [ ] Define `create_ingestion_service()` factory function
- [ ] Define `create_index_service()` factory function
- [ ] Define `create_render_service()` factory function
- [ ] Define `create_pipeline_runner()` factory function
- [ ] Update CLI commands to use factories instead of direct instantiation
- [ ] Add `--config` flag to override config path
- [ ] Add CLI test suite using mock factories
- [ ] Update CLI documentation

---

## Priority 3: High Impact, High Effort (3-4 weeks)

### 3.1 Storage Abstraction Layer (2 weeks)

**Goal**: Enable swapping storage backends (SQLite → PostgreSQL/DuckDB)

**Dependencies**: 1.2 (Storage Repository), 1.3 (Protocols)

#### Protocol Definition
- [ ] Extend `StorageBackend` protocol in `protocols.py`
- [ ] Define `get_conversation(id) -> ConversationRecord | None`
- [ ] Define `list_conversations(filter) -> list[ConversationRecord]`
- [ ] Define `save_conversation(record) -> None`
- [ ] Define `get_messages(conversation_id) -> list[MessageRecord]`
- [ ] Define `save_messages(records) -> None`
- [ ] Define `get_attachments(conversation_id) -> list[AttachmentRecord]`
- [ ] Define `save_attachments(records) -> None`
- [ ] Define `search_messages(query) -> list[MessageRecord]`
- [ ] Define transaction methods: `begin()`, `commit()`, `rollback()`

#### SQLite Backend Implementation
- [ ] Create `storage/backends/` package
- [ ] Create `storage/backends/sqlite.py` with `SQLiteBackend` class
- [ ] Move all SQL queries from scattered modules to `SQLiteBackend`
- [ ] Implement `StorageBackend` protocol in `SQLiteBackend`
- [ ] Move schema management to `SQLiteBackend`
- [ ] Move migration logic to `SQLiteBackend`
- [ ] Add connection pooling to `SQLiteBackend`
- [ ] Update `StorageRepository` to accept `StorageBackend` parameter

#### Integration & Migration
- [ ] Create backend factory in `storage/backends/__init__.py`
- [ ] Add `backend_type` field to config
- [ ] Update `storage/repository.py` to use backend abstraction
- [ ] Remove direct SQL from `lib/repository.py`
- [ ] Remove direct SQL from `pipeline/ingest.py`
- [ ] Remove direct SQL from CLI commands
- [ ] Update all tests to use backend fixtures
- [ ] Create backend integration test suite
- [ ] Run full test suite with SQLite backend
- [ ] Document backend abstraction in architecture docs

#### Future Backend Support (Optional)
- [ ] Create `storage/backends/postgres.py` (stub implementation)
- [ ] Create `storage/backends/duckdb.py` (stub implementation)
- [ ] Document backend interface for contributors

### 3.2 Advanced Dependency Injection (1 week)

**Goal**: Implement proper DI framework for better testability

**Dependencies**: 2.2 (Service Decomposition), 2.4 (CLI DI)

#### Framework Setup
- [ ] Add `dependency-injector` to dependencies
- [ ] Create `polylogue/container.py` with DI container
- [ ] Define `ConfigProvider` in container
- [ ] Define `StorageProvider` in container
- [ ] Define `ServiceProvider` in container
- [ ] Configure container singleton/factory scopes

#### Service Registration
- [ ] Register `Config` as singleton
- [ ] Register `StorageRepository` as singleton
- [ ] Register `IngestionService` as factory
- [ ] Register `IndexService` as factory
- [ ] Register `RenderService` as factory
- [ ] Register `PipelineRunner` as factory
- [ ] Register `ConversationRepository` as factory

#### CLI Integration
- [ ] Update `cli/__main__.py` to use container
- [ ] Wire CLI commands to container
- [ ] Remove manual factory functions from `cli/container.py`
- [ ] Update CLI tests to use test container
- [ ] Add CLI test suite with mocked services

#### Server Integration
- [ ] Update FastAPI app to use DI container
- [ ] Wire API routes to container
- [ ] Update server tests to use test container
- [ ] Document DI patterns for contributors

### 3.3 Renderer Abstraction (4 days)

**Goal**: Support multiple output formats through renderer plugins

**Dependencies**: 1.3 (Protocols), 2.2 (Service Decomposition)

#### Renderer Protocol
- [ ] Define `OutputRenderer` protocol in `protocols.py`
- [ ] Add `render(conversation, output_path) -> RenderResult` method
- [ ] Add `supports_format() -> list[str]` method

#### Renderer Implementations
- [ ] Create `rendering/renderers/` package
- [ ] Create `rendering/renderers/markdown.py` with `MarkdownRenderer`
- [ ] Create `rendering/renderers/html.py` with `HTMLRenderer`
- [ ] Move Jinja2 logic from `render.py` to `HTMLRenderer`
- [ ] Implement `OutputRenderer` protocol in both renderers

#### Integration
- [ ] Create renderer factory in `rendering/renderers/__init__.py`
- [ ] Update `RenderService` to accept renderer list
- [ ] Add `--format` CLI flag to `render` command
- [ ] Update tests with renderer mocks
- [ ] Add renderer-specific test suites

#### Future Renderers (Optional)
- [ ] Document renderer plugin interface
- [ ] Create `rendering/renderers/pdf.py` (stub)
- [ ] Create `rendering/renderers/epub.py` (stub)

---

## Priority 4: Code Quality & Maintenance

### 4.1 Documentation Updates

- [ ] Update `CLAUDE.md` with new architecture
- [ ] Document module boundaries in `docs/architecture.md`
- [ ] Create `docs/dependency-injection.md` guide
- [ ] Create `docs/adding-backends.md` guide
- [ ] Create `docs/adding-renderers.md` guide
- [ ] Update API documentation
- [ ] Update CLI documentation
- [ ] Add migration guide for breaking changes

### 4.2 Type Safety Hardening

- [ ] Enable `mypy` strict mode in CI
- [ ] Add `--disallow-untyped-defs` flag
- [ ] Add `--disallow-any-generics` flag
- [ ] Fix all mypy errors in strict mode
- [ ] Add type stubs for external dependencies
- [ ] Add `py.typed` marker file
- [ ] Update type hints in all public APIs

### 4.3 Test Coverage Improvements

- [ ] Identify modules below 80% coverage
- [ ] Add integration tests for service layer
- [ ] Add contract tests for protocols
- [ ] Add end-to-end CLI tests
- [ ] Add performance regression tests
- [ ] Add property-based tests for storage layer
- [ ] Maintain overall coverage >85%

### 4.4 Performance Optimizations

- [ ] Profile pipeline performance with large datasets
- [ ] Optimize content hashing (cache results)
- [ ] Optimize FTS5 indexing (batch operations)
- [ ] Optimize rendering (parallel processing)
- [ ] Add performance benchmarks to test suite
- [ ] Document performance characteristics

---

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)
1. Complete Priority 1 tasks (module organization, repository, protocols)
2. Update tests incrementally
3. Maintain backward compatibility

### Phase 2: Services (Weeks 3-4)
1. Complete Priority 2 tasks (config objects, service decomposition)
2. Refactor CLI to use services
3. Update integration tests

### Phase 3: Abstractions (Weeks 5-8)
1. Complete Priority 3 tasks (storage abstraction, DI framework)
2. Add backend/renderer plugins
3. Full test suite update

### Phase 4: Stabilization (Week 9+)
1. Complete Priority 4 tasks (docs, type safety, coverage)
2. Performance testing
3. Release preparation

---

## Success Criteria

- [ ] All tests passing (>85% coverage maintained)
- [ ] Mypy strict mode passing
- [ ] No regression in performance benchmarks
- [ ] Documentation updated and reviewed
- [ ] Migration guide validated
- [ ] Zero production incidents during rollout

---

## Risk Mitigation

**Risk**: Breaking changes impact users
**Mitigation**: Maintain backward compatibility, add deprecation warnings, version bump to 2.0

**Risk**: Test suite becomes slow
**Mitigation**: Use test parallelization, mock external dependencies, separate unit/integration tests

**Risk**: Increased complexity
**Mitigation**: Document patterns, provide examples, maintain simple public APIs

**Risk**: Incomplete migration
**Mitigation**: Track progress with this checklist, review dependencies before starting tasks

---

## Notes

- Each checkbox represents a concrete, testable task
- Dependencies noted where task order matters
- Effort estimates are approximate (may vary based on complexity discovered)
- Priority can be adjusted based on business needs
- This is a living document - update as work progresses
