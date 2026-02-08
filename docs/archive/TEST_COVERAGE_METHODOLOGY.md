# Test Coverage Improvement Methodology

> A reproducible playbook for systematic test coverage sprints. Proven on polylogue (68% → 71% in one session, 4 blocking failures fixed, 53 tests added).

---

## Pre-Sprint Analysis Protocol

### 1. Identify Current State

```bash
# Get baseline coverage with term-missing for uncovered lines
uv run pytest tests/ --cov=<package> --cov-report=term-missing --ignore=tests/test_<external>.py

# Count current tests
uv run pytest tests/ --collect-only -q | tail -1
```

### 2. Categorize Issues

| Category | Priority | Action |
|----------|----------|--------|
| **Blocking failures** | P0 | Fix immediately (tests that fail, not skip) |
| **0% coverage modules** | P1 | High-value targets for new tests |
| **< 30% coverage** | P2 | Second-tier improvements |
| **External integrations** | P3 | Often deliberately low (mock complexity) |
| **UI/presentation** | P4 | Low ROI, visual QA preferred |

### 3. Prioritization Criteria

Score each module (1-5) on:
- **Business criticality**: Core logic vs. utility
- **Bug probability**: Complex state, external I/O, edge cases
- **Test difficulty**: Simple functions vs. complex mocking
- **Coverage gap size**: 0% more impactful than 60% → 70%

**Formula**: `Priority = (Criticality × Bug_Probability) / Test_Difficulty`

---

## Phase-Based Execution Framework

### Phase 1: Fix Blocking Failures (MUST complete first)

**Why first?** Blocking tests indicate real bugs. Until fixed:
- CI/CD is broken
- Regression detection is impossible
- Coverage metrics are unreliable

**Process:**
1. Run failing tests in isolation to get full error output
2. Trace root cause (read implementation code)
3. Fix production code (not the test, unless test is wrong)
4. Verify fix: single test → full regression suite

**Common blocking failure patterns:**
| Pattern | Symptom | Typical Fix |
|---------|---------|-------------|
| Attribute mismatch | `has no attribute 'x'` | Check model/protocol definitions |
| Missing implementation | `NotImplementedError` | Implement the method |
| Thread safety | Intermittent failures | Add locks, use thread-local |
| Timezone issues | Off-by-N-hours | Use UTC everywhere |
| Path confusion | File not found | Check fixture paths match implementation |

### Phase 2: High-Value Targets (0% coverage)

Focus on modules with:
- 0% coverage + business logic
- Core abstractions (repositories, services, providers)
- Data transformation (formatters, parsers, serializers)

**Test discovery process:**
```python
# 1. Read the module
# 2. List all public functions/methods
# 3. For each function, identify:
#    - Happy path (normal input → expected output)
#    - Edge cases (empty, None, boundary values)
#    - Error conditions (invalid input → expected exception)
```

### Phase 3: Quick Wins (utility modules)

Target simple modules that can reach 70%+ quickly:
- Formatting utilities
- Configuration helpers
- Pure functions without side effects

**These often have 8-15 tests covering:**
- Normal cases
- Empty/None inputs
- Boundary conditions
- Environment variable overrides

### Phase 4: Verification

```bash
# Module-specific coverage
uv run pytest tests/test_<module>.py --cov=<package>.<module> --cov-report=term-missing

# Full regression (must pass with no new failures)
uv run pytest tests/ --ignore=tests/test_<external>.py

# Final coverage report
uv run pytest tests/ --cov=<package> --cov-report=html --ignore=tests/test_<external>.py
```

---

## Test Discovery Patterns

### Pattern 1: Service/Repository Tests

```python
class TestServiceName:
    @pytest.fixture
    def service(self, workspace_env, storage_repository):
        """Create service with test dependencies."""
        return ServiceName(repository=storage_repository)

    def test_happy_path(self, service):
        """Normal operation with valid input."""
        result = service.do_thing(valid_input)
        assert result.success is True

    def test_empty_input(self, service):
        """Empty input returns empty result."""
        result = service.do_thing([])
        assert result.count == 0

    def test_invalid_input_raises(self, service):
        """Invalid input raises appropriate error."""
        with pytest.raises(ValueError, match="must be"):
            service.do_thing(invalid_input)
```

### Pattern 2: Formatter/Utility Tests

```python
class TestFormatFunction:
    def test_formats_normal_input(self):
        """Standard input produces expected output."""
        result = format_thing({"key": "value"})
        assert "key" in result

    def test_handles_empty(self):
        """Empty input returns default/None."""
        assert format_thing({}) is None

    def test_truncates_long_lists(self):
        """Long lists are truncated with indicator."""
        result = format_thing(list(range(100)))
        assert "+X more" in result
```

### Pattern 3: Database-Dependent Tests

```python
def test_stores_and_retrieves(self, workspace_env, storage_repository):
    """Data survives round-trip through storage."""
    # Use workspace_env for isolated database
    db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

    # Create test data
    record = RecordType(id="test-1", ...)
    storage_repository.save(record)

    # Retrieve and verify
    retrieved = storage_repository.get("test-1")
    assert retrieved.id == record.id
```

### Pattern 4: Provider/Protocol Tests

```python
class TestProviderName:
    @pytest.fixture
    def provider(self, workspace_env):
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"
        return ProviderName(db_path=db_path)

    def test_index_creates_structure(self, provider, storage_repository):
        """Indexing creates expected storage structure."""
        # Arrange: create data via repository
        storage_repository.save_conversation(...)

        # Act: index
        provider.index(messages)

        # Assert: structure exists
        # (query underlying storage to verify)

    def test_search_finds_indexed(self, provider):
        """Search returns indexed content."""
        # After indexing...
        results = provider.search("keyword")
        assert len(results) > 0

    def test_search_empty_when_no_index(self, workspace_env):
        """Search on missing index returns empty."""
        provider = ProviderName(db_path=Path("/nonexistent.db"))
        assert provider.search("anything") == []
```

---

## Bug Discovery Protocol

Tests often reveal production bugs. When a test fails unexpectedly:

### 1. Verify Test is Correct

```
Is the test asserting the right thing?
   YES → Production bug
   NO  → Fix the test
```

### 2. Trace the Root Cause

```python
# Add debug output or step through:
# - What does the function actually return?
# - What type is it? (AttributeError = wrong type/model)
# - What state was the dependency in?
```

### 3. Fix Production Code

Common discoveries:
| Test Symptom | Production Bug |
|--------------|----------------|
| `'X' has no attribute 'y'` | Using wrong model/class |
| `assert 'x' == 'y'` but equal | String encoding, whitespace |
| Off-by-one in counts | Loop boundary, early return |
| Works alone, fails in suite | Shared state, test isolation |

### 4. Document the Fix

```python
# In the production code:
# Fix: Changed conv.provider_name → conv.provider
# (Conversation model uses 'provider' not 'provider_name')
```

---

## Fixture Patterns

### Isolated Workspace (Database Tests)

```python
@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    """Isolated workspace with temp directories."""
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_dir / "config.json"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

    return {
        "config_path": config_dir / "config.json",
        "archive_root": archive_root,
        "state_root": state_dir,
    }
```

### Repository with Workspace

```python
@pytest.fixture
def storage_repository(workspace_env):
    """Repository using workspace database."""
    # workspace_env sets env vars that create_default_backend() uses
    from package.storage.backends.sqlite import create_default_backend
    from package.storage.repository import StorageRepository
    return StorageRepository(backend=create_default_backend())
```

### Fixture Ordering

When using multiple fixtures, **order matters**:
```python
def test_something(self, workspace_env, storage_repository):
    #                     ^^ evaluated first (sets env vars)
    #                                  ^^ evaluated second (uses env vars)
```

---

## Verification Commands

### Per-Module Coverage

```bash
# Check specific module reached target
uv run pytest tests/test_<name>.py \
    --cov=<package>.<subpackage>.<module> \
    --cov-report=term-missing
```

### Full Regression

```bash
# Must pass with no new failures
uv run pytest tests/ -v --ignore=tests/test_<external>.py

# With fail-fast for quick iteration
uv run pytest tests/ -x --ignore=tests/test_<external>.py
```

### Coverage Report

```bash
# HTML report for detailed analysis
uv run pytest tests/ \
    --cov=<package> \
    --cov-report=html \
    --cov-report=term \
    --ignore=tests/test_<external>.py

# Open report
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html       # macOS
```

---

## Success Criteria

A coverage sprint is successful when:

- [ ] All blocking test failures resolved
- [ ] No regressions (existing tests still pass)
- [ ] Target coverage achieved (or justified shortfall)
- [ ] New test files follow project patterns
- [ ] Bugs discovered during testing are fixed in production code
- [ ] Documentation updated if patterns changed

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why Bad | Instead |
|--------------|---------|---------|
| Testing private methods | Brittle, implementation detail | Test public API |
| Mocking everything | Tests don't catch real bugs | Use real dependencies where feasible |
| Skipping hard-to-test code | False security | Document why or simplify code |
| Testing trivial getters | Low value | Focus on logic/transformation |
| Fixing tests to match bugs | Hides bugs | Fix production code |
| Copy-paste test data | Maintenance burden | Use fixtures/factories |

---

## Quick Reference

```bash
# Baseline
uv run pytest tests/ --cov=pkg --cov-report=term-missing

# Blocking tests only
uv run pytest tests/test_X.py::test_failing -v

# New module tests
uv run pytest tests/test_new_module.py -v

# Coverage for module
uv run pytest tests/test_X.py --cov=pkg.X --cov-report=term-missing

# Full regression
uv run pytest tests/ --ignore=tests/test_external.py

# Final report
uv run pytest tests/ --cov=pkg --cov-report=html
```
