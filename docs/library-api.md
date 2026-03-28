[← Back to README](../README.md)

# Library API

Polylogue is designed library-first. The CLI wraps the Python API.

## Basic Usage

```python
from polylogue.lib.filters import ConversationFilter
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.backends.sqlite import SQLiteBackend

backend = SQLiteBackend()
repo = ConversationRepository(backend=backend)

# Search
results = ConversationFilter(repo).contains("error").provider("claude").list()
for conv in results:
    print(f"{conv.id}: {conv.display_title}")

# Single conversation
conv = ConversationFilter(repo).id("abc123").first()
```

## Filter Chain API

```python
# Chainable, lazy evaluation
results = (ConversationFilter(repo)
    .contains("error")
    .contains("python")             # AND
    .provider("claude", "chatgpt")  # OR
    .since("2025-01-01")
    .has("thinking")
    .limit(10)
    .list())                        # Terminal: list(), first(), count(), delete()

# Exclusion filters
results = (ConversationFilter(repo)
    .contains("error")
    .no_contains("warning")
    .no_provider("gemini")
    .no_tag("archived")
    .list())

# Lightweight summaries (no message loading)
summaries = (ConversationFilter(repo)
    .provider("claude")
    .since("2025-01-01")
    .list_summaries())              # Returns ConversationSummary (no messages)

# Check if summaries are sufficient
f = ConversationFilter(repo).provider("claude")
if f.can_use_summaries():
    results = f.list_summaries()    # Fast path
else:
    results = f.list()              # Loads full conversations

# Custom predicates
results = (ConversationFilter(repo)
    .where(lambda c: len(c.messages) > 50)
    .list())

# Sorting and sampling
results = (ConversationFilter(repo)
    .sort("tokens")
    .reverse()
    .sample(10)
    .list())

# Conversation structure filters
roots = ConversationFilter(repo).is_root().list()
continuations = ConversationFilter(repo).is_continuation().list()
```

## Available Filter Methods

| Method | Description |
|--------|-------------|
| `.contains(text)` | FTS term (chainable = AND) |
| `.no_contains(text)` | Exclude FTS term |
| `.provider(*names)` | Include providers |
| `.no_provider(*names)` | Exclude providers |
| `.tag(*tags)` | Include tags |
| `.no_tag(*tags)` | Exclude tags |
| `.has(*types)` | Content types: `thinking`, `tools`, `summary`, `attachments` |
| `.title(pattern)` | Title contains pattern |
| `.id(prefix)` | ID prefix match |
| `.since(date)` | After date (str or datetime) |
| `.until(date)` | Before date (str or datetime) |
| `.similar(text)` | Semantic similarity (requires vector index) |
| `.sort(field)` | Sort: `date`, `tokens`, `messages`, `words`, `longest`, `random` |
| `.reverse()` | Reverse sort order |
| `.limit(n)` | Max results |
| `.sample(n)` | Random sample |
| `.where(predicate)` | Custom filter predicate |
| `.is_root()` | Root conversations only |
| `.is_continuation()` | Continuation conversations only |
| `.is_sidechain()` | Sidechain conversations only |
| `.has_branches()` | Conversations with branching messages |
| `.parent(id)` | Children of a given parent |

## Terminal Methods

| Method | Description |
|--------|-------------|
| `.list()` | Execute and return `list[Conversation]` |
| `.list_summaries()` | Execute and return `list[ConversationSummary]` (lightweight, no messages) |
| `.first()` | Execute and return first match or `None` |
| `.count()` | Execute and return count (uses SQL fast path when possible) |
| `.delete()` | Delete matching conversations (returns count deleted) |
| `.pick()` | Interactive picker (TTY) or first match (non-TTY) |
| `.can_use_summaries()` | Check if `list_summaries()` is valid for current filters |

## Pipeline (Run)

```python
from polylogue.pipeline.runner import run_sources
from polylogue.services import get_service_config

config = get_service_config()
result = run_sources(config=config, stage="all")
```

---

**See also:** [CLI Reference](cli-reference.md) · [Data Model](data-model.md) · [Configuration](configuration.md)
