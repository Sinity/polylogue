# Design: Semantic Projections & View Command

## Architectural Context

From the Lynchpin design principles:

> **Polylogue (Library):** Responsible purely for ingesting, normalizing, and providing a **semantic API** for conversation logs. It has no UI, no database, and no daemon. It is a dependency.
>
> **Lynchpin (Consumer):** Imports the Polylogue library to ingest data into its warehouse, run its watchers, and display its data in the dashboard.

This means:
1. **Polylogue's core value is semantic projections**, not storage or presentation
2. The `view` command is a **CLI wrapper** around these projections
3. Lynchpin (and other consumers) use the **same projection API**
4. Storage (SQLite) is a current implementation detail, not the API surface

## Core Insight

The user's workflow is a two-stage pipeline:
1. **Select** - Filter the set of conversations/messages of interest
2. **Project** - Apply semantic transformations to extract meaning

This is analogous to SQL's `SELECT <projection> FROM <source> WHERE <filter>`.

## Proposed Command: `polylogue view`

```
polylogue view <entity> [filters...] [--transform ...] [--fields ...] [--format ...]
```

### Entity Types (Source)

| Entity | Description | Granularity |
|--------|-------------|-------------|
| `conversations` | Full conversation objects | Coarse |
| `messages` | Individual messages | Medium |
| `turns` | User+assistant pairs | Semantic |
| `stats` | Aggregate statistics | Meta |

### Filter Predicates

Filters narrow the result set. All filters are AND'd together.

```bash
# Provider/source filtering
--provider claude-code     # By provider name
--source inbox             # By config source name

# Role filtering (messages/turns only)
--role user                # user, assistant, system, tool
--role assistant

# Time filtering
--since 2024-01-01         # ISO date or relative (7d, 2w, 1m)
--until 2024-06-01
--today                    # Shorthand for --since today

# Content filtering
--search "error handling"  # FTS5 text search
--match "regex pattern"    # Regex match on content
--has-attachments          # Only messages with attachments
--has-tools                # Only messages with tool use

# Metadata filtering
--min-messages 10          # Conversations with at least N messages
--max-messages 100
--title "project.*"        # Regex on conversation title

# Limit/offset
--limit 100
--offset 50
```

### Transforms (Content Modification)

Transforms modify the content before output. Multiple transforms chain.

```bash
--transform strip-tools    # Remove tool_use/tool_result blocks
--transform strip-thinking # Remove thinking blocks
--transform extract-code   # Extract only code blocks
--transform summarize      # AI-powered summarization (requires API key)
--transform annotate       # Add pattern annotations (see below)
```

### Projections (Field Selection)

Control which fields appear in output.

```bash
--fields id,role,text             # Specific fields
--fields +timestamp               # Add to defaults
--fields -provider_meta           # Remove from defaults
--fields all                      # Everything
--fields minimal                  # id, role, text only
```

### Output Formats

```bash
--format table             # Rich terminal table (default for interactive)
--format jsonl             # JSON Lines for machine consumption
--format json              # Single JSON array
--format markdown          # Markdown document
--format csv               # CSV export
--format stream            # Streaming output as found (no buffering)
```

### Pattern Annotations

Built-in pattern detectors for analysis:

```bash
--annotate frustration     # "why did you", "undo", corrections
--annotate verbosity       # "Let me...", announcements
--annotate questions       # User questions
--annotate praise          # Positive feedback
--annotate all             # All patterns
```

Adds `_annotations` field to output with detected patterns.

---

## Example Usage

### Basic Browsing

```bash
# List recent conversations
polylogue view conversations --since 7d

# Browse messages from a specific provider
polylogue view messages --provider claude-code --limit 50

# View user messages only
polylogue view messages --role user --provider claude-code
```

### Analysis Workflows

```bash
# Find frustration patterns for hook development
polylogue view messages \
  --role user \
  --match "why did you|undo|wrong|not what I" \
  --annotate frustration \
  --format jsonl

# Extract clean assistant text (no tool blocks)
polylogue view messages \
  --role assistant \
  --provider claude-code \
  --transform strip-tools \
  --format markdown

# Conversation statistics
polylogue view stats \
  --provider claude-code \
  --group-by role
```

### Machine-Readable Output for Agents

```bash
# Export for analysis agent
polylogue view messages \
  --provider claude-code \
  --since 30d \
  --fields id,conversation_id,role,text,timestamp \
  --format jsonl \
  > /tmp/messages.jsonl

# Pipe to analysis
polylogue view messages --format jsonl | agent-analyze frustration
```

### Interactive Exploration

```bash
# Page through conversations interactively
polylogue view conversations --interactive

# Open selected conversation in editor
polylogue view messages --conversation-id "claude-code:abc123" --open
```

---

## Implementation Architecture

### Layer 1: Semantic Projection API (Library Core)

This is the **actual value** of polylogue - the API that Lynchpin and CLI both consume.

```python
# polylogue/lib/projections.py

class ConversationView:
    """Semantic projection over a conversation."""

    def __init__(self, conversation: Conversation):
        self._conv = conversation

    # === Selection Projections ===

    def messages(self) -> Iterator[Message]:
        """All messages in order."""
        yield from self._conv.messages

    def user_messages(self) -> Iterator[Message]:
        """Only user messages."""
        yield from (m for m in self._conv.messages if m.role == "user")

    def assistant_messages(self) -> Iterator[Message]:
        """Only assistant messages."""
        yield from (m for m in self._conv.messages if m.role == "assistant")

    # === Semantic Projections (the real value) ===

    def dialogue_pairs(self) -> Iterator[tuple[Message, Message]]:
        """User-assistant turn pairs, filtering out system context."""
        user_msg = None
        for msg in self._conv.messages:
            if msg.role == "user" and not msg.is_context_dump:
                user_msg = msg
            elif msg.role == "assistant" and user_msg:
                yield (user_msg, msg)
                user_msg = None

    def clean_dialogue(self) -> Iterator[Message]:
        """Messages with tool blocks, thinking, and noise stripped."""
        for msg in self._conv.messages:
            yield msg.strip_tools().strip_thinking()

    def hide_noise(self) -> Iterator[Message]:
        """Filter out context dumps, tool results, system prompts."""
        for msg in self._conv.messages:
            if msg.is_noise:
                continue
            yield msg

    # === Analysis Projections ===

    def with_annotations(self, patterns: list[str]) -> Iterator[AnnotatedMessage]:
        """Add pattern detection annotations to messages."""
        for msg in self._conv.messages:
            annotations = detect_patterns(msg.text, patterns)
            yield AnnotatedMessage(msg, annotations)

    def frustration_indicators(self) -> Iterator[AnnotatedMessage]:
        """Messages showing user frustration."""
        return self.with_annotations(["frustration"])

    # === Aggregation Projections ===

    def stats(self) -> ConversationStats:
        """Aggregate statistics."""
        return ConversationStats(
            message_count=len(self._conv.messages),
            user_message_count=sum(1 for m in self.user_messages()),
            assistant_message_count=sum(1 for m in self.assistant_messages()),
            tool_use_count=sum(1 for m in self._conv.messages if m.has_tool_use),
            # ...
        )
```

### Layer 2: Repository with Projection Support

```python
# polylogue/lib/repository.py (extended)

class ConversationRepository:
    def view(self, conversation_id: str) -> ConversationView | None:
        """Get a conversation wrapped in semantic projection interface."""
        conv = self.get(conversation_id)
        return ConversationView(conv) if conv else None

    def query(self, filter_spec: FilterSpec) -> Iterator[ConversationView]:
        """Query conversations with filters, returning projection views."""
        for conv in self._query_raw(filter_spec):
            yield ConversationView(conv)

    def query_messages(self, filter_spec: FilterSpec) -> Iterator[Message]:
        """Query at message level with filters."""
        # Efficient: query messages directly, not via conversations
        for msg in self._query_messages_raw(filter_spec):
            yield msg
```

### Layer 3: CLI Wrapper

The CLI is just a thin presentation layer over the projection API.

```
polylogue/cli/commands/view.py
    └── view_command()
        ├── parse_filters() → FilterSpec
        ├── repo.query() → Iterator[ConversationView]
        ├── apply_projection() → Iterator[Message]  # calls conv.clean_dialogue() etc.
        └── format_output() → None                  # table, jsonl, markdown
```

### Core Types

```python
@dataclass
class FilterSpec:
    entity: Literal["conversations", "messages", "turns", "stats"]
    provider: str | None
    source: str | None
    role: str | None
    since: datetime | None
    until: datetime | None
    search: str | None
    match: str | None
    limit: int
    offset: int
    # ... additional filters

@dataclass
class TransformSpec:
    strip_tools: bool = False
    strip_thinking: bool = False
    extract_code: bool = False
    annotate: list[str] = field(default_factory=list)

@dataclass
class OutputSpec:
    format: Literal["table", "jsonl", "json", "markdown", "csv", "stream"]
    fields: list[str] | None
    interactive: bool = False
```

### Query Execution

```python
def execute_view(filter_spec: FilterSpec, transform: TransformSpec, output: OutputSpec):
    # 1. Build SQL query from filters
    query, params = build_query(filter_spec)

    # 2. Execute and stream results
    with connection_context() as conn:
        for row in conn.execute(query, params):
            entity = hydrate_entity(row, filter_spec.entity)

            # 3. Apply transforms
            if transform.strip_tools:
                entity = strip_tool_blocks(entity)
            if transform.annotate:
                entity = annotate_patterns(entity, transform.annotate)

            # 4. Project fields
            entity = project_fields(entity, output.fields)

            # 5. Format and output
            yield format_entity(entity, output.format)
```

### Pattern Detectors

```python
# polylogue/analysis/patterns.py

PATTERNS = {
    "frustration": [
        r"why did you",
        r"that's not what I",
        r"undo that",
        r"wrong",
        r"no,? (?:I said|I meant|I wanted)",
        r"try again",
    ],
    "verbosity": [
        r"^(?:Let me|I'll|I will|I'm going to)",
        r"^(?:Sure|Certainly|Of course|Absolutely)[,!]",
    ],
    "questions": [
        r"\?$",
        r"^(?:How|What|Why|When|Where|Which|Can you|Could you)",
    ],
    "praise": [
        r"(?:perfect|great|excellent|awesome|thanks|thank you)",
    ],
}

def detect_patterns(text: str, pattern_names: list[str]) -> dict[str, list[Match]]:
    results = {}
    for name in pattern_names:
        if name in PATTERNS:
            matches = []
            for pattern in PATTERNS[name]:
                matches.extend(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches:
                results[name] = [{"start": m.start(), "end": m.end(), "match": m.group()} for m in matches]
    return results
```

### Tool Block Stripping

```python
# polylogue/analysis/transforms.py

def strip_tool_blocks(message: MessageRecord) -> MessageRecord:
    """Remove tool_use and tool_result blocks from message text."""
    if not message.provider_meta:
        return message

    meta = message.provider_meta
    raw = meta.get("raw", {})

    # Handle Claude-style content blocks
    if isinstance(raw.get("content"), list):
        text_parts = []
        for block in raw["content"]:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return message.copy(update={"text": "\n".join(text_parts)})

    return message
```

---

---

## Lynchpin Integration (Library Consumer Pattern)

Lynchpin imports polylogue as a library and uses the projection API:

```python
# lynchpin/ingest/conversations.py

from polylogue.lib import ConversationRepository, FilterSpec
from polylogue.lib.projections import ConversationView

def sync_conversations_to_warehouse(warehouse_conn, since: datetime):
    """Ingest polylogue conversations into Lynchpin warehouse."""
    repo = ConversationRepository()

    for conv_view in repo.query(FilterSpec(since=since)):
        # Use semantic projections to extract what we need
        for pair in conv_view.dialogue_pairs():
            user_msg, assistant_msg = pair
            warehouse_conn.execute("""
                INSERT INTO dialogue_turns (conversation_id, user_text, assistant_text, timestamp)
                VALUES (?, ?, ?, ?)
            """, (conv_view.id, user_msg.text, assistant_msg.text, user_msg.timestamp))

        # Store annotations for analysis dashboard
        for annotated in conv_view.with_annotations(["frustration", "verbosity"]):
            if annotated.annotations:
                warehouse_conn.execute("""
                    INSERT INTO message_annotations (message_id, annotations)
                    VALUES (?, ?)
                """, (annotated.message.id, json.dumps(annotated.annotations)))
```

### Dashboard Queries (Views on Warehouse)

```python
# lynchpin/dashboard/api.py

@app.get("/api/frustration-patterns")
def frustration_patterns(since: str = "7d"):
    """Dashboard endpoint: Recent frustration patterns."""
    # Query warehouse, not polylogue directly
    return db.execute("""
        SELECT m.text, m.timestamp, a.annotations
        FROM messages m
        JOIN message_annotations a ON m.id = a.message_id
        WHERE a.annotations LIKE '%frustration%'
        AND m.timestamp > datetime('now', ?)
        ORDER BY m.timestamp DESC
        LIMIT 50
    """, (f"-{since}",))
```

### Key Architectural Points

1. **Polylogue is stateless from Lynchpin's perspective** - it provides projections over data, Lynchpin owns persistence
2. **Warehouse is the source of truth** - artifacts are views/queries, not files
3. **Semantic projections are the API contract** - `clean_dialogue()`, `dialogue_pairs()`, `with_annotations()`
4. **CLI and Lynchpin use identical projection methods** - no divergence

---

## Comparison with Original Proposal

| Original `analyze` | New `view` | Improvement |
|--------------------|------------|-------------|
| `--strip-tools` flag | `--transform strip-tools` | Composable transforms |
| `--role user` | `--role user` | Same |
| `--pattern` pre-defined | `--annotate` + `--match` | Both built-in and custom |
| `--output jsonl` | `--format jsonl` | More formats |
| Separate `stats` subcommand | `view stats` entity | Unified interface |
| Fixed field output | `--fields` projection | Flexible |

### Key Improvements

1. **Unified command** - One command handles browsing, filtering, and analysis
2. **Composable transforms** - Chain multiple content modifications
3. **Field projection** - Control exactly what fields appear
4. **Multiple output formats** - Table, JSONL, markdown, CSV
5. **Entity abstraction** - Same filters work on conversations, messages, turns
6. **Pattern annotations** - Non-destructive; adds metadata rather than filtering
7. **Streaming** - Can process large datasets without loading all into memory

---

## Migration Path

Phase 1: Implement core `view` command with basic filters and formats
Phase 2: Add transforms (strip-tools, strip-thinking)
Phase 3: Add pattern annotations
Phase 4: Add `turns` entity and semantic grouping
Phase 5: Add interactive mode and editor integration

---

## Future Extensions

### Query Language (v2)

For power users, support a mini query language:

```bash
polylogue query "messages where provider='claude-code' and role='user' | strip_tools | limit 100"
```

### Saved Views

Save commonly used filter combinations:

```bash
polylogue view --save my-analysis \
  --provider claude-code \
  --role user \
  --transform strip-tools \
  --format jsonl

# Later:
polylogue view --use my-analysis
```

### Piping to Agents

```bash
# Direct agent integration
polylogue view messages --format jsonl | polylogue agent analyze-frustration

# Or with Task tool spawning
polylogue view messages --analyze-with frustration-detector
```

---

## Architectural Evolution: Assimilate, Don't Aggregate

### Current State (Polylogue as Application)

```
Sources (JSON exports) → Polylogue SQLite → Rendered Files (MD/HTML)
                              ↓
                         FTS5 Index → CLI Search
```

**Problems:**
- Rendered files are "encapsulating the mess"
- SQLite is an implementation detail exposed as the interface
- CLI is tightly coupled to storage layer

### Target State (Polylogue as Library)

```
Sources (JSON exports) → Polylogue Projection API
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              CLI (`view`)        Lynchpin Warehouse
                                        ↓
                                  Dashboard (views/queries)
```

**Benefits:**
- Projections are the API, not storage
- CLI and Lynchpin use identical interface
- Warehouse becomes single source of truth
- Rendered files become optional (or eliminated)

### Migration Path

1. **Phase 1 (Now):** Add projection API (`polylogue/lib/projections.py`)
2. **Phase 2:** Implement `view` command using projections
3. **Phase 3:** Lynchpin imports polylogue library, syncs to warehouse
4. **Phase 4:** Deprecate rendered file generation (make optional)
5. **Phase 5:** Dashboard reads from warehouse, not polylogue directly

### What Stays in Polylogue

- **Ingestion:** Parse ChatGPT, Claude, Codex exports
- **Normalization:** Convert to common `Conversation`/`Message` model
- **Semantic Projections:** `dialogue_pairs()`, `clean_dialogue()`, `with_annotations()`
- **Pattern Detection:** Frustration, verbosity, etc.

### What Moves to Lynchpin

- **Persistence:** Long-term storage in warehouse
- **Indexing:** FTS, analytics indexes
- **Presentation:** Dashboard, visualizations
- **Aggregation:** Cross-conversation statistics

### Current Polylogue Storage as Transition

During transition, polylogue's SQLite can serve as:
- Local cache for projections
- Development/testing data source
- Standalone usage without Lynchpin

But it should not be the architectural contract - projections are.
