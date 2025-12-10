# Database as Source of Truth - Implementation Plan

## Current Architecture (Dual-Write)

```
Import Flow:
JSON/ZIP → Parser → [Markdown Files + SQLite] (parallel writes)
                    ↓                    ↓
              Filesystem            Database
              (source of truth)     (index)
```

**Problems:**
- State drift: Filesystem and DB can get out of sync
- No single source of truth
- Can't regenerate markdown with different templates
- Hard to reprocess with updated logic

## Target Architecture (DB-First)

```
Import Flow:
JSON/ZIP → raw_imports → Parser → messages/conversations → Renderer → Markdown
           (tier 1)                (tier 2)                          (view)
                                    ↓
                                Database
                            (source of truth)
```

**Benefits:**
- Single source of truth
- Regenerate markdown anytime with new templates
- Reprocess historical data without re-importing
- State always consistent
- True "forever archive"

## Implementation Phases

### Phase 1: Enhanced Database Schema ✅ PARTIALLY DONE

**Status:** raw_imports table exists, but we need conversations/messages tables

**What exists:**
- ✅ `raw_imports` table (tier 1 - the sarcophagus)
- ✅ Hash-based deduplication
- ✅ Parse status tracking

**What's needed:**
- [ ] `conversations` table (tier 2 - structured data)
- [ ] `messages` table (tier 2 - structured data)
- [ ] `attachments` table (tier 2 - asset tracking)
- [ ] Relationships and foreign keys

**Schema Design:**

```sql
-- Tier 2: Structured Conversation Data
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    conversation_id TEXT NOT NULL,  -- Provider's ID
    slug TEXT NOT NULL,
    title TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    metadata_json TEXT,  -- Flexible JSON for provider-specific data
    UNIQUE(provider, conversation_id)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    message_id TEXT NOT NULL,  -- Provider's message ID
    parent_id INTEGER REFERENCES messages(id),  -- For threading
    branch_id TEXT,  -- For branch tracking
    position INTEGER,  -- Order in conversation
    role TEXT NOT NULL,  -- user, assistant, system, tool
    content TEXT,  -- Main text content
    content_json TEXT,  -- Structured content (for complex formats)
    model TEXT,  -- Which model generated this
    timestamp INTEGER,
    metadata_json TEXT,  -- Flexible JSON
    UNIQUE(conversation_id, message_id)
);

CREATE TABLE attachments (
    id INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id),
    hash TEXT NOT NULL,  -- Content hash for deduplication
    filename TEXT,
    mime_type TEXT,
    size_bytes INTEGER,
    blob BLOB,  -- Actual file data
    extracted_text TEXT,  -- OCR/extracted content
    UNIQUE(hash)
);

-- Indexes for performance
CREATE INDEX idx_conversations_provider ON conversations(provider);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_attachments_hash ON attachments(hash);
```

### Phase 2: Modify Importers (DB-Only Writes)

**Current:** Importers call both markdown writer AND database functions

**Target:** Importers write ONLY to database

**Files to modify:**
- `polylogue/importers/chatgpt.py`
- `polylogue/importers/claude_ai.py`
- `polylogue/importers/claude_code.py`
- `polylogue/importers/codex.py`
- `polylogue/drive.py`

**Changes:**
```python
# OLD (dual-write):
def import_conversation(data):
    # 1. Parse JSON
    messages = parse_messages(data)

    # 2. Write to filesystem
    write_markdown(messages, output_dir)

    # 3. Write to database
    insert_messages(conn, messages)

# NEW (DB-only):
def import_conversation(data):
    # 1. Store raw first (already done)
    raw_hash = store_raw_import(data, provider="chatgpt")

    # 2. Parse JSON
    try:
        messages = parse_messages(data)

        # 3. Write ONLY to database
        conversation_id = insert_conversation(conn, metadata)
        insert_messages(conn, conversation_id, messages)

        # 4. Mark parse successful
        mark_parse_success(raw_hash)
    except Exception as e:
        mark_parse_failed(raw_hash, str(e))
        raise
```

### Phase 3: Database-First Renderer

**Create new:** `polylogue/renderers/db_renderer.py`

**Purpose:** Read from database, generate markdown/HTML

```python
class DatabaseRenderer:
    """Renders markdown/HTML from database conversations."""

    def render_conversation(
        self,
        conversation_id: int,
        output_dir: Path,
        *,
        format: str = "markdown",  # or "html"
        include_attachments: bool = True,
        branch: Optional[str] = None,
    ) -> Path:
        """Render a single conversation from DB to disk."""

        # 1. Load from database
        conversation = self.load_conversation(conversation_id)
        messages = self.load_messages(conversation_id, branch=branch)
        attachments = self.load_attachments(conversation_id)

        # 2. Build document
        doc = MarkdownDocument()
        doc.add_frontmatter(conversation.metadata)
        for msg in messages:
            doc.add_message(msg)

        # 3. Write to disk
        output_path = output_dir / conversation.slug / "conversation.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(doc))

        # 4. Extract attachments if requested
        if include_attachments:
            self.extract_attachments(attachments, output_dir)

        return output_path

    def render_all(
        self,
        output_dir: Path,
        *,
        provider: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Path]:
        """Render all conversations matching criteria."""
        conversations = self.query_conversations(
            provider=provider,
            since=since,
        )

        results = []
        for conv in conversations:
            path = self.render_conversation(conv.id, output_dir)
            results.append(path)

        return results
```

### Phase 4: Update `polylogue render` Command

**Modify:** `polylogue/cli/render.py`

**Old behavior:** Read JSON file, parse, write markdown
**New behavior:**
1. Check if already in DB → skip
2. If not in DB → import first (via importer)
3. Render from DB

```python
def run_render_cli(args, env):
    """Render command - now DB-first."""

    input_path = args.input
    output_dir = args.out

    # 1. Import to database if needed
    if input_path.is_file():
        # Single file import
        import_to_database(input_path, provider=detect_provider(input_path))
    elif input_path.is_dir():
        # Batch import
        import_directory_to_database(input_path)

    # 2. Render from database
    renderer = DatabaseRenderer(db_path=env.db_path)

    if args.force:
        # Regenerate everything
        paths = renderer.render_all(output_dir)
    else:
        # Only render new/updated
        paths = renderer.render_incremental(
            output_dir,
            since=get_last_render_time(output_dir),
        )

    # 3. Report results
    env.ui.console.print(f"[green]Rendered {len(paths)} conversations")
```

### Phase 5: Migration Path

**For existing users with filesystem data:**

```python
# polylogue/cli/migrate.py

def migrate_filesystem_to_db(
    markdown_dir: Path,
    db_path: Path,
) -> None:
    """Migrate existing markdown files to database.

    Reads frontmatter and content, reconstructs messages,
    stores in database.
    """

    for conv_dir in markdown_dir.glob("*/"):
        md_file = conv_dir / "conversation.md"
        if not md_file.exists():
            continue

        # Parse markdown
        frontmatter, content = parse_markdown_file(md_file)
        messages = extract_messages_from_markdown(content)

        # Insert into DB
        conversation_id = insert_conversation(
            conn,
            provider=frontmatter.get("provider"),
            conversation_id=frontmatter.get("id"),
            slug=conv_dir.name,
            title=frontmatter.get("title"),
            metadata_json=json.dumps(frontmatter),
        )

        for msg in messages:
            insert_message(conn, conversation_id, msg)

        print(f"✓ Migrated {conv_dir.name}")
```

## Async I/O Implementation

### Phase 1: Replace requests with httpx

**Files to modify:**
- `polylogue/drive.py`
- `polylogue/drive_client.py`

**Changes:**
```python
# OLD:
import requests

response = requests.get(url)
data = response.json()

# NEW:
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(url)
    data = response.json()
```

### Phase 2: Parallel Downloads

**Current:** Downloads files sequentially
**Target:** Download multiple files concurrently

```python
import asyncio
import httpx

async def download_batch(
    urls: List[str],
    max_concurrent: int = 5,
) -> List[bytes]:
    """Download multiple files concurrently."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_one(client, url):
        async with semaphore:
            response = await client.get(url)
            return response.content

    async with httpx.AsyncClient() as client:
        tasks = [download_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage in Drive sync:
async def sync_drive_async(chat_ids: List[str]):
    # Get all download URLs
    urls = [get_download_url(chat_id) for chat_id in chat_ids]

    # Download in parallel
    contents = await download_batch(urls, max_concurrent=10)

    # Process
    for chat_id, content in zip(chat_ids, contents):
        process_chat(chat_id, content)
```

### Phase 3: Async File I/O

**Use aiofiles for file operations:**

```python
import aiofiles

# OLD:
with open(path, 'wb') as f:
    f.write(data)

# NEW:
async with aiofiles.open(path, 'wb') as f:
    await f.write(data)
```

## Success Criteria

### Database as Source of Truth
- [ ] All importers write ONLY to database
- [ ] Markdown files generated from database
- [ ] Can delete all markdown and regenerate from DB
- [ ] `polylogue render` reads from DB, not files
- [ ] Migration tool for existing users
- [ ] Schema includes all necessary data
- [ ] Attachments stored in database

### Async I/O
- [ ] httpx replaces requests
- [ ] Parallel downloads implemented
- [ ] Configurable concurrency limits
- [ ] Progress tracking for batch operations
- [ ] Proper error handling and retries
- [ ] Benchmarks show improvement

## Implementation Order

1. **Week 1: Database Schema**
   - Design and create conversations/messages/attachments tables
   - Add migration script for schema updates
   - Test with sample data

2. **Week 2: Database Renderer**
   - Implement DatabaseRenderer class
   - Test rendering from existing DB data
   - Verify output matches current format

3. **Week 3: Modify Importers**
   - Update one importer (chatgpt) to DB-only
   - Test end-to-end import → render flow
   - Rollout to other importers

4. **Week 4: Async I/O**
   - Replace requests with httpx
   - Implement parallel downloads
   - Add progress tracking

5. **Week 5: Migration & Testing**
   - Create filesystem → DB migration tool
   - Comprehensive testing
   - Update documentation

## Rollback Strategy

**During transition:**
- Keep old code paths available behind feature flag
- `POLYLOGUE_DB_FIRST=1` enables new behavior
- Can revert to filesystem-first if issues arise

**After migration:**
- Remove old dual-write code
- Make DB-first the only path
