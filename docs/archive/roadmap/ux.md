# Polylogue UX & Workflow Evolution Roadmap

## Overview

This roadmap outlines 15 UX/workflow improvements spanning CLI enhancements to potential GUI interfaces. Current UX is CLI-only with basic server option. These enhancements enable interactive browsing, persistent state, automated workflows, and multi-platform support.

**Current State**:
- CLI-only primary interface
- Basic FastAPI server (experimental)
- No interactive result browser
- Manual workflow (explicit `polylogue run`)
- No shell integration
- Static HTML rendering

---

## Tier 1: High Impact, Lower Complexity (Quick Wins)

### 1. Interactive Search Browser (TUI)

**Files Affected**: New `ui/search_browser.py`, `cli/commands/search.py`

**Problem**: Search returns list, user must manually inspect each result or use `--open` (slow with browser).

**Solution**: TUI-based result browser using `textual` library:

```bash
# Launch interactive browser
polylogue search "python error handling" --interactive
```

**Features**:
- Scrollable result list (left pane)
- Live preview pane (right pane)
- Arrow keys to navigate
- Enter to open full conversation
- `/` to refine query live
- `?` for help
- `q` to quit

**Implementation** (using `textual`):
```python
# ui/search_browser.py
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, ListView, ListItem
from textual.binding import Binding

class SearchBrowser(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        ("/", "focus_search", "Search"),
        ("enter", "open_result", "Open"),
        ("?", "show_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        """Layout: results list + preview pane."""
        with Container(id="search_browser"):
            yield ResultsList(id="results_list")
            yield PreviewPane(id="preview")

    def on_list_item_selected(self, event: ListView.ItemSelected) -> None:
        """Update preview on selection."""
        result = event.item.data
        self.query_one(PreviewPane).display_result(result)
```

**Success Criteria**:
- Search results loaded in <500ms
- Navigate 100+ results smoothly
- Preview updates instantly on selection

---

### 2. Persistent Search History

**Files Affected**: New `storage/search_history.py`, `cli/commands/search.py`

**Problem**: Users repeat the same searches but have no history.

**Solution**: Store recent queries locally:

```python
# storage/search_history.py
class SearchHistoryRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def record_search(self, query: str, result_count: int, duration_ms: int) -> None:
        """Log search query."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO search_history (query, result_count, timestamp, duration_ms)
            VALUES (?, ?, ?, ?)
        """, (query, result_count, datetime.now().isoformat(), duration_ms))
        conn.commit()

    def get_recent(self, limit: int = 20) -> list[SearchQuery]:
        """Get recent searches."""
        conn = sqlite3.connect(str(self.db_path))
        return conn.execute("""
            SELECT query, result_count, timestamp
            FROM search_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
```

**CLI Integration**:
```bash
# View recent searches
polylogue search --history
# Output:
# 1. "python async" (12 results, 2 hours ago)
# 2. "rust debugging" (5 results, 1 day ago)
# 3. "nix flake" (8 results, 3 days ago)

# Fuzzy search history
polylogue search --history | fzf | xargs polylogue search

# Ctrl+R in interactive mode for quick recall
```

**Success Criteria**:
- Store 100+ search queries
- Fast fuzzy search (<100ms)

---

### 3. Rich Progress Reporting

**Files Affected**: `cli/commands/run.py`, `ui/facade.py`

**Current State**: Basic spinner via `rich.progress`

**Enhancement**: Multi-stage progress bars with metrics:

```bash
polylogue run --preview

# Output:
# Polylogue Pipeline Progress
# ═══════════════════════════════════════════════════════════
#
# ⧖ Ingestion [████████░░░░░░░░░░░░░░░░░░░░░░] 45%
#   └─ 450/1000 conversations | 1.2s elapsed | ~1.5s remaining
#      Throughput: 375 conv/s | Peak memory: 45MB
#
# ⧖ Rendering [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 12%
#   └─ 12/100 conversations rendered
#
# ⧖ Indexing [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%
#   └─ Waiting...
#
# Overall: 19% complete | 8.3s / 45s estimated
```

**Implementation**:
```python
# ui/progress.py
class PipelineProgress:
    def __init__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

    def create_ingest_task(self, total: int) -> TaskID:
        return self.progress.add_task("[cyan]Ingestion", total=total)

    def update_metrics(self, task_id: TaskID, **kwargs) -> None:
        """Update custom metrics."""
        # Throughput, memory, etc.
```

**Success Criteria**:
- Show 4+ metrics per stage (throughput, memory, ETA)
- Update every 500ms (smooth animation)

---

### 4. Colorized Output Themes

**Files Affected**: `ui/facade.py`, `cli/click_app.py`

**Current State**: Single theme in facade.py

**Enhancement**: Theme selection via `--theme` flag:

```bash
polylogue run --theme dark  # Default
polylogue run --theme light
polylogue run --theme minimal  # Monochrome
polylogue view --theme dark

# Persist preference
polylogue config set theme=light
```

**Themes**:
- **dark**: Bold colors (default for terminal)
- **light**: Inverted for light backgrounds
- **minimal**: Monochrome, no colors

**Implementation**:
```python
# ui/themes.py
THEMES = {
    "dark": Theme({
        "info": "cyan",
        "success": "green bold",
        "warning": "yellow",
        "error": "red bold",
        "highlight": "magenta bold",
    }),
    "light": Theme({
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "highlight": "magenta",
    }),
    "minimal": Theme({}),  # No colors
}
```

**Success Criteria**:
- Switch themes without restart
- Persist preference in config

---

### 5. Conversation Bookmarks

**Files Affected**: New `storage/bookmarks.py`, `cli/commands/bookmark.py`

**Problem**: Users have "favorite" conversations but no quick access mechanism.

**Solution**: Bookmarks with tags:

```python
# storage/bookmarks.py
@dataclass
class Bookmark:
    bookmark_id: str
    conversation_id: str
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
```

**CLI Commands**:
```bash
# Add bookmark
polylogue bookmark add abc123 --tag "important" --tag "python"
polylogue bookmark add abc123 --notes "Async patterns deep dive"

# List bookmarks
polylogue bookmark list
# Output:
# 1. [python] "Understanding async/await" - abc123
# 2. [important] "Debugging session" - def456

# List by tag
polylogue bookmark list --tag important

# Search bookmarks
polylogue bookmark search "async"

# Remove bookmark
polylogue bookmark remove abc123
```

**Success Criteria**:
- Fast bookmark operations (<100ms)
- Support 100+ bookmarks

---

## Tier 2: Medium Impact, Medium Complexity

### 6. Watch Mode for Auto-Import

**Files Affected**: New `daemon/watcher.py`, `cli/commands/watch.py`

**Problem**: Manual `polylogue run` workflow; users forget to import new exports.

**Solution**: Daemon that auto-imports on file changes:

```bash
# Start watch daemon
polylogue watch --inbox ~/Downloads/chats
# Watches for new files, auto-imports

# With notifications
polylogue watch --inbox ~/Downloads/chats --notify
# Desktop notification: "Imported 3 conversations from ChatGPT export"

# Scheduled import
polylogue watch --schedule "0 */6 * * *"  # Every 6 hours
```

**Implementation** (using `watchfiles`):
```python
# daemon/watcher.py
from watchfiles import run_process, watch
from pathlib import Path

class ArchiveWatcher:
    def __init__(self, inbox_path: Path, service: IngestionService):
        self.inbox_path = inbox_path
        self.service = service

    def start(self) -> None:
        """Watch inbox for changes."""
        for changes in watch(self.inbox_path):
            for change_type, path in changes:
                if change_type == Change.added:
                    self._on_file_added(Path(path))

    def _on_file_added(self, path: Path) -> None:
        """Handle new file."""
        logger.info(f"New file detected: {path}")
        result = self.service.ingest_source(path)
        logger.info(f"Imported {result.conversation_count} conversations")

        # Desktop notification
        notify(
            title="Polylogue Import",
            message=f"Imported {result.conversation_count} conversations"
        )
```

**Success Criteria**:
- Detect file changes <500ms
- Auto-import within 1 second
- Desktop notifications working on Linux/macOS/Windows

---

### 7. Workspace/Profile Support

**Files Affected**: `config.py`, `cli/click_app.py`, `storage/db.py`

**Problem**: Single global config; can't separate personal vs work conversations.

**Solution**: Named profiles with separate archives:

```bash
# Create profiles
polylogue profile create personal
polylogue profile create work

# Switch profile
polylogue --profile work run
polylogue --profile personal search "python"

# Set default
polylogue profile set-default personal

# List profiles
polylogue profile list
# Output:
# - personal (default, 450 conversations)
# - work (1200 conversations)
# - projects (230 conversations)
```

**Configuration Structure**:
```
~/.config/polylogue/
├── config.json (global settings)
├── profiles/
│   ├── personal.json (personal archive config)
│   ├── work.json (work archive config)
│   └── projects.json
```

**Implementation**:
```python
# config.py
@dataclass
class ProfileConfig:
    name: str
    archive_root: Path
    default: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class Config:
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)
    default_profile: str = "default"

    def get_profile(self, name: str) -> ProfileConfig:
        return self.profiles.get(name) or ProfileConfig(name=name, archive_root=...)
```

**Success Criteria**:
- Support 10+ profiles
- Switch profiles instantly (<100ms)
- Separate archives without conflicts

---

### 8. Interactive HTML with Client-Side Search

**Files Affected**: `rendering/renderers/html.py`, new `rendering/static/search.js`

**Problem**: Static HTML output; users must use CLI for search.

**Enhancement**: Embed client-side search using lunr.js:

```html
<!-- Generated HTML includes embedded search -->
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js"></script>
    <style>
        #search-box {
            position: sticky;
            top: 0;
        }
    </style>
</head>
<body>
    <input type="text" id="search-box" placeholder="Search conversations...">
    <div id="search-results"></div>
    <div id="conversations">
        <!-- Conversation content -->
    </div>

    <script>
        // Build search index from message text
        const index = lunr(function() {
            this.field('text');
            this.field('role');
            // ... add all messages
        });

        // Live search on keyup
        document.getElementById('search-box').addEventListener('keyup', (e) => {
            const results = index.search(e.target.value);
            displayResults(results);
        });
    </script>
</body>
</html>
```

**Implementation**:
```python
# rendering/renderers/html.py
def render(self, conversation_id: str, output_path: Path) -> Path:
    """Render with embedded search index."""
    conv = self.repository.get_conversation(conversation_id)

    # Generate search index
    search_index = {
        "messages": [
            {"id": m.message_id, "text": m.text, "role": m.role}
            for m in conv.messages
        ]
    }

    # Render template with index
    html = self.template.render(
        conversation=conv,
        search_index=json.dumps(search_index),
    )

    output_path.write_text(html)
    return output_path
```

**Success Criteria**:
- Search 1000-message conversation: <100ms
- Keyboard navigation (arrow keys, Enter)
- Filter by role (user/assistant)

---

### 9. Selective Export with Redaction

**Files Affected**: `cli/commands/export.py`, `export/exporter.py`

**Problem**: Export exports everything; can't share subsets without exposing sensitive data.

**Solution**: Selective export with optional redaction:

```bash
# Export specific conversations
polylogue export --conversations abc123,def456 --out subset.jsonl

# Export with filtering
polylogue export --provider claude --since "2024-01" --out monthly.jsonl

# Redact sensitive patterns
polylogue export --redact-pattern "API_KEY|password" --out safe.jsonl

# Combine
polylogue export --conversations abc123 --redact strict --out shared.jsonl
```

**Implementation**:
```python
# export/exporter.py
class ExportFilter:
    def __init__(self,
                 conversation_ids: list[str] | None = None,
                 providers: list[str] | None = None,
                 since: datetime | None = None,
                 redact_mode: Literal["off", "warn", "redact", "strict"] = "off"):
        self.conversation_ids = conversation_ids
        self.providers = providers
        self.since = since
        self.redact_mode = redact_mode

    def should_export(self, conversation: Conversation) -> bool:
        """Check if conversation matches filters."""
        if self.conversation_ids and conversation.conversation_id not in self.conversation_ids:
            return False
        if self.providers and conversation.provider_name not in self.providers:
            return False
        if self.since and conversation.created_at < self.since:
            return False
        return True
```

**Success Criteria**:
- Export specific conversations: <1 second
- Redaction accurate: >95% precision

---

### 10. Shell Integration (fzf/rofi)

**Files Affected**: `cli/commands/pick.py`, shell completions

**Problem**: No keyboard-friendly access to conversations.

**Solution**: `polylogue pick` for shell integration:

```bash
# Interactive picker
polylogue pick
# Opens fzf with conversation list, outputs ID to stdout

# Usage examples
polylogue pick | xargs polylogue view  # Pick then view

polylogue search "python" | polylogue pick | xargs polylogue view

# With aliases
alias pp='polylogue pick | xargs polylogue view'
alias ps='polylogue search'

# Integrate with shell
# In ~/.bashrc:
# source <(polylogue completions bash)
# Or in ~/.config/fish/config.fish:
# polylogue completions fish | source
```

**Implementation**:
```python
# cli/commands/pick.py
@click.command()
@click.option('--provider', help='Filter by provider')
@click.option('--since', help='Filter by date')
def pick(provider, since):
    """Pick a conversation interactively."""
    convs = repository.iter_conversations(
        provider_name=provider,
        after=parse_date(since) if since else None
    )

    # Format for fzf
    lines = [f"{c.conversation_id} {c.title}" for c in convs]

    # Pipe to fzf
    result = click.prompt("Search", default="")  # Fallback if fzf unavailable

    # Output ID only (for xargs)
    if result:
        conversation_id = result.split()[0]
        click.echo(conversation_id)
```

**Success Criteria**:
- Integrate with fzf/rofi
- Fast (<100ms) for 1000+ conversations

---

## Tier 3: Strategic Features (Higher Impact, Higher Complexity)

### 11. TUI Browser (Full Application)

**Files Affected**: New `ui/tui_app.py`, entirely new TUI interface

**Problem**: Full potential requires GUI, but TUI is more accessible.

**Solution**: Full-featured TUI using `textual`:

```
┌─ Polylogue Browser ─────────────────────────────────────────────┐
│ Search: _________________________ [Filters ▼] [Bookmarks ▼]    │
├─ Conversations ─────────────────────────────────┬─ Preview ────┐
│ python async: 45 msgs, claude, 2h ago          │ From abc123  │
│ rust debugging: 23 msgs, chatgpt, 1d ago       │              │
│ > nix flake: 12 msgs, claude, 3d ago           │ What's the   │
│ deployment: 8 msgs, gemini, 1w ago             │ fastest way  │
│                                                 │ to learn    │
│ [1/1234 conversations]                        │ Rust?       │
├─────────────────────────────────────────────────┴──────────────┤
│ [Enter: view] [Ctrl+B: bookmark] [Ctrl+S: save] [Q: quit]    │
└─────────────────────────────────────────────────────────────────┘
```

**Features**:
- Left panel: Conversation list with filters
- Main panel: Message viewer with semantic projections toggle
- Bottom: Command bar
- Keyboard-driven (no mouse required)
- Syntax highlighting for code
- Collapsible code blocks

**Implementation**: Full `textual` application with bindings

**Success Criteria**:
- Smooth scrolling for 1000+ conversations
- All search/filter features accessible
- Mouse optional, not required

---

### 12. Local Web UI (Enhanced Server)

**Files Affected**: `server/api.py`, `server/web.py`, new frontend SPA

**Problem**: Current server is experimental; web UI needs polish.

**Enhancement**: Modern web interface with full feature parity:

```
┌─ Polylogue Web ─────────────────────────────────────────────┐
│  [🔍] Search _________________________                       │
│        Filters: [Provider ▼] [Date ▼] [Topics ▼]          │
│                                                             │
│  Results: 42 conversations                                 │
│                                                             │
│  ┌─ Claude: Async Patterns ─────────────────────────────┐ │
│  │ 45 messages | 3 days ago | Tags: python, async     │ │
│  │ Preview: What's the relationship between asyncio... │ │
│  │ [View Full] [Bookmark] [Similar]                   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ ChatGPT: Debugging Session ──────────────────────────┐ │
│  │ 23 messages | 1 week ago | Tags: debugging         │ │
│  │ Preview: I'm getting a KeyError in my code...      │ │
│  │ [View Full] [Bookmark] [Similar]                   │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Features**:
- Full-text search box with autocomplete
- Filter sidebar (provider, date range, topics)
- Conversation preview cards
- Responsive mobile layout
- Keyboard shortcuts
- Permalink anchors for sharing

**Tech Stack**:
- Backend: FastAPI (existing)
- Frontend: HTMX + TailwindCSS (lightweight, no heavy JS framework)
- Real-time: WebSocket for live updates

**Success Criteria**:
- Load conversation list: <500ms
- Search results: <1 second
- Mobile responsive

---

### 13. Obsidian/Logseq Integration

**Files Affected**: New `export/obsidian_exporter.py`

**Problem**: Markdown output doesn't integrate with knowledge management tools.

**Solution**: Export with Obsidian/Logseq-compatible metadata:

```bash
polylogue export --format obsidian --output ~/vault/chats/

# Generates:
# ~/vault/chats/
# ├── Claude/
# │   ├── 2024-01-15 - Async Patterns.md
# │   └── 2024-01-16 - Debugging.md
# ├── ChatGPT/
# └── _index.md (MOC)
```

**Features**:
- Wikilinks between related conversations: `[[Async Patterns]]`
- YAML frontmatter with metadata
- Tag extraction: `#python`, `#async`
- Backlink-friendly filenames: `YYYY-MM-DD - Title.md`
- Master MOC (Map of Contents)
- Dataview support for statistics

**Implementation**:
```markdown
---
title: "Async Patterns Deep Dive"
date: 2024-01-15
provider: claude
topics: [python, async, concurrency]
messages: 45
tags: [python, async, learning]
---

# Async Patterns Deep Dive

Related: [[Debugging]], [[Concurrency Primitives]]

## Conversation

**User**: What's the relationship between asyncio and threading?

**Assistant**: asyncio is a single-threaded concurrency model...

---
Generated by Polylogue | [[Index]]
```

**Success Criteria**:
- Export with wikilinks
- Tags extracted from content
- MOC auto-generated

---

### 14. Scheduled Import + Digest Emails

**Files Affected**: New `scheduler/`, `notifications/email.py`

**Problem**: Users don't track conversation patterns; no summary.

**Solution**: Scheduled imports + weekly digest emails:

```bash
# Configure schedule
polylogue schedule --import "0 */6 * * *"  # Every 6 hours
polylogue schedule --digest "0 9 * * MON"  # Monday 9am

# Digest email includes:
# - New conversations (past week)
# - Most active topics
# - Keyword alerts ("python" if not discussed recently)
# - Statistics
```

**Digest Email Content**:
```
Subject: Polylogue Digest - Week of Jan 15

Your AI Chat Summary
═════════════════════

New Conversations (7)
- Claude: "Async/await patterns" (45 messages)
- ChatGPT: "Debugging session" (23 messages)
- ...

Most Active Topics
1. Python (98 messages)
2. Rust (45 messages)
3. DevOps (32 messages)

Statistics
- Total conversations: 1,234
- This week: +7 conversations
- Most used provider: Claude (65%)

Keyword Alerts
- "Testing" not discussed in 30 days (usually weekly)
- Suggestion: Review testing approaches

[View Archive] [Configure] [Unsubscribe]
```

**Implementation**:
```python
# scheduler/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler

class ScheduledTasks:
    def __init__(self, config: Config):
        self.scheduler = BackgroundScheduler()
        self._register_jobs(config)

    def _register_jobs(self, config: Config) -> None:
        """Register scheduled tasks."""
        if config.schedule_import:
            self.scheduler.add_job(
                self._scheduled_import,
                'cron',
                **parse_cron(config.schedule_import),
                id='scheduled_import'
            )

        if config.schedule_digest:
            self.scheduler.add_job(
                self._scheduled_digest,
                'cron',
                **parse_cron(config.schedule_digest),
                id='scheduled_digest'
            )

        self.scheduler.start()
```

**Success Criteria**:
- Scheduled imports working reliably
- Digest emails sent at correct time
- Configuration UI

---

### 15. Browser Extension (One-Click Export)

**Files Affected**: New `browser_extension/`, `server/api.py`

**Problem**: Manual export from ChatGPT/Claude web UI is tedious.

**Solution**: Browser extension with one-click export:

```
ChatGPT Web UI Button: [Export to Polylogue ↓]

Clicking opens sidebar:
- Shows connection status to local Polylogue daemon
- Export format options: Full conversation, Latest N messages
- Auto-import option
- Send button
```

**Architecture**:
```
Browser Extension
    ↓ (HTTP POST to localhost:8000/api/import)
    ↓
Polylogue Daemon
    ↓
Import Pipeline
    ↓
SQLite Database
```

**Extension Code** (manifest.json):
```json
{
  "manifest_version": 3,
  "name": "Polylogue Export",
  "permissions": ["activeTab", "scripting", "storage"],
  "host_permissions": ["http://localhost:8000/*"],
  "action": {
    "default_popup": "popup.html",
    "default_scripts": ["popup.js"]
  }
}
```

**Implementation**:
```javascript
// popup.js
document.getElementById('export-btn').addEventListener('click', async () => {
    // Extract conversation from page
    const conversation = extractConversation();

    // Send to Polylogue daemon
    const response = await fetch('http://localhost:8000/api/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(conversation)
    });

    if (response.ok) {
        showNotification('Exported to Polylogue');
    } else {
        showError('Failed to export');
    }
});
```

**Success Criteria**:
- One-click export from ChatGPT/Claude web
- Works with Chrome/Firefox/Safari
- Daemon auto-detection

---

## Implementation Roadmap

### Phase 1 (Weeks 1-2): CLI Polish
- Interactive search browser
- Search history
- Rich progress
- Colorized themes

**Effort**: 40 hours | **Dependencies**: None

### Phase 2 (Weeks 3-4): Workflow Automation
- Bookmarks
- Watch mode
- Profiles

**Effort**: 30 hours | **Dependencies**: Phase 1 optional

### Phase 3 (Weeks 5-6): Advanced Export
- Interactive HTML
- Selective export
- Shell integration
- Obsidian export

**Effort**: 50 hours | **Dependencies**: Phase 1 (for UI polish)

### Phase 4 (Weeks 7-10): GUI & Automation
- TUI browser
- Web UI enhancement
- Scheduled tasks
- Digest emails

**Effort**: 80 hours | **Dependencies**: Phase 1-2

### Phase 5 (Weeks 11+): Browser Extension
- Extension development
- Sidebar UX
- Multi-browser support

**Effort**: 60 hours | **Dependencies**: Phase 4

---

## Priority Matrix

| # | Feature | Impact | Complexity | ROI | Target Users |
|---|---------|--------|-----------|-----|--------------|
| 1 | Interactive Browser | High | Low | 10:1 | Power users |
| 2 | Search History | High | Low | 8:1 | All |
| 3 | Progress Reporting | Medium | Low | 6:1 | All |
| 4 | Color Themes | Low | Low | 4:1 | Customizers |
| 5 | Bookmarks | Medium | Low | 5:1 | Researchers |
| 6 | Watch Mode | High | Medium | 6:1 | Heavy users |
| 7 | Profiles | Medium | Medium | 4:1 | Multi-use |
| 8 | Interactive HTML | High | Medium | 5:1 | Web users |
| 9 | Selective Export | Medium | Medium | 4:1 | Sharers |
| 10 | Shell Integration | Medium | Low | 5:1 | CLI power users |
| 11 | TUI Browser | High | High | 4:1 | Terminal users |
| 12 | Web UI | High | High | 4:1 | Web users |
| 13 | Obsidian Export | Medium | Medium | 3:1 | PKM users |
| 14 | Scheduled Tasks | Medium | Medium | 3:1 | Automators |
| 15 | Browser Ext | High | High | 3:1 | Heavy exporters |

---

## Success Metrics

- **Discoverability**: Users find UI features without documentation
- **Workflow**: Typical task completion time reduced by 50%
- **Satisfaction**: UI NPS score 8+/10
- **Adoption**: 60%+ use interactive features vs CLI-only
- **Accessibility**: Works with keyboard-only (no mouse required)
