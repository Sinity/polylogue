# UX Improvements Roadmap

This document analyzes the remaining UX improvement opportunities identified during the comprehensive streamlining initiative. Each improvement is evaluated for complexity, effort, user impact, and implementation approach.

## Quick Reference Matrix

| Improvement | Complexity | Effort | Impact | Priority | Status |
|------------|------------|--------|--------|----------|--------|
| Parent Parser | Low | 2-4h | Medium | High | ⭐ Quick Win |
| --examples Flag | Low | 2-3h | Medium | Medium | ⭐ Quick Win |
| --open Flag | Medium | 4-6h | Medium | Medium | ⚡ Medium Value |
| Path Standardization | Medium | 6-8h | Medium | Medium | ⚡ Medium Value |
| config init | Medium | 6-10h | High | Medium | ⚡ Medium Value |
| JSON Error Messages | Medium-High | 8-12h | Medium | Low | 🔧 Refactor |
| Progress Bars | Medium-High | 10-16h | High | High | 🚀 High Value |
| --all-providers | High | 12-20h | High | Low | 🏗️ Feature |

## 1. Parent Parser for Common Flags ⭐

**Complexity:** Low
**Estimated Effort:** 2-4 hours
**User Impact:** Medium (DRY principle, easier maintenance)
**Priority:** High - Quick win with ongoing benefits

### Problem
Common flags (`--json`, `--html`, `--collapse-threshold`, `--force`, `--allow-dirty`) are defined separately for each command, creating duplication and maintenance burden.

### Solution
Create parent parsers in `arg_helpers.py` for flag groups:
- `common_output_parser`: `--json`, `--html`, `--out`
- `common_write_parser`: `--force`, `--allow-dirty`, `--dry-run`
- `common_render_parser`: `--collapse-threshold`, `--theme`

### Implementation
```python
# polylogue/cli/arg_helpers.py
def create_output_parent_parser() -> ArgumentParser:
    """Parent parser for common output flags."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    add_html_option(parser)
    return parser

def create_write_parent_parser() -> ArgumentParser:
    """Parent parser for common write flags."""
    parser = ArgumentParser(add_help=False)
    add_force_option(parser)
    add_allow_dirty_option(parser)
    add_dry_run_option(parser)
    return parser
```

Usage:
```python
# polylogue/cli/app.py
output_parent = create_output_parent_parser()
write_parent = create_write_parent_parser()

p_render = sub.add_parser("render", parents=[output_parent, write_parent], ...)
```

### Benefits
- Reduces code duplication (~50 lines)
- Ensures flag consistency automatically
- Easier to add new common flags
- Clearer command structure

### Risks
- Low - argparse supports parent parsers well
- Need to verify flag ordering doesn't break existing tests

---

## 2. --examples Flag for Help Command ⭐

**Complexity:** Low
**Estimated Effort:** 2-3 hours
**User Impact:** Medium (improved onboarding)
**Priority:** Medium - Quick win for discoverability

### Problem
Users need concrete usage examples to understand command combinations and workflows.

### Solution
Add `polylogue help <command> --examples` to show real-world usage patterns.

### Implementation
```python
# polylogue/cli/help.py (new file or add to app.py)
EXAMPLES = {
    "sync": [
        ("Sync all Drive chats", "polylogue sync drive --all"),
        ("Sync specific folder", "polylogue sync drive --folder-name 'Work Chats'"),
        ("Preview before sync", "polylogue sync codex --dry-run"),
    ],
    "import": [
        ("Import ChatGPT export", "polylogue import chatgpt export.zip --html"),
        ("Import with picker", "polylogue import claude-code pick"),
        ("Import specific conversation", "polylogue import chatgpt export.zip --conversation-id abc123"),
    ],
    # ... more commands
}

def show_examples(command: str) -> None:
    if command not in EXAMPLES:
        print(f"No examples available for '{command}'")
        return

    print(f"\n{command.upper()} EXAMPLES\n")
    for desc, cmd in EXAMPLES[command]:
        print(f"  # {desc}")
        print(f"  $ {cmd}\n")
```

### Benefits
- Improves onboarding experience
- Reduces documentation lookups
- Shows best practices

### Risks
- Very low - purely additive
- Need to maintain examples as flags change

---

## 3. --open Flag for Editor Integration ⚡

**Complexity:** Medium
**Estimated Effort:** 4-6 hours
**User Impact:** Medium (workflow enhancement)
**Priority:** Medium

### Problem
Users often want to immediately edit search results or view conversation markdown in their editor after running inspect commands.

### Solution
Add `--open` flag to `inspect search` and `inspect branches` that opens the result in `$EDITOR`.

### Implementation
```python
# polylogue/cli/editor.py (new file)
import os
import subprocess
from pathlib import Path
from typing import Optional

def get_editor() -> Optional[str]:
    """Get user's preferred editor from environment."""
    return os.environ.get("EDITOR") or os.environ.get("VISUAL")

def open_in_editor(path: Path, line: Optional[int] = None) -> bool:
    """Open file in user's editor, optionally at specific line."""
    editor = get_editor()
    if not editor:
        return False

    # Handle editor-specific line syntax
    if line:
        if "vim" in editor or "nvim" in editor:
            cmd = [editor, f"+{line}", str(path)]
        elif "code" in editor or "subl" in editor:
            cmd = [editor, f"{path}:{line}"]
        else:
            cmd = [editor, str(path)]
    else:
        cmd = [editor, str(path)]

    try:
        subprocess.run(cmd)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

# In app.py:
p_inspect_search.add_argument("--open", action="store_true",
                             help="Open result in $EDITOR after search")

# In search handler:
if getattr(args, "open", False) and result_path:
    if not open_in_editor(result_path):
        console.print("[yellow]Warning: Could not open editor (set $EDITOR)")
```

### Benefits
- Smoother workflow integration
- Reduces context switching
- Common in CLI tools (git, grep, etc.)

### Risks
- Medium - Editor support varies widely
- Need to handle missing $EDITOR gracefully
- Different editors have different line-jump syntax

---

## 4. Path Handling Standardization ⚡

**Complexity:** Medium
**Estimated Effort:** 6-8 hours
**User Impact:** Medium (consistency)
**Priority:** Medium

### Problem
Inconsistent behavior when paths don't exist:
- Some commands error immediately
- Some create directories silently
- Some prompt for confirmation
- Error messages vary

### Current State
```python
# stats.py - errors immediately
if not directory.exists():
    console.print(f"[red]Directory not found: {directory}")
    raise SystemExit(1)

# imports.py - creates with flag
if create_base_dir:
    base_dir.mkdir(parents=True, exist_ok=True)

# render.py - creates silently
output_dir.mkdir(parents=True, exist_ok=True)
```

### Solution
Standardize via helper function:
```python
# polylogue/cli/arg_helpers.py
@dataclass
class PathPolicy:
    """Policy for handling missing paths."""
    should_exist: bool = True       # Error if missing
    create_if_missing: bool = False # Auto-create
    prompt_create: bool = False     # Ask before creating

    @staticmethod
    def must_exist() -> PathPolicy:
        """Path must exist (read operations)."""
        return PathPolicy(should_exist=True)

    @staticmethod
    def create_ok() -> PathPolicy:
        """Auto-create if missing (write operations)."""
        return PathPolicy(should_exist=False, create_if_missing=True)

    @staticmethod
    def prompt_create() -> PathPolicy:
        """Ask before creating (interactive operations)."""
        return PathPolicy(should_exist=False, prompt_create=True)

def resolve_path(path: Path, policy: PathPolicy, ui) -> Optional[Path]:
    """Resolve path according to policy."""
    if path.exists():
        return path

    if policy.should_exist:
        ui.console.print(f"[red]Error: Path not found: {path}")
        ui.console.print(f"[dim]Create it with: mkdir -p {path}")
        return None

    if policy.create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
        return path

    if policy.prompt_create and not ui.plain:
        if ui.console.confirm(f"Create directory {path}?"):
            path.mkdir(parents=True, exist_ok=True)
            return path
        return None

    return path
```

Usage:
```python
# For stats (read operation)
directory = resolve_path(Path(args.dir), PathPolicy.must_exist(), ui)
if not directory:
    raise SystemExit(1)

# For render (write operation)
output_dir = resolve_path(Path(args.out), PathPolicy.create_ok(), ui)
```

### Benefits
- Consistent UX across all commands
- Clear error messages with hints
- Easier to understand code intent

### Risks
- Medium - Touches many commands
- Need thorough testing of all path-handling code
- May change existing behavior (document in commit)

---

## 5. polylogue config init Command ⚡

**Complexity:** Medium
**Estimated Effort:** 6-10 hours
**User Impact:** High (onboarding)
**Priority:** Medium

### Problem
New users don't know:
- Where config files live
- What settings are available
- How to set up Drive credentials
- Best practices for directory structure

### Solution
Interactive first-run setup:
```
$ polylogue config init

Welcome to Polylogue! Let's set up your configuration.

Configuration will be saved to: ~/.config/polylogue/

[1/5] Output Directory
Where should rendered conversations be saved?
  > ~/polylogue-data (recommended)
    ~/Documents/polylogue
    Custom path...

[2/5] HTML Previews
Generate HTML previews by default?
  > Yes (recommended for browsing)
    No (Markdown only)

[3/5] Theme
Choose HTML theme:
    Light
  > Dark

[4/5] Google Drive (optional)
Set up Drive sync now?
  > Skip for now
    Set up credentials

[5/5] Summary
✓ Output directory: ~/polylogue-data
✓ HTML previews: enabled (dark theme)
✓ Drive: not configured (run 'polylogue sync drive' when ready)

Configuration saved! Try:
  polylogue import chatgpt export.zip
  polylogue sync codex --all
  polylogue help
```

### Implementation
```python
# polylogue/cli/init.py (new file)
def run_init_cli(args, env):
    """Interactive configuration wizard."""
    ui = env.ui

    if SETTINGS_PATH.exists() and not args.force:
        if not ui.console.confirm("Config exists. Overwrite?", default=False):
            return

    # Step 1: Output directory
    default_dir = Path.home() / "polylogue-data"
    output_dir = ui.console.input(
        f"Output directory [{default_dir}]: ",
        default=str(default_dir)
    )

    # Step 2: HTML previews
    html_enabled = ui.console.confirm("Enable HTML previews?", default=True)

    # Step 3: Theme
    theme = ui.console.choose("Choose theme:", ["light", "dark"])

    # Step 4: Drive setup (optional)
    setup_drive = ui.console.confirm("Set up Google Drive?", default=False)
    if setup_drive:
        # Guide through Drive setup...
        pass

    # Save settings
    settings = Settings(html_previews=html_enabled, html_theme=theme)
    persist_settings(settings)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    ui.console.print("\n[green]✓ Configuration saved!")
    ui.console.print(f"\nTry these commands:")
    ui.console.print("  polylogue help")
    ui.console.print("  polylogue import chatgpt export.zip")
```

### Benefits
- Drastically improved onboarding
- Reduces support questions
- Encourages best practices
- Builds user confidence

### Risks
- Medium complexity
- Need to handle both interactive and `--json` modes
- Should be optional (not required)

---

## 6. JSON Mode Error Messages 🔧

**Complexity:** Medium-High
**Estimated Effort:** 8-12 hours
**User Impact:** Medium (automation/scripting)
**Priority:** Low (works but could be better)

### Problem
Errors in `--json` mode print to console as colored text instead of being included in JSON output. Scripts can't parse errors programmatically.

Current behavior:
```bash
$ polylogue import chatgpt missing.zip --json
[red]Error: File not found: missing.zip
# JSON output missing
```

Desired behavior:
```bash
$ polylogue import chatgpt missing.zip --json
{"status": "error", "code": "file_not_found", "message": "File not found: missing.zip", "path": "missing.zip"}
```

### Solution
Create error handling wrapper:
```python
# polylogue/cli/json_output.py (new file)
from typing import Any, Dict, Optional
import json
import sys

class JSONModeError(Exception):
    """Error that should be rendered as JSON."""
    def __init__(self, code: str, message: str, **kwargs):
        self.code = code
        self.message = message
        self.details = kwargs
        super().__init__(message)

def json_error(code: str, message: str, **details) -> Dict[str, Any]:
    """Format error as JSON."""
    return {
        "status": "error",
        "code": code,
        "message": message,
        **details
    }

def emit_json_or_error(payload: Optional[Dict], error: Optional[Exception], ui):
    """Emit JSON output or error."""
    if ui.plain or getattr(ui, "json_mode", False):
        if error:
            if isinstance(error, JSONModeError):
                output = json_error(error.code, error.message, **error.details)
            else:
                output = json_error("unknown", str(error))
        else:
            output = payload or {}

        print(json.dumps(output, indent=2))
    elif error:
        ui.console.print(f"[red]Error: {error}")

# Wrap command handlers:
def safe_json_handler(handler):
    """Decorator to handle JSON mode errors."""
    def wrapper(args, env):
        try:
            return handler(args, env)
        except JSONModeError as e:
            emit_json_or_error(None, e, env.ui)
            raise SystemExit(1)
        except Exception as e:
            if getattr(args, "json", False):
                emit_json_or_error(None, e, env.ui)
            raise
    return wrapper
```

Usage:
```python
# In imports.py:
@safe_json_handler
def run_import_chatgpt(args, env):
    export_path = Path(args.source[0])
    if not export_path.exists():
        raise JSONModeError("file_not_found",
                           f"File not found: {export_path}",
                           path=str(export_path))
```

### Benefits
- Machine-readable errors
- Better for CI/CD pipelines
- Consistent error format
- Error codes for programmatic handling

### Risks
- Medium-high effort (touches many commands)
- Need to preserve existing UX for non-JSON mode
- Requires identifying all error paths
- Exit codes need to be consistent

---

## 7. Progress Bars for Multi-file Operations 🚀

**Complexity:** Medium-High
**Estimated Effort:** 10-16 hours
**User Impact:** High (user experience during long operations)
**Priority:** High for large-scale users

### Problem
Long operations (Drive sync with 100+ chats, batch imports) provide no progress feedback:
- User doesn't know if process is working
- No ETA for completion
- Can't see which files are being processed

### Solution
Integrate Rich progress bars:
```python
# polylogue/cli/progress.py (new file)
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn
)

def create_sync_progress(ui) -> Progress:
    """Create progress bar for sync operations."""
    if ui.plain:
        return None

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

# In sync handler:
with create_sync_progress(ui) as progress:
    task = progress.add_task("Syncing conversations...", total=len(conversations))

    for conv in conversations:
        progress.update(task, description=f"Syncing {conv.title}")
        # ... process conversation ...
        progress.advance(task)
```

Enhanced version with multiple tasks:
```python
with Progress(...) as progress:
    download_task = progress.add_task("Downloading...", total=len(drive_chats))
    render_task = progress.add_task("Rendering...", total=len(drive_chats))

    for chat in drive_chats:
        # Download
        progress.update(download_task, description=f"⬇️  {chat.name}")
        download_chat(chat)
        progress.advance(download_task)

        # Render
        progress.update(render_task, description=f"📝 {chat.name}")
        render_chat(chat)
        progress.advance(render_task)
```

### Implementation Points
1. **Conditional display**: Only show in interactive mode
2. **Graceful degradation**: Fall back to simple print in `--plain`
3. **JSON mode**: No progress bars, use event logging
4. **Nested progress**: Support for multi-stage operations

### Benefits
- Massively improved UX for long operations
- Reduces perceived wait time
- Shows ETA for planning
- Identifies slow operations (bottleneck detection)

### Risks
- Medium-high effort (needs to touch sync, import, render)
- Progress bars can break in some terminals
- Need to handle interruption (Ctrl+C) gracefully
- Testing is harder with progress bars

---

## 8. sync --all-providers Batch Command 🏗️

**Complexity:** High
**Estimated Effort:** 12-20 hours
**User Impact:** High (for power users with multiple providers)
**Priority:** Low (niche use case)

### Problem
Users with multiple providers need to run separate sync commands:
```bash
polylogue sync codex --all
polylogue sync claude-code --all
polylogue sync chatgpt --all
polylogue sync claude --all
polylogue sync drive --all
```

### Desired Behavior
```bash
# Sync all configured providers
polylogue sync --all-providers

# Or be selective
polylogue sync --providers codex,claude-code,drive
```

### Implementation
```python
# In app.py:
p_sync.add_argument("--all-providers", action="store_true",
                   help="Sync all local providers (codex, claude-code, chatgpt, claude)")
p_sync.add_argument("--providers", type=str,
                   help="Comma-separated list of providers to sync")

# In sync.py:
def run_multi_provider_sync(args, env):
    """Sync multiple providers in sequence."""
    if args.all_providers:
        providers = LOCAL_SYNC_PROVIDER_NAMES
    elif args.providers:
        providers = [p.strip() for p in args.providers.split(",")]
    else:
        providers = [args.provider]  # Single provider

    results = {}
    total_written = 0
    total_skipped = 0

    for provider_name in providers:
        ui.console.print(f"\n[bold]Syncing {provider_name}...[/bold]")

        try:
            result = sync_single_provider(provider_name, args, env)
            results[provider_name] = {"status": "success", "result": result}
            total_written += len(result.written)
            total_skipped += result.skipped
        except Exception as e:
            results[provider_name] = {"status": "error", "error": str(e)}
            if not args.continue_on_error:
                break

    # Summary
    ui.console.print("\n[bold]Multi-Provider Sync Summary[/bold]")
    for provider, result in results.items():
        if result["status"] == "success":
            ui.console.print(f"  ✓ {provider}: {result['result'].written} written")
        else:
            ui.console.print(f"  ✗ {provider}: {result['error']}")
```

### Challenges
1. **Provider-specific flags**: Some flags only apply to certain providers (Drive flags)
2. **Error handling**: Should failure in one provider stop others?
3. **Progress display**: Need to show which provider is active
4. **Parallelization**: Could sync multiple providers concurrently
5. **Configuration**: May need per-provider settings

### Benefits
- One command for daily sync routine
- Useful for automation/cron jobs
- Reduces repetition

### Risks
- High complexity
- Provider incompatibilities
- Long-running operation (needs progress bars)
- Harder to debug failures

### Alternative Approach
Create a simple shell script helper:
```bash
# polylogue-sync-all.sh (installed with CLI)
#!/bin/bash
polylogue sync codex --all "$@"
polylogue sync claude-code --all "$@"
polylogue sync chatgpt --all "$@"
polylogue sync claude --all "$@"
```

Much simpler, achieves 80% of the benefit.

---

## Implementation Priority Recommendations

### Phase 1: Quick Wins (4-7 hours)
1. ✅ Parent Parser (2-4h) - Immediate maintenance benefit
2. ✅ --examples Flag (2-3h) - Improves documentation

**Why:** Low effort, immediate value, low risk

### Phase 2: Medium Value (16-24 hours)
3. ✅ --open Flag (4-6h) - Workflow enhancement
4. ✅ Path Standardization (6-8h) - Consistency improvement
5. ✅ config init (6-10h) - Onboarding improvement

**Why:** Medium effort, high user impact, manageable risk

### Phase 3: High Value Features (10-16 hours)
6. ✅ Progress Bars (10-16h) - Biggest UX improvement for heavy users

**Why:** Higher effort but transforms experience for key use case

### Phase 4: Systematic Improvements (8-12 hours)
7. ⏭️  JSON Error Messages (8-12h) - If scripting becomes priority

**Why:** Lower priority unless automation users request it

### Phase 5: Complex Features (12-20 hours)
8. ⏭️  --all-providers (12-20h or shell script alternative)

**Why:** High complexity, consider simpler alternatives first

---

## Next Steps

1. **Get user feedback**: Which improvements matter most to actual users?
2. **Implement Phase 1**: Quick wins for immediate benefit
3. **Prototype Progress Bars**: High-impact feature worth early validation
4. **Defer Phase 5**: Consider shell script alternative for --all-providers

