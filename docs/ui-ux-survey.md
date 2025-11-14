# Polylogue CLI User Interface Survey Report

## Executive Summary

Polylogue is a comprehensive CLI toolkit for archiving and managing AI/LLM conversations from multiple providers (Google Drive/Gemini, ChatGPT, Claude.ai, Codex, Claude Code). The CLI exhibits a modern, thoughtfully-designed interactive-first UX approach with fallbacks to plain mode, extensive JSON output support, and sophisticated interactive features powered by `gum` and `skim`.

**Repository:** `/home/user/polylogue`  
**Main CLI Module:** `/home/user/polylogue/polylogue/cli/app.py`  
**Version:** Recent improvements to CLI help and completions UX

---

## 1. COMMAND STRUCTURE & ORGANIZATION

### 1.1 Command Hierarchy Overview

**Primary Commands (8):**
- `render` - Convert local JSON logs to Markdown
- `sync` - Synchronize provider archives (drive, codex, claude-code, chatgpt, claude)
- `import` - Import provider exports
- `inspect` - Analyze archives (branches, search, stats)
- `watch` - Monitor local sessions for changes
- `status` - Show cached Drive info and run history
- `doctor` - Check data directories for issues
- `settings` - Configure default preferences
- `prune` - Clean up legacy single-file outputs

**Support Commands (4):**
- `help` - Show command help (with optional topic)
- `env` - Display resolved config/output paths
- `completions` - Generate shell completion scripts
- Internal: `_complete`, `_search-preview` (hidden, for completions/search)

### 1.2 Subcommand Structure

**Multi-level commands (`inspect`):**
```
inspect
  ├── branches   - Explore branch graphs
  ├── search     - Query FTS index
  └── stats      - Summarize output dirs
```

**Provider-specific commands:**
- `sync [drive|codex|claude-code|chatgpt|claude]`
- `import [chatgpt|claude|codex|claude-code]` (source args vary)
- `watch [codex|claude-code|chatgpt|claude]`

### 1.3 Command Naming & Consistency

**Strengths:**
- Clear verb-noun structure (render, sync, import, watch, doctor, prune)
- Consistent provider names: `drive`, `codex`, `claude-code`, `chatgpt`, `claude`
- Subcommand naming maps to actions: `branches`, `search`, `stats`
- Past tense for status-like commands: rendered results shown in `status`

**Potential Issues:**
- Dashed vs underscore inconsistency in some contexts:
  - Commands: `claude-code` (dashed)
  - Flag names: `--dry-run`, `--html-out`, `--base-dir` (dashed)
  - Python variable names: `html_mode`, `collapse_threshold` (underscored)
  - This is semantically correct but can be visually confusing

**Command Registry:**
- Implemented via `CommandRegistry` class (`/home/user/polylogue/polylogue/cli/registry.py`)
- Supports command aliases (though none currently registered)
- Lazy registration pattern: commands registered once at first use

**Code Location:**
- `lines 79-90`: COMMAND_REGISTRY and _register_default_commands()
- `lines 836-861`: _register_default_commands() implementation
- `lines 863-1065`: build_parser() creates argparse structure

### 1.4 Flag Consistency

**Universal flags:**
```
--plain              # Disable interactive UI (all commands)
--interactive        # Force interactive UI (all commands)
--json               # Machine-readable output (render, sync, import, inspect, status, doctor, settings, env)
```

**Common optional flags:**
```
--out PATH           # Output directory override
--collapse-threshold N # Conversation folding threshold
--html [on|off|auto] # HTML preview generation
--force              # Reprocess/overwrite
--dry-run            # Simulate without writing
--diff               # Write delta diffs
--links-only         # Link attachments instead of downloading
```

**Provider-specific flags:**
- Drive: `--folder-name`, `--folder-id`, `--chat-id`, `--since`, `--until`, `--name-filter`, `--list-only`
- Local: `--base-dir`, `--session`, `--all`
- Import: `--conversation-id`, `--to-clipboard`

**Strengths:**
- DRY principle applied via `add_out_option()`, `add_collapse_option()`, `add_force_option()`, etc. helpers
- File location: `/home/user/polylogue/polylogue/cli/arg_helpers.py` (lines 7-39)
- Clear help text for each flag

---

## 2. HELP SYSTEM & DOCUMENTATION

### 2.1 Help Text Quality

**Top-level help:**
- Descriptive: `"Render local provider JSON logs"`, `"Watch local session stores and sync on changes"`
- Consistent format in both short (`help=`) and long (`description=`) forms
- Example (render): Lines 874-885 in app.py

**Command-specific help:**
- Every subcommand has both `help=` (brief) and `description=` (full)
- Example structure:
  ```python
  p_render = sub.add_parser("render", 
      help="Render local provider JSON logs",
      description="Render local provider JSON logs"
  )
  ```
- Lines 874-1065: Comprehensive parser setup

### 2.2 Custom Help Command

**Implementation:** `run_help_cli()` (lines 135-152)

**Features:**
- `polylogue help` - Shows top-level help + command listing
- `polylogue help COMMAND` - Shows detailed help for specific command
- Error handling: `"Unknown command: {topic}"` with available commands listed
- Reformatted descriptions: stripped/normalized spacing (line 98)

**Strengths:**
- Better UX than raw `--help` (integrates with UI system)
- Avoids deep nesting when inspecting subcommands
- Example: `polylogue help sync` shows sync-specific options clearly

**Limitations:**
- No support for `--help` per-subcommand (standard argparse behavior remains available)
- Subcommand help only works one level deep (inspect.branches → help inspect branches)

### 2.3 Help Output Formatting

**Code:** Lines 103-122 (_print_command_listing)

**Dual-mode output:**

**Rich mode (interactive, when not --plain):**
```
Commands:
╒════════════════════════════════════════════════════════════════════╕
│ Command       │ Description                                         │
├───────────────┼─────────────────────────────────────────────────────┤
│ doctor        │ Check local data directories for common issues       │
│ env           │ Show resolved configuration/output paths             │
│ help          │ Show command help                                    │
│ import        │ Import provider exports into the archive             │
│ ... (alphabetical)                                                  │
╘════════════════════════════════════════════════════════════════════╛
```

**Plain mode:**
```
Commands:
  doctor      Check local data directories for common issues
  env         Show resolved configuration/output paths
  import      Import provider exports into the archive
```

**Table implementation:**
- Rich library: `from rich.table import Table` (line 115)
- Styled header: `header_style="bold cyan"`
- Color: cyan for command names

### 2.4 Inline Documentation & Examples

**Lacking in most commands.** Examples:

**Good example (inspect branches):**
```
-min-branches, --min-branches TYPE=int
    Only include conversations with at least this many branches
    (default 1)
```

**Minimal example (render --links-only):**
```
--links-only
    Link attachments instead of downloading
```

**No usage examples** in help text (e.g., "polylogue sync drive --all" or "polylogue watch codex --debounce 1.0")

### 2.5 Error Message Quality

**Examples of error messages by severity:**

**[RED] Critical errors (user action required):**
- `"[red]Unknown command: {topic}"` (line 144) - Unknown help topic
- `"[red]Failed to remove {path}: {exc}"` (line 311) - File deletion failure
- `"[red]Choose only one of --with-attachments or --without-attachments."` (line 556) - Conflicting flags
- `"[red]Failed to {action} HTML explorer: {exc}"` (line 451) - HTML generation failure
- `"[red]Import callable not configured."` (imports.py:39) - Internal misconfiguration

**[YELLOW] Non-blocking warnings/informational:**
- `"[yellow][dry-run] Would prune {len(legacy)} path(s) in {root}"` (line 298) - Dry-run indication
- `"[yellow]No results found."` (line 609) - Search with no hits
- `"[yellow]No branchable conversations found{detail}."` (line 347) - Branches not available
- `"[yellow]Branch explorer cancelled."` (line 365) - User cancellation
- `"[yellow]Sync cancelled; no chats selected."` (sync.py:135) - User cancellation

**[GREEN] Success messages:**
- `"[green]Wrote {len(records)} run(s) to {path}"` (status.py:26) - File written
- `"[green]Copied {target_file.slug} to clipboard."` (render.py:140) - Clipboard success
- `"[green]Pruned {removed_here} legacy path(s) in {root}"` (app.py:313) - Cleanup summary

**[CYAN] Informational:**
- `"[cyan]Branch {branch_id} matches the canonical transcript."` (app.py:502) - Status info
- `"[cyan](Message body empty)"` (app.py:723) - Empty state message

**Strengths:**
- Semantic color coding (red=error, yellow=warning/info, green=success, cyan=status)
- Rich markup enables styling beyond color

**Weaknesses:**
- No standardized prefix for error severity/category (e.g., "ERROR:", "WARNING:")
- Some errors are SystemExit without console output first:
  - `raise SystemExit("Provide an export path for this import.")` (imports.py:100)
  - `raise SystemExit(f"Input path not found: {path}")` (app.py:1072)
  - Better: Print [red] message first, then exit
- No suggestion/remediation for many errors (e.g., missing required args)

---

## 3. INTERACTIVE FEATURES

### 3.1 Pickers & Selection Interfaces

**Primary implementation:** `skim (sk)` - fuzzy finder  
**Secondary implementation:** `gum` - specialized UI toolkit  
**Fallback:** Plain mode (returns first item or None)

**Code locations:**
- `cli_common.py` (lines 41-99): `sk_select()` and `choose_single_entry()`
- `ui/facade.py` (lines 126-134): `_interactive_choose()` using gum

#### 3.1.1 Skim-based Pickers

**Function signature:**
```python
def sk_select(
    lines: Sequence[str],
    *,
    multi: bool = True,              # Multi-select mode
    preview: Optional[str] = None,    # Preview command (bat, glow)
    header: Optional[str] = None,     # Header text
    bindings: Optional[Sequence[str]] = None,  # Custom key bindings
    prompt: Optional[str] = None,     # Prompt text
    cycle: bool = True,               # Wrap at edges
) -> Optional[List[str]]
```

**Usage examples:**

1. **Chat selection for sync** (sync.py:129-140):
```
lines = [
    f"{c.get('name') or '(untitled)'}\t{c.get('modifiedTime') or ''}\t{c.get('id')}"
    for c in filtered
]
selection = sk_select(lines, preview="printf '%s' {+}")
```
Output: Tab-separated display with ID column

2. **File selection for render** (app.py:1079-1088):
```
lines = [str(p) for p in candidates]
selection = sk_select(
    lines,
    preview="bat --style=plain {}",
    bindings=["ctrl-g:execute(glow --style=dark {+})"],
)
```
Features: Bat preview with Ctrl+G for Glow markdown rendering

3. **Branch picker** (app.py:351-363):
```
def _format_conv(entry, idx):
    return f"{entry.provider}:{entry.conversation_id}\t{entry.slug}\tbranches={branch_total}\t{title}"

chosen, cancelled = choose_single_entry(
    ui,
    conversations,
    format_line=_format_conv,
    header="idx\tprovider:id\tslug\tbranches\ttitle",
    prompt="branch>",
)
```
Features: Structured data display, custom prompt

**Default key bindings** (cli_common.py:41-48):
```
tab           - Toggle selection
shift-tab     - Toggle selection (up)
ctrl-a        - Select all
ctrl-d        - Deselect all
ctrl-space    - Toggle current item
alt-a         - Select all + accept
```

**Error handling:**
- FileNotFoundError → RuntimeError: `"Required command 'sk' is not available in PATH."` (cli_common.py:92)
- CalledProcessError (user abort) → Returns None
- Empty selection → Returns empty list []

#### 3.1.2 Gum-based Interactive UI

**Code location:** `ui/facade.py` (lines 76-144)

**Interactive methods:**

1. **Banner display:**
```python
def _interactive_banner(title, subtitle=None)
# Uses: gum style --border rounded --margin 1 --padding 1
```

2. **Summary/info panels:**
```python
def _interactive_summary(title, text)
# Uses: gum format (Markdown-to-terminal)
# Fallback: Rich Panel if gum unavailable
```

3. **Confirmations:**
```python
def _interactive_confirm(prompt, *, default=True)
# Uses: gum confirm --prompt TEXT [--default]
# Returns: True if yes, False if no
```

4. **Choice menus:**
```python
def _interactive_choose(prompt, options)
# Uses: gum choose --header PROMPT OPTION1 OPTION2 ...
# Returns: selected option string or None
```

5. **Input prompts:**
```python
def _interactive_input(prompt, *, default=None)
# Uses: gum input --placeholder TEXT [--value DEFAULT]
```

**Example usage in inspect (app.py:507-540):**
```python
def _prompt_branch_followups(ui, conversation, args, html_path, settings):
    options = ["Diff a branch", "Write HTML explorer", "Done"]
    choice = ui.choose("Next action?", options)  # → gum choose
    # Looping menu for follow-up actions
```

### 3.2 Dependency Fallback Strategy

**Architecture:** ConsoleFacade classes in `ui/facade.py`

**Plain mode (--plain flag):**
```python
class PlainConsoleFacade(ConsoleFacade):
    def confirm(self, prompt, *, default=True):
        return default  # Non-interactive
    
    def choose(self, prompt, options):
        return None  # No choice available
    
    def input(self, prompt, *, default=None):
        return default  # No input available
```

**Interactive mode fallback logic** (lines 158-168):
```python
class InteractiveConsoleFacade(ConsoleFacade):
    def __post_init__(self):
        missing = [cmd for cmd in ("gum", "sk") 
                   if shutil.which(cmd) is None]
        if Console is None or missing:
            # Fall back to plain mode
            self.console = PlainConsole()
            self.plain = True
            warning = " and ".join(missing) + " command(s)"
            self.console.print(f"[plain]Interactive UI unavailable...")
```

**Graceful degradation:**
- Missing `gum` → Fall back to gum format (stdout)
- Missing `skim` → No multi-select picker (takes first/none)
- Missing Rich library → Use PlainConsole (basic print)
- All at once → --plain behavior auto-enabled with warning

### 3.3 Progress Indicators

**Not implemented via explicit progress bars.**

Observable progress:
- Console output during sync: `"[cyan]{title}: no new Markdown files."` (sync.py:35)
- Gum format blocks for long-running operations
- Status output shows count of items processed

**Opportunity:** Could add Rich progress bars for large sync operations (thousands of chats).

### 3.4 Plain Mode vs Interactive Behavior

**Activation:**
```python
plain_mode = args.plain and not getattr(args, "interactive", False)
ui = create_ui(plain_mode)
```

**Impact across commands:**

| Scenario | Interactive | Plain |
|----------|-----------|--------|
| File picker (render/import) | Skim fuzzy search | Return all candidates |
| Chat selection (sync drive) | Skim multi-select | Skip, require --chat-id |
| Session selection (sync local) | Skim multi-select | Skip, require --session or --all |
| Output formatting | Rich tables, gum format | Plaintext with indentation |
| Confirmations (attach?) | `gum confirm` | Default assumed |
| Errors | Rich styled panels | Plain text |

**Code locations:**
- Render: `render.py:50-53` checks `ui.plain`
- Sync drive: `sync.py:120` checks `not ui.plain and not json_mode`
- Import: `imports.py` checks `getattr(ui, "plain", False)`

---

## 4. OUTPUT & FORMATTING

### 4.1 Table Formatting

**Library:** Rich (when available) → Falls back to plain text

**Examples:**

1. **Doctor issues table** (doctor.py:64-75):
```python
table = Table(title="Doctor Issues", show_lines=False)
table.add_column("Provider")
table.add_column("Severity")
table.add_column("Path")
table.add_column("Message")
for issue in report.issues:
    table.add_row(issue.provider, issue.severity, str(issue.path), issue.message)
console.print(table)
```

2. **Import summary table** (summaries.py:24-49):
```python
table = Table(title=title, show_lines=False)
table.add_column("File")
table.add_column("Attachments", justify="right")
table.add_column("Attachment MiB", justify="right")
table.add_column("Tokens (~words)", justify="right")
for res in written:
    table.add_row(res.slug, str(att_count), f"{att_bytes / (1024 * 1024):.2f}", ...)
```

3. **Help command table** (app.py:115-122):
```python
table = Table(show_header=True, header_style="bold cyan")
table.add_column("Command", style="cyan")
table.add_column("Description")
for name, description in entries:
    table.add_row(name, description)
```

**Plain mode fallback:**
```python
width = max(len(name) for name, _ in entries) + 2
for name, description in entries:
    console.print(f"  {name.ljust(width)}{description}")
```

### 4.2 Color Usage & Theming

**Color palette:**
- `cyan` - Command names, headers, informational
- `red` - Errors, critical issues
- `yellow` - Warnings, cancellations, dry-run
- `green` - Success messages
- `bold cyan` - Table headers

**HTML theme support:**
- `--theme light` or `--theme dark` (render/sync/import/watch/inspect branches)
- Stored in settings: `settings.html_theme`
- Applied to generated HTML previews (not CLI output)
- Default: light theme

**No theme customization for CLI output itself** (colors are hardcoded).

### 4.3 JSON Output Support

**Commands with JSON mode:**
- `render --json`
- `sync --json`
- `import --json`
- `inspect search --json`
- `inspect stats --json`
- `status --json`
- `doctor --json`
- `settings --json`
- `env --json`

**Example structure (render):**
```json
{
  "cmd": "render",
  "count": 2,
  "out": "/path/to/output",
  "files": [
    {
      "output": "/path/to/file.md",
      "slug": "conversation-1",
      "attachments": 5,
      "stats": {
        "totalTokensApprox": 2000,
        "totalWordsApprox": 400,
        "chunkCount": 3,
        "userTurns": 5,
        "modelTurns": 5
      },
      "html": "/path/to/file.html",
      "diff": "/path/to/file.diff"
    }
  ],
  "total_stats": {
    "attachments": 5,
    "totalTokensApprox": 2000,
    "totalWordsApprox": 400,
    "chunkCount": 3,
    "userTurns": 5,
    "modelTurns": 5
  }
}
```

**Implementation pattern:**
```python
if getattr(args, "json", False):
    payload = {...}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return
# Otherwise, interactive/plain output
```

**Strength:** Consistent across all commands, enables piping to other tools

**Limitation:** No `--json-pretty` or `--json-compact` option (always indented)

### 4.4 Verbosity Levels

**Not explicitly implemented.** Available mechanisms:

1. **--plain mode** - Suppresses colors, tables, interactive prompts
2. **--json mode** - Structured output instead of human-readable
3. **Command-specific flags:**
   - `status --watch` - Continuous refresh
   - `status --interval SECONDS` - Control refresh rate
   - `prune --dry-run` - Preview without acting
   - `sync --list-only` - List without syncing

**Missing:** `--verbose` / `-v` flags for detailed debug output

---

## 5. USER WORKFLOWS

### 5.1 Common Use Cases & Their UX

#### Use Case 1: Initial Setup (First Time)

**Steps:**
1. `polylogue env --json` → Verify config paths
2. `polylogue doctor` → Check for data sources
3. Manually configure `~/.config/polylogue/config.json`
4. Run first sync: `polylogue sync drive` → Walks through OAuth setup

**UX Observations:**
- Good: Doctor provides clear missing config guidance
- Good: env command outputs paths for quick inspection
- Missing: Setup wizard or `polylogue init` command

#### Use Case 2: Sync from Google Drive (Interactive)

**Steps:**
1. `polylogue sync drive` (no flags)
2. Interactive skim picker shows all Drive chats
3. User selects via tab/ctrl-a
4. Sync runs, shows table of results
5. `polylogue status` shows recent run stats

**Code:** sync.py:100-141

**UX Strengths:**
- No friction for first-time users
- Preview of chat names + dates
- Tab/ctrl-a shortcuts for all/deselect

**Limitations:**
- Picker shows name + modifiedTime + id only (no description)
- Can't filter by folder within picker
- Large account (1000+ chats) → slow picker load

#### Use Case 3: Batch Import (Scripted)

**Steps:**
1. `polylogue import chatgpt export.zip --all --out ~/chats --json`
2. Script receives JSON output
3. Parses `results[].markdown`, `results[].html` paths

**UX Strengths:**
- --json avoids parsing human-readable output
- --all flag skips picker for automation
- --out ensures output location is known

#### Use Case 4: Continuous Monitoring (Watch Mode)

**Steps:**
1. `polylogue watch claude-code` → Starts monitoring
2. IDE sessions written to ~/.claude/projects/
3. Auto-syncs on change (debounced)
4. Ctrl+C to stop

**Code:** watch.py (lines 35-112)

**UX Strengths:**
- Zero-configuration (uses provider defaults)
- Banner shows watch directory
- Debounce prevents thrashing
- Can combine with --html, --diff

#### Use Case 5: Search & Inspect (Iterative Exploration)

**Steps:**
1. `polylogue inspect search "AI safety" --limit 20`
2. Interactive skim picker with preview
3. Select result → Shows full message context
4. Follow-up: `polylogue inspect branches --slug my-chat --diff`

**Code:** app.py:553-639 (run_inspect_search)

**UX Strengths:**
- FTS query with multiple filters (provider, date, model)
- Preview shows snippet + metadata
- Can diff branches without additional queries

### 5.2 Multi-Step Operations

#### Scenario: "Re-render existing conversation with new settings"

1. User runs: `polylogue sync drive --force --diff --html on`
2. Conversation detected as "dirty"
3. Forces re-render
4. --diff generates side-by-side diff
5. --html outputs interactive HTML explorer

**Problem:** No way to resume interrupted operation (error mid-sync = restart all)

#### Scenario: "Clean up and reorganize"

1. `polylogue prune --dry-run` → Previews legacy files
2. Manual inspection: `ls ~/polylogue-data/render/`
3. `polylogue prune` → Actually removes
4. `polylogue status` → Confirms cleanup

**Issue:** Prune doesn't interactively select (can't prune only certain dirs)

### 5.3 Flag Consistency

**Strengths:**
- `--collapse-threshold N` consistent across render/sync/import/watch
- `--html MODE` consistent with same choices [auto|on|off]
- `--force`, `--dry-run`, `--diff` reused appropriately

**Weaknesses:**
- `--out` vs `--out-dir` inconsistency:
  - render, sync, import: `--out`
  - inspect stats: `--dir` (not `--out`)
  - watch, doctor, prune: custom (--base-dir, --codex-dir)

- Naming: `--html` (singular), `--html-out` (with suffix)
  - Could be `--html-output` for clarity
  - Or `--html-out-path`

- Provider selection:
  - sync: positional argument `sync PROVIDER`
  - watch: positional argument `watch PROVIDER`
  - import: positional argument `import PROVIDER`
  - Consistency achieved, but order matters (no --provider flag)

### 5.4 Default Behaviors

**Good defaults:**
- `--html auto` → Generates HTML only when branches > 1
- `--collapse-threshold 25` (configurable in config.json)
- `--debounce 2.0` (reasonable for local file watching)
- `--limit 20` (inspect search, sensible for terminal)

**Overridable via settings:**
```bash
polylogue settings --html on --theme dark
# Future runs inherit these defaults
```

---

## 6. EDGE CASES & ERROR HANDLING

### 6.1 Missing Required Arguments

**Scenario: User runs `polylogue sync` without provider**

**Current behavior:**
```
usage: ... sync {drive,codex,claude-code,chatgpt,claude} ...
error: the following arguments are required: provider
```

**Observation:** Argparse default error message (not custom)

**Better:** Custom error message suggesting:
```
Error: Sync requires a provider.
Usage: polylogue sync PROVIDER [OPTIONS]
Available: drive, codex, claude-code, chatgpt, claude
```

**Scenario: User runs `polylogue import chatgpt` without source**

**Code:** imports.py:98-101
```python
def _ensure_path() -> Path:
    if not sources:
        raise SystemExit("Provide an export path for this import.")
    return Path(sources[0])
```

**Issue:** No [red] coloring, just raw exit

**Better:**
```python
env.ui.console.print("[red]Error: Provide an export path for this import.")
raise SystemExit(1)
```

### 6.2 Invalid Inputs

**Scenario: User provides invalid collapse threshold**

**Input:** `polylogue sync drive --collapse-threshold abc`

**Result:** Argparse error:
```
error: argument --collapse-threshold: invalid int value: 'abc'
```

**Observation:** Type validation via `type=int` in argparse (good)

**Scenario: User provides invalid provider**

**Input:** `polylogue sync invalid-provider`

**Code:** sync.py:87-97
```python
if provider == "drive":
    ...
elif provider in LOCAL_SYNC_PROVIDER_NAMES:
    ...
else:
    raise SystemExit(f"Unsupported provider for sync: {provider}")
```

**Issue:** No [red] marker, unclear where to find valid providers

### 6.3 File Not Found Scenarios

**Scenario: Render input directory doesn't exist**

**Code:** app.py:1068-1072
```python
def resolve_inputs(path: Path, plain: bool):
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input path not found: {path}")
```

**Issue:** Plain SystemExit, no color, no suggestion to check `polylogue env`

**Better:**
```python
env.ui.console.print(f"[red]Error: Input path not found: {path}")
env.ui.console.print("[cyan]Hint: Use 'polylogue env' to see configured paths")
raise SystemExit(1)
```

**Scenario: Output directory can't be created**

**Code:** No explicit handling in most commands

**Risk:** Silent failure if parent directory doesn't exist or lacks permissions

**Current behavior:** `Path.mkdir(parents=True, exist_ok=True)` called, but failures not caught:
```python
output_dir.mkdir(parents=True, exist_ok=True)  # May raise PermissionError
```

### 6.4 Permission Issues

**Scenario: No write permission to output directory**

**Current:** PermissionError propagates to user

**Better:** Catch and provide actionable message:
```python
try:
    output_dir.mkdir(parents=True, exist_ok=True)
except PermissionError as e:
    env.ui.console.print(f"[red]Cannot write to {output_dir}: permission denied")
    raise SystemExit(1)
```

**Scenario: Credentials missing (Drive sync)**

**Code:** doctor.py suggests locations

**Good:** Doctor reports missing credentials.json and token.json with paths

**Issue:** Error doesn't suggest running `polylogue doctor` when credentials fail

### 6.5 Conflicting Flags

**Scenario: User provides both --with-attachments and --without-attachments**

**Code:** app.py:555-557
```python
if args.with_attachments and args.without_attachments:
    ui.console.print("[red]Choose only one of --with-attachments or --without-attachments.")
    return
```

**Good:** Clear error message

**Issue:** Returns without error code (should raise SystemExit(1))

**Scenario: User provides --list-only with --chat-id**

**Current:** No validation, list runs and returns early, ignoring chat-id

**Better:** Warn user that chat-id is ignored in list mode

### 6.6 Dry-Run Guarantees

**Scenario: Dry-run claims to show actions but some are skipped**

**Code:** prune.py:298-301
```python
if dry_run:
    ui.console.print(f"[yellow][dry-run] Would prune {len(legacy)} path(s) in {root}")
    for path in legacy:
        ui.console.print(f"  rm {'-r ' if path.is_dir() else ''}{path}")
```

**Good:** Explicit `[dry-run]` prefix

**Issue:** Doesn't show "would skip X because Y" cases (e.g., in-progress downloads)

### 6.7 Network/API Error Handling

**Scenario: Google Drive API rate-limit hit during sync**

**Code:** Not shown in CLI layer; handled by drive_client

**Observable:** `polylogue status` shows `driveRetries` and `driveFailures`

**User experience:** Operation continues with partial results, error reported in status

**Better:** Immediate feedback during operation (progress indicator)

---

## 7. STRENGTHS & BEST PRACTICES

### 7.1 Architectural Strengths

1. **Separation of Concerns:**
   - CLI layer (`cli/` module) clean and focused
   - Business logic in `commands.py` and service classes
   - UI abstraction in `ui/facade.py` enables easy testing

2. **Graceful Fallback Strategy:**
   - `--plain` flag available everywhere
   - Interactive features optional (gum/skim)
   - Rich library optional (fallback to PlainConsole)

3. **Consistent Command Structure:**
   - Provider names standardized
   - Flag patterns repeated (--html, --collapse-threshold, --out)
   - Subcommand hierarchy clear (inspect → branches/search/stats)

4. **Comprehensive Help System:**
   - `polylogue help COMMAND` custom implementation
   - Descriptions in parser definitions
   - Shell completions dynamic (zsh callback to CLI)

5. **JSON Output Ubiquity:**
   - All informational commands support --json
   - Consistent structure (cmd, count, results/items, stats)
   - Enables scripting and integration

### 7.2 UX Improvements Made Recently

From git log (84b879f "Improve CLI help and completions UX"):
- Enhanced help text quality
- Improved completion engine
- Better command listing

### 7.3 Testing Coverage

**Test files indicate:**
- `test_cli_help_env.py` - Help command, env, completions
- `test_cli_flags.py` - Flag validation and resolution
- `test_cli_wrappers.py` - Individual command UX
- `test_watch_cli.py` - Watch-mode-specific UX

---

## 8. WEAKNESSES & OPPORTUNITIES FOR IMPROVEMENT

### 8.1 Critical Issues

1. **No setup/initialization flow:**
   - Users must manually discover `polylogue env` and `polylogue doctor`
   - No `polylogue init` or interactive first-run wizard
   - **Recommendation:** Add guided setup for first-time use

2. **Error messages without color in some paths:**
   - `raise SystemExit(...)` without console output first
   - **Recommendation:** Standardize error output pattern
   - **Code location to refactor:** imports.py:100, app.py:1072, sync.py:97

3. **Incomplete error recovery:**
   - Interrupted operations (ctrl-c mid-sync) don't resume
   - No checkpoint system
   - **Recommendation:** Add resumable sync capability

### 8.2 High-Impact Improvements

1. **Progress bars for long operations:**
   - No indication of progress during large Drive syncs
   - **Would improve:** User confidence, time estimation
   - **Implementation:** Rich.progress with live update

2. **Unified --output-dir naming:**
   - Some commands use `--out`, others `--dir` or `--base-dir`
   - **Recommendation:** Standardize to one pattern (suggest `--output` or `--out`)

3. **Validation and clear feedback for conflicting flags:**
   - `--list-only` + `--chat-id` silently ignores latter
   - **Recommendation:** Detect and warn before execution

4. **Example usage in help text:**
   - Commands lack concrete examples
   - **Recommendation:** Add "Examples:" section to top-level help
   - **Code location:** app.py:874+ (parser definitions)

### 8.3 Medium-Impact Improvements

1. **Verbosity levels:**
   - No --verbose/-v flag for debug output
   - **Recommendation:** Add optional structured logging

2. **Resumable/Incremental operations:**
   - Large imports can't resume after interrupt
   - **Recommendation:** Implement checkpoints for provider imports

3. **Interactive branch selection:**
   - Inspect branches requires --branch flag for automated diff
   - **Recommendation:** Offer skim picker for branch selection

4. **Unified output field naming:**
   - Sometimes `attachments`, sometimes `attachmentCount`
   - **Recommendation:** Standardize JSON output keys across commands

### 8.4 Low-Impact Improvements

1. **Shell completion for arguments:**
   - Zsh dynamic completion only covers commands/subcommands
   - Could suggest conversation IDs, dates, etc.
   - **Current:** completion_engine.py partially handles this

2. **Color theme customization:**
   - Colors are hardcoded (no --color-scheme flag)
   - **Note:** HTML theming exists (light/dark), CLI doesn't

3. **Confirmation prompts for destructive actions:**
   - `polylogue prune` warns but doesn't confirm
   - **Recommendation:** Add --force to bypass prompt

---

## 9. DETAILED FILE LOCATIONS & CODE EXAMPLES

### 9.1 Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `/home/user/polylogue/polylogue/cli/app.py` | 1-1115 | Main CLI app, parser, command dispatch |
| `/home/user/polylogue/polylogue/cli/registry.py` | 1-70 | Command registration and lookup |
| `/home/user/polylogue/polylogue/cli/arg_helpers.py` | 1-39 | Reusable flag helpers |
| `/home/user/polylogue/polylogue/cli_common.py` | 1-175 | Skim/gum integration, pickers |
| `/home/user/polylogue/polylogue/ui/facade.py` | 1-177 | Console abstraction, interactive modes |
| `/home/user/polylogue/polylogue/cli/sync.py` | 1-361 | Sync command implementations |
| `/home/user/polylogue/polylogue/cli/imports.py` | 1-400+ | Import command implementations |
| `/home/user/polylogue/polylogue/cli/render.py` | 1-167 | Render command implementation |
| `/home/user/polylogue/polylogue/cli/watch.py` | 1-115 | Watch command implementation |
| `/home/user/polylogue/polylogue/cli/status.py` | 1-300+ | Status command implementation |
| `/home/user/polylogue/polylogue/cli/doctor.py` | 1-85 | Doctor command implementation |
| `/home/user/polylogue/polylogue/cli/completion_engine.py` | 1-200+ | Completion engine |
| `/home/user/polylogue/polylogue/cli/summaries.py` | 1-59 | Summary table formatting |
| `/home/user/polylogue/polylogue/cli/settings_cli.py` | 1-62 | Settings command |
| `/home/user/polylogue/polylogue/cli/context.py` | 1-129 | Context resolution helpers |

### 9.2 Command Registration Example

**Code:** app.py:836-861

```python
def _register_default_commands() -> None:
    global _REGISTRATION_COMPLETE
    if _REGISTRATION_COMPLETE:
        return

    def _ensure(name: str, handler: Callable, help_text: str) -> None:
        if COMMAND_REGISTRY.resolve(name) is None:
            COMMAND_REGISTRY.register(name, handler, help_text=help_text)

    _ensure("render", _dispatch_render, "Render local provider JSON logs")
    _ensure("sync", _dispatch_sync, "Synchronize provider archives")
    # ... more registrations
    _REGISTRATION_COMPLETE = True
```

### 9.3 Interactive Feature Example

**Code:** sync.py:129-140 (Chat picker)

```python
lines = [
    f"{c.get('name') or '(untitled)'}\t{c.get('modifiedTime') or ''}\t{c.get('id')}"
    for c in filtered
]
selection = sk_select(
    lines,
    preview="printf '%s' {+}",  # Preview selected chat IDs
)
if selection is None:
    console.print("[yellow]Sync cancelled; no chats selected.")
    return
if not selection:
    console.print("[yellow]No chats selected; nothing to sync.")
    return
selected_ids = [line.split("\t")[-1] for line in selection]
```

### 9.4 Error Message Example

**Code:** app.py:555-557 (Conflicting flags)

```python
if args.with_attachments and args.without_attachments:
    ui.console.print("[red]Choose only one of --with-attachments or --without-attachments.")
    return  # BUG: Should raise SystemExit(1)
```

### 9.5 JSON Output Example

**Code:** render.py:79-98

```python
if json_output:
    payload = {
        "cmd": "render",
        "count": result.count,
        "out": str(result.output_dir),
        "files": [
            {
                "output": str(f.output),
                "slug": f.slug,
                "attachments": f.attachments,
                "stats": f.stats,
                "html": str(f.html) if f.html else None,
                "diff": str(f.diff) if f.diff else None,
            }
            for f in result.files
        ],
        "total_stats": result.total_stats,
    }
    print(json.dumps(payload, indent=2))
    return
```

---

## 10. CONCLUSION & RECOMMENDATIONS

### Summary of Findings

Polylogue's CLI is a **well-architected, modern interface** that prioritizes user experience through:
- Interactive-first design with graceful fallbacks
- Consistent command naming and flag patterns
- Comprehensive help system and documentation
- JSON output support for automation

**UX Maturity Level:** Advanced (8/10)
- Strong: Command structure, help system, fallback strategy, JSON output
- Weak: Progress indication, error messages consistency, error recovery

### Priority Recommendations

**Tier 1 (Critical):**
1. Standardize error output to use console.print([red]) before SystemExit
2. Add progress bars for long-running operations (large Drive syncs)
3. Create `polylogue init` or guided first-run setup

**Tier 2 (Important):**
1. Unify `--output` flag naming across all commands
2. Add error recovery/resumable operations for imports and syncs
3. Enhance error messages with actionable suggestions

**Tier 3 (Nice-to-have):**
1. Add `--verbose` flag for debug output
2. Implement confirmation prompt for destructive operations
3. Extend zsh completion to include conversation IDs, dates, etc.

### Files to Reference for Implementation

- **Error standardization:** app.py:1072, imports.py:100, sync.py:97
- **Flag helpers:** arg_helpers.py (expand with new unified pattern)
- **Progress bars:** Integrate Rich.progress into commands.py render/sync
- **First-run setup:** New file cli/init.py with interactive prompts

---

## Appendix: Test Files Index

**Test coverage for CLI UX:**
- `/home/user/polylogue/tests/test_cli_help_env.py` - Help, env, completions
- `/home/user/polylogue/tests/test_cli_flags.py` - Flags and resolution
- `/home/user/polylogue/tests/test_cli_wrappers.py` - Command output/formatting
- `/home/user/polylogue/tests/test_cli_integration.py` - End-to-end workflows
- `/home/user/polylogue/tests/test_watch_cli.py` - Watch mode specifics
- `/home/user/polylogue/tests/test_ui.py` - UI facade/console behavior
