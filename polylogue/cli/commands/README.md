# CLI Command Modules

This directory contains command dispatchers extracted from the old `app.py`.
The CLI is now Click-based (`polylogue/cli/click_app.py`), and these modules are called from Click callbacks.

## Pattern

Each command module must export:

### `dispatch(args, env)`
Executes the command logic.

**Parameters:**
- `args`: `argparse.Namespace` constructed by Click callbacks.
- `env`: `CommandEnv` object with config, UI, database, etc.

### Legacy: `setup_parser(...)` (optional)
Some modules still include an argparse `setup_parser` from the pre‑Click era.
It is retained only for reference and is not used by the current CLI.

## Example: sync.py

```python
def setup_parser(subparsers, _add_command_parser, add_helpers):
    p = _add_command_parser(subparsers, "sync", help="...", ...)
    p.add_argument("provider", choices=[...])
    add_helpers["out_option"](p, ...)
    # ... more arguments

def dispatch(args, env):
    if args.watch:
        run_watch_cli(args, env)
    else:
        run_sync_cli(args, env)
```

## Current Commands

All major commands live here and are wired by Click:

1. **sync.py** - Sync provider archives (watch mode, Drive/local sync)
2. **render.py** - Render JSON exports to Markdown/HTML
3. **config.py** - Configuration management (init/set/show)
4. **attachments.py** - Attachment utilities (stats/extract)
5. **browse.py** - Browse data (branches/stats/status/runs/inbox)
6. **import_cmd.py** - Import provider exports
7. **search.py** - Search rendered transcripts (FTS5)
8. **maintain.py** - System maintenance (prune/doctor/index/restore)
9. **status.py** - Show cached Drive info and recent runs

## Notes

- `polylogue/cli/app.py` is kept only for shared helpers and backwards‑compat shims. It no longer owns argument parsing.

## Benefits

1. **Modularity** - Each command is self-contained
2. **Testability** - Easy to test individual commands
3. **Maintainability** - Find/modify command logic quickly
4. **Collaboration** - Fewer merge conflicts (separate files)
5. **Readability** - ~200 line files instead of 2163 line monster

## How to Extract a Command

1. Find parser setup in `app.py` (search for command name)
2. Copy argument setup to `setup_parser()`
3. Find dispatch function (search for `_dispatch_<command>`)
4. Copy dispatch logic to `dispatch()`
5. Fix imports (move to top of file)
6. Update `app.py` to use new module
7. Test that command still works
