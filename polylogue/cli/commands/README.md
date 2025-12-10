# CLI Command Modules

This directory contains extracted command modules from `app.py` (2163 lines → modular structure).

## Pattern

Each command module exports two functions:

### 1. `setup_parser(subparsers, _add_command_parser, add_helpers)`
Sets up argparse subparser for the command.

**Parameters:**
- `subparsers`: The main `_SubParsersAction` object
- `_add_command_parser`: Helper function to add command with examples epilog
- `add_helpers`: Dict of common argument helper functions:
  - `out_option(parser, default_path, help_text)`
  - `dry_run_option(parser)`
  - `force_option(parser, help_text)`
  - `collapse_option(parser)`
  - `html_option(parser)`
  - `diff_option(parser, help_text)`
  - `examples_epilog(command_name)`

### 2. `dispatch(args, env)`
Executes the command logic.

**Parameters:**
- `args`: `argparse.Namespace` with parsed arguments
- `env`: `CommandEnv` object with config, UI, database, etc.

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

## Migration Status

### ✅ Completed (9/9 Core Commands)

All major commands have been extracted into modular files:

1. **sync.py** - Sync provider archives (watch mode, Drive/local sync)
2. **render.py** - Render JSON exports to Markdown/HTML
3. **config.py** - Configuration management (init/set/show)
4. **attachments.py** - Attachment utilities (stats/extract)
5. **browse.py** - Browse data (branches/stats/status/runs/inbox)
6. **import_cmd.py** - Import provider exports
7. **search.py** - Search rendered transcripts (FTS5)
8. **maintain.py** - System maintenance (prune/doctor/index/restore)
9. **status.py** - Show cached Drive info and recent runs

### 📝 Next Steps

- **Update app.py** - Integrate command modules, remove old parser code
- **Test** - Verify all commands still work
- **Target** - Reduce app.py from 2163 lines to <300 lines

## After Extraction

Once all commands are extracted, `app.py` should be < 300 lines:

```python
# app.py (simplified)
from .commands import sync, render, maintain, config, attachments, ...

def build_parser():
    parser = argparse.ArgumentParser(...)
    sub = parser.add_subparsers(...)

    # Setup all command parsers
    sync.setup_parser(sub, _add_command_parser, helpers)
    render.setup_parser(sub, _add_command_parser, helpers)
    maintain.setup_parser(sub, _add_command_parser, helpers)
    # ... etc

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    env = CommandEnv(...)

    # Dispatch to appropriate command
    if args.cmd == "sync":
        sync.dispatch(args, env)
    elif args.cmd == "render":
        render.dispatch(args, env)
    # ... etc
```

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
