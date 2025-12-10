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

### ✅ Completed
- `sync.py` - Sync command (example implementation)

### ⏳ TODO (Extract from app.py)

Extract these commands following the sync.py pattern:

1. **render.py** (~80 lines)
   - Parser setup: lines 1807-1850
   - Dispatch: `_dispatch_render()` at line 1343

2. **maintain.py** (~150 lines)
   - Parser setup: lines 1924-1990
   - Dispatch: Multiple functions for maintain/index subcommands
   - Includes: `maintain index`, `maintain vacuum`, etc.

3. **config.py** (~100 lines)
   - Parser setup: lines 1994-2015
   - Dispatch: `_dispatch_config()` at line 1368
   - Includes: `config init`, `config set`, `config show`

4. **attachments.py** (~80 lines)
   - Parser setup: lines 2018-2040
   - Dispatch: `_dispatch_attachments()` at line 1383

5. **browse.py** (~100 lines)
   - Parser setup: lines 1808-1850
   - Dispatch: Browse command logic

6. **import_cmd.py** (~60 lines)
   - Parser setup: lines 1774-1806
   - Dispatch: `_dispatch_import()` at line 1334

7. **search.py** (~40 lines)
   - Parser setup: lines ~1850-1870
   - Dispatch: `_dispatch_search()` at line 1338

8. **status.py** (~50 lines)
   - Parser setup and dispatch for status command

9. **prefs.py** (~70 lines)
   - Parser setup: lines 2090-2115
   - Dispatch: Preferences commands

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
