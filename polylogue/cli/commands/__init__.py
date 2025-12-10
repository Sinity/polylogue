"""Command modules for polylogue CLI.

Each command module exports:
- setup_parser(subparsers) - Adds the command's argparse subparser
- dispatch(args, env) - Executes the command
"""

from __future__ import annotations

__all__ = []
