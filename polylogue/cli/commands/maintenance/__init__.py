"""Maintenance command group: archive inspection and guarded recovery verbs.

Each subcommand lives in its own submodule and is attached lazily (the
``_LazyCommand`` pattern already used for the root CLI's own dispatch,
:mod:`polylogue.cli.click_command_registration`), so importing this
package -- which happens on *any* ``ops maintenance ...`` invocation,
including a bare ``--help`` listing -- never imports a specific
subcommand's own heavy runtime dependencies (``ArchiveStore``,
``blob_gc``, ``migration_runner``, ``embeddings.reconcile``, ...). Those
load only when that specific subcommand is actually resolved (dispatched
or its own ``--help`` requested). polylogue-sod7.
"""

from __future__ import annotations

import click

from polylogue.cli.click_command_registration import _LazyCommand, _NestedLazyGroup
from polylogue.maintenance.declarations import MAINTENANCE_COMMAND_DECLARATIONS

# The command table is *derived*, never transcribed: every name, submodule,
# handler attribute, short help, and nested-group flag comes from
# ``polylogue/maintenance/declarations.py``, whose kernel records are resolved
# against the live checkout by ``devtools gate declaration-bindings``. Adding a
# maintenance command means adding one declaration, not editing this file.


@click.group("maintenance")
@click.pass_context
def maintenance_group(ctx: click.Context) -> None:
    """Preview and run maintenance backfill operations.

    Prints the resolved archive root and its provenance (env override /
    config file / default) before dispatching to any subcommand -- see
    ``_shared.print_archive_root_provenance`` (polylogue-l1qg). This runs
    for every real subcommand invocation, including ``--help`` on a
    subcommand, but not for ``maintenance``/``maintenance --help`` on their
    own (no subcommand resolves, so there is nothing to warn about yet).
    """
    from polylogue.cli.commands.maintenance._shared import print_archive_root_provenance
    from polylogue.cli.shared.types import AppEnv

    env = ctx.obj
    if isinstance(env, AppEnv):
        print_archive_root_provenance(env)


for _declaration in MAINTENANCE_COMMAND_DECLARATIONS:
    _command_type = _NestedLazyGroup if _declaration.nested_group else _LazyCommand
    maintenance_group.add_command(
        _command_type(
            _declaration.cli_name,
            _declaration.module,
            _declaration.attribute,
            short_help=_declaration.short_help,
        )
    )

del _declaration

__all__ = ["maintenance_group"]
