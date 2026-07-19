"""``polylogue manual`` — the discoverable human entry point for CLI docs.

Renders the exact same live-help tree as ``polylogue --help-markdown``
(``polylogue/cli/help_markdown.py``) so there is exactly one generator for
"the whole CLI surface as one document" instead of a hand-authored manual
drifting out of sync with the real command tree (polylogue-jnj.8). The
content itself is generated, not authored here — this command only owns
discoverability and routing: a human typing ``polylogue manual`` should not
need to already know about the ``--help-markdown`` flag.

Distinct from ``polylogue agent manual`` (polylogue-3gd.2), which packages a
separate, curated, MCP-tool-oriented standing manual for coding agents.
"""

from __future__ import annotations

import json

import click

from polylogue.version import POLYLOGUE_VERSION


@click.command("manual")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.pass_context
def manual_command(ctx: click.Context, output_format: str) -> None:
    """Render the installed CLI manual offline (generated from the live command tree).

    Every command and option shown is read live from the installed
    ``polylogue`` package — the same source of truth ``--help`` and
    ``docs/cli-reference.md`` use, so this never drifts from what actually
    runs. Works without network access or a configured archive.
    """
    from polylogue.cli.help_markdown import render_help_markdown

    root = ctx.find_root()
    prog_name = root.info_name or "polylogue"
    body = render_help_markdown(root.command, prog_name=prog_name)

    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "source": f"installed `{prog_name}` command tree",
                    "content_version": POLYLOGUE_VERSION,
                    "content": body,
                },
                ensure_ascii=False,
            )
        )
        return

    click.echo(f"# Polylogue manual — generated from the installed `{prog_name}` v{POLYLOGUE_VERSION} command tree.")
    click.echo("# Same content as `--help-markdown`; nothing here is hand-authored separately.")
    click.echo("")
    click.echo(body, nl=False)


__all__ = ["manual_command"]
