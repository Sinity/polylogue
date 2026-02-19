"""Static site generation command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv


@click.command("site")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for generated site (default: ~/.local/share/polylogue/site)",
)
@click.option(
    "--title",
    default="Polylogue Archive",
    help="Site title",
)
@click.option(
    "--search/--no-search",
    default=True,
    help="Enable client-side search (default: enabled)",
)
@click.option(
    "--search-provider",
    type=click.Choice(["pagefind", "lunr"]),
    default="pagefind",
    help="Search index provider (default: pagefind)",
)
@click.option(
    "--dashboard/--no-dashboard",
    default=True,
    help="Generate dashboard page (default: enabled)",
)
@click.pass_obj
def site_command(
    env: AppEnv,
    output: Path | None,
    title: str,
    search: bool,
    search_provider: str,
    dashboard: bool,
) -> None:
    """Generate a static HTML site from the archive.

    Creates a browsable website with:

    \b
    - Index page with recent conversations
    - Per-provider index pages
    - Dashboard with archive statistics
    - Client-side search (pagefind or lunr.js)

    \b
    Examples:
        polylogue site                       # Build to default location
        polylogue site -o ./public           # Build to custom directory
        polylogue site --title "My Archive"  # Custom site title
        polylogue site --no-search           # Disable search index
    """
    from polylogue.paths import data_home
    from polylogue.site import SiteBuilder, SiteConfig

    # Determine output path
    output_path = output or (data_home() / "site")

    config = SiteConfig(
        title=title,
        enable_search=search,
        search_provider=search_provider,
        include_dashboard=dashboard,
    )

    builder = SiteBuilder(output_dir=output_path, config=config)

    click.echo(f"Building site to {output_path}...")

    try:
        result = builder.build()
        click.echo(
            "Site generated: "
            f"{result['conversations']} conversations, "
            f"{result['index_pages']} index pages"
        )
        click.echo(f"Output: {output_path}")

        if search and search_provider == "pagefind":
            click.echo(
                "\nTo enable search, run:\n"
                f"  npx pagefind --site {output_path}"
            )

        click.echo(
            "\nTo preview locally:\n"
            f"  python -m http.server -d {output_path}"
        )

    except Exception as exc:
        click.echo(f"Error building site: {exc}", err=True)
        raise click.Abort() from exc


__all__ = ["site_command"]
