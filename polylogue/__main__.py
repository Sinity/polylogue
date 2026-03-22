"""Main entry point for Polylogue CLI."""

from polylogue.cli.click_app import main as cli_main


def main() -> None:
    """Main entry point."""
    cli_main()


if __name__ == "__main__":
    main()
