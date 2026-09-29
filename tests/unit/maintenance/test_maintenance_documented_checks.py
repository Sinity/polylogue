"""Operator check names must resolve to declared verification owners."""

from pathlib import Path

from polylogue.maintenance.archive_verification import archive_verification_domain_adapters


def test_documented_archive_checks_have_live_owners(tmp_path: Path) -> None:
    """Restoring a removed check to the operator table makes this fail."""
    document = (Path(__file__).resolve().parents[3] / "docs" / "maintenance.md").read_text()
    table = document.split("| Check | Proves |", 1)[1].split("\n\n", 1)[0]
    names = {line.split("`", 2)[1] for line in table.splitlines() if line.startswith("| `")}
    owners = {owner.name for owner in archive_verification_domain_adapters(tmp_path)}
    assert names
    assert names <= owners
