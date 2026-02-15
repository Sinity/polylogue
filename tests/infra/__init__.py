"""Shared test infrastructure — helpers, strategies, mocks, and path constants."""

from pathlib import Path

TESTS_ROOT = Path(__file__).parent.parent
DATA_DIR = TESTS_ROOT / "data"
GOLDEN_DIR = DATA_DIR / "golden"
