"""Fixture module for the gate-registration AST scan.

Its only purpose is to give ``test_at_least_one_gate_module_declares_a_required_argument``
a known ``required=True`` argument to find, so the scan cannot pass by finding
nothing at all.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--needed", required=True)
    parser.add_argument("--optional")
    return parser
