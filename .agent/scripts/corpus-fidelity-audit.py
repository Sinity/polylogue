#!/usr/bin/env python3
"""Compatibility entrypoint for the production corpus-fidelity gate."""

from __future__ import annotations

import sys

from polylogue.maintenance.corpus_fidelity import main

if __name__ == "__main__":
    sys.exit(main())
