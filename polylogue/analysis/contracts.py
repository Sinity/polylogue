"""Public analysis/evidence contract vocabulary.

The implementation lives in :mod:`polylogue.core.analysis_contracts` so query,
storage, and analysis owners can share the protocol without importing a
surface.  This module is the discoverable analysis-facing import path.
"""

from polylogue.core import analysis_contracts as _contracts
from polylogue.core.analysis_contracts import *  # noqa: F403

__all__ = _contracts.__all__
