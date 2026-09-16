"""Health-threshold loaders read the typed config accessors, not ``cfg.raw``.

polylogue-v6xh: the ``health_convergence_debt`` / ``health_cursor_lag``
accessors were dead because every consumer reached into ``cfg.raw``.

Anti-vacuity: the config used here exposes the tables *only* through the
typed properties -- its ``raw`` mapping is empty. Reverting any loader to
``cfg.raw.get(...)`` (or deleting the properties) makes it fall back to the
built-in defaults and the assertions go red.
"""

from __future__ import annotations

from polylogue.config import PolylogueConfig
from polylogue.daemon.convergence_debt_alert import (
    load_thresholds_from_config as load_debt_thresholds,
)
from polylogue.daemon.cursor_lag_alert import load_thresholds_from_config as load_lag_thresholds
from polylogue.daemon.cursor_lag_anomaly import load_anomaly_thresholds_from_config


class _AccessorOnlyConfig(PolylogueConfig):
    """A config whose health tables exist only behind the typed accessors."""

    @property
    def health_convergence_debt(self) -> dict[str, object]:
        return {"default_warning": 7}

    @property
    def health_cursor_lag(self) -> dict[str, object]:
        return {"default_warning_s": 11, "anomaly_min_lag_s": 13}


def test_health_threshold_loaders_use_typed_accessors() -> None:
    cfg = _AccessorOnlyConfig()

    assert "health_convergence_debt" not in cfg.raw
    assert "health_cursor_lag" not in cfg.raw

    assert load_debt_thresholds(cfg).default_warning == 7
    assert load_lag_thresholds(cfg).default_warning_s == 11
    assert load_anomaly_thresholds_from_config(cfg).min_lag_s == 13
