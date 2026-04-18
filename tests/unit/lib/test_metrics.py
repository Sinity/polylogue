from __future__ import annotations

from pytest import MonkeyPatch

from polylogue.lib import metrics


def test_read_peak_rss_self_mb_prefers_proc_vmhwm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_read_proc_status_kb", lambda field_name: 15360 if field_name == "VmHWM:" else None)
    monkeypatch.setattr(metrics, "_read_rusage_peak_rss_mb", lambda scope: 999.9)

    assert metrics.read_peak_rss_self_mb() == 15.0


def test_read_peak_rss_self_mb_falls_back_to_rusage(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_read_proc_status_kb", lambda field_name: None)
    monkeypatch.setattr(metrics, "_read_rusage_peak_rss_mb", lambda scope: 42.5)

    assert metrics.read_peak_rss_self_mb() == 42.5
