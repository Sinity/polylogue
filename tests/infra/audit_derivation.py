"""Neutral paged domain for actual composed-audit kernel regressions."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon.derivation import BaseDerivation, DerivationFrame, KeyPage, KeyStatus, Replacement


class AuditDerivation(BaseDerivation):
    prerequisites: tuple[str, ...] = ()

    def __init__(self, domain: str, failure: str) -> None:
        self.domain = domain
        self.failure: str | None = failure
        self.keys = tuple(f"key-{index:03d}" for index in range(133))
        self.inspected: set[str] = set()
        self.pages = 0
        self.prefix_inspections = 0

    def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        if frame.profile_demand_only:
            return KeyPage()
        self.pages += 1
        # An unchanged-prefix spin fails on actual acquisition, with no sleep
        # or wall-clock threshold. This is a regression guard, not a product cap.
        assert self.pages < 40, "an unsettled sweep was immediately repeated"
        start = int(cursor or 0)
        stop = min(start + limit, len(self.keys))
        return KeyPage(self.keys[start:stop], str(stop) if stop < len(self.keys) else None)

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
        self.inspected.update(keys)
        self.prefix_inspections += int(self.keys[0] in keys)
        if self.failure == "fault" and self.keys[0] in keys:
            raise ValueError("synthetic per-key inspection fault")
        return {
            key: KeyStatus.MISSING if self.failure == "pending" and key == self.keys[0] else KeyStatus.VALID
            for key in keys
        }

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        return self.failure == "pending" and key == self.keys[0]

    def compute(self, frame: DerivationFrame, key: str) -> Replacement:
        raise AssertionError("valid or quiet keys must not compute")

    def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
        raise AssertionError("valid or quiet keys must not publish")


def install_audit_derivations(monkeypatch: pytest.MonkeyPatch, failure: str) -> list[AuditDerivation]:
    from polylogue.daemon import session_profile_composition as composition
    from polylogue.storage.derived.session.derivation import SESSION_PROFILE_DOMAIN
    from polylogue.storage.derived.session.marker_domain import SESSION_MARKER_DOMAIN
    from polylogue.storage.derived.session.summary import SESSION_SUMMARY_DOMAIN
    from polylogue.storage.derived.session.usage_rollup import SESSION_USAGE_ROLLUP_DOMAIN

    factories = (
        ("make_session_summary_derivation", SESSION_SUMMARY_DOMAIN),
        ("make_session_usage_rollup_derivation", SESSION_USAGE_ROLLUP_DOMAIN),
        ("make_session_profile_derivation", SESSION_PROFILE_DOMAIN),
        ("make_session_marker_derivation", SESSION_MARKER_DOMAIN),
    )
    adapters = [AuditDerivation(domain, failure) for _factory, domain in factories]

    def factory(adapter: AuditDerivation) -> Callable[..., AuditDerivation]:
        def make(_path: Path, **_kwargs: Any) -> AuditDerivation:
            return adapter

        return make

    for (name, _domain), adapter in zip(factories, adapters, strict=True):
        monkeypatch.setattr(composition, name, factory(adapter))
    return adapters
