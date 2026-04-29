"""Tests for ``polylogue.proof.suppressions``."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from polylogue.proof.suppressions import Suppression, is_expired, load_suppressions, validate_suppressions


def _registry(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "suppressions.yaml"
    p.write_text(content)
    return p


def _sup(
    *,
    id: str = "test-sup",
    reason: str = "test reason",
    expires_at: str = "2099-12-31",
    issue: str | None = "#1",
    paths: tuple[str, ...] = (),
    claims: tuple[str, ...] = (),
) -> Suppression:
    return Suppression(
        id=id,
        reason=reason,
        expires_at=expires_at,
        issue=issue,
        paths=paths,
        claims=claims,
    )


class TestLoadSuppressions:
    def test_empty_registry(self, tmp_path: Path) -> None:
        p = _registry(tmp_path, "suppressions: []\n")
        result = load_suppressions(registry=p)
        assert result == []

    def test_no_suppressions_key(self, tmp_path: Path) -> None:
        p = _registry(tmp_path, "other: 42\n")
        result = load_suppressions(registry=p)
        assert result == []

    def test_not_a_dict(self, tmp_path: Path) -> None:
        p = _registry(tmp_path, "just a string\n")
        result = load_suppressions(registry=p)
        assert result == []

    def test_single_suppression(self, tmp_path: Path) -> None:
        p = _registry(
            tmp_path,
            """suppressions:
  - id: my-sup
    reason: known issue
    expires_at: "2099-12-31"
    issue: "#123"
    paths:
      - polylogue/foo.py
    claims:
      - claim-bar
""",
        )
        result = load_suppressions(registry=p)
        assert len(result) == 1
        s = result[0]
        assert s.id == "my-sup"
        assert s.reason == "known issue"
        assert s.expires_at == "2099-12-31"
        assert s.issue == "#123"
        assert s.paths == ("polylogue/foo.py",)
        assert s.claims == ("claim-bar",)

    def test_multiple_suppressions(self, tmp_path: Path) -> None:
        p = _registry(
            tmp_path,
            """suppressions:
  - id: sup-a
    reason: reason a
    expires_at: "2099-01-01"
  - id: sup-b
    reason: reason b
    expires_at: "2099-02-01"
""",
        )
        result = load_suppressions(registry=p)
        assert len(result) == 2
        assert result[0].id == "sup-a"
        assert result[1].id == "sup-b"

    def test_optional_fields(self, tmp_path: Path) -> None:
        p = _registry(
            tmp_path,
            """suppressions:
  - id: minimal
    reason: just because
    expires_at: "2099-12-31"
""",
        )
        result = load_suppressions(registry=p)
        assert len(result) == 1
        s = result[0]
        assert s.issue is None
        assert s.paths == ()
        assert s.claims == ()


class TestIsExpired:
    def test_not_expired(self) -> None:
        s = _sup(expires_at="2099-12-31")
        assert not is_expired(s, now=date(2026, 1, 1))

    def test_expired(self) -> None:
        s = _sup(expires_at="2025-01-01")
        assert is_expired(s, now=date(2026, 1, 1))

    def test_expires_today_not_expired(self) -> None:
        """Same day as expiry is still current (strictly < comparison)."""
        s = _sup(expires_at="2026-01-15")
        assert not is_expired(s, now=date(2026, 1, 15))

    def test_missing_expires_at(self) -> None:
        s = _sup(expires_at="")
        assert is_expired(s)

    def test_invalid_date_format(self) -> None:
        s = _sup(expires_at="not-a-date")
        assert is_expired(s)

    def test_iso_datetime_with_tz(self) -> None:
        s = _sup(expires_at="2099-12-31T23:59:59+00:00")
        assert not is_expired(s, now=date(2026, 1, 1))

    def test_iso_datetime_expired(self) -> None:
        s = _sup(expires_at="2025-06-15T12:00:00Z")
        assert is_expired(s, now=date(2026, 1, 1))

    def test_default_now_is_today(self) -> None:
        """Without an explicit ``now``, the expiry check uses today's date."""
        today = datetime.now(timezone.utc).date()
        s = _sup(expires_at=today.isoformat())
        assert not is_expired(s)


class TestValidateSuppressions:
    def test_empty_list_is_valid(self) -> None:
        """An empty registry (no suppressions) is a valid state."""
        errors = validate_suppressions([])
        assert errors == []

    def test_valid_suppression_no_errors(self) -> None:
        s = _sup()
        errors = validate_suppressions([s], now=date(2026, 1, 1))
        assert errors == []

    def test_expired_reported(self) -> None:
        s = _sup(expires_at="2025-01-01")
        errors = validate_suppressions([s], now=date(2026, 1, 1))
        assert len(errors) == 1
        assert "expired at" in errors[0]
        assert s.id in errors[0]

    def test_missing_id(self) -> None:
        s = _sup(id="", expires_at="2099-12-31")
        errors = validate_suppressions([s])
        assert len(errors) == 1
        assert "missing 'id'" in errors[0]

    def test_missing_expires_at(self) -> None:
        s = _sup(expires_at="")
        errors = validate_suppressions([s])
        assert len(errors) == 1
        assert "missing 'expires_at'" in errors[0]

    def test_invalid_expires_at(self) -> None:
        s = _sup(expires_at="bad-date")
        errors = validate_suppressions([s])
        assert len(errors) == 1
        assert "invalid 'expires_at'" in errors[0]

    def test_mixed_valid_and_invalid(self) -> None:
        s1 = _sup(id="good", expires_at="2099-12-31")
        s2 = _sup(id="bad", expires_at="2025-01-01")
        s3 = _sup(id="empty", expires_at="")
        errors = validate_suppressions([s1, s2, s3], now=date(2026, 1, 1))
        assert len(errors) == 2
        error_texts = "\n".join(errors)
        assert "bad" in error_texts
        assert "empty" in error_texts
        assert "good" not in error_texts


class TestLoadAndValidateIntegration:
    def test_committed_registry_is_valid(self) -> None:
        """The committed suppressions.yaml should parse and pass validation."""
        suppressions = load_suppressions()
        errors = validate_suppressions(suppressions)
        assert not errors, f"committed suppressions.yaml has errors: {errors}"
