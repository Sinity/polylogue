from __future__ import annotations

import pytest

from polylogue.core.config_validation import validate_config_payload


def test_validate_config_rejects_bad_theme():
    with pytest.raises(SystemExit):
        validate_config_payload({"defaults": {"html_theme": "purple"}})


def test_validate_config_accepts_partial():
    validate_config_payload({"defaults": {"collapse_threshold": 50}})
