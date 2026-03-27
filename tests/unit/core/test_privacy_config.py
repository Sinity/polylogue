"""Direct tests for polylogue.schemas.privacy_config module.

Covers PrivacyConfig presets, threshold computation, field overrides,
value allow/deny patterns, and TOML config loading.
"""
from __future__ import annotations

from polylogue.schemas.privacy_config import PrivacyConfig, load_privacy_config


class TestPrivacyConfigPresets:
    def test_strict_preset_values(self) -> None:
        config = PrivacyConfig(level="strict")
        assert config.safe_enum_max_length == 30
        assert config.high_entropy_min_length == 8
        assert config.cross_conv_min_count == 5
        assert config.cross_conv_proportional is True

    def test_standard_preset_values(self) -> None:
        config = PrivacyConfig(level="standard")
        assert config.safe_enum_max_length == 50
        assert config.high_entropy_min_length == 10
        assert config.cross_conv_min_count == 3
        assert config.cross_conv_proportional is False

    def test_permissive_preset_values(self) -> None:
        config = PrivacyConfig(level="permissive")
        assert config.safe_enum_max_length == 80
        assert config.high_entropy_min_length == 16
        assert config.cross_conv_min_count == 1


class TestEffectiveCrossConvThreshold:
    def test_proportional_mode(self) -> None:
        config = PrivacyConfig(level="strict")  # proportional=True
        # max(3, int(1000 * 0.02)) = max(3, 20) = 20
        assert config.effective_cross_conv_threshold(1000) == 20

    def test_proportional_mode_small_corpus(self) -> None:
        config = PrivacyConfig(level="strict")
        # max(3, int(50 * 0.02)) = max(3, 1) = 3
        assert config.effective_cross_conv_threshold(50) == 3

    def test_fixed_mode(self) -> None:
        config = PrivacyConfig(level="standard")  # proportional=False
        assert config.effective_cross_conv_threshold(1000) == 3
        assert config.effective_cross_conv_threshold(50) == 3


class TestFieldOverride:
    def test_glob_matching(self) -> None:
        config = PrivacyConfig(field_overrides={"$.mapping.*": "allow"})
        assert config.field_override("$.mapping.abc") == "allow"
        assert config.field_override("$.other") is None

    def test_deny_override(self) -> None:
        config = PrivacyConfig(field_overrides={"$.secret*": "deny"})
        assert config.field_override("$.secret_key") == "deny"

    def test_no_overrides(self) -> None:
        config = PrivacyConfig()
        assert config.field_override("$.anything") is None


class TestValueAllowDeny:
    def test_deny_beats_allow(self) -> None:
        config = PrivacyConfig(
            deny_value_patterns=["secret*"],
            allow_value_patterns=["secret_public"],
        )
        # Deny is checked first, so "secret_public" is denied
        assert config.is_value_allowed("secret_public") is False

    def test_allow_pattern(self) -> None:
        config = PrivacyConfig(allow_value_patterns=["safe_*"])
        assert config.is_value_allowed("safe_value") is True

    def test_deny_pattern(self) -> None:
        config = PrivacyConfig(deny_value_patterns=["bad_*"])
        assert config.is_value_allowed("bad_token") is False

    def test_no_match_returns_none(self) -> None:
        config = PrivacyConfig(
            deny_value_patterns=["bad_*"],
            allow_value_patterns=["good_*"],
        )
        assert config.is_value_allowed("neutral") is None


class TestLoadPrivacyConfig:
    def test_cli_overrides_take_effect(self) -> None:
        config = load_privacy_config(
            cli_overrides={"level": "strict", "safe_enum_max_length": 100},
            project_path=None,
        )
        # CLI override should win
        assert config.safe_enum_max_length == 100

    def test_toml_cascade(self, tmp_path) -> None:
        toml_path = tmp_path / "polylogue-schemas.toml"
        toml_path.write_text(
            '[schema.privacy]\nlevel = "permissive"\nsafe_enum_max_length = 99\n'
        )
        config = load_privacy_config(project_path=tmp_path)
        assert config.level == "permissive"
        assert config.safe_enum_max_length == 99

    def test_field_overrides_merge(self, tmp_path) -> None:
        toml_path = tmp_path / "polylogue-schemas.toml"
        toml_path.write_text(
            '[schema.privacy]\n[schema.privacy.field_overrides]\n"$.a" = "allow"\n'
        )
        config = load_privacy_config(
            project_path=tmp_path,
            cli_overrides={"field_overrides": {"$.b": "deny"}},
        )
        assert config.field_overrides == {"$.a": "allow", "$.b": "deny"}

    def test_default_with_no_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nonexistent"))
        config = load_privacy_config(project_path=tmp_path)
        assert config.level == "standard"
