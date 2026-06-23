"""Regression tests for ``polylogue config`` secret redaction (#1748).

The config command is the artifact users paste into bug reports and share
with coding agents. A secret value must never appear in cleartext in any of
its three output surfaces: the default TOML, ``--format json``, and
``--show-layers`` (in both TOML and JSON forms).
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner
from rich.console import Console

from polylogue.cli.commands.config import config_command
from tests.infra.app_env import make_app_env

_VOYAGE_SECRET = "sk-voyage-LEAKME-0123456789"
_WEBHOOK_SECRET = "wh-LEAKME-secret-token"
_EMAIL_PASSWORD = "email-LEAKME-passw0rd"
_AUTH_TOKEN = "auth-LEAKME-bearer"

_CONFIG_BODY = f"""\
[embedding]
enabled = true
voyage_api_key = "{_VOYAGE_SECRET}"

[notifications]
webhook_secret = "{_WEBHOOK_SECRET}"

[notifications.email]
password = "{_EMAIL_PASSWORD}"

[daemon.api]
auth_token = "{_AUTH_TOKEN}"
"""

_ALL_SECRETS = (_VOYAGE_SECRET, _WEBHOOK_SECRET, _EMAIL_PASSWORD, _AUTH_TOKEN)


@pytest.fixture()
def config_with_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "polylogue.toml"
    cfg.write_text(_CONFIG_BODY, encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg))
    # Avoid a site-layer file leaking into the resolved config.
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))
    # The flat ``voyage_api_key`` secret is populated from the ``VOYAGE_API_KEY``
    # env var (config._apply_env_overrides), which the autouse env-stripping
    # fixture removes — set it so the secret is present for redaction assertions.
    monkeypatch.setenv("VOYAGE_API_KEY", _VOYAGE_SECRET)
    return cfg


def _run(args: list[str]) -> str:
    env = make_app_env()
    result = CliRunner().invoke(config_command, obj=env, args=args)
    assert result.exit_code == 0, (result.output, result.exception)
    return cast(StringIO, cast(Console, env.ui.console).file).getvalue()


@pytest.mark.parametrize(
    "args",
    [
        [],  # default toml
        ["-f", "toml"],
        ["-f", "json"],
        ["--show-layers"],
        ["--show-layers", "-f", "json"],
    ],
    ids=["default", "toml", "json", "layers-toml", "layers-json"],
)
def test_no_secret_appears_in_config_output(config_with_secrets: Path, args: list[str]) -> None:
    output = _run(args)
    for secret in _ALL_SECRETS:
        assert secret not in output, f"secret leaked in `polylogue config {' '.join(args)}`"


def test_redacted_secrets_show_presence_placeholder(config_with_secrets: Path) -> None:
    # A set secret renders as the presence placeholder, not its value.
    toml_out = _run(["-f", "toml"])
    assert "<set>" in toml_out
    json_out = _run(["-f", "json"])
    payload = json.loads(json_out)
    assert payload["values"]["voyage_api_key"]["value"] == "<set>"
    assert payload["values"]["notification_webhook_secret"]["value"] == "<set>"


def test_unset_secret_renders_unset_in_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "polylogue.toml"
    cfg.write_text("[embedding]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))
    # Present-but-empty secret: ``VOYAGE_API_KEY=""`` lands in the config as an
    # empty value, which must render as ``<unset>`` (never null/cleartext).
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    payload = json.loads(_run(["-f", "json"]))
    # An unset secret is reported as <unset>, never as null masquerading as a value.
    assert payload["values"]["voyage_api_key"]["value"] == "<unset>"


def test_non_secret_values_are_not_redacted(config_with_secrets: Path) -> None:
    payload = json.loads(_run(["-f", "json"]))
    # Non-secret config keys must still be visible verbatim.
    assert payload["values"]["embedding_enabled"]["value"] is True


def test_json_effective_config_includes_inventory_and_layer_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "polylogue.toml"
    cfg.write_text("[daemon.api]\nport = 9001\n", encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))
    monkeypatch.setenv("POLYLOGUE_API_PORT", "9002")
    monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", _AUTH_TOKEN)

    payload = json.loads(_run(["-f", "json"]))

    api_port = payload["values"]["api_port"]
    assert api_port["value"] == 9002
    assert api_port["source_layer"] == "env"
    assert api_port["toml_path"] == "daemon.api.port"
    assert api_port["env_var"] == "POLYLOGUE_API_PORT"
    assert api_port["owner_class"] == "network-security"
    assert api_port["reload_behavior"] == "startup-bound"

    api_token = payload["values"]["api_auth_token"]
    assert api_token["value"] == "<set>"
    assert api_token["secret"] is True
    assert api_token["secret_present"] is True

    inventory_by_key = {entry["key"]: entry for entry in payload["inventory"]}
    assert inventory_by_key["api_port"]["default"] == 8766
