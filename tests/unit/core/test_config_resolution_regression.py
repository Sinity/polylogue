"""Fixture-driven regression coverage for polylogue-9gh1.

The epic (config.py: close the gap between documented 5-layer precedence and
actual runtime behavior) named three structurally distinct bug shapes found
by the dogfood-2 round-2 config investigation (``investigations/config-resolution.md``):

1. **fd2s-class (split-system delegation)** -- a setting is inventoried with a
   ``toml_path`` (implying full 5-layer precedence) but its real runtime
   consumer resolves the value some other way (legacy ``Config``/env-only
   reads), so TOML precedence is silently dead for that consumer no matter how
   many call sites get migrated to read "the config".
2. **cxlk-class (nested-table full-replace)** -- ``_merge_toml`` treats a
   nested TOML table as an opaque leaf value, so a later layer's partial
   override of the table silently discards earlier-layer sibling keys instead
   of deep-merging.
3. **uu8r-class (direct os.environ bypass)** -- a file reads a genuinely
   POLYLOGUE_*-namespaced, TOML-backed setting straight from
   ``os.environ`` instead of routing through :func:`load_polylogue_config`,
   so TOML/site/user layering never reaches that call site even though the
   setting is fully inventoried.

Each test class below pins one bug shape against a real runtime consumer (not
a synthetic replica) and states, in its docstring, the exact mutation that
reverts the corresponding fix and makes the test fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _disable_site(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty POLYLOGUE_SITE_CONFIG disables site discovery so tests do not
    # accidentally read /etc/polylogue/polylogue.toml from the host.
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")


class TestNestedTableDeepMergePreservesSiblings:
    """cxlk-class: health.convergence_debt / health.cursor_lag deep-merge.

    Reverted-mutation witness: replace ``_deep_merge_table``'s recursive
    branch in ``polylogue/config.py`` with a plain
    ``{**existing, **incoming}`` (or, equivalently, have ``_merge_toml``'s
    ``toml_kind == "table"`` branch do ``cfg[entry.key] = dict(value)``
    instead of ``_deep_merge_table(base, value)``) -- both tests below then
    fail because the site layer's ``default_warning`` and family overrides
    are silently dropped by the user layer's partial override.
    """

    def test_health_convergence_debt_retains_site_siblings_after_user_partial_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.convergence_debt_alert import load_thresholds_from_config

        _disable_site(monkeypatch)
        site = tmp_path / "site.toml"
        site.write_text(
            """
[health.convergence_debt]
default_warning = 1
default_error = 10

[health.convergence_debt.families.claude-code-session]
warning = 1
error = 5

[health.convergence_debt.families.chatgpt-export]
warning = 25
error = 200
""",
            encoding="utf-8",
        )
        user = tmp_path / "user.toml"
        user.write_text(
            """
[health.convergence_debt]
default_error = 20
""",
            encoding="utf-8",
        )
        cfg = load_polylogue_config(site_config_path=site, config_path=user)

        # Raw-dict assertion: the user layer's override survives, but the
        # site layer's sibling keys are NOT clobbered.
        merged = cfg.raw["health_convergence_debt"]
        assert isinstance(merged, dict)
        assert merged["default_warning"] == 1, "site-layer default_warning was dropped by user partial override"
        assert merged["default_error"] == 20, "user-layer override did not win"
        assert merged["families"]["claude-code-session"] == {"warning": 1, "error": 5}
        assert merged["families"]["chatgpt-export"] == {"warning": 25, "error": 200}

        # Real-consumer assertion: the daemon alert evaluator actually reads
        # both the tuned global default and the per-family overrides.
        thresholds = load_thresholds_from_config(cfg)
        assert thresholds.default_warning == 1
        assert thresholds.default_error == 20
        assert thresholds.for_family("claude-code-session") == (1, 5)
        assert thresholds.for_family("chatgpt-export") == (25, 200)
        # A family with no override still falls back to the tuned defaults,
        # not the hardcoded module defaults.
        assert thresholds.for_family("codex-session") == (1, 20)

    def test_health_cursor_lag_retains_site_siblings_after_user_partial_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.cursor_lag_alert import load_thresholds_from_config

        _disable_site(monkeypatch)
        site = tmp_path / "site.toml"
        site.write_text(
            """
[health.cursor_lag]
default_warning_s = 60
default_error_s = 600

[health.cursor_lag.families.claude-code-session]
warning_s = 30
error_s = 300
""",
            encoding="utf-8",
        )
        user = tmp_path / "user.toml"
        user.write_text(
            """
[health.cursor_lag]
default_error_s = 900
""",
            encoding="utf-8",
        )
        cfg = load_polylogue_config(site_config_path=site, config_path=user)

        merged = cfg.raw["health_cursor_lag"]
        assert isinstance(merged, dict)
        assert merged["default_warning_s"] == 60, "site-layer default_warning_s was dropped by user partial override"
        assert merged["default_error_s"] == 900
        assert merged["families"]["claude-code-session"] == {"warning_s": 30, "error_s": 300}

        thresholds = load_thresholds_from_config(cfg)
        assert thresholds.default_warning_s == 60
        assert thresholds.default_error_s == 900


class TestDelegatedSettingsReachRealConsumers:
    """fd2s-class: an inventoried, toml_path-bearing setting must reach the
    real object every runtime consumer uses, not a parallel ambient-read
    authority.

    Reverted-mutation witness for the voyage_api_key test: restore
    ``polylogue/storage/search_providers/__init__.py``'s
    ``voyage_key = os.environ.get("VOYAGE_API_KEY")`` fallback in place of
    ``load_polylogue_config().voyage_api_key`` -- the test then fails because
    no ``VOYAGE_API_KEY`` env var is set (TOML-only configuration).
    """

    def test_archive_root_toml_reaches_resolved_runtime_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from polylogue.config import resolve_runtime_config

        _disable_site(monkeypatch)
        for key in ("POLYLOGUE_ARCHIVE_ROOT", "XDG_DATA_HOME"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "unused-data-home"))
        configured_root = tmp_path / "toml-configured-archive"
        user = tmp_path / "user.toml"
        user.write_text(f'[archive]\nroot = "{configured_root.as_posix()}"\n', encoding="utf-8")

        runtime = resolve_runtime_config(config_path=user)

        assert runtime.paths.archive_root == configured_root
        assert runtime.paths.index_db == configured_root / "index.db"
        assert runtime.settings.layer_of("archive_root") == "user"

    def test_voyage_api_key_toml_only_reaches_vector_provider_construction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        pytest.importorskip("sqlite_vec")
        from polylogue.storage.search_providers import create_vector_provider

        _disable_site(monkeypatch)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        user = tmp_path / "user.toml"
        user.write_text('[embedding]\nvoyage_api_key = "toml-only-fixture-key"\n', encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        provider = create_vector_provider(db_path=tmp_path / "embeddings.db")

        assert provider is not None, "vector provider construction failed even though TOML supplied a voyage key"
        assert getattr(provider, "voyage_key", None) == "toml-only-fixture-key"


class TestDirectEnvBypassCallersRouteThroughResolver:
    """uu8r-class: already-inventoried settings whose real call site read
    ``os.environ`` directly instead of the layered resolver.

    Reverted-mutation witness (backup): restore
    ``env_tmpdir = os.environ.get("POLYLOGUE_BACKUP_VERIFY_TMPDIR")`` in
    ``polylogue/daemon/backup.py::_backup_verification_scratch_parent`` --
    the test then fails because no environment variable is set (TOML-only
    configuration) and the scratch parent falls back to ``/realm/tmp``
    instead of the configured directory.

    Reverted-mutation witness (antigravity): restore
    ``env_path = os.environ.get("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER")`` in
    ``polylogue/sources/parsers/antigravity.py::discover_language_server`` --
    the test then fails because the TOML-only fixture path is never found and
    discovery falls through to the ``shutil.which``/glob probes.
    """

    def test_backup_verify_tmpdir_toml_only_reaches_scratch_parent_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.daemon.backup import _backup_verification_scratch_parent

        _disable_site(monkeypatch)
        monkeypatch.delenv("POLYLOGUE_BACKUP_VERIFY_TMPDIR", raising=False)
        configured_scratch = tmp_path / "toml-configured-scratch"
        user = tmp_path / "user.toml"
        user.write_text(f'[maintenance]\nbackup_verify_tmpdir = "{configured_scratch.as_posix()}"\n', encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        # A backup path whose own parent cannot be used (missing), so
        # resolution must fall through to the configured scratch directory
        # rather than the final hardcoded /realm/tmp candidate.
        unusable_backup_path = Path("/nonexistent-root-for-test") / "backup"

        result = _backup_verification_scratch_parent(unusable_backup_path)

        assert result == configured_scratch
        assert configured_scratch.is_dir()

    def test_antigravity_language_server_toml_only_reaches_discovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.sources.parsers.antigravity import discover_language_server

        _disable_site(monkeypatch)
        monkeypatch.delenv("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER", raising=False)
        configured_binary = tmp_path / "fixture-language-server"
        configured_binary.write_text("#!/bin/sh\n", encoding="utf-8")
        user = tmp_path / "user.toml"
        user.write_text(
            f'[sources.antigravity]\nlanguage_server = "{configured_binary.as_posix()}"\n', encoding="utf-8"
        )
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        result = discover_language_server()

        assert result == configured_binary


class TestNewlyInventoriedSettingsRouteThroughResolver:
    """polylogue-uu8r: settings that had NO config-inventory entry at all
    before this change (not merely a bypassing caller for an
    already-inventoried key, unlike the class above) -- each test pins one
    setting family against its real runtime consumer.
    """

    def test_hook_provider_toml_only_forces_detection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        """Reverted-mutation witness: restore
        ``forced = os.environ.get("POLYLOGUE_HOOK_PROVIDER")`` in
        ``polylogue/hooks/__init__.py::_detect_hook_provider`` -- the test
        then fails because no environment variable is set (TOML-only
        configuration) and detection falls through to the payload-shape
        sniffing branches below, which do not match this ambiguous payload.
        """
        from polylogue.hooks import _detect_hook_provider

        _disable_site(monkeypatch)
        monkeypatch.delenv("POLYLOGUE_HOOK_PROVIDER", raising=False)
        user = tmp_path / "user.toml"
        user.write_text('[sources]\nhook_provider = "codex"\n', encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        # Ambiguous payload: none of the shape-sniffing branches (turn_id,
        # permission_mode/model, source) match, so an unforced detection
        # would return None.
        result = _detect_hook_provider({})

        assert result == "codex"

    def test_raw_authority_commit_batch_size_toml_only_reaches_repair_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        """Reverted-mutation witness: restore
        ``raw = os.environ.get("POLYLOGUE_RAW_AUTHORITY_COMMIT_BATCH_SIZE")``
        in ``polylogue/storage/repair.py::_resolve_raw_authority_commit_batch_size``
        -- the test then fails because no environment variable is set
        (TOML-only configuration) and resolution falls back to the module's
        hardcoded ``RAW_MATERIALIZATION_COMMIT_BATCH_SIZE`` default instead
        of the configured value.
        """
        from polylogue.storage.repair import _resolve_raw_authority_commit_batch_size

        _disable_site(monkeypatch)
        monkeypatch.delenv("POLYLOGUE_RAW_AUTHORITY_COMMIT_BATCH_SIZE", raising=False)
        user = tmp_path / "user.toml"
        user.write_text("[pipeline.raw_authority]\ncommit_batch_size = 4242\n", encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        assert _resolve_raw_authority_commit_batch_size(None) == 4242
        # Explicit caller override still wins over the configured value.
        assert _resolve_raw_authority_commit_batch_size(7) == 7

    def test_revision_parse_dispatch_thresholds_toml_only_reach_dispatch_partitioning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        """Reverted-mutation witness: restore either
        ``os.environ.get("POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES")`` or
        ``os.environ.get("POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES")`` in
        ``polylogue/sources/revision_backfill.py`` -- the test then fails
        because no environment variable is set (TOML-only configuration) and
        both helpers fall back to their hardcoded module defaults
        (262144 bytes / 48 MiB) instead of the configured values.
        """
        from polylogue.sources.revision_backfill import (
            _parse_dispatch_max_bytes,
            _parse_pool_min_aggregate_bytes,
        )

        _disable_site(monkeypatch)
        monkeypatch.delenv("POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES", raising=False)
        monkeypatch.delenv("POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES", raising=False)
        user = tmp_path / "user.toml"
        user.write_text(
            "[pipeline.revision_parse]\ndispatch_max_bytes = 1024\npool_min_bytes = 2048\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        assert _parse_dispatch_max_bytes() == 1024
        assert _parse_pool_min_aggregate_bytes() == 2048

    def test_daemon_parse_stage_knobs_toml_only_reach_prefetch_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        """Reverted-mutation witness: restore any of the four
        ``os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_...")`` reads in
        ``polylogue/daemon/parse_prefetch.py`` -- the corresponding assertion
        then fails because no environment variable is set (TOML-only
        configuration) and that helper falls back to its adaptive
        physical-RAM-derived or hardcoded default instead of the configured
        value. These four knobs are read from a daemon-owned
        ``ThreadPoolExecutor`` in-process (never inside a spawned/forked
        worker), so ``load_polylogue_config()`` at this call site carries no
        multiprocessing-spawn hazard.
        """
        from polylogue.daemon.parse_prefetch import (
            daemon_parse_stage_max_cached_tree_bytes,
            daemon_parse_stage_max_inflight_bytes,
            daemon_parse_stage_warm_timeout_seconds,
            daemon_parse_stage_worker_count,
        )

        _disable_site(monkeypatch)
        for env_var in (
            "POLYLOGUE_DAEMON_PARSE_STAGE_WORKERS",
            "POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES",
            "POLYLOGUE_DAEMON_PARSE_STAGE_MAX_CACHED_TREE_BYTES",
            "POLYLOGUE_DAEMON_PARSE_STAGE_WARM_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(env_var, raising=False)
        user = tmp_path / "user.toml"
        user.write_text(
            """
[daemon.raw_materialization]
parse_stage_workers = 3
parse_stage_max_inflight_bytes = 999999
parse_stage_max_cached_tree_bytes = 8888888
parse_stage_warm_timeout_seconds = 12.5
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))

        assert daemon_parse_stage_worker_count() == 3
        assert daemon_parse_stage_max_inflight_bytes() == 999999
        assert daemon_parse_stage_max_cached_tree_bytes() == 8888888
        assert daemon_parse_stage_warm_timeout_seconds() == 12.5
