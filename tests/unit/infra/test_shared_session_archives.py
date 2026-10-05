"""The controller warm-up covers every archive the session fixtures share.

Anti-vacuity: drop an entry from ``shared_session_archives()``, or declare a
builder in ``tests.infra.shared_session_archives`` without registering it, and
:func:`test_every_declared_builder_is_warmed` goes red. Bind the named-profile
entries to the loop variable instead of a default argument and
:func:`test_named_entries_build_their_own_profile` goes red.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast

import pytest

from polylogue.scenarios import CorpusSpec
from tests.infra import shared_session_archives as registry
from tests.infra.workload_artifacts import (
    SeededArchiveArtifact,
    seeded_archive_key,
)
from tests.infra.workload_declarations import (
    NAMED_WORKLOAD_PROFILES,
    named_corpus_specs,
    schema_coverage_corpus_specs,
)

pytest_plugins = ("tests.infra.corpus_fixtures",)

#: Registry accessors, not builders of an archive.
_NON_BUILDERS = frozenset({"shared_session_archives", "warm_shared_session_archives"})


def _zero_argument_builders() -> dict[str, object]:
    """Every public builder the module declares that the warm-up could call."""
    return {
        name: value
        for name in registry.__all__
        if name not in _NON_BUILDERS
        and callable(value := getattr(registry, name))
        and not inspect.isclass(value)
        and not inspect.signature(value).parameters
    }


def test_every_declared_builder_is_warmed() -> None:
    registered = {entry.build for entry in registry.shared_session_archives()}
    declared = _zero_argument_builders()
    assert declared, "module declares no zero-argument builders to check"
    assert set(declared.values()) <= registered


def test_every_named_workload_profile_is_warmed() -> None:
    names = {entry.name for entry in registry.shared_session_archives()}
    assert {f"named-{profile.name}" for profile in NAMED_WORKLOAD_PROFILES} <= names


def test_registry_entry_names_are_unique() -> None:
    entries = registry.shared_session_archives()
    assert len({entry.name for entry in entries}) == len(entries)


def test_named_entries_build_their_own_profile() -> None:
    by_name = {entry.name: entry for entry in registry.shared_session_archives()}
    for profile in NAMED_WORKLOAD_PROFILES:
        artifact = by_name[f"named-{profile.name}"].build()
        assert isinstance(artifact, SeededArchiveArtifact)
        assert artifact.manifest.key == seeded_archive_key(named_corpus_specs(profile.name)).value


def test_warm_up_accounts_for_every_entry_and_publishes_what_it_warmed() -> None:
    """One warm-up call: each entry is either published or reported with its reason."""
    outcome = registry.warm_shared_session_archives()
    by_name = {entry.name: entry for entry in registry.shared_session_archives()}
    accounted = [*outcome.warmed, *(name for name, _ in outcome.failed)]
    assert sorted(accounted) == sorted(by_name)
    assert outcome.warmed, f"nothing warmed; failures: {outcome.failed}"
    for name in outcome.warmed:
        built = by_name[name].build()
        assert isinstance(built, SeededArchiveArtifact)
        assert built.root.is_dir()


def test_warm_up_survives_an_archive_that_cannot_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken archive is charged to its own fixture, never to the whole run.

    Anti-vacuity: let the exception escape ``warm_shared_session_archives`` and
    this raises instead of reporting, which is the collection-time abort the
    guard exists to prevent.
    """

    def explode() -> object:
        raise RuntimeError("recipe moved and the rebuild is red")

    built: list[str] = []
    monkeypatch.setattr(
        registry,
        "_REGISTRY",
        (
            registry.SharedSessionArchive("broken", explode),
            registry.SharedSessionArchive("after", lambda: built.append("after")),
        ),
    )
    outcome = registry.warm_shared_session_archives()
    assert outcome.warmed == ("after",)
    assert outcome.failed == (("broken", "RuntimeError: recipe moved and the rebuild is red"),)
    assert built == ["after"], "a failed entry must not stop the entries behind it"


@pytest.mark.parametrize(
    ("builder", "specs"),
    [
        pytest.param(registry.schema_coverage_archive, schema_coverage_corpus_specs, id="schema-coverage"),
        pytest.param(registry.semantic_archive, None, id="semantic"),
    ],
)
def test_builders_resolve_the_artifact_their_fixture_reads(
    builder: Callable[[], SeededArchiveArtifact],
    specs: Callable[[], tuple[CorpusSpec, ...]] | None,
) -> None:
    artifact = builder()
    assert isinstance(artifact, SeededArchiveArtifact)
    if specs is not None:
        assert artifact.manifest.key == seeded_archive_key(specs()).value


def test_seeded_archive_fixture_resolves_through_the_registry(
    seeded_archive: SeededArchiveArtifact,
) -> None:
    assert seeded_archive.manifest.key == registry.schema_coverage_archive().manifest.key


@pytest.mark.parametrize("fixture_name", ["seeded_archive", "named_seeded_artifact", "pilot_artifact"])
def test_session_fixture_pins_cache_before_acquisition_until_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_name: str
) -> None:
    """Removing the original cache-domain hold deletes an aged artifact under the actual fixture."""
    import dataclasses
    import io
    import json
    import os

    from devtools import seeded_archive_cache_gc as command
    from tests.infra import corpus_fixtures, integration_profile, pilot_resources, workload_artifacts
    from tests.infra.workload_declarations import c03_semantic_corpus_spec

    cache = tmp_path / "cache"
    artifact = workload_artifacts.build_seeded_archive(
        (
            dataclasses.replace(
                c03_semantic_corpus_spec(), seed=997, count=2, session_native_ids=("c03-target", "c03-irrelevant-000")
            ),
        ),
        cache_root=cache,
    )
    os.utime(artifact.root, (0, 0))
    os.utime(artifact.root / "manifest.json", (0, 0))
    monkeypatch.setattr(workload_artifacts, "default_cache_root", lambda: cache)

    def collect() -> dict[str, object]:
        output = io.StringIO()
        assert (
            command.main(["--cache-root", str(cache), "--grace-period-s", "1", "--apply", "--json"], stdout=output) == 0
        )
        return cast(dict[str, object], json.loads(output.getvalue()))

    def acquire(*_args: object, **_kwargs: object) -> SeededArchiveArtifact:
        assert collect()["dispositions"] == {"active-lock": 1}
        assert artifact.root.exists()
        return artifact

    if fixture_name == "seeded_archive":
        monkeypatch.setattr(corpus_fixtures, "schema_coverage_archive", acquire)
        fixture = cast(Callable[[], Generator[object, None, None]], inspect.unwrap(corpus_fixtures.seeded_archive))()
    elif fixture_name == "named_seeded_artifact":
        monkeypatch.setattr(corpus_fixtures, "build_seeded_archive", acquire)
        fixture = cast(
            Callable[[], Generator[object, None, None]], inspect.unwrap(corpus_fixtures.named_seeded_artifact)
        )()
    else:
        monkeypatch.setattr(integration_profile, "build_integration_archive", acquire)
        fixture = cast(Callable[[], Generator[object, None, None]], inspect.unwrap(pilot_resources.pilot_artifact))()
    try:
        value = next(fixture)
        if fixture_name == "named_seeded_artifact":
            assert callable(value) and value(NAMED_WORKLOAD_PROFILES[0].name) is artifact
        else:
            assert value is artifact
        assert collect()["dispositions"] == {"active-lock": 1}
        assert artifact.root.exists()
    finally:
        fixture.close()
    assert collect()["dispositions"] == {"deleted": 1}
    assert not artifact.root.exists()
