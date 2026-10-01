"""Production routes retain actual producers across declared asymmetric retrieval."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from polylogue.storage.embeddings.derivation import EmbeddingDerivationAdapter
from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingRequestSpec
from polylogue.storage.embeddings.materialization import (
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.search_providers.sqlite_vec import SqliteVecError, SqliteVecProvider
from tests.infra.embedding_compatibility import _NEW_TEXT, _TEXT, _Documents, _rows, _session


@pytest.mark.parametrize("selected", ["voyage-4-lite", "voyage-4-large"])
def test_compatible_switch_keeps_exact_outputs_and_occurrences_without_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    """Removing compatibility from inspector/selector or projection makes this red."""
    root = tmp_path / "archive"
    sid, ids = _session(root)
    original = _Documents("voyage-4")
    assert embed_archive_session_sync(root / "index.db", original, sid).status == "embedded"
    before = _rows(root)
    document_recipe = EmbeddingRecipe.current(model=selected, dimensions=1024)
    chosen = _Documents(selected)
    adapter = EmbeddingDerivationAdapter(root / "index.db", chosen)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    assert adapter.inspect(frame, [f"message:{mid}" for mid in ids]) == {f"message:{mid}": "valid" for mid in ids}
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(root / "embeddings.db"),))
        assert (
            select_pending_archive_session_window(
                conn, status_table="embeddings.embedding_status", recipe=document_recipe
            )
            == []
        )
    assert embed_archive_session_sync(root / "index.db", chosen, sid).status == "embedded"
    assert chosen.calls == []
    assert _rows(root) == before
    from polylogue.storage.embeddings.preflight import read_embedding_work_counts
    from polylogue.storage.embeddings.status_payload import embedding_status_payload
    from tests.infra.embedding_config import embedding_config

    assert read_embedding_work_counts(root / "index.db", recipe=document_recipe) == (1, 0, 0, 0)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config", lambda **kwargs: embedding_config(embedding_model=selected)
    )
    status = embedding_status_payload(
        SimpleNamespace(config=SimpleNamespace(db_path=root / "index.db", archive_root=root)), include_detail=True
    )
    assert status is not None
    assert status["status"] == "complete"
    assert status["compute_missing_messages"] == status["binding_pending_messages"] == 0

    requests: list[dict[str, object]] = []
    client_type = httpx.Client

    def serve(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024}]})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(serve), **kwargs))
    query_recipe = EmbeddingRecipe.current(model=selected, dimensions=1024, input_type="query")
    from tests.infra.embedding_compatibility import assert_embedding_handles_settled, embedding_file_header

    header_before = embedding_file_header(root)

    def refuse_writable_query(*args: object, **kwargs: object) -> sqlite3.Connection:
        raise AssertionError("text similarity must use the existing retained read snapshot")

    monkeypatch.setattr("polylogue.storage.search_providers.sqlite_vec_runtime.open_connection", refuse_writable_query)
    search = SqliteVecProvider(
        "synthetic-key", db_path=root / "embeddings.db", archive_root=root, model="voyage-4", query_recipe=query_recipe
    )
    hits = search.query("Why retain the producer identity?", limit=10)
    assert {mid for mid, distance in hits} == set(ids)
    assert requests[0]["model"] == selected
    assert requests[0]["input_type"] == "query"
    assert _rows(root) == before
    assert embedding_file_header(root) == header_before
    assert_embedding_handles_settled(root)

    _session(root, extra=True)
    outcome = embed_archive_session_sync(root / "index.db", chosen, sid)
    assert outcome.status == "embedded", outcome
    assert chosen.calls == [(_NEW_TEXT,)]
    meta, refs = _rows(root)
    assert {row[1] for row in meta} == {"voyage-4", selected}
    assert len(refs) == 3
    assert before[0][0] in meta


@pytest.mark.parametrize(
    "change",
    [
        {"model": "voyage-context-4"},
        {"provider": "different-endpoint"},
        {"dimensions": 512},
        {"model_revision": "new-revision"},
        {"normalization": "different"},
        {"request_options": (("truncation", False),)},
        {"canonicalization": "new-input-segmentation"},
        {"chunking_version": "different-segmentation"},
        {"input_type": "document"},
        {"element_type": "int8"},
    ],
)
def test_query_recipe_refuses_unproven_space_before_dispatch(change: dict[str, object], tmp_path: Path) -> None:
    """Matching model labels/dimensions cannot erase endpoint/revision/input contracts."""
    query = EmbeddingRecipe.current(model="voyage-4-lite", dimensions=1024, input_type="query")
    with pytest.raises((SqliteVecError, ValueError)):
        SqliteVecProvider(
            None, db_path=tmp_path / "embeddings.db", model="voyage-4", query_recipe=replace(query, **cast(Any, change))
        )


def test_compatible_space_does_not_change_exact_request_identity() -> None:
    doc = EmbeddingRecipe.current(model="voyage-4", dimensions=1024)
    other = replace(doc, model="voyage-4-lite")
    query = replace(other, input_type="query")
    assert doc.retrieval_compatible(query)
    assert doc.recipe_hash != other.recipe_hash != query.recipe_hash
    assert (
        len({EmbeddingRequestSpec(recipe=r, input_text=_TEXT).vector_derivation_hash for r in (doc, other, query)}) == 3
    )


def test_missing_bindings_use_exact_outputs_and_report_free_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing exact-output lookup spends on unchanged prose or hides binding debt."""
    from polylogue.storage.embeddings.preflight import read_embedding_work_counts
    from polylogue.storage.embeddings.status_payload import embedding_status_payload
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.embedding_config import embedding_config

    root = tmp_path / "archive"
    sid, ids = _session(root)
    provider = _Documents("voyage-4")
    assert embed_archive_session_sync(root / "index.db", provider, sid).status == "embedded"
    provider.calls.clear()
    from tests.infra.embedding_compatibility import clear_embedding_refs

    clear_embedding_refs(root)
    assert read_embedding_work_counts(
        root / "index.db", recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024)
    ) == (1, 0, 0, 2)
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda **kwargs: embedding_config())
    status = embedding_status_payload(
        SimpleNamespace(config=SimpleNamespace(db_path=root / "index.db", archive_root=root)), include_detail=True
    )
    assert status is not None
    assert status["compute_missing_messages"] == 0
    assert status["binding_pending_messages"] == 2
    assert status["status"] == "partial"
    adapter = EmbeddingDerivationAdapter(root / "index.db", provider)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    assert set(adapter.inspect(frame, [f"message:{mid}" for mid in ids]).values()) == {"missing"}
    for mid in ids:
        with write_lease("test.compatible-binding-reservation", archive_root=root):
            replacement = adapter.compute(frame, f"message:{mid}")
        with write_lease("test.compatible-binding-publication", archive_root=root):
            assert adapter.publish(frame, replacement)
    assert provider.calls == []
    assert len(_rows(root)[1]) == 2
    assert set(adapter.inspect(frame, [f"message:{mid}" for mid in ids]).values()) == {"valid"}


@pytest.mark.parametrize("damage", ["source", "recipe", "request", "payload"])
def test_compatible_selection_refuses_stale_or_incomplete_evidence(tmp_path: Path, damage: str) -> None:
    """Model-family compatibility cannot certify a stale source or absent output."""
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    root = tmp_path / "archive"
    sid, ids = _session(root)
    assert embed_archive_session_sync(root / "index.db", _Documents("voyage-4"), sid).status == "embedded"
    if damage == "source":
        with closing(sqlite3.connect(root / "index.db")) as conn, conn:
            conn.execute("UPDATE blocks SET text = ?", (_NEW_TEXT,))
    else:
        with closing(sqlite3.connect(root / "embeddings.db")) as conn, conn:
            assert try_load_sqlite_vec(conn)[0]
            if damage == "recipe":
                conn.execute("UPDATE message_embeddings_meta SET recipe_hash = ?", (b"x" * 32,))
            elif damage == "request":
                conn.execute("UPDATE message_embedding_refs SET vector_derivation_hash = ?", (b"y" * 32,))
            else:
                conn.execute("DELETE FROM message_embeddings")
    provider = _Documents("voyage-4-lite")
    adapter = EmbeddingDerivationAdapter(root / "index.db", provider)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    assert set(adapter.inspect(frame, [f"message:{mid}" for mid in ids]).values()) == {"stale"}
    assert provider.calls == []


def test_restart_retains_published_producer_proof_without_recipe_registry(tmp_path: Path) -> None:
    """A fresh process proves stored outputs through the production read projection."""
    import subprocess
    import sys

    root = tmp_path / "archive"
    sid, ids = _session(root)
    producer = _Documents("voyage-4")
    assert embed_archive_session_sync(root / "index.db", producer, sid).status == "embedded"
    before = _rows(root)
    completed = subprocess.run(
        [sys.executable, "-m", "tests.infra.embedding_restart_probe", str(root), sid],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert int(completed.stdout.strip()) == 1
    assert _rows(root) == before


def test_unproven_purchased_provenance_refuses_repurchase(tmp_path: Path) -> None:
    """Unknown exact producer metadata cannot become a paid computation miss."""
    from polylogue.storage.embeddings.materialization import EmbeddingProvenanceError
    from polylogue.storage.sqlite.write_lease import write_lease

    root = tmp_path / "archive"
    sid, ids = _session(root)
    assert embed_archive_session_sync(root / "index.db", _Documents("voyage-4"), sid).status == "embedded"
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore

    store = EmbeddingGenerationStore(root)
    with write_lease("test.unproven-producer-fixture", archive_root=root), store.writer_lock() as binding:
        conn = sqlite3.connect(binding.database_path)
        try:
            conn.execute("UPDATE message_embeddings_meta SET recipe_hash = ?", (b"x" * 32,))
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        store.refresh_binding_contract(binding)
    selected = _Documents("voyage-4-lite")
    outcome = embed_archive_session_sync(root / "index.db", selected, sid)
    assert outcome.status == "error"
    assert selected.calls == []
    adapter = EmbeddingDerivationAdapter(root / "index.db", selected)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    with write_lease("test.unproven-output-reservation", archive_root=root):
        with pytest.raises(EmbeddingProvenanceError):
            adapter.compute(frame, f"message:{ids[0]}")
    assert selected.calls == []


def test_provider_wire_keeps_independent_actual_roles_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the provider payload to a shared family label makes this red."""
    import json

    from polylogue.storage.search_providers import create_vector_provider

    captured: list[dict[str, object]] = []
    client_type = httpx.Client

    def serve(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024}]})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(serve), **kwargs))
    query = EmbeddingRecipe.current(model="voyage-4-large", dimensions=1024, input_type="query")
    provider = create_vector_provider(
        voyage_api_key="synthetic-key",
        db_path=tmp_path / "embeddings.db",
        archive_root=tmp_path,
        model="voyage-4",
        dimension=1024,
        query_recipe=query,
    )
    assert isinstance(provider, SqliteVecProvider)
    provider.model = "voyage-4-lite"
    assert len(provider._get_embeddings([_TEXT], input_type="document")) == 1
    assert len(provider._get_embeddings([_TEXT], input_type="query")) == 1
    assert [(payload["model"], payload["input_type"]) for payload in captured] == [
        ("voyage-4-lite", "document"),
        ("voyage-4-large", "query"),
    ]
    with pytest.raises(SqliteVecError):
        provider._get_embeddings([_TEXT], input_type="unknown")
    assert len(captured) == 2


def test_implicit_query_recipe_follows_actual_document_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caching an implicit query model at construction makes this red."""
    import json

    from polylogue.storage.search_providers import create_vector_provider

    captured: list[dict[str, object]] = []
    client_type = httpx.Client

    def serve(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024}]})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(serve), **kwargs))
    provider = create_vector_provider(
        voyage_api_key="synthetic-key",
        db_path=tmp_path / "embeddings.db",
        archive_root=tmp_path,
        model="voyage-4",
        dimension=1024,
    )
    assert isinstance(provider, SqliteVecProvider)
    provider.model = "voyage-4-lite"
    assert len(provider._get_embeddings([_TEXT], input_type="document")) == 1
    assert len(provider._get_embeddings([_TEXT], input_type="query")) == 1
    assert [(payload["model"], payload["input_type"]) for payload in captured] == [
        ("voyage-4-lite", "document"),
        ("voyage-4-lite", "query"),
    ]
    with pytest.raises(SqliteVecError):
        provider._get_embeddings([_TEXT], input_type="unknown")
    assert len(captured) == 2


def test_adapter_label_drift_publishes_only_refs_and_preserves_purchased_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing purchased bytes through computed replacement relabels metadata and fails this."""
    from polylogue.storage.embeddings import identity
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.embedding_compatibility import output_rows

    root = tmp_path / "archive"
    sid, ids = _session(root)
    provider = _Documents("voyage-4")
    assert embed_archive_session_sync(root / "index.db", provider, sid).status == "embedded"
    original = output_rows(root)
    from tests.infra.embedding_compatibility import clear_embedding_refs

    clear_embedding_refs(root)
    monkeypatch.setattr(identity, "EMBEDDING_INPUT_SCHEMA_VERSION", "archive-index-v79-relabelled")
    provider.calls.clear()
    adapter = EmbeddingDerivationAdapter(root / "index.db", provider)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    for mid in ids:
        with write_lease("test.label-drift-reservation", archive_root=root):
            replacement = adapter.compute(frame, f"message:{mid}")
        assert replacement.vector is None
        assert replacement.retained_output is not None
        with write_lease("test.label-drift-publication", archive_root=root):
            assert adapter.publish(frame, replacement)
        assert output_rows(root) == original
    assert provider.calls == []
    assert len(_rows(root)[1]) == 2


def test_settled_preflight_aggregates_without_materializing_session_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring the preflight session-list selection makes this production read fail."""
    from polylogue.storage.embeddings.preflight import read_embedding_work_counts
    from tests.infra.embedding_compatibility import add_settled_sessions

    root = tmp_path / "archive"
    sid, ids = _session(root)
    provider = _Documents("voyage-4")
    assert embed_archive_session_sync(root / "index.db", provider, sid).status == "embedded"
    add_settled_sessions(root, count=256)

    def materialized_membership(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("preflight must aggregate its canonical SQL window")

    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.select_pending_archive_session_window", materialized_membership
    )
    assert read_embedding_work_counts(root / "index.db") == (257, 0, 0, 0)
    assert read_embedding_work_counts(root / "index.db", rebuild=True) == (257, 0, 0, 0)
    assert read_embedding_work_counts(root / "index.db", max_sessions=3, max_messages=3) == (257, 0, 0, 0)
    assert len(provider.calls) == 1


def test_ref_only_publication_refuses_changed_output_currency(tmp_path: Path) -> None:
    """A missing purchased payload between reservation and publication cannot create a ref."""
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.embedding_compatibility import clear_embedding_refs

    root = tmp_path / "archive"
    sid, ids = _session(root)
    provider = _Documents("voyage-4")
    assert embed_archive_session_sync(root / "index.db", provider, sid).status == "embedded"
    clear_embedding_refs(root)
    provider.calls.clear()
    adapter = EmbeddingDerivationAdapter(root / "index.db", provider)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    with write_lease("test.retained-output-reservation", archive_root=root):
        replacement = adapter.compute(frame, f"message:{ids[0]}")
    assert replacement.retained_output is not None
    with closing(sqlite3.connect(root / "embeddings.db")) as conn:
        assert try_load_sqlite_vec(conn)[0]
        conn.execute("DELETE FROM message_embeddings")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with write_lease("test.retained-output-publication", archive_root=root):
        assert not adapter.publish(frame, replacement)
    assert provider.calls == []
    assert _rows(root)[1] == []


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        {},
        {"demo_only": True},
        {"demo_only": True, "demo_session_ids": [7], "demo_raw_ids": [], "demo_assertion_ids": []},
        {"demo_only": False, "demo_session_ids": ["unrelated"], "demo_raw_ids": [], "demo_assertion_ids": []},
        {
            "demo_only": True,
            "demo_session_ids": ["stale-recorded-session"],
            "demo_raw_ids": [],
            "demo_assertion_ids": [],
        },
    ],
)
def test_incomplete_or_stale_demo_membership_does_not_exclude_real_acquisition(
    tmp_path: Path, manifest: dict[str, object] | None
) -> None:
    """Only exact completed ownership membership can remove a real work key."""
    import json

    from polylogue.storage.archive_identity import DEMO_OWNERSHIP_MANIFEST_FILENAME

    root = tmp_path / "archive"
    sid, _ids = _session(root)
    path = root / DEMO_OWNERSHIP_MANIFEST_FILENAME
    path.write_text("invalid JSON" if manifest is None else json.dumps(manifest))
    with closing(sqlite3.connect(root / "index.db")) as conn:
        selected = select_pending_archive_session_window(conn, status_table="")
    assert [item.session_id for item in selected] == [sid]


def test_unreadable_demo_ownership_keeps_acquisition_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read failure must not grant acquisition by dropping an exclusion."""
    from polylogue.storage.archive_identity import DEMO_OWNERSHIP_MANIFEST_FILENAME

    root = tmp_path / "archive"
    _session(root)
    original = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == DEMO_OWNERSHIP_MANIFEST_FILENAME:
            raise PermissionError("synthetic ownership read denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with closing(sqlite3.connect(root / "index.db")) as conn:
        with pytest.raises(PermissionError):
            select_pending_archive_session_window(conn, status_table="")


@pytest.mark.parametrize("mode", ["computed", "provider_error", "policy_refusal", "source_read_error"])
def test_derivation_releases_actual_read_handles_before_provider_and_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Removing explicit reader closure leaves captured physical handles usable."""
    import json

    from polylogue.storage.archive_identity import DEMO_OWNERSHIP_MANIFEST_FILENAME
    from polylogue.storage.embeddings import derivation
    from polylogue.storage.embeddings.materialization import EmbeddingAcquisitionExcludedError
    from polylogue.storage.sqlite.write_lease import write_lease

    root = tmp_path / "archive"
    sid, ids = _session(root)
    from tests.infra.embedding_reader_probe import EmbeddingReadProbe

    probe = EmbeddingReadProbe(root, monkeypatch)

    class Provider(_Documents):
        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            probe.assert_settled()
            if mode == "provider_error":
                raise RuntimeError("synthetic acquisition failure")
            return super()._get_embeddings(texts, input_type)

    monkeypatch.setattr("polylogue.storage.embeddings.derivation.open_readonly_connection", probe.open)
    provider = Provider("voyage-4")
    adapter = EmbeddingDerivationAdapter(root / "index.db", provider)
    frame = SimpleNamespace(
        source_revision=f"index-generation:{root / 'index.db'}",
        scope=None,
        recipe_version=lambda domain: adapter.recipe_version,
    )
    keys = [f"message:{mid}" for mid in ids]
    adapter.required_page(frame, cursor=None, limit=10)
    probe.assert_settled()
    adapter.excess_page(frame, cursor=None, limit=10)
    probe.assert_settled()
    adapter.inspect(frame, keys)
    probe.assert_settled()
    assert adapter.barrier_sessions(frame, keys) == dict.fromkeys(keys, sid)
    probe.assert_settled()
    current = derivation._current_input(
        root / "index.db",
        ids[0],
        provider_recipe := EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
        None,
        frame.source_revision,
    )
    assert current is not None
    probe.assert_settled()
    with write_lease("test.physical-reader-reservation", archive_root=root):
        reserved = derivation.reserve_embedding_message(
            root / "index.db", root / "embeddings.db", ids[0], provider_recipe, frame.source_revision
        )
    assert reserved is not None
    probe.assert_settled()
    if mode == "policy_refusal":
        (root / DEMO_OWNERSHIP_MANIFEST_FILENAME).write_text(
            json.dumps({"demo_only": True, "demo_session_ids": [sid], "demo_raw_ids": [], "demo_assertion_ids": []})
        )
        expected_error: type[Exception] = EmbeddingAcquisitionExcludedError
    elif mode == "source_read_error":

        def read_failure(*args: Any, **kwargs: Any) -> None:
            raise sqlite3.OperationalError("synthetic source read failure")

        monkeypatch.setattr(derivation, "_message_input", read_failure)
        with pytest.raises(sqlite3.OperationalError):
            adapter.inspect(frame, keys)
        probe.assert_settled()
        expected_error = sqlite3.OperationalError
    else:
        expected_error = RuntimeError
    with write_lease("test.physical-reader-computation", archive_root=root):
        if mode == "computed":
            replacement = adapter.compute(frame, keys[0])
            assert replacement.vector is not None
        else:
            with pytest.raises(expected_error):
                adapter.compute(frame, keys[0])
    probe.assert_settled()
    assert provider.calls == ([(_TEXT,)] if mode == "computed" else [])
