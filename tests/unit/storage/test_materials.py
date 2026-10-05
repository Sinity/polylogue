from __future__ import annotations

import sqlite3
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.storage.blob_liveness import inspect_blob_liveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import (
    link_material,
    list_material_links,
    list_materials,
    prepare_material,
    prepare_material_acquisition,
    prepare_material_file,
    read_material,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.material_preparation import apply_material_preparation, material_publisher


def _source_db(tmp_path: Path) -> sqlite3.Connection:
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass
    return sqlite3.connect(tmp_path / "source.db")


def test_material_retains_bytes_and_links_without_session(tmp_path: Path) -> None:
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/result.zip",
            referrer_ref="message:codex-session:abc:m1",
            payload=b"not a session export",
            media_type="application/zip",
            privacy_classification="private",
        ),
        observed_at_ms=100,
    )
    link_material(
        conn,
        observation.material_id,
        "work-attempt:attempt-1",
        relation="supports",
        authority="provider",
        confidence=0.8,
        observed_at_ms=101,
        source_diagnostic="linked detail " * 2000 + "exact linked terminus",
    )
    assert list_material_links(conn, observation.material_id)[0].source_diagnostic == (
        "linked detail " * 2000 + "exact linked terminus"
    )
    row = conn.execute("SELECT acquisition_state, custody, byte_size, blob_hash FROM material_observations").fetchone()
    assert row[:3] == ("malformed", "retained", len(b"not a session export"))
    assert len(row[3]) == 32
    assert conn.execute("SELECT COUNT(*) FROM material_evidence_links").fetchone()[0] == 1
    assert [link.evidence_ref for link in list_material_links(conn, observation.material_id)] == [
        "work-attempt:attempt-1"
    ]
    assert list_materials(conn, evidence_ref="work-attempt:attempt-1")[0].material_id == observation.material_id


@pytest.mark.parametrize("diagnostic", ["HTTP 410 Gone", "detail " * 3000 + "exact terminal detail"])
def test_failed_claim_is_queryable_and_synthetic_raw_bytes_are_rejected(tmp_path: Path, diagnostic: str) -> None:
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://expired.example/file",
            referrer_ref="agent:worker-1",
            state="expired",
            diagnostic=diagnostic,
            retryable=True,
        ),
        observed_at_ms=200,
    )
    assert observation.blob_hash is None
    assert conn.execute("SELECT acquisition_state, diagnostic, retryable FROM material_observations").fetchone() == (
        "expired",
        diagnostic,
        1,
    )
    with pytest.raises(ValueError, match="synthetic"):
        apply_material_preparation(
            conn,
            prepared=prepare_material(
                blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
                source_uri="synthetic:test",
                referrer_ref="test",
                payload=b"raw",
                privacy_classification="synthetic",
            ),
            observed_at_ms=201,
        )


def test_duplicate_bytes_remain_separate_observations_and_are_liveness_protected(tmp_path: Path) -> None:
    conn = _source_db(tmp_path)
    store = BlobStore(tmp_path / "blobs")
    first = apply_material_preparation(
        conn,
        prepared=prepare_material(
            blob_store=material_publisher(conn, store),
            source_uri="https://one.example/file",
            referrer_ref="message:one",
            payload=b"same bytes",
            media_type="text/plain",
        ),
        observed_at_ms=1,
    )
    second = apply_material_preparation(
        conn,
        prepared=prepare_material(
            blob_store=material_publisher(conn, store),
            source_uri="https://two.example/file",
            referrer_ref="message:two",
            payload=b"same bytes",
            media_type="text/plain",
        ),
        observed_at_ms=2,
    )
    assert first.material_id != second.material_id
    assert second.acquisition_state == "duplicate"
    assert read_material(conn, first.material_id, blob_store=store) == b"same bytes"
    assert (
        inspect_blob_liveness(
            conn,
            blob_hash=first.blob_hash or "",
            index_conn=None,
        ).state.value
        == "live"
    )


def test_text_manifest_does_not_copy_raw_content(tmp_path: Path) -> None:
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="file:///private/notes.txt",
            referrer_ref="agent:worker",
            payload=b"secret text that must remain in the blob",
            media_type="text/plain",
        ),
        observed_at_ms=3,
    )
    assert "text_prefix" not in observation.extraction_manifest
    assert observation.extraction_manifest["text"] == {"available": True, "encoding": "text/plain"}


class _StubResponse:
    """Minimal stand-in for the object ``_open_url`` returns; never touches a socket."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        location: str | None = None,
        content_length: str | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._done = False

        def _header(key: str, default: object = None) -> object:
            if key.lower() == "location":
                return location
            if key.lower() == "content-length":
                return content_length if content_length is not None else default
            return default

        self.headers = SimpleNamespace(
            get_content_type=lambda: "text/markdown",
            get_content_charset=lambda: "utf-8",
            get=_header,
        )

    def __enter__(self) -> _StubResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, size: int) -> bytes:
        if self._done:
            return b""
        self._done = True
        return self._body


def _public_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every test hostname to a documentation-range public address."""
    monkeypatch.setattr(
        "polylogue.storage.materials._resolve_addresses",
        lambda host, port: ["93.184.216.34"],
    )


def _refuse_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any actual connection attempt an explicit test failure."""

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("acquisition connected to a destination the policy must refuse")

    monkeypatch.setattr("polylogue.storage.materials._open_url", _boom)


def test_url_acquisition_keeps_redirect_provenance_and_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Control case: a public destination still acquires, so the policy is not over-broad.

    Anti-vacuity: refusing any destination unconditionally, or dropping the
    per-hop redirect provenance, makes this red.
    """
    hops: list[str] = []

    def fake_open(url: str, address: str, timeout: float) -> _StubResponse:
        hops.append(url)
        assert address == "93.184.216.34"
        if url == "https://example.test/result.md":
            return _StubResponse(status=302, location="https://cdn.example/result.md")
        return _StubResponse(body=b"# retained")

    _public_resolver(monkeypatch)
    monkeypatch.setattr("polylogue.storage.materials._open_url", fake_open)
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/result.md",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=10,
    )
    assert hops == ["https://example.test/result.md", "https://cdn.example/result.md"]
    assert observation.acquisition_state == "acquired"
    assert observation.diagnostic == "redirected to https://cdn.example/result.md"
    assert read_material(conn, observation.material_id, blob_store=BlobStore(tmp_path / "blobs")) == b"# retained"


def test_loopback_material_url_is_permanently_refused_without_connecting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A loopback literal is refused before connecting and retains no bytes.

    Anti-vacuity: removing the pre-connect address check makes the stubbed
    opener raise AssertionError (a connection attempt), turning this red.
    """
    _refuse_connections(monkeypatch)
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="http://127.0.0.1:9/secrets",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=10,
    )
    assert observation.acquisition_state == "access_denied"
    assert observation.retryable is False
    assert observation.blob_hash is None
    assert "127.0.0.1" in observation.diagnostic
    with pytest.raises(FileNotFoundError):
        read_material(conn, observation.material_id, blob_store=BlobStore(tmp_path / "blobs"))


def test_link_local_metadata_url_is_permanently_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The cloud metadata link-local address is refused permanently.

    Anti-vacuity: removing the link-local branch of the address check makes the
    stubbed opener raise on a connection attempt, turning this red.
    """
    _refuse_connections(monkeypatch)
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="http://169.254.169.254/latest/meta-data/",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=11,
    )
    assert observation.acquisition_state == "access_denied"
    assert observation.retryable is False
    assert observation.blob_hash is None
    assert "169.254.169.254" in observation.diagnostic


def test_ipv6_loopback_and_mapped_ipv4_hosts_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IPv6 loopback and IPv4-mapped IPv6 resolutions are refused like their v4 forms.

    Anti-vacuity: dropping the ``ipv4_mapped`` unwrapping, or the IPv6 cases,
    lets one of these connect and raise AssertionError.
    """
    _refuse_connections(monkeypatch)
    conn = _source_db(tmp_path)
    monkeypatch.setattr(
        "polylogue.storage.materials._resolve_addresses",
        lambda host, port: ["::ffff:10.0.0.5"] if host == "mapped.example" else [host],
    )
    mapped = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://mapped.example/resource",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=12,
    )
    literal = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="http://[::1]:9/resource",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=13,
    )
    assert mapped.acquisition_state == "access_denied"
    assert mapped.retryable is False
    assert literal.acquisition_state == "access_denied"
    assert literal.retryable is False


def test_redirect_to_private_address_is_refused_at_the_hop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A public host redirecting to a private address is refused at the hop, not followed.

    Anti-vacuity: restoring automatic redirect following, or skipping the
    re-check on hops after the first, lets the second hop open and makes the
    recorded state ``acquired`` instead of ``access_denied``.
    """
    opened: list[str] = []

    def fake_open(url: str, address: str, timeout: float) -> _StubResponse:
        opened.append(url)
        if url == "https://example.test/start":
            return _StubResponse(status=302, location="http://127.0.0.1:9/secrets")
        return _StubResponse(body=b"leaked")

    monkeypatch.setattr(
        "polylogue.storage.materials._resolve_addresses",
        lambda host, port: ["93.184.216.34"] if host == "example.test" else [host],
    )
    monkeypatch.setattr("polylogue.storage.materials._open_url", fake_open)
    conn = _source_db(tmp_path)
    observation = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/start",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=14,
    )
    assert opened == ["https://example.test/start"]
    assert observation.acquisition_state == "access_denied"
    assert observation.retryable is False
    assert observation.blob_hash is None


def test_url_and_file_failures_are_queryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Transport failures stay retryable observations distinct from policy refusals.

    Anti-vacuity: routing HTTP status failures through the permanent refusal
    path would change ``expired``/``unavailable`` and turn this red.
    """
    conn = _source_db(tmp_path)
    _public_resolver(monkeypatch)
    monkeypatch.setattr(
        "polylogue.storage.materials._open_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.HTTPError("url", 410, "Gone", Message(), None)),
    )
    expired = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/old",
            referrer_ref="agent:a",
        ),
        observed_at_ms=1,
    )
    missing = apply_material_preparation(
        conn,
        prepared=prepare_material_file(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            path=tmp_path / "missing.patch",
            referrer_ref="agent:a",
        ),
        observed_at_ms=2,
    )
    assert expired.acquisition_state == "expired"
    assert missing.acquisition_state == "unavailable"
    material_ids = [row[0] for row in conn.execute("SELECT material_id FROM material_observations")]
    assert len(material_ids) == 2
    assert all(material_ids)


def test_material_boundaries_validate_link_targets(tmp_path: Path) -> None:
    conn = _source_db(tmp_path)
    with pytest.raises(KeyError, match="missing"):
        link_material(conn, "missing", "message:1", relation="refers_to", observed_at_ms=1)


def test_short_body_against_a_declared_length_is_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A server that advertises a length and closes early yields partial evidence.

    ``HTTPResponse.read`` returns the prefix and then ``b""`` with no exception,
    so the acquisition loop saw an ordinary EOF and persisted the truncated
    bytes as ``acquired`` -- authoritative retained evidence for bytes that were
    never sent.

    Anti-vacuity: reverting the declared-length branch in ``prepare_material_acquisition``
    makes the first assertion read ``acquired``. The undeclared-length case
    below pins the other direction, so classifying every response as partial
    cannot pass.
    """
    _public_resolver(monkeypatch)
    monkeypatch.setattr(
        "polylogue.storage.materials._open_url",
        lambda url, address, timeout: _StubResponse(body=b"# short", content_length="1000"),
    )
    conn = _source_db(tmp_path)
    truncated = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/truncated.md",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=10,
    )
    assert truncated.acquisition_state == "partial"
    assert truncated.retryable is True
    assert "declared 1000 bytes" in truncated.diagnostic

    monkeypatch.setattr(
        "polylogue.storage.materials._open_url",
        lambda url, address, timeout: _StubResponse(body=b"# whole", content_length="7"),
    )
    complete = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/whole.md",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=11,
    )
    assert complete.acquisition_state == "acquired"

    monkeypatch.setattr(
        "polylogue.storage.materials._open_url",
        lambda url, address, timeout: _StubResponse(body=b"# undeclared"),
    )
    undeclared = apply_material_preparation(
        conn,
        prepared=prepare_material_acquisition(
            blob_store=material_publisher(conn, BlobStore(tmp_path / "blobs")),
            source_uri="https://example.test/undeclared.md",
            referrer_ref="message:codex:1",
        ),
        observed_at_ms=12,
    )
    assert undeclared.acquisition_state == "acquired"


def test_admission_keeps_bytes_in_own_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A byte-bearing admission must write into *this* connection's archive.

    An unspecified publisher previously fell through to the process-wide active store,
    so a scratch, probe or test source tier committed a durable
    ``material_observations`` row in one archive while its bytes landed in
    another: archive-local reads, backup, integrity checks and GC could not
    treat the pair consistently, and private bytes leaked into the operator's
    live archive.

    Anti-vacuity: resolve the publisher from the active archive and the bytes
    appear under ``active/blob`` instead of ``scratch/blob``, and the
    ``read_material`` assertion below fails because the scratch archive has no
    such blob. The opposite direction -- ignoring an explicitly supplied store
    -- is pinned by every other test in this file, which passes one.
    """
    scratch = tmp_path / "scratch"
    active = tmp_path / "active"
    scratch.mkdir()
    (active / "blob").mkdir(parents=True)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(active))

    with ArchiveStore(scratch, initialize=True, read_only=False):
        pass
    conn = sqlite3.connect(scratch / "source.db")
    try:
        observation = apply_material_preparation(
            conn,
            prepared=prepare_material(
                blob_store=material_publisher(conn, BlobStore(scratch / "blob")),
                source_uri="https://example.test/private.bin",
                referrer_ref="message:codex-session:abc:m1",
                payload=b"private scratch bytes",
            ),
            observed_at_ms=100,
        )
        assert observation.blob_hash is not None
        assert read_material(conn, observation.material_id) == b"private scratch bytes"
    finally:
        conn.close()

    assert list((scratch / "blob").rglob("*")), "scratch archive retained no bytes"
    assert not [path for path in (active / "blob").rglob("*") if path.is_file()], "bytes leaked into the active archive"


def test_acquisition_above_former_size_cap_retains_complete_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    total = 65 * 1024 * 1024 + 7

    class GeneratedResponse(_StubResponse):
        def __init__(self) -> None:
            super().__init__(body=b"", content_length=str(total))
            self.remaining = total

        def read(self, size: int) -> bytes:
            assert 0 < size <= 1024 * 1024
            count = min(size, self.remaining)
            self.remaining -= count
            return b"x" * count

    response = GeneratedResponse()
    _public_resolver(monkeypatch)
    monkeypatch.setattr("polylogue.storage.materials._open_url", lambda *args, **kwargs: response)
    conn = _source_db(tmp_path)
    publisher = material_publisher(conn, BlobStore(tmp_path / "blobs"))
    prepared = prepare_material_acquisition(
        blob_store=publisher, source_uri="https://example.test/large", referrer_ref="message:synthetic-stream"
    )
    assert prepared.blob is not None
    assert prepared.blob.size_bytes == total
    observation = apply_material_preparation(conn, prepared=prepared, observed_at_ms=1)
    assert observation.acquisition_state == "acquired"
    assert observation.blob_hash == prepared.blob.hash_hex
    assert response.remaining == 0
    assert publisher.blob_path(prepared.blob.hash_hex).stat().st_size == total
