"""The vector provider factory owns the embedding recipe contract.

``resolve_optional_vector_provider`` -- the route behind
``SessionRepository.embed_session`` / ``similarity_search`` -- calls
``create_vector_provider`` with ``config=None``. The factory loaded only the
ambient Voyage key and skipped the recipe block, so a declared model was
silently replaced by the library default and the repository wrote or queried
vectors under a recipe the archive never configured, while the CLI/API route
(which passes a ``Config``) honoured it.

Anti-vacuity, per test:

* ``test_repository_route_uses_declared_model`` is red the moment the ambient
  recipe load is removed: the provider reports ``voyage-4-lite`` again.
* ``test_default_recipe_survives_a_key_only_config`` is the opposite direction:
  a factory that invented a recipe, rather than reading the configured one,
  fails it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION


def _isolated_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, toml_body: str) -> Path:
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    user = tmp_path / "user.toml"
    user.write_text(toml_body, encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))
    archive_root = tmp_path / "archive"
    archive_root.mkdir(exist_ok=True)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    return archive_root


def test_repository_route_uses_declared_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    from polylogue.storage.repository.vectors.repository_vectors import resolve_optional_vector_provider

    archive_root = _isolated_archive(
        monkeypatch,
        tmp_path,
        f'[embedding]\nvoyage_api_key = "toml-fixture-key"\nmodel = "voyage-4"\ndimension = {EMBEDDING_DIMENSION}\n',
    )
    provider = resolve_optional_vector_provider(None, db_path=archive_root / "embeddings.db")
    assert provider is not None
    assert getattr(provider, "model", None) == "voyage-4"
    assert getattr(provider, "dimension", None) == EMBEDDING_DIMENSION


def test_default_recipe_survives_a_key_only_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.storage.repository.vectors.repository_vectors import resolve_optional_vector_provider

    pytest.importorskip("sqlite_vec")
    archive_root = _isolated_archive(
        monkeypatch,
        tmp_path,
        '[embedding]\nvoyage_api_key = "toml-fixture-key"\n',
    )
    provider = resolve_optional_vector_provider(None, db_path=archive_root / "embeddings.db")
    assert provider is not None
    assert getattr(provider, "model", None) == "voyage-4-lite"
    assert getattr(provider, "dimension", None) == EMBEDDING_DIMENSION
