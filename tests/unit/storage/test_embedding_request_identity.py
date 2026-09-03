"""Production contract tests for complete embedding request identity."""

from pytest import MonkeyPatch

from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingRequestSpec


def _recipe(**changes: object) -> EmbeddingRecipe:
    values: dict[str, object] = {
        "model": "voyage-4",
        "dimensions": 1024,
    }
    values.update(changes)
    return EmbeddingRecipe.current(**values)  # type: ignore[arg-type]


def test_request_payload_and_address_share_one_spec() -> None:
    spec = EmbeddingRequestSpec(
        recipe=_recipe(request_options=(("truncation", " ಒ"),)),
        input_text="Cafe\u0301",
    )
    assert spec.provider_request["input"] == ["Café"]
    assert spec.provider_request["truncation"] == " ಒ"
    assert (
        spec.vector_derivation_hash
        == EmbeddingRequestSpec(recipe=spec.recipe, input_text="Café").vector_derivation_hash
    )


def test_only_request_shaping_fields_change_address() -> None:
    """The address follows the provider request; recipe labels only change recipe_hash.

    Anti-vacuity: folding any label field back into the hashed document makes
    the second block fail for that field.
    """
    baseline = EmbeddingRequestSpec(recipe=_recipe(), input_text="same")
    request_changes = (
        {"model": "voyage-4-lite"},
        {"input_type": "query"},
        {"request_options": (("truncation", "none"),)},
        {"dimensions": 512},
    )
    label_changes = (
        {"provider": "other"},
        {"model_revision": "rev-2"},
        {"task": "classification"},
        {"canonicalization": "v2"},
        {"chunking_version": "v2"},
        {"normalization": "unit"},
        {"tool_implementation": "tool-v2"},
        {"element_type": "float16"},
        {"input_schema_version": "schema-v2"},
    )
    for change in request_changes:
        changed = EmbeddingRequestSpec(recipe=_recipe(**change), input_text="same")
        assert changed.vector_derivation_hash != baseline.vector_derivation_hash, change
    for change in label_changes:
        changed = EmbeddingRequestSpec(recipe=_recipe(**change), input_text="same")
        assert changed.vector_derivation_hash == baseline.vector_derivation_hash, change
        assert changed.recipe.recipe_hash != baseline.recipe.recipe_hash, change


def test_default_shaped_request_address_is_pinned() -> None:
    """A default-shaped request hashes as ``domain || len(model)||model || len(text)||text``.

    The literal is the address every vector embedded under the original
    content-addressing formula carries; a stored archive's vectors are reused
    only while this holds. Anti-vacuity: any change to the domain string,
    segment layout, or the set of fields folded into a default-shaped address
    changes the digest.
    """
    spec = EmbeddingRequestSpec(recipe=_recipe(), input_text="same")
    assert spec.vector_derivation_hash.hex() == "d6042147df31c0513ff15869dce130f425a6c6a9139ffdd502f2145247de6027"
    # ``input_type=document`` is the default shape, not an extra segment.
    explicit = EmbeddingRequestSpec(recipe=_recipe(input_type="document"), input_text="same")
    assert explicit.vector_derivation_hash == spec.vector_derivation_hash


def test_complete_requests_deduplicate_without_message_identity() -> None:
    recipe = _recipe()
    first = EmbeddingRequestSpec(recipe=recipe, input_text="same")
    second = EmbeddingRequestSpec(recipe=recipe, input_text="same")
    assert first.vector_derivation_hash == second.vector_derivation_hash
    assert (
        EmbeddingRequestSpec(recipe=_recipe(input_type="query"), input_text="same").vector_derivation_hash
        != first.vector_derivation_hash
    )


def test_default_recipe_is_independent_of_index_schema_version(monkeypatch: MonkeyPatch) -> None:
    """Fresh identity imports retain cache addresses for unchanged provider input."""
    import importlib

    import polylogue.storage.embeddings.identity as identity
    import polylogue.storage.sqlite.archive_tiers.index as index

    before = identity.EmbeddingRecipe.current(model="voyage-4", dimensions=1024)
    original_version = index.INDEX_SCHEMA_VERSION
    monkeypatch.setattr(index, "INDEX_SCHEMA_VERSION", index.INDEX_SCHEMA_VERSION + 1)
    importlib.reload(identity)
    after = identity.EmbeddingRecipe.current(model="voyage-4", dimensions=1024)

    assert before.input_schema_version == "archive-index-v79"
    assert after.input_schema_version == before.input_schema_version
    assert after.recipe_hash == before.recipe_hash
    assert (
        identity.EmbeddingRequestSpec(recipe=after, input_text="same").vector_derivation_hash
        == identity.EmbeddingRequestSpec(recipe=before, input_text="same").vector_derivation_hash
    )

    monkeypatch.undo()
    assert original_version == index.INDEX_SCHEMA_VERSION
    importlib.reload(identity)
