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


def test_every_provider_or_output_field_changes_address() -> None:
    baseline = EmbeddingRequestSpec(recipe=_recipe(), input_text="same")
    changes = (
        {"provider": "other"},
        {"model_revision": "rev-2"},
        {"input_type": "query"},
        {"task": "classification"},
        {"request_options": (("truncation", "none"),)},
        {"canonicalization": "v2"},
        {"chunking_version": "v2"},
        {"normalization": "unit"},
        {"tool_implementation": "tool-v2"},
        {"dimensions": 512},
        {"element_type": "float16"},
        {"input_schema_version": "schema-v2"},
    )
    assert all(
        EmbeddingRequestSpec(recipe=_recipe(**change), input_text="same").vector_derivation_hash
        != baseline.vector_derivation_hash
        for change in changes
    )


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
    """Fresh index rebuilds retain cache addresses for unchanged provider input."""
    import polylogue.storage.sqlite.archive_tiers.index as index

    before = _recipe()
    monkeypatch.setattr(index, "INDEX_SCHEMA_VERSION", index.INDEX_SCHEMA_VERSION + 1)
    after = _recipe()

    assert before.input_schema_version == "archive-index-v79"
    assert after.input_schema_version == before.input_schema_version
    assert after.recipe_hash == before.recipe_hash
    assert (
        EmbeddingRequestSpec(recipe=after, input_text="same").vector_derivation_hash
        == EmbeddingRequestSpec(recipe=before, input_text="same").vector_derivation_hash
    )
