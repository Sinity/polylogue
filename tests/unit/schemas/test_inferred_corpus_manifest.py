from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.maintenance.schema_inference_gate import (
    run_schema_inference_gate,
    schema_inference_gate_receipt_digest,
)
from polylogue.schemas.operator.receipt import (
    SchemaInferenceUnsupportedDecision,
    build_schema_inference_receipt,
)
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.wire_formats import (
    PROVIDER_WIRE_FORMATS,
    WireSupportEntry,
    WireSupportReceipt,
    build_wire_support_receipt,
)
from polylogue.sources.parsers.base_models import ParsedSession
from tests.infra import inferred_corpus as inferred_corpus_module
from tests.infra.inferred_corpus import (
    CorpusManifestKey,
    InferredCorpusManifest,
    assert_inferred_corpus_convergence_handoff_complete,
    assert_inferred_corpus_manifest_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
    read_inferred_corpus_manifest,
    write_inferred_corpus_manifest,
)
from tests.unit.maintenance.test_schema_inference_gate import _seed_archive


def _registry() -> SchemaRegistry:
    return SchemaRegistry(storage_root=SCHEMA_DIR)


def _authoritative_gate(tmp_path: Path) -> tuple[Path, Path, str]:
    archive_root = tmp_path / "archive"
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    _seed_archive(archive_root)
    result = run_schema_inference_gate(
        archive_root,
        receipt_path=receipt_path,
        ground_truth_roots={"codex-session": (tmp_path / "archive-codex-ground-truth",)},
    )
    assert result.passed
    return archive_root, receipt_path, schema_inference_gate_receipt_digest(result.payload)


def _catalog_keys(registry: SchemaRegistry) -> set[CorpusManifestKey]:
    keys: set[CorpusManifestKey] = set()
    for provider in registry.list_providers():
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        for package in catalog.packages:
            for element in package.elements:
                keys.add(CorpusManifestKey(provider, package.version, element.element_kind))
    return keys


class _RegistryProxy:
    def __init__(self, base: SchemaRegistry) -> None:
        self.base = base
        self.catalog_overrides: dict[str, object] = {}
        self.schema_overrides: dict[tuple[str, str, str], object] = {}
        self.provider_order: list[str] | None = None

    def list_providers(self) -> list[str]:
        return self.provider_order if self.provider_order is not None else self.base.list_providers()

    def load_package_catalog(self, provider: str) -> object:
        return self.catalog_overrides.get(provider, self.base.load_package_catalog(provider))

    def get_element_schema(self, provider: str, *, version: str = "default", element_kind: str | None = None) -> object:
        assert element_kind is not None
        package = self.base.get_package(provider, version=version)
        assert package is not None
        key = (provider, package.version, element_kind)
        return self.schema_overrides.get(
            key,
            self.base.get_element_schema(provider, version=package.version, element_kind=element_kind),
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)


def test_manifest_covers_every_persisted_package_version_element() -> None:
    registry = _registry()

    manifest = compile_inferred_corpus_manifest(registry=registry)

    assert {
        CorpusManifestKey(entry.key.provider, entry.key.package_version, entry.key.element_kind)
        for entry in manifest.entries
    } == _catalog_keys(registry)
    assert len(manifest.entries) > len(registry.list_providers())
    assert manifest.receipt_state == "catalog_only"
    assert manifest.manifest_id.startswith("manifest:sha256:")
    assert len(manifest.payload_sha256) == 64


def test_persisted_manifest_round_trip_validates_identity_and_integrity(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"

    write_inferred_corpus_manifest(manifest, path)

    assert read_inferred_corpus_manifest(path) == manifest


def test_manifest_can_bind_every_selection_to_the_exact_wire_support_receipt() -> None:
    registry = _registry()
    support = build_wire_support_receipt(registry=registry)

    manifest = compile_inferred_corpus_manifest(registry=registry, wire_support_receipt=support)

    assert manifest.entries
    assert all(entry.unsupported is None for entry in manifest.entries if entry.spec is not None)
    assert {(entry.key.provider, entry.key.package_version, entry.key.element_kind) for entry in manifest.entries} == {
        (entry.provider, entry.package_version, entry.element_kind) for entry in support.entries
    }


def test_wire_support_receipt_is_canonical_across_catalog_reordering() -> None:
    registry = _registry()
    reordered = _RegistryProxy(registry)
    reordered.provider_order = list(reversed(registry.list_providers()))
    for provider in registry.list_providers():
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        reordered.catalog_overrides[provider] = replace(
            catalog,
            packages=[
                replace(package, elements=list(reversed(package.elements))) for package in reversed(catalog.packages)
            ],
        )

    assert (
        build_wire_support_receipt(registry=registry).to_dict()
        == build_wire_support_receipt(registry=reordered).to_dict()
    )


def test_wire_support_receipt_rejects_conflicting_duplicate_identity_at_all_boundaries() -> None:
    registry = _registry()
    receipt = build_wire_support_receipt(registry=registry, providers=("codex",))
    original = receipt.entries[0]
    conflicting = replace(original, reason="conflicting duplicate")

    with pytest.raises(ValueError, match="duplicate wire support entry key"):
        WireSupportReceipt(
            catalog_providers=receipt.catalog_providers,
            entries=(original, conflicting),
            missing_routes=receipt.missing_routes,
            witness_seed=receipt.witness_seed,
        )

    malformed = object.__new__(WireSupportReceipt)
    object.__setattr__(malformed, "catalog_providers", receipt.catalog_providers)
    object.__setattr__(malformed, "entries", (original, conflicting))
    object.__setattr__(malformed, "missing_routes", receipt.missing_routes)
    object.__setattr__(malformed, "witness_seed", receipt.witness_seed)
    with pytest.raises(ValueError, match="duplicate wire support entry key"):
        compile_inferred_corpus_manifest(registry=registry, wire_support_receipt=malformed)

    manifest = compile_inferred_corpus_manifest(registry=registry, wire_support_receipt=receipt)
    assert manifest.wire_support_receipt is not None
    persisted_receipt = cast(dict[str, object], dict(manifest.wire_support_receipt))
    persisted_entries = list(cast(list[dict[str, object]], persisted_receipt["entries"]))
    persisted_entries.append(dict(persisted_entries[0], reason="conflicting duplicate"))
    persisted_receipt["entries"] = persisted_entries
    persisted_manifest = replace(manifest, wire_support_receipt=cast(JSONDocument, persisted_receipt))

    with pytest.raises(ValueError, match="duplicate wire support entry key"):
        inferred_corpus_module._wire_support_entry_index(persisted_manifest)
    with pytest.raises(ValueError, match="duplicate wire support entry key"):
        InferredCorpusManifest.from_payload(persisted_manifest.to_payload())


def test_all_provider_campaign_round_trip_preserves_unsupported_wire_authority(tmp_path: Path) -> None:
    registry = _registry()
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    package_receipts = [
        build_schema_inference_receipt(registry, provider=provider, gate_receipt_digest=gate_digest)
        for provider in registry.list_providers()
    ]
    package_receipt = package_receipts[0]
    for other in package_receipts[1:]:
        package_receipt = package_receipt.merged_with(other)
    wire_support = build_wire_support_receipt(registry=registry)

    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        package_receipt=package_receipt.to_payload(),
        wire_support_receipt=wire_support,
        campaign_mode=True,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )

    antigravity_entries = [entry for entry in manifest.entries if entry.key.provider == "antigravity"]
    assert antigravity_entries
    assert all(entry.unsupported is not None for entry in antigravity_entries)
    assert all(
        entry.unsupported.reason == "unsupported_wire_route" for entry in antigravity_entries if entry.unsupported
    )

    path = tmp_path / "all-provider-campaign.json"
    write_inferred_corpus_manifest(manifest, path)
    restored = read_inferred_corpus_manifest(
        path,
        campaign_mode=True,
        registry=registry,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )
    assert restored == manifest
    assert restored.wire_support_receipt == wire_support.to_dict()


def test_campaign_indexes_persisted_wire_support_entries_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    package_receipt = build_schema_inference_receipt(
        registry,
        provider="codex",
        gate_receipt_digest=gate_digest,
    )
    wire_support = build_wire_support_receipt(registry=registry, providers=("codex",))
    index_calls = 0
    payload_calls = 0
    original_index = inferred_corpus_module._wire_support_entry_index
    original_payload = inferred_corpus_module._wire_support_entry_from_payload

    def count_index(
        manifest: InferredCorpusManifest,
    ) -> dict[tuple[str, str | None, str | None], WireSupportEntry]:
        nonlocal index_calls
        index_calls += 1
        return original_index(manifest)

    def count_payload(payload: object) -> object:
        nonlocal payload_calls
        payload_calls += 1
        return original_payload(cast(dict[str, object], payload))

    monkeypatch.setattr(inferred_corpus_module, "_wire_support_entry_index", count_index)
    monkeypatch.setattr(inferred_corpus_module, "_wire_support_entry_from_payload", count_payload)

    compile_inferred_corpus_manifest(
        registry=registry,
        providers=("codex",),
        package_receipt=package_receipt.to_payload(),
        wire_support_receipt=wire_support,
        campaign_mode=True,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )

    assert index_calls == 1
    assert payload_calls == len(wire_support.entries)


def test_campaign_read_rejects_wire_route_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry()
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    package_receipt = build_schema_inference_receipt(
        registry,
        provider="codex",
        gate_receipt_digest=gate_digest,
    )
    wire_support = build_wire_support_receipt(registry=registry, providers=("codex",))
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        package_receipt=package_receipt.to_payload(),
        wire_support_receipt=wire_support,
        providers=("codex",),
        campaign_mode=True,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )
    path = tmp_path / "campaign.json"
    write_inferred_corpus_manifest(manifest, path)

    from polylogue.sources import dispatch as dispatch_module

    original_parse_payload = dispatch_module.parse_payload

    def drifted_parse_payload(*args: object, **kwargs: object) -> list[ParsedSession]:
        if args and args[0] == "codex":
            raise ValueError("simulated parser drift")
        return original_parse_payload(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dispatch_module, "parse_payload", drifted_parse_payload)
    with pytest.raises(ValueError, match=r"wire-support receipt changed.*changed_fields=.*entries"):
        read_inferred_corpus_manifest(
            path,
            campaign_mode=True,
            registry=registry,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )


def test_manifest_refuses_a_selection_missing_from_bound_wire_support_receipt() -> None:
    registry = _registry()
    support = build_wire_support_receipt(registry=registry)
    missing = next(entry for entry in support.entries if entry.status == "supported")
    reduced_support = replace(
        support,
        entries=tuple(
            entry
            for entry in support.entries
            if (entry.provider, entry.package_version, entry.element_kind)
            != (missing.provider, missing.package_version, missing.element_kind)
        ),
    )

    manifest = compile_inferred_corpus_manifest(registry=registry, wire_support_receipt=reduced_support)

    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (missing.provider, missing.package_version, missing.element_kind)
    )
    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "wire_support_selection_unwitnessed"


def test_campaign_read_revalidates_live_schema_and_classifier(tmp_path: Path) -> None:
    registry = _registry()
    provider = "codex"
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    receipt = build_schema_inference_receipt(registry, provider=provider, gate_receipt_digest=gate_digest)
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        providers=(provider,),
        package_receipt=receipt.to_payload(),
        campaign_mode=True,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )
    path = tmp_path / "campaign.json"

    supported = next(entry for entry in manifest.entries if entry.spec is not None)
    tampered_schema = replace(supported, generator_schema={"type": "string"})
    tampered = replace(
        manifest,
        entries=tuple(
            sorted(
                (tampered_schema if entry is supported else entry for entry in manifest.entries),
                key=lambda entry: entry.key,
            )
        ),
    )
    write_inferred_corpus_manifest(tampered, path)
    with pytest.raises(ValueError, match="package/version/element hashes|generator schema changed"):
        read_inferred_corpus_manifest(
            path,
            campaign_mode=True,
            registry=registry,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )

    tampered_key = replace(supported.key, construct_support=())
    tampered_entry = replace(supported, key=tampered_key)
    tampered = replace(
        manifest,
        entries=tuple(
            sorted(
                (tampered_entry if entry is supported else entry for entry in manifest.entries),
                key=lambda entry: entry.key,
            )
        ),
    )
    with pytest.raises(ValueError, match="classifier output changed"):
        build_inferred_corpus_convergence_handoff(
            tampered,
            campaign_mode=True,
            registry=registry,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )


def test_campaign_mode_rejects_catalog_only_manifest(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "catalog-only.json"
    write_inferred_corpus_manifest(manifest, path)

    with pytest.raises(ValueError, match="catalog-only"):
        read_inferred_corpus_manifest(path, campaign_mode=True)
    with pytest.raises(ValueError, match="handoff"):
        compile_inferred_corpus_manifest(registry=_registry(), campaign_mode=True)


def test_campaign_receipt_rejects_tampered_gate_package_and_unsupported_decisions(tmp_path: Path) -> None:
    registry = _registry()
    provider = "codex"
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    receipt = build_schema_inference_receipt(
        registry,
        provider=provider,
        gate_receipt_digest=gate_digest,
    )
    compile_inferred_corpus_manifest(
        registry=registry,
        providers=(provider,),
        package_receipt=receipt.to_payload(),
        campaign_mode=True,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )

    tampered_gate = replace(receipt, gate_receipt_digest="b" * 64)
    with pytest.raises(ValueError, match="different gate receipt digests"):
        tampered_gate.merged_with(receipt)

    tampered_package = replace(
        receipt,
        packages=(replace(receipt.packages[0], package_hash="b" * 64), *receipt.packages[1:]),
    )
    with pytest.raises(ValueError, match="package/version/element hashes"):
        compile_inferred_corpus_manifest(
            registry=registry,
            providers=(provider,),
            package_receipt=tampered_package.to_payload(),
            campaign_mode=True,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )


def test_campaign_rejects_fabricated_gate_digest_even_when_shape_is_valid(tmp_path: Path) -> None:
    registry = _registry()
    archive_root, gate_receipt_path, _gate_digest = _authoritative_gate(tmp_path)
    receipt = build_schema_inference_receipt(registry, provider="codex", gate_receipt_digest="a" * 64)

    with pytest.raises(ValueError, match="does not match the authoritative PASS receipt"):
        compile_inferred_corpus_manifest(
            registry=registry,
            providers=("codex",),
            package_receipt=receipt.to_payload(),
            campaign_mode=True,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )


def test_campaign_rejects_tampered_ground_truth_denominators_after_digest_recompute(tmp_path: Path) -> None:
    registry = _registry()
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    receipt = build_schema_inference_receipt(registry, provider="codex", gate_receipt_digest=gate_digest)

    valid_manifest = compile_inferred_corpus_manifest(
        registry=registry,
        providers=("codex",),
        package_receipt=receipt.to_payload(),
        campaign_mode=True,
        gate_receipt_path=gate_receipt_path,
        archive_root=archive_root,
    )
    assert valid_manifest.receipt_state == "package_receipt_attached"

    tampered_gate = json.loads(gate_receipt_path.read_text(encoding="utf-8"))
    denominators = dict(tampered_gate["ground_truth_denominators"])
    denominators["documents_known"] += 1
    tampered_gate["ground_truth_denominators"] = denominators
    gate_receipt_path.write_text(json.dumps(tampered_gate, sort_keys=True) + "\n", encoding="utf-8")
    tampered_digest = schema_inference_gate_receipt_digest(tampered_gate)
    tampered_receipt = build_schema_inference_receipt(
        registry,
        provider="codex",
        gate_receipt_digest=tampered_digest,
    )

    with pytest.raises(ValueError, match="ground-truth denominators changed"):
        compile_inferred_corpus_manifest(
            registry=registry,
            providers=("codex",),
            package_receipt=tampered_receipt.to_payload(),
            campaign_mode=True,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )


def test_bundled_registry_relation_annotations_share_one_receipt_classification(tmp_path: Path) -> None:
    registry = _registry()
    provider = "chatgpt"
    archive_root, gate_receipt_path, gate_digest = _authoritative_gate(tmp_path)
    receipt = build_schema_inference_receipt(registry, provider=provider, gate_receipt_digest=gate_digest)
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        providers=(provider,),
        package_receipt=receipt.to_payload(),
        campaign_mode=False,
    )

    expected_annotations = {
        "x-polylogue-foreign-keys",
        "x-polylogue-mutually-exclusive",
        "x-polylogue-string-lengths",
        "x-polylogue-time-deltas",
    }
    observed = {
        item.construct
        for entry in manifest.entries
        for item in entry.key.construct_support
        if item.construct in expected_annotations
    }
    assert observed == expected_annotations
    receipt_decisions = {
        (item.provider, item.package_version, item.element_kind, item.decision, item.reason, item.details)
        for item in receipt.unsupported_decisions
        if item.provider == provider
    }
    manifest_decisions: set[tuple[str, str, str, str, str, tuple[str, ...]]] = set()
    for entry in manifest.entries:
        if entry.unsupported is None:
            continue
        unsupported = entry.unsupported
        manifest_decisions.add(
            (
                entry.key.provider,
                entry.key.package_version,
                entry.key.element_kind,
                "nonrepresentable" if unsupported.reason == "unsupported_json_schema_construct" else "unsupported",
                unsupported.reason,
                unsupported.details,
            )
        )
    assert receipt_decisions == manifest_decisions
    assert all(
        annotation in details
        for *_identity, details in receipt_decisions
        for annotation in expected_annotations
        if annotation in observed
    )

    package = receipt.packages[0]
    if receipt.unsupported_decisions:
        first = receipt.unsupported_decisions[0]
        changed = replace(first, decision="unsupported" if first.decision == "nonrepresentable" else "nonrepresentable")
        tampered_decisions = (changed, *receipt.unsupported_decisions[1:])
    else:
        changed = SchemaInferenceUnsupportedDecision(
            provider=provider,
            package_version=package.package_version,
            element_kind="tampered-element",
            decision="nonrepresentable",
            reason="tampered decision",
            details=("tampered_construct",),
        )
        tampered_decisions = (changed,)
    tampered_unsupported = replace(receipt, unsupported_decisions=tuple(sorted(tampered_decisions)))
    with pytest.raises(ValueError, match="no executable synthetic corpus selection"):
        compile_inferred_corpus_manifest(
            registry=registry,
            providers=(provider,),
            package_receipt=tampered_unsupported.to_payload(),
            campaign_mode=True,
            gate_receipt_path=gate_receipt_path,
            archive_root=archive_root,
        )


@pytest.mark.parametrize("field", ["manifest_id", "payload_sha256"])
def test_persisted_manifest_rejects_tampered_hash_fields(tmp_path: Path, field: str) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="(identity|integrity) mismatch"):
        read_inferred_corpus_manifest(path)


def test_persisted_manifest_rejects_unknown_schema_version(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_inferred_corpus_manifest(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"manifest_id": 1, "manifest_id": 2}', "duplicate"),
        ('{"manifest_id": NaN}', "non-finite"),
    ],
)
def test_persisted_manifest_rejects_noncanonical_json(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "noncanonical.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_inferred_corpus_manifest(path)


def test_persisted_manifest_rejects_unhashed_extra_fields(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed"):
        read_inferred_corpus_manifest(path)


def test_persisted_manifest_rejects_spec_identity_tampering(tmp_path: Path) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "inferred-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    supported = next(entry for entry in payload["entries"] if entry["supported"] is True)
    supported["spec"]["provider"] = "wrong-provider"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        read_inferred_corpus_manifest(path)


def test_persisted_selection_preserves_workload_profile(tmp_path: Path) -> None:
    base = compile_inferred_corpus_manifest(registry=_registry())
    supported = next(entry for entry in base.entries if entry.spec is not None)
    profile: SchemaRecord = {"elements": {supported.key.element_kind: {"structural_variants": []}}}
    profiled = InferredCorpusManifest(
        entries=tuple(
            replace(entry, workload_profile=profile) if entry is supported else entry for entry in base.entries
        )
    )
    path = tmp_path / "profiled-manifest.json"
    write_inferred_corpus_manifest(profiled, path)

    persisted = read_inferred_corpus_manifest(path)
    handoff = build_inferred_corpus_convergence_handoff(path)

    persisted_supported = next(entry for entry in persisted.entries if entry.spec is not None)
    assert persisted_supported.workload_profile == profile
    assert handoff.selections[0].workload_profile == profile


@pytest.mark.parametrize("extra_field", ["workload_profile", "spec"])
def test_persisted_manifest_rejects_noncanonical_optional_entry_fields(tmp_path: Path, extra_field: str) -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    path = tmp_path / "noncanonical-manifest.json"
    write_inferred_corpus_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if extra_field == "workload_profile":
        supported = next(entry for entry in payload["entries"] if entry["supported"] is True)
        supported[extra_field] = None
    else:
        unsupported = next(entry for entry in payload["entries"] if entry["supported"] is False)
        unsupported[extra_field] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed|workload_profile"):
        read_inferred_corpus_manifest(path)


def test_manifest_is_independent_of_default_version_selection() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    for provider in registry.list_providers():
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        if len(catalog.packages) > 1:
            proxy.catalog_overrides[provider] = replace(
                catalog,
                default_version=catalog.packages[0].version,
            )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]

    assert {
        CorpusManifestKey(entry.key.provider, entry.key.package_version, entry.key.element_kind)
        for entry in manifest.entries
    } == _catalog_keys(registry)
    assert {entry.key.package_version for entry in manifest.entries} >= {"v1", "v2"}


def test_completeness_guard_detects_a_removed_enumerated_entry() -> None:
    registry = _registry()
    manifest = compile_inferred_corpus_manifest(registry=registry)
    reduced = InferredCorpusManifest(entries=manifest.entries[:-1])

    with pytest.raises(AssertionError, match="coverage mismatch"):
        assert_inferred_corpus_manifest_complete(reduced, registry)


def test_missing_element_schema_becomes_explicit_unsupported_record() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    target_provider = registry.list_providers()[0]
    catalog = registry.load_package_catalog(target_provider)
    assert catalog is not None
    target_package = catalog.packages[0]
    target_element = target_package.elements[0]
    proxy.schema_overrides[(target_provider, target_package.version, target_element.element_kind)] = None

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
        )
        == (target_provider, target_package.version, target_element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "missing_schema"


def test_missing_element_schema_precedes_missing_wire_format() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    target_provider = registry.list_providers()[0]
    catalog = registry.load_package_catalog(target_provider)
    assert catalog is not None
    target_package = catalog.packages[0]
    target_element = target_package.elements[0]
    proxy.schema_overrides[(target_provider, target_package.version, target_element.element_kind)] = None

    manifest = compile_inferred_corpus_manifest(registry=proxy, wire_formats={})  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (target_provider, target_package.version, target_element.element_kind)
    )

    assert target.unsupported is not None
    assert target.unsupported.reason == "missing_schema"


def test_catalog_element_marked_unsupported_is_retained_as_a_typed_record() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider = registry.list_providers()[0]
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.packages[0]
    element = package.elements[0]
    unsupported_element = replace(element, supported=False)
    proxy.catalog_overrides[provider] = replace(
        catalog,
        packages=[
            replace(
                package,
                elements=[unsupported_element, *package.elements[1:]],
            ),
            *catalog.packages[1:],
        ],
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
        )
        == (provider, package.version, element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_element"


def test_removed_wire_format_is_explicit_and_does_not_drop_provider_entries() -> None:
    registry = _registry()
    provider = next(name for name in registry.list_providers() if name in PROVIDER_WIRE_FORMATS)
    formats = dict(PROVIDER_WIRE_FORMATS)
    formats.pop(provider)

    manifest = compile_inferred_corpus_manifest(registry=registry, wire_formats=formats)

    provider_entries = [entry for entry in manifest.entries if entry.key.provider == provider]
    assert provider_entries
    assert all(entry.spec is None for entry in provider_entries)
    assert {entry.unsupported.reason for entry in provider_entries if entry.unsupported is not None} == {
        "provider_without_wire_format"
    }


def test_unsupported_json_schema_construct_is_keyed_and_receiptable() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider = next(name for name in registry.list_providers() if name in PROVIDER_WIRE_FORMATS)
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.packages[0]
    element = package.elements[0]
    schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
    assert isinstance(schema, dict)
    mutated = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties = dict(raw_properties)
    properties["manifest_unsupported"] = {"enum": ["future"]}
    mutated["properties"] = properties
    proxy.schema_overrides[(provider, package.version, element.element_kind)] = mutated
    receipt = {"receipt_id": "tnqqt-package-receipt-placeholder", "status": "pending"}

    manifest = compile_inferred_corpus_manifest(registry=proxy, package_receipt=receipt)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (
            entry.key.provider,
            entry.key.package_version,
            entry.key.element_kind,
        )
        == (provider, package.version, element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_json_schema_construct"
    assert {"enum", "required"} <= set(target.unsupported.details)
    assert ("type", "supported") in {(item.construct, item.state) for item in target.key.construct_support}
    assert manifest.package_receipt == receipt
    assert manifest.receipt_state == "package_receipt_attached"
    assert manifest.to_payload()["package_receipt"] == receipt


def test_every_actual_schema_keyword_is_keyed_and_unhandled_constraints_fail_closed() -> None:
    registry = _registry()
    manifest = compile_inferred_corpus_manifest(registry=registry)

    observed_constructs: set[str] = set()
    for entry in manifest.entries:
        observed_constructs.update(item.construct for item in entry.key.construct_support)

    assert {"$id", "$schema", "maxLength", "minLength", "required"} <= observed_constructs
    browser_entry = next(entry for entry in manifest.entries if entry.key.provider == "browser-capture")
    construct_states = {item.construct: item.state for item in browser_entry.key.construct_support}
    assert construct_states["minLength"] == "unsupported"
    assert construct_states["maxLength"] == "unsupported"
    assert construct_states["required"] == "unsupported"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("pattern", "^[A-Z]+$"),
        ("format", "uuid"),
        ("minimum", 1),
        ("maxItems", 1),
    ],
)
def test_unhandled_standard_constraints_are_typed_unsupported_records(keyword: str, value: object) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider = next(name for name in registry.list_providers() if name in PROVIDER_WIRE_FORMATS)
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.packages[0]
    element = package.elements[0]
    schema = registry.get_element_schema(provider, version=package.version, element_kind=element.element_kind)
    assert isinstance(schema, dict)
    mutated = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["unhandled_constraint"] = {"type": "string", keyword: cast(JSONValue, value)}
    mutated["properties"] = properties
    proxy.schema_overrides[(provider, package.version, element.element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package.version, element.element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_json_schema_construct"
    assert keyword in target.unsupported.details
    assert (keyword, "unsupported") in {(item.construct, item.state) for item in target.key.construct_support}


@pytest.mark.parametrize(
    ("annotation", "schema_type", "value"),
    [
        ("x-polylogue-range", "string", [1, 2]),
        ("x-polylogue-array-lengths", "string", [1, 2]),
        ("x-polylogue-multiline", "integer", True),
        (
            "x-polylogue-foreign-keys",
            "string",
            [{"source": "$.id", "target": "$.parent"}],
        ),
    ],
)
def test_annotation_wrong_node_shape_or_path_fails_closed(
    annotation: str,
    schema_type: str,
    value: JSONValue,
) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["annotation_probe"] = {"type": schema_type, annotation: value}
    mutated["properties"] = properties
    proxy.schema_overrides[(provider, package_version, element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert annotation in target.unsupported.details


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        ("x-polylogue-range", [3, 2]),
        (
            "x-polylogue-time-deltas",
            [{"field_a": "$.a", "field_b": "$.b", "min_delta": 5, "max_delta": 2, "avg_delta": 3}],
        ),
        ("x-polylogue-string-lengths", [{"path": "$.text", "min": 10, "max": 2, "avg": 4, "stddev": -1}]),
        (
            "x-polylogue-observed-distribution",
            {
                "numeric": {"histogram": [[1, 2]], "log_base": 1.1},
                "array_length": {"histogram": [[1, 0]], "log_base": 1.1},
            },
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[1, 2]], "log_base": 1.1, "stddev": -1}},
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[1, 2]], "log_base": 1.1, "min": 0, "max": 10, "p50": 100}},
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[10**1000, 1]], "log_base": 1.1}},
        ),
        (
            "x-polylogue-observed-distribution",
            {"numeric": {"histogram": [[711, 1]], "log_base": 2.718281828459045}},
        ),
    ],
)
def test_invalid_numeric_relation_or_partial_distribution_fails_closed(
    annotation: str,
    value: JSONValue,
) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    if annotation in {"x-polylogue-time-deltas", "x-polylogue-string-lengths"}:
        mutated[annotation] = value
    else:
        raw_properties = mutated.get("properties")
        assert isinstance(raw_properties, dict)
        properties: dict[str, JSONValue] = dict(raw_properties)
        properties["annotation_probe"] = {
            "type": "number",
            annotation: value,
        }
        mutated["properties"] = properties
    proxy.schema_overrides[(provider, package_version, element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert annotation in target.unsupported.details


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (
            "x-polylogue-string-lengths",
            [{"path": "not-a-generated-path", "min": 1, "max": 4, "avg": 2, "stddev": 1}],
        ),
        (
            "x-polylogue-foreign-keys",
            [{"source": "$.missing", "target": "$.missing_id"}],
        ),
        (
            "x-polylogue-time-deltas",
            [{"field_a": "$.missing_a", "field_b": "$.missing_b", "min_delta": 1, "max_delta": 2, "avg_delta": 1.5}],
        ),
    ],
)
def test_relation_annotation_paths_must_resolve_in_schema(
    annotation: str,
    value: JSONValue,
) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_root_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation=annotation,
        value=value,
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert annotation in target.unsupported.details


def test_time_delta_paths_must_have_compatible_types() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["time_delta_text"] = {"type": "string"}
    properties["time_delta_number"] = {"type": "integer"}
    mutated["properties"] = properties
    mutated["x-polylogue-time-deltas"] = [
        {
            "field_a": "$.time_delta_text",
            "field_b": "$.time_delta_number",
            "min_delta": 1,
            "max_delta": 2,
            "avg_delta": 1.5,
        }
    ]
    proxy.schema_overrides[(provider, package_version, element_kind)] = mutated

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.unsupported is not None
    assert "x-polylogue-time-deltas" in target.unsupported.details


def test_convergence_handoff_rejects_an_omitted_supported_spec() -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert handoff.specs == manifest.supported_specs
    assert handoff.specs
    omitted = replace(handoff, specs=())
    with pytest.raises(AssertionError, match="omitted or substituted"):
        assert_inferred_corpus_convergence_handoff_complete(manifest, omitted)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)


def _first_wired_catalog_entry(registry: SchemaRegistry) -> tuple[str, str, str]:
    for provider in registry.list_providers():
        if provider not in PROVIDER_WIRE_FORMATS:
            continue
        catalog = registry.load_package_catalog(provider)
        assert catalog is not None
        package = catalog.packages[0]
        return provider, package.version, package.elements[0].element_kind
    raise AssertionError("expected a persisted provider with a wire format")


def test_persisted_codex_package_route_is_nonempty_and_uses_catalog_artifact() -> None:
    schema_path = SCHEMA_DIR / "codex" / "versions" / "v1" / "elements" / "session_record_stream.schema.json.gz"
    assert schema_path.is_file()
    manifest = compile_inferred_corpus_manifest(registry=_registry())

    assert manifest.supported_specs
    entry = next(entry for entry in manifest.entries if entry.key.provider == "codex")
    assert (entry.key.provider, entry.key.package_version, entry.key.element_kind) == (
        "codex",
        "v1",
        "session_record_stream",
    )
    assert entry.generator_schema is not None
    assert entry.generator_schema == _registry().get_element_schema(
        "codex", version="v1", element_kind="session_record_stream"
    )
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    codex_selection = next(selection for selection in handoff.selections if selection.provider == "codex")
    assert codex_selection.schema == entry.generator_schema


def _schema_with_annotation(
    registry: SchemaRegistry,
    *,
    provider: str,
    package_version: str,
    element_kind: str,
    annotation: str,
) -> dict[str, JSONValue]:
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    raw_properties = mutated.get("properties")
    assert isinstance(raw_properties, dict)
    properties: dict[str, JSONValue] = dict(raw_properties)
    properties["annotation_probe"] = {"type": "string", annotation: "uuid"}
    mutated["properties"] = properties
    return mutated


def _schema_with_root_annotation(
    registry: SchemaRegistry,
    *,
    provider: str,
    package_version: str,
    element_kind: str,
    annotation: str,
    value: JSONValue,
) -> dict[str, JSONValue]:
    schema = registry.get_element_schema(provider, version=package_version, element_kind=element_kind)
    assert isinstance(schema, dict)
    mutated: dict[str, JSONValue] = dict(schema)
    mutated[annotation] = value
    return mutated


def test_known_generator_annotation_remains_supported_and_is_keyed() -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation="x-polylogue-format",
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.spec is None
    assert ("x-polylogue-format", "supported") in {
        (item.construct, item.state) for item in target.key.construct_support
    }


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        ("x-polylogue-format", "markdown"),
        ("x-polylogue-semantic-role", "identifier"),
        ("x-polylogue-foreign-keys", [{"source": "", "target": "$.id"}]),
        ("x-polylogue-time-deltas", [{"field_a": "$.a", "field_b": "$.b", "min_delta": "bad"}]),
    ],
)
def test_unenforced_annotation_values_are_typed_unsupported_records(annotation: str, value: JSONValue) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_root_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation=annotation,
        value=value,
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert annotation in target.unsupported.details
    assert (annotation, "unsupported") in {(item.construct, item.state) for item in target.key.construct_support}


@pytest.mark.parametrize(
    "annotation",
    [
        "x-polylogue-unknown-constraint",
        "x-third-party-constraint",
    ],
)
def test_unenforced_x_annotation_becomes_a_typed_unsupported_record(annotation: str) -> None:
    registry = _registry()
    proxy = _RegistryProxy(registry)
    provider, package_version, element_kind = _first_wired_catalog_entry(registry)
    proxy.schema_overrides[(provider, package_version, element_kind)] = _schema_with_annotation(
        registry,
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        annotation=annotation,
    )

    manifest = compile_inferred_corpus_manifest(registry=proxy)  # type: ignore[arg-type]
    target = next(
        entry
        for entry in manifest.entries
        if (entry.key.provider, entry.key.package_version, entry.key.element_kind)
        == (provider, package_version, element_kind)
    )

    assert target.spec is None
    assert target.unsupported is not None
    assert target.unsupported.reason == "unsupported_json_schema_construct"
    assert annotation in target.unsupported.details
    assert (annotation, "unsupported") in {(item.construct, item.state) for item in target.key.construct_support}


def test_live_catalog_provenance_annotations_are_not_generator_constraints() -> None:
    manifest = compile_inferred_corpus_manifest(registry=_registry())

    score_entries = [
        entry
        for entry in manifest.entries
        if any(item.construct == "x-polylogue-score" for item in entry.key.construct_support)
    ]
    assert not score_entries
    assert any(spec.provider == "codex" for spec in manifest.supported_specs)
