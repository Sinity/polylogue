"""One full-catalog wire-support receipt and one generated corpus per process.

``build_wire_support_receipt`` runs the synthetic corpus generator and the
production parser over every catalogued package element. That is ~22s of work
in a quiet process, and ``verify --all`` runs eight workers, so each test that
builds the whole catalog for itself pays it again under contention -- which is
what pushed this family past its 120s timeout.

Two things are shared here.

``shared_wire_support_receipt`` returns a whole receipt for tests that only
read one. The receipt is a pure function of the registry, the provider
selection and the seed. A test whose subject is the *building* (determinism,
catalog ordering, a mutated route or handler) must still build its own.

``shared_wire_generation`` shares the parts of a *rebuild* that a parser
mutation cannot reach. Generation, schema validation and construct coverage
are ~85% of a build and depend only on the selected schema, the seed and the
live construct handlers; the parse, the artifact evidence and every witness
verdict are recomputed on every build. A test that mutates ``parse_payload``
therefore pays the generator once per process instead of once per test.

The memo keys carry schema content and the live handler set, so an injected
schema and a removed construct handler both miss. What they do not carry is
any change to the *generator itself*: a test that patches a builder, a runtime
handler body, a corpus or ``SchemaValidator`` must build outside this.

``test_support_receipt_is_deterministic`` is the anti-vacuity condition -- it
compares a shared, memo-built receipt with a fresh build that runs the real
generator, and goes red the moment the memo answers with anything a full build
would not produce.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.archive.raw_payload.decode import JSONValue
    from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticGenerationBatch
    from polylogue.schemas.synthetic.wire_formats import ConstructCoverage, WireSupportReceipt
    from polylogue.schemas.validator import ValidationResult

__all__ = ["shared_wire_generation", "shared_wire_support_receipt"]


#: Digests of the objects a memo key names repeatedly. A selected schema runs
#: to megabytes and every key in a build carries the same one, so hashing it
#: per call would cost more than the work the memo saves. Each entry holds its
#: own strong reference, so an id is never reused underneath it, and the bound
#: keeps the retained schemas to the handful one build has live at once.
_IDENTITY_DIGESTS: dict[int, tuple[object, str]] = {}
_IDENTITY_DIGEST_LIMIT = 16


def _content_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


def _stable_digest(value: object) -> str:
    """The content digest of a long-lived object, computed once per object."""
    entry = _IDENTITY_DIGESTS.get(id(value))
    if entry is not None and entry[0] is value:
        return entry[1]
    digest = _content_digest(value)
    if len(_IDENTITY_DIGESTS) >= _IDENTITY_DIGEST_LIMIT:
        _IDENTITY_DIGESTS.pop(next(iter(_IDENTITY_DIGESTS)))
    _IDENTITY_DIGESTS[id(value)] = (value, digest)
    return digest


def _handler_key() -> tuple[str, ...]:
    """The live construct-handler names, which decide generation and coverage."""
    from polylogue.schemas.synthetic.runtime import SCHEMA_CONSTRUCT_HANDLERS

    return tuple(sorted(SCHEMA_CONSTRUCT_HANDLERS))


#: One element's schema obligations run to thousands of keyword strings, and
#: every artifact of that element reports the same ones. Held per artifact the
#: memo would retain hundreds of megabytes, so the strings and the tuples are
#: pooled before a coverage result is kept.
_KEYWORDS: dict[str, str] = {}
_KEYWORD_TUPLES: dict[tuple[str, ...], tuple[str, ...]] = {}


def _pooled(keywords: tuple[str, ...]) -> tuple[str, ...]:
    pooled = tuple(_KEYWORDS.setdefault(keyword, keyword) for keyword in keywords)
    return _KEYWORD_TUPLES.setdefault(pooled, pooled)


def _pooled_coverage(coverage: ConstructCoverage) -> ConstructCoverage:
    return replace(
        coverage,
        schema_keywords=_pooled(coverage.schema_keywords),
        exercised_keywords=_pooled(coverage.exercised_keywords),
        missing_keywords=_pooled(coverage.missing_keywords),
        nonrepresentable_keywords=_pooled(coverage.nonrepresentable_keywords),
    )


_GENERATED_WITNESSES: dict[tuple[Any, ...], list[bytes]] = {}
_GENERATED_BATCHES: dict[tuple[Any, ...], SyntheticGenerationBatch] = {}
_CONSTRUCT_COVERAGE: dict[tuple[Any, ...], ConstructCoverage] = {}
_VALIDATIONS: dict[tuple[Any, ...], ValidationResult] = {}

_ACTIVE = 0


def _corpus_key(corpus: Any) -> tuple[Any, ...]:
    return (
        corpus.provider,
        corpus.package_version,
        corpus.element_kind,
        _stable_digest(corpus.schema),
        _stable_digest(corpus.workload_profile),
        _handler_key(),
    )


@contextmanager
def shared_wire_generation() -> Iterator[None]:
    """Answer this process's generated wire corpus from a memo while inside.

    Only a build run inside the block reads the memo; a build outside it runs
    the real generator, which is what keeps the determinism comparison a
    comparison. Nesting is a no-op.
    """
    global _ACTIVE
    if _ACTIVE:
        _ACTIVE += 1
        try:
            yield
        finally:
            _ACTIVE -= 1
        return

    from polylogue.schemas.synthetic import wire_formats
    from polylogue.schemas.synthetic.core import SyntheticCorpus
    from polylogue.schemas.validator import SchemaValidator, ValidationResult

    real_witnesses = wire_formats.generate_coverage_witnesses
    real_coverage = wire_formats.construct_coverage
    real_batch = SyntheticCorpus.generate_batch
    real_validate = SchemaValidator.validate

    def memo_witnesses(corpus: Any, *, seed: int, max_witnesses: int = 128) -> list[bytes]:
        key = (*_corpus_key(corpus), seed, max_witnesses)
        witnesses = _GENERATED_WITNESSES.get(key)
        if witnesses is None:
            witnesses = real_witnesses(corpus, seed=seed, max_witnesses=max_witnesses)
            _GENERATED_WITNESSES[key] = witnesses
        return list(witnesses)

    def memo_batch(self: Any, *args: Any, **kwargs: Any) -> SyntheticGenerationBatch:
        # A witness corpus carries its branch, type and null choices in
        # instance state no key here names, and the whole witness run is
        # already memoized one level up. A positional call names arguments
        # the key does not, so it goes straight through.
        if args or self._coverage_witness_mode:
            return real_batch(self, *args, **kwargs)
        key = (*_corpus_key(self), tuple(sorted((name, repr(value)) for name, value in kwargs.items())))
        batch = _GENERATED_BATCHES.get(key)
        if batch is None:
            batch = real_batch(self, **kwargs)
            _GENERATED_BATCHES[key] = batch
        return batch

    def memo_coverage(
        schema: SchemaRecord,
        payloads: Sequence[JSONValue],
        *,
        handler_names: Collection[str] | None = None,
        **kwargs: Any,
    ) -> ConstructCoverage:
        key = (
            _stable_digest(schema),
            _content_digest(list(payloads)),
            _handler_key() if handler_names is None else tuple(sorted(handler_names)),
            _content_digest(kwargs),
        )
        coverage = _CONSTRUCT_COVERAGE.get(key)
        if coverage is None:
            coverage = _pooled_coverage(real_coverage(schema, payloads, handler_names=handler_names, **kwargs))
            _CONSTRUCT_COVERAGE[key] = coverage
        return coverage

    def memo_validate(self: Any, data: object, *, include_drift: bool | None = None) -> ValidationResult:
        key = (_stable_digest(self.schema), self.strict, include_drift, _content_digest(data))
        result = _VALIDATIONS.get(key)
        if result is None:
            result = real_validate(self, data, include_drift=include_drift)
            _VALIDATIONS[key] = result
        # ValidationResult carries mutable lists; hand every caller its own.
        return ValidationResult(
            is_valid=result.is_valid,
            errors=list(result.errors),
            drift_warnings=list(result.drift_warnings),
        )

    wire_formats.generate_coverage_witnesses = memo_witnesses
    wire_formats.construct_coverage = memo_coverage
    SyntheticCorpus.generate_batch = memo_batch  # type: ignore[method-assign]
    SchemaValidator.validate = memo_validate  # type: ignore[method-assign]
    _ACTIVE = 1
    try:
        yield
    finally:
        _ACTIVE = 0
        wire_formats.generate_coverage_witnesses = real_witnesses
        wire_formats.construct_coverage = real_coverage
        SyntheticCorpus.generate_batch = real_batch  # type: ignore[method-assign]
        SchemaValidator.validate = real_validate  # type: ignore[method-assign]


_RECEIPTS: dict[tuple[str, tuple[str, ...] | None, int], WireSupportReceipt] = {}


def shared_wire_support_receipt(
    *,
    storage_root: Path | None = None,
    providers: Sequence[str] | None = None,
    seed: int | None = None,
) -> WireSupportReceipt:
    """Return the receipt for one immutable input set, built once per process.

    The default storage root is the packaged catalog, never the ambient user
    schema directory, so every caller of the default shares one build. The
    receipt is frozen, so callers share the object rather than a copy.
    """
    from polylogue.schemas.runtime_registry import SCHEMA_DIR, SchemaRegistry
    from polylogue.schemas.synthetic.wire_formats import build_wire_support_receipt

    root = str(SCHEMA_DIR if storage_root is None else storage_root)
    selection = None if providers is None else tuple(providers)
    witness_seed = _default_receipt_seed() if seed is None else seed
    key = (root, selection, witness_seed)
    receipt = _RECEIPTS.get(key)
    if receipt is None:
        with shared_wire_generation():
            receipt = build_wire_support_receipt(
                registry=SchemaRegistry(storage_root=Path(root)),
                providers=selection,
                seed=witness_seed,
            )
        _RECEIPTS[key] = receipt
    return receipt


def _default_receipt_seed() -> int:
    from inspect import signature

    from polylogue.schemas.synthetic.wire_formats import build_wire_support_receipt

    default = signature(build_wire_support_receipt).parameters["seed"].default
    assert isinstance(default, int)
    return default
