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
live construct handlers. Unmodified parser calls are content-memoized per
process and returned as fresh values; a test that mutates ``parse_payload``
bypasses that memo, so its parser, artifact evidence and witness verdicts are
recomputed on every build. Such a test therefore pays the generator once per
process instead of once per test.

The memo keys carry schema content, the live handler set and the entry point
each memo stands in front of, so an injected schema, a removed construct
handler and a monkeypatched generator, validator or coverage function all
miss. What they do not carry is a patch *below* those four entry points: a
test that replaces a builder or a runtime handler body must build outside
this.

``test_support_receipt_is_deterministic`` is the anti-vacuity condition -- it
compares a shared, memo-built receipt with a fresh build that runs the real
generator, and goes red the moment the memo answers with anything a full build
would not produce.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from polylogue.sources import dispatch as _dispatch

_ORIGINAL_PARSE_PAYLOAD = _dispatch.parse_payload

if TYPE_CHECKING:
    from polylogue.archive.raw_payload.decode import JSONValue
    from polylogue.core.enums import Provider
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticGenerationBatch
    from polylogue.schemas.synthetic.wire_formats import ConstructCoverage, WireSupportReceipt
    from polylogue.schemas.validator import ValidationResult
    from polylogue.sources.parsers.base_models import ParsedSession
    from polylogue.sources.sidecar_evidence import SidecarResolver

__all__ = ["shared_wire_generation", "shared_wire_support_receipt"]


#: Digests of the objects a memo key names repeatedly. A selected schema runs
#: to megabytes and every key in a build carries the same one, so hashing it
#: per call would cost more than the work the memo saves. Each entry holds its
#: own strong reference, so an id is never reused underneath it, and the bound
#: keeps the retained schemas to the handful one build has live at once.
#:
#: MEASURED 2026-09-22 at 6fa844051, because the previous bound of 16 was a
#: guess at "a handful" and cost 162.6 MiB to be wrong. One full-catalog build
#: calls ``_stable_digest`` 2,120 times over 73 distinct objects. Replaying
#: that access sequence through an LRU of each size:
#:
#:   limit  digest misses   retained by this memo
#:       2            144   0.9 MiB
#:       4             73   1.3 MiB
#:       8             73   34.0 MiB
#:      16             73   163.8 MiB  (1.67M objects)
#:      32             73   475.0 MiB
#:
#: 4 is where the miss count reaches its floor: every size from 4 upward costs
#: the same 73 digests, so 16 bought nothing and retained 162.5 MiB for it. 2
#: is not enough -- it doubles the digests. Measured end to end, the live
#: reachable heap after one build is 282.4 MiB over 2,188,458 objects at
#: limit 16 and 119.8 MiB over 534,043 at limit 4, reproduced twice, while
#: ``_content_digest`` calls move 3,874 -> 3,887 (+13, ~12 ms at 0.9 ms each).
#: Nothing else holds those schemas: the whole difference is freed.
#:
#: Confirmed end to end through the managed harness on one selection
#: (``devtools test tests/unit/core``, 2,839 tests, one worker, the same nine
#: inherited failures either side). This memo alone falls 175.32 -> 23.77 MiB,
#: the worker's live reachable heap 684.3 -> 475.2 MiB over 2.1M fewer
#: objects, and its resident memory at session end 1001.2 -> 849.7 MiB.
#: What did NOT move is the per-test PEAK -- 1145.7 -> 1291.0 MiB, a transient
#: single sample on a contended host (the 50-test sampled maximum moved only
#: 1015 -> 1058 MiB, and mean sampled resident fell 686.2 -> 663.2). Peak is
#: what ``devtools/worker_memory.py`` sizes width from, so this reduces
#: retention without yet being evidence for a wider corpus run.
_IDENTITY_DIGESTS: dict[int, tuple[object, str]] = {}
_IDENTITY_DIGEST_LIMIT = 4


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
#:
#: The string pool is left unbounded deliberately, and its size is declared
#: rather than assumed: it held 19,560 entries / 4.35 MiB after one catalog
#: build and *the same* 19,560 / 4.35 MiB after six, because its key space is
#: the catalog's schema vocabulary, not the number of calls. Bounding it would
#: only break the interning that keeps the coverage results small.
_KEYWORDS: dict[str, str] = {}
#: The tuple pool does grow with distinct keyword combinations (937 after one
#: build, 3,696 after six), so it carries a bound. 2048 is twice one build's
#: live set; above it the oldest tuple stops being deduplicated, which costs
#: memory only for a combination nothing has asked for recently.
_KEYWORD_TUPLE_LIMIT = 2048
_KEYWORD_TUPLES: OrderedDict[tuple[str, ...], tuple[str, ...]] = OrderedDict()


def _pooled(keywords: tuple[str, ...]) -> tuple[str, ...]:
    pooled = tuple(_KEYWORDS.setdefault(keyword, keyword) for keyword in keywords)
    shared = _KEYWORD_TUPLES.setdefault(pooled, pooled)
    _KEYWORD_TUPLES.move_to_end(shared)
    _evict_to(_KEYWORD_TUPLES, _KEYWORD_TUPLE_LIMIT)
    return shared


def _pooled_coverage(coverage: ConstructCoverage) -> ConstructCoverage:
    return replace(
        coverage,
        schema_keywords=_pooled(coverage.schema_keywords),
        exercised_keywords=_pooled(coverage.exercised_keywords),
        missing_keywords=_pooled(coverage.missing_keywords),
        nonrepresentable_keywords=_pooled(coverage.nonrepresentable_keywords),
    )


#: Every memo below is bounded, and every bound is the measurement that chose
#: it. Measured 2026-09-22 at 6fa844051 by driving ``shared_wire_support_receipt``
#: over the packaged catalog (52 entries) once, and then over six distinct
#: seeds, reading each memo's deep size back with ``tests.infra.retention_probe``:
#:
#: memo                  1 build          6 builds        per build
#: _GENERATED_WITNESSES  36 / 34.3 MiB    216 / 209.6 MiB  +35 MiB
#: _KEYWORD_TUPLES       937 / 7.6 MiB    3696 / 32.5 MiB  +5 MiB
#: _CONSTRUCT_COVERAGE   654 / 1.8 MiB    3580 / 10.6 MiB  +1.8 MiB
#: _RECEIPTS             1 / 0.8 MiB      6 / 4.5 MiB      +0.8 MiB
#: _VALIDATIONS          1321 / 0.5 MiB   7351 / 2.9 MiB   +0.5 MiB
#: _GENERATED_BATCHES    36 / 0.5 MiB     216 / 2.9 MiB    +0.5 MiB
#:
#: (Sizes are marginal -- what a memo retains that no earlier memo in the same
#: walk already claimed -- so they sum rather than double-count the pooled
#: keyword strings the coverage results share.)
#:
#: Every limit holds AT LEAST TWO full builds. That is the quantity that
#: matters, and it is measured rather than assumed: replaying two real builds'
#: recorded key sequences through an LRU shows these memos almost never hit
#: WITHIN a build (coverage 1,310 accesses over 1,306 distinct keys,
#: validations 2,642 over 2,642, witnesses 72 over 72). Their whole value is
#: that the NEXT test asking for the same build answers from the memo, so a
#: limit below one build's key set would turn every repeat into a full
#: recomputation, and one at two builds leaves that hit rate untouched. The
#: keyword tuple pool is the exception that does reuse within a build --
#: 5,224 accesses over 1,607 distinct tuples -- and 2,048 covers two builds of
#: those. What the bounds stop is the second, sixth and hundredth distinct
#: corpus a long session accumulates and never reads again.

#: Generated witness payloads dominate this module's retention, and the budget
#: is in BYTES rather than entries because an entry is a list of wire payloads
#: whose size follows the schema it came from -- 0.97 MiB each on the packaged
#: catalog, but nothing declares that it stays there. 96 MiB holds roughly
#: 2.8 full catalog builds against a 34.3 MiB per-build working set.
_GENERATED_WITNESS_BYTES_LIMIT = 96 * 1024 * 1024
_GENERATED_WITNESSES: OrderedDict[tuple[Any, ...], list[bytes]] = OrderedDict()
_GENERATED_WITNESS_BYTES = 0
#: A batch is small (13 KiB measured); the bound exists so the count cannot run
#: away with the number of distinct corpora a session touches.
_GENERATED_BATCH_LIMIT = 128
_GENERATED_BATCHES: OrderedDict[tuple[Any, ...], SyntheticGenerationBatch] = OrderedDict()
#: Coverage results are pooled (see ``_pooled``), so their own retention is
#: small; 2048 is three full catalog builds' worth of distinct results.
_CONSTRUCT_COVERAGE_LIMIT = 2048
_CONSTRUCT_COVERAGE: OrderedDict[tuple[Any, ...], ConstructCoverage] = OrderedDict()
#: One build validates 1,321 payloads; 4096 is three of those.
_VALIDATION_LIMIT = 4096
_VALIDATIONS: OrderedDict[tuple[Any, ...], ValidationResult] = OrderedDict()


def _evict_to(cache: OrderedDict[Any, Any], limit: int) -> None:
    """Keep ``cache`` at ``limit`` entries, oldest use first."""
    while len(cache) > limit:
        cache.popitem(last=False)


def _retain_witnesses(key: tuple[Any, ...], witnesses: list[bytes]) -> None:
    """Memoize one generated corpus under the module's declared byte budget.

    The newest entry is never the one evicted: a corpus larger than the whole
    budget still answers the call that produced it, because a memo that
    discards what it was just asked for is a slower no-op, not a bound.
    """
    global _GENERATED_WITNESS_BYTES
    _GENERATED_WITNESSES[key] = witnesses
    _GENERATED_WITNESSES.move_to_end(key)
    _GENERATED_WITNESS_BYTES += sum(len(payload) for payload in witnesses)
    while _GENERATED_WITNESS_BYTES > _GENERATED_WITNESS_BYTES_LIMIT and len(_GENERATED_WITNESSES) > 1:
        _, evicted = _GENERATED_WITNESSES.popitem(last=False)
        _GENERATED_WITNESS_BYTES -= sum(len(payload) for payload in evicted)


# Parser results are immutable input evidence for an unmodified production
# route, but the ParsedSession model contains mutable lists.  Keep only one
# receipt-sized working set and hand callers deep copies.  Parser-mutating
# tests deliberately bypass this memo (see ``memo_parse`` below), preserving
# their red/green witnesses.
_PARSED_PAYLOAD_CACHE_LIMIT = 512
_PARSED_PAYLOADS: OrderedDict[tuple[Any, ...], list[Any]] = OrderedDict()

_ACTIVE = 0


def _corpus_key(corpus: Any, entry_point: Any) -> tuple[Any, ...]:
    return (
        entry_point,
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
    from polylogue.sources import dispatch as dispatch_module

    real_witnesses = wire_formats.generate_coverage_witnesses
    real_coverage = wire_formats.construct_coverage
    real_batch = SyntheticCorpus.generate_batch
    real_validate = SchemaValidator.validate
    real_parse = dispatch_module.parse_payload

    def memo_witnesses(corpus: Any, *, seed: int, max_witnesses: int = 128) -> list[bytes]:
        key = (*_corpus_key(corpus, real_witnesses), seed, max_witnesses)
        witnesses = _GENERATED_WITNESSES.get(key)
        if witnesses is None:
            witnesses = real_witnesses(corpus, seed=seed, max_witnesses=max_witnesses)
            _retain_witnesses(key, witnesses)
        else:
            _GENERATED_WITNESSES.move_to_end(key)
        return list(witnesses)

    def memo_batch(self: Any, *args: Any, **kwargs: Any) -> SyntheticGenerationBatch:
        # A witness corpus carries its branch, type and null choices in
        # instance state no key here names, and the whole witness run is
        # already memoized one level up. A positional call names arguments
        # the key does not, so it goes straight through.
        if args or self._coverage_witness_mode:
            return real_batch(self, *args, **kwargs)
        key = (
            *_corpus_key(self, real_batch),
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
        )
        batch = _GENERATED_BATCHES.get(key)
        if batch is None:
            batch = real_batch(self, **kwargs)
            _GENERATED_BATCHES[key] = batch
            _evict_to(_GENERATED_BATCHES, _GENERATED_BATCH_LIMIT)
        else:
            _GENERATED_BATCHES.move_to_end(key)
        return batch

    def memo_coverage(
        schema: SchemaRecord,
        payloads: Sequence[JSONValue],
        *,
        handler_names: Collection[str] | None = None,
        **kwargs: Any,
    ) -> ConstructCoverage:
        key = (
            real_coverage,
            _stable_digest(schema),
            _content_digest(list(payloads)),
            _handler_key() if handler_names is None else tuple(sorted(handler_names)),
            _content_digest(kwargs),
        )
        coverage = _CONSTRUCT_COVERAGE.get(key)
        if coverage is None:
            coverage = _pooled_coverage(real_coverage(schema, payloads, handler_names=handler_names, **kwargs))
            _CONSTRUCT_COVERAGE[key] = coverage
            _evict_to(_CONSTRUCT_COVERAGE, _CONSTRUCT_COVERAGE_LIMIT)
        else:
            _CONSTRUCT_COVERAGE.move_to_end(key)
        return coverage

    def memo_validate(self: Any, data: object, *, include_drift: bool | None = None) -> ValidationResult:
        key = (real_validate, _stable_digest(self.schema), self.strict, include_drift, _content_digest(data))
        result = _VALIDATIONS.get(key)
        if result is None:
            # Forward only what arrived: a narrower stand-in for ``validate``
            # need not accept the keyword this signature declares.
            result = (
                real_validate(self, data)
                if include_drift is None
                else real_validate(self, data, include_drift=include_drift)
            )
            _VALIDATIONS[key] = result
            _evict_to(_VALIDATIONS, _VALIDATION_LIMIT)
        else:
            _VALIDATIONS.move_to_end(key)
        # ValidationResult carries mutable lists; hand every caller its own.
        return ValidationResult(
            is_valid=result.is_valid,
            errors=list(result.errors),
            drift_warnings=list(result.drift_warnings),
        )

    def memo_parse(
        provider: str | Provider,
        payload: object,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: SchemaResolution | None = None,
        source_path: str | None = None,
        sidecar_resolver: SidecarResolver | None = None,
    ) -> list[ParsedSession]:
        """Reuse only the unmodified parser's immutable-input result.

        The witness tests monkeypatch ``dispatch.parse_payload`` to model
        parser loss.  Their function object is therefore not ``real_parse``
        and they continue through the real call on every build.  Ordinary
        builds use the content-addressed payload key and receive a fresh deep
        copy, so callers cannot mutate a later assertion through the memo.
        """
        if real_parse is not _ORIGINAL_PARSE_PAYLOAD or sidecar_resolver is not None:
            if sidecar_resolver is None:
                # Existing witness wrappers intentionally mirror the call
                # shape used by the receipt builder and do not accept the
                # optional sidecar keyword.
                return real_parse(
                    provider,
                    payload,
                    fallback_id,
                    _depth,
                    schema_resolution=schema_resolution,
                    source_path=source_path,
                )
            return real_parse(
                provider,
                payload,
                fallback_id,
                _depth,
                schema_resolution=schema_resolution,
                source_path=source_path,
                sidecar_resolver=sidecar_resolver,
            )
        key = (
            provider,
            _content_digest(payload),
            fallback_id,
            _depth,
            _content_digest(schema_resolution),
            source_path,
        )
        cached = _PARSED_PAYLOADS.get(key)
        if cached is None:
            cached = real_parse(
                provider,
                payload,
                fallback_id,
                _depth,
                schema_resolution=schema_resolution,
                source_path=source_path,
                sidecar_resolver=sidecar_resolver,
            )
            _PARSED_PAYLOADS[key] = cached
            _PARSED_PAYLOADS.move_to_end(key)
            while len(_PARSED_PAYLOADS) > _PARSED_PAYLOAD_CACHE_LIMIT:
                _PARSED_PAYLOADS.popitem(last=False)
        else:
            _PARSED_PAYLOADS.move_to_end(key)
        return deepcopy(cached)

    wire_formats.generate_coverage_witnesses = memo_witnesses
    wire_formats.construct_coverage = memo_coverage
    SyntheticCorpus.generate_batch = memo_batch  # type: ignore[method-assign]
    SchemaValidator.validate = memo_validate  # type: ignore[method-assign]
    # ``build_wire_support_receipt`` imports this attribute at call time.  A
    # monkeypatched parser is intentionally left untouched by ``memo_parse``
    # so mutation witnesses still exercise the parser seam on every input.
    dispatch_module.parse_payload = memo_parse
    _ACTIVE = 1
    try:
        yield
    finally:
        _ACTIVE = 0
        wire_formats.generate_coverage_witnesses = real_witnesses
        wire_formats.construct_coverage = real_coverage
        SyntheticCorpus.generate_batch = real_batch  # type: ignore[method-assign]
        SchemaValidator.validate = real_validate  # type: ignore[method-assign]
        dispatch_module.parse_payload = real_parse


#: A whole receipt is 5.8 MiB of typed entries; six distinct ones retained
#: 35.1 MiB. Eight is well above the handful of (root, selection, seed) triples
#: the suite actually asks for, and a ninth costs one rebuild, not a wrong
#: answer -- the receipt is a pure function of its key.
_RECEIPT_LIMIT = 8
_RECEIPTS: OrderedDict[tuple[str, tuple[str, ...] | None, int], WireSupportReceipt] = OrderedDict()

_CACHE_VERSION = 1


def _run_cache_path(*, root: Path, selection: tuple[str, ...] | None, seed: int) -> Path | None:
    """A per-harness-run receipt path shared by xdist workers.

    A run id is deliberately mandatory: cached synthetic parser evidence must
    never survive into a later source revision.  The managed harness assigns
    one id to every worker in a pytest invocation; ordinary Python use keeps
    the existing process-local memo semantics.
    """
    run_id = os.environ.get("POLYLOGUE_PYTEST_RUN_ID")
    if not run_id:
        return None
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    input_digest = _content_digest({"root": str(root.resolve()), "providers": selection, "seed": seed})
    return Path.cwd() / ".cache" / "pytest-wire-support" / run_digest / f"{input_digest}.json"


@contextmanager
def _receipt_cache_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _receipt_from_cache_payload(payload: object) -> WireSupportReceipt:
    """Rebuild typed immutable receipt data from the private JSON cache."""
    from polylogue.schemas.synthetic.conservation import ConservationFinding, ConservationResult
    from polylogue.schemas.synthetic.wire_formats import (
        ConstructCoverage,
        WireParserWitness,
        WireSupportEntry,
        WireSupportReceipt,
    )

    if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
        raise ValueError("unsupported wire-support cache payload")
    raw_receipt = payload.get("receipt")
    if not isinstance(raw_receipt, dict):
        raise ValueError("wire-support cache receipt is not an object")

    def strings(value: object) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("wire-support cache expected strings")
        return tuple(value)

    def integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("wire-support cache expected integer")
        return value

    def optional_string(value: object) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError("wire-support cache expected string or null")
        return value

    entries: list[WireSupportEntry] = []
    raw_entries = raw_receipt.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("wire-support cache entries are not a list")
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("wire-support cache entry is not an object")
        raw_coverage = raw_entry.get("construct_coverage")
        coverage = None
        if raw_coverage is not None:
            if not isinstance(raw_coverage, dict):
                raise ValueError("wire-support cache coverage is not an object")
            raw_reasons = raw_coverage.get("nonrepresentable_reasons")
            if not isinstance(raw_reasons, list):
                raise ValueError("wire-support cache reasons are not a list")
            reasons = tuple(
                (item[0], item[1])
                for item in raw_reasons
                if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], str)
            )
            if len(reasons) != len(raw_reasons):
                raise ValueError("wire-support cache reason is malformed")
            coverage = ConstructCoverage(
                schema_keywords=strings(raw_coverage.get("schema_keywords")),
                exercised_keywords=strings(raw_coverage.get("exercised_keywords")),
                missing_keywords=strings(raw_coverage.get("missing_keywords")),
                nonrepresentable_keywords=strings(raw_coverage.get("nonrepresentable_keywords")),
                nonrepresentable_reasons=reasons,
            )
        witnesses: list[WireParserWitness] = []
        raw_witnesses = raw_entry.get("parser_witnesses")
        if not isinstance(raw_witnesses, list):
            raise ValueError("wire-support cache witnesses are not a list")
        for raw_witness in raw_witnesses:
            if not isinstance(raw_witness, dict):
                raise ValueError("wire-support cache witness is not an object")
            raw_conservation = raw_witness.get("conservation")
            conservation = None
            if raw_conservation is not None:
                if not isinstance(raw_conservation, dict):
                    raise ValueError("wire-support cache conservation is not an object")
                raw_findings = raw_conservation.get("findings")
                if not isinstance(raw_findings, list):
                    raise ValueError("wire-support cache findings are not a list")
                findings: list[ConservationFinding] = []
                for finding in raw_findings:
                    if not isinstance(finding, dict):
                        raise ValueError("wire-support cache finding is not an object")
                    path, role, verdict, detail = (
                        finding.get("path"),
                        finding.get("role"),
                        finding.get("verdict"),
                        finding.get("detail"),
                    )
                    if not all(isinstance(value, str) for value in (path, role, verdict, detail)):
                        raise ValueError("wire-support cache finding is malformed")
                    if verdict not in {"loss", "duplication", "mutation"}:
                        raise ValueError("wire-support cache finding verdict is invalid")
                    assert isinstance(path, str) and isinstance(role, str) and isinstance(detail, str)
                    findings.append(ConservationFinding(path=path, role=role, verdict=verdict, detail=detail))
                conservation = ConservationResult(
                    planted_count=integer(raw_conservation.get("planted_count")),
                    findings=tuple(findings),
                    excluded_paths=strings(raw_conservation.get("excluded_paths")),
                )
            artifact_kind = raw_witness.get("artifact_kind")
            if artifact_kind not in {"baseline", "coverage"}:
                raise ValueError("wire-support cache artifact kind is invalid")
            witnesses.append(
                WireParserWitness(
                    index=integer(raw_witness.get("index")),
                    exercised_keywords=strings(raw_witness.get("exercised_keywords")),
                    parsed_session_count=integer(raw_witness.get("parsed_session_count")),
                    parsed_message_count=integer(raw_witness.get("parsed_message_count")),
                    validation_error=optional_string(raw_witness.get("validation_error")),
                    artifact_kind=artifact_kind,
                    artifact_evidence=strings(raw_witness.get("artifact_evidence")),
                    conservation=conservation,
                    conservation_enforced=raw_witness.get("conservation_enforced") is True,
                )
            )
        status = raw_entry.get("status")
        if status not in {"supported", "unsupported"}:
            raise ValueError("wire-support cache status is invalid")
        schema_valid = raw_entry.get("schema_valid")
        if schema_valid is not None and not isinstance(schema_valid, bool):
            raise ValueError("wire-support cache schema validity is malformed")
        provider = raw_entry.get("provider")
        if not isinstance(provider, str):
            raise ValueError("wire-support cache provider is invalid")
        entries.append(
            WireSupportEntry(
                provider=provider,
                status=status,
                reason=optional_string(raw_entry.get("reason")),
                package_version=optional_string(raw_entry.get("package_version")),
                element_kind=optional_string(raw_entry.get("element_kind")),
                schema_valid=schema_valid,
                parsed_session_count=integer(raw_entry.get("parsed_session_count")),
                parsed_message_count=integer(raw_entry.get("parsed_message_count")),
                construct_coverage=coverage,
                validation_error=optional_string(raw_entry.get("validation_error")),
                parser_witnesses=tuple(witnesses),
            )
        )
    scope = raw_receipt.get("catalog_scope")
    if scope not in {"registry-default", "explicit"}:
        raise ValueError("wire-support cache scope is invalid")
    return WireSupportReceipt(
        catalog_providers=strings(raw_receipt.get("catalog_providers")),
        entries=tuple(entries),
        missing_routes=strings(raw_receipt.get("missing_routes")),
        witness_seed=integer(raw_receipt.get("witness_seed")),
        catalog_scope=scope,
    )


def _read_cached_receipt(path: Path) -> WireSupportReceipt | None:
    try:
        return _receipt_from_cache_payload(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_receipt(path: Path, receipt: WireSupportReceipt) -> None:
    payload = json.dumps({"version": _CACHE_VERSION, "receipt": asdict(receipt)}, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
        cache_path = _run_cache_path(root=Path(root), selection=selection, seed=witness_seed)
        if cache_path is None:
            with shared_wire_generation():
                receipt = build_wire_support_receipt(
                    registry=SchemaRegistry(storage_root=Path(root)),
                    providers=selection,
                    seed=witness_seed,
                )
        else:
            with _receipt_cache_lock(cache_path.with_suffix(".lock")):
                receipt = _read_cached_receipt(cache_path)
                if receipt is None:
                    with shared_wire_generation():
                        receipt = build_wire_support_receipt(
                            registry=SchemaRegistry(storage_root=Path(root)),
                            providers=selection,
                            seed=witness_seed,
                        )
                    _write_cached_receipt(cache_path, receipt)
        _RECEIPTS[key] = receipt
        _evict_to(_RECEIPTS, _RECEIPT_LIMIT)
    else:
        _RECEIPTS.move_to_end(key)
    return receipt


def _default_receipt_seed() -> int:
    from inspect import signature

    from polylogue.schemas.synthetic.wire_formats import build_wire_support_receipt

    default = signature(build_wire_support_receipt).parameters["seed"].default
    assert isinstance(default, int)
    return default
