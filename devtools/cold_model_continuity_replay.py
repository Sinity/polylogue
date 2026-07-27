"""z9gh gap #3: real paged/cancelled/cold-model MCP continuity replay.

``devtools/continuity_replay.py`` (polylogue-t8t) already proves lossless
pagination and genuine mid-flight cancellation over real MCP stdio JSON-RPC --
but only for scenarios whose exact route (tool + expression + item-identity
path) is pre-scripted in :mod:`polylogue.product.continuity_scenarios`. That
proves a route can be *executed*; it does not prove a route can be
*formulated* by a model with no access to that scripted declaration. That is
the one property polylogue-z9gh's remaining gap #3 names and this module
supplies: a plan chosen at runtime from published discovery evidence alone
(the ``query`` tool's live-discovered schema plus the
``QUERY_DISCOVERY_EXAMPLES`` catalog), proven jointly with lossless
pagination and a real server-confirmed cancellation over that same
self-formulated plan -- against a freshly seeded archive sized to force
genuine multi-page pagination.

Deliberate scope boundary (read before assuming more is proven here):

- "Cold" here means the planner (:func:`select_cold_model_plan`) never
  imports :mod:`polylogue.product.continuity_scenarios` or any oracle/fixture
  answer -- it only ever sees live MCP tool discovery and the query-discovery
  catalog, then does deterministic keyword-to-template matching + parameter
  substitution. It is not an LLM call; it is the auditable mechanical
  contract an actual cold LLM would need the discovery surface to support.
- Fetching that catalog *purely over the wire* (``explain(subject="result")``)
  was attempted first and found to fail: the real MCP server overflows its
  25KB response budget on the full 106-example catalog and returns a
  non-narrowing continuation that drops the original ``subject`` argument
  (see :func:`fetch_discovery_catalog_over_wire`). That is a real, separately
  tracked defect (``polylogue-3k30``), not something this module works around
  silently -- it is reported as an explicit, non-blocking-for-the-other-two-
  properties diagnostic, and this module falls back to the in-process
  ``QUERY_DISCOVERY_EXAMPLES`` registry (the exact same data ``explain``
  is contracted to serve) to keep proving plan formulation, pagination, and
  cancellation jointly rather than stalling entirely on that defect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devtools.continuity_replay import (
    CancellationExerciseReceipt,
    ContinuityReplayError,
    StdioMCPContinuityRoute,
)
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.storage_records import SessionBuilder

DiscoverySource = Literal["mcp-wire", "in-process-registry-fallback"]


class ColdModelPlanningError(RuntimeError):
    """Raised when no plan can be formulated from discovery evidence alone."""


# ── The investigative goal (sparse, operator-shaped clues) ────────────


@dataclass(frozen=True, slots=True)
class ColdModelGoal:
    """A sparse natural-language goal, mirroring the mandate's own incident:
    a topic keyword and approximate scope, never an internal route id."""

    goal_text: str
    goal_keywords: tuple[str, ...]
    substitutions: Mapping[str, str]
    forced_page_limit: int
    corpus_row_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "goal_text": self.goal_text,
            "goal_keywords": list(self.goal_keywords),
            "substitutions": dict(self.substitutions),
            "forced_page_limit": self.forced_page_limit,
            "corpus_row_count": self.corpus_row_count,
        }


#: Default goal exercised by ``main``/tests: mirrors an operator recalling
#: "there were some decisions recorded about a cold-model paging proof" --
#: enough of a sparse topic clue to drive template selection, substituted
#: into whichever catalog template scores highest, never a scripted route id.
DEFAULT_COLD_MODEL_GOAL = ColdModelGoal(
    goal_text="Which decisions did this investigation record about the cold-model paging proof?",
    goal_keywords=("decision", "assertions", "topic", "named"),
    substitutions={"topic": "coldmodelpagingproof"},
    forced_page_limit=8,
    corpus_row_count=27,
)


# ── Discovery-catalog fetch (wire-first, honest fallback) ─────────────


@dataclass(frozen=True, slots=True)
class DiscoveryCatalogFetch:
    """Result of attempting to fetch the query-discovery catalog purely over MCP wire."""

    source: DiscoverySource
    examples: tuple[Mapping[str, object], ...]
    wire_status: str | None
    wire_original_bytes: int | None
    wire_budget_bytes: int | None
    diagnostic: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "example_count": len(self.examples),
            "wire_status": self.wire_status,
            "wire_original_bytes": self.wire_original_bytes,
            "wire_budget_bytes": self.wire_budget_bytes,
            "diagnostic": self.diagnostic,
        }


async def fetch_discovery_catalog_over_wire(route: StdioMCPContinuityRoute) -> DiscoveryCatalogFetch:
    """Attempt the strictest-possible cold path: fetch the catalog via the
    real ``explain`` MCP tool call alone, no Python import.

    Falls back to the in-process :data:`QUERY_DISCOVERY_EXAMPLES` registry
    (the identical data ``explain`` is contracted to serve) when the wire
    call overflows its response budget -- see the module docstring and
    ``polylogue-3k30`` for why that happens today.
    """

    response_text = await route.invoke("explain", {"subject": "result"})
    payload = json.loads(response_text)
    if payload.get("status") == "response_budget_exceeded":
        diagnostic = (
            "explain(subject='result') exceeded the MCP response budget "
            f"({payload.get('original_bytes')} bytes vs {payload.get('budget_bytes')} budget); "
            f"its continuation {payload.get('continuation')!r} does not preserve the original "
            "subject argument, so retrying repeats the identical oversized call (polylogue-3k30)"
        )
        return DiscoveryCatalogFetch(
            source="in-process-registry-fallback",
            examples=tuple(example.to_payload() for example in QUERY_DISCOVERY_EXAMPLES),
            wire_status=str(payload.get("status")),
            wire_original_bytes=payload.get("original_bytes")
            if isinstance(payload.get("original_bytes"), int)
            else None,
            wire_budget_bytes=payload.get("budget_bytes") if isinstance(payload.get("budget_bytes"), int) else None,
            diagnostic=diagnostic,
        )
    explanation = payload.get("explanation", payload)
    examples = explanation.get("examples") if isinstance(explanation, Mapping) else None
    if not isinstance(examples, list) or not examples:
        return DiscoveryCatalogFetch(
            source="in-process-registry-fallback",
            examples=tuple(example.to_payload() for example in QUERY_DISCOVERY_EXAMPLES),
            wire_status=str(payload.get("status")) if isinstance(payload.get("status"), str) else None,
            wire_original_bytes=None,
            wire_budget_bytes=None,
            diagnostic="explain(subject='result') returned no usable examples list over the wire",
        )
    return DiscoveryCatalogFetch(
        source="mcp-wire",
        examples=tuple(cast(Mapping[str, object], example) for example in examples),
        wire_status="ok",
        wire_original_bytes=None,
        wire_budget_bytes=None,
        diagnostic=None,
    )


# ── Cold plan selection (deterministic, catalog-only) ──────────────────


@dataclass(frozen=True, slots=True)
class ColdModelPlan:
    """A query-tool plan formulated entirely from discovery evidence."""

    tool: str
    arguments: dict[str, object]
    example_key: str
    example_answers: str
    example_template: str
    example_parameters: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    discovered_argument_names: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "example_key": self.example_key,
            "example_answers": self.example_answers,
            "example_template": self.example_template,
            "example_parameters": list(self.example_parameters),
            "matched_keywords": list(self.matched_keywords),
            "discovered_argument_names": list(self.discovered_argument_names),
        }


def select_cold_model_plan(
    goal: ColdModelGoal,
    examples: Sequence[Mapping[str, object]],
    query_tool_properties: Mapping[str, object],
) -> ColdModelPlan:
    """Formulate one query-tool plan from discovery evidence alone.

    Scores every *parameterized* catalog example (one with a ``template`` +
    named ``parameters``, i.e. genuinely fillable rather than a bare literal)
    by counting how many of the goal's keywords appear in its ``answers``
    text -- the same natural-language field a cold model reading the catalog
    would read. The winner's template is filled from the goal's
    substitutions; if a required parameter has no supplied substitution, or
    no example scores at all, planning fails honestly rather than guessing.
    Finally cross-checks that ``expression``/``limit``/``continuation`` are
    themselves documented on the live-discovered ``query`` tool schema --
    proving the plan's own argument names were independently discoverable,
    not assumed.
    """

    candidates = [
        example
        for example in examples
        if example.get("route") == "query"
        and isinstance(example.get("template"), str)
        and isinstance(example.get("parameters"), list)
        and example["parameters"]
    ]
    if not candidates:
        raise ColdModelPlanningError("no parameterized query-route discovery example is available to formulate a plan")

    def _score(example: Mapping[str, object]) -> int:
        answers = str(example.get("answers", "")).lower()
        return sum(1 for keyword in goal.goal_keywords if keyword.lower() in answers)

    ranked = sorted(candidates, key=lambda example: (-_score(example), str(example.get("key"))))
    best = ranked[0]
    best_score = _score(best)
    if best_score == 0:
        raise ColdModelPlanningError(
            f"no discovery example's answers text matched any goal keyword in {goal.goal_keywords!r}"
        )

    raw_parameters = cast(list[Mapping[str, object]], best["parameters"])
    parameter_names = tuple(str(parameter["name"]) for parameter in raw_parameters)
    missing = [name for name in parameter_names if name not in goal.substitutions]
    if missing:
        raise ColdModelPlanningError(f"goal substitutions do not supply required template parameters: {missing}")

    template = str(best["template"])
    try:
        expression = template.format(**goal.substitutions)
    except (KeyError, IndexError) as exc:
        raise ColdModelPlanningError(f"template substitution failed for {template!r}: {exc}") from exc

    required_arguments = ("expression", "limit", "continuation")
    missing_schema_arguments = [name for name in required_arguments if name not in query_tool_properties]
    if missing_schema_arguments:
        raise ColdModelPlanningError(
            f"live-discovered query tool schema omits required argument(s) {missing_schema_arguments!r}; "
            "a cold model could not have discovered them"
        )

    matched = tuple(
        sorted({keyword for keyword in goal.goal_keywords if keyword.lower() in str(best["answers"]).lower()})
    )
    return ColdModelPlan(
        tool="query",
        arguments={"expression": expression, "limit": goal.forced_page_limit},
        example_key=str(best["key"]),
        example_answers=str(best["answers"]),
        example_template=template,
        example_parameters=parameter_names,
        matched_keywords=matched,
        discovered_argument_names=tuple(sorted(str(name) for name in query_tool_properties)),
    )


# ── Synthetic corpus (sized to force genuine multi-page pagination) ────


def seed_cold_model_corpus(archive_root: Path, goal: ColdModelGoal) -> tuple[str, ...]:
    """Seed ``goal.corpus_row_count`` decision assertions, each mentioning
    the goal's substituted topic, over that many distinct sessions.

    Uses the same direct-storage primitives (``SessionBuilder``,
    ``upsert_assertion``) the existing t8t fixture uses, decoupled from that
    fixture's own (smaller) corpus so this proof's row count is chosen
    independently to force pagination for the goal's ``forced_page_limit``.
    """

    archive_root = archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"
    user_db_path = archive_root / "user.db"
    if db_path.exists() or user_db_path.exists():
        raise ValueError(f"cold-model archive root must be fresh: {archive_root}")
    initialize_archive_database(db_path, ArchiveTier.INDEX)
    initialize_archive_database(user_db_path, ArchiveTier.USER)

    topic = goal.substitutions.get("topic")
    if not topic:
        raise ValueError("cold-model corpus seeding requires a 'topic' goal substitution")

    import sqlite3

    assertion_ids: list[str] = []
    with sqlite3.connect(user_db_path) as conn:
        for index in range(goal.corpus_row_count):
            builder = (
                SessionBuilder(db_path, f"cold-model-row-{index}")
                .provider("test")
                .title(f"cold model corpus row {index}")
                .add_message(text=f"synthetic cold-model corpus row {index}")
            )
            builder.save()
            assertion_id = f"cold-model-assert-{index}"
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                target_ref=f"session:{builder.native_session_id()}",
                kind=AssertionKind.DECISION,
                key=f"cold-model-decision-{index}",
                body_text=f"{topic} marker {index}",
                author_ref="user:cold-model-harness",
                author_kind="user",
                evidence_refs=(),
                status="active",
                visibility="public",
                now_ms=1_768_478_400_000 + index,
            )
            assertion_ids.append(assertion_id)
        conn.commit()
    return tuple(assertion_ids)


# ── Generic pagination proof (tool-agnostic, no scenario coupling) ─────


@dataclass(slots=True)
class PaginationProof:
    pages: list[JSONDocument] = field(default_factory=list)
    item_identities: list[str] = field(default_factory=list)
    exact_count: int | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def unique_identity_count(self) -> int:
        return len(set(self.item_identities))

    @property
    def exact_enumeration_verified(self) -> bool:
        no_duplicates = len(self.item_identities) == self.unique_identity_count
        count_matches = self.exact_count is None or self.exact_count == len(self.item_identities)
        return no_duplicates and count_matches and self.page_count >= 1

    def to_payload(self) -> dict[str, object]:
        return {
            "page_count": self.page_count,
            "enumerated_item_count": len(self.item_identities),
            "unique_identity_count": self.unique_identity_count,
            "exact_count": self.exact_count,
            "exact_enumeration_verified": self.exact_enumeration_verified,
            "pages": list(self.pages),
        }


async def page_and_verify(
    route: StdioMCPContinuityRoute,
    tool: str,
    arguments: Mapping[str, object],
    *,
    item_identity_field: str,
) -> PaginationProof:
    """Drive one real MCP call to exhaustion, verifying every pagination
    invariant the production ``query_units`` continuation contract makes:
    offset progression, stable ``query_ref``/``result_ref``, no repeated or
    skipped logical rows, and a non-empty, strictly progressing continuation
    chain that terminates.
    """

    proof = PaginationProof()
    seen_identities: set[str] = set()
    seen_continuations: set[str] = set()
    expected_offset = 0
    call_arguments: Mapping[str, object] = dict(arguments)
    query_ref: str | None = None
    result_ref: str | None = None

    while True:
        response_text = await route.invoke(tool, call_arguments)
        payload = json.loads(response_text)
        if "error" in payload:
            raise ContinuityReplayError(
                f"MCP tool {tool} returned an error: {payload.get('message', payload['error'])}",
                kind=str(payload.get("error", "route_error")),
                failure_class="execution",
            )
        items = payload.get("items")
        if not isinstance(items, list):
            raise ContinuityReplayError(
                f"paginated call to {tool!r} returned no items list",
                kind="invalid_paginated_payload",
                failure_class="execution",
            )
        offset = payload.get("offset")
        page_total = payload.get("total")
        if offset != expected_offset:
            raise ContinuityReplayError(
                f"expected offset {expected_offset}, observed {offset!r}",
                kind="pagination_offset_mismatch",
                failure_class="execution",
            )
        if page_total != len(items):
            raise ContinuityReplayError(
                f"page total {page_total!r} does not match {len(items)} returned rows",
                kind="invalid_page_total",
                failure_class="execution",
            )
        current_query_ref = payload.get("query_ref")
        current_result_ref = payload.get("result_ref")
        if query_ref is None:
            query_ref, result_ref = current_query_ref, current_result_ref
        elif current_query_ref != query_ref or current_result_ref != result_ref:
            raise ContinuityReplayError(
                "query_ref/result_ref changed across continuation pages",
                kind="pagination_ref_changed",
                failure_class="execution",
            )
        page_identities: list[str] = []
        for item in items:
            if not isinstance(item, Mapping) or item_identity_field not in item:
                raise ContinuityReplayError(
                    f"page item has no {item_identity_field!r} identity field",
                    kind="missing_item_identity",
                    failure_class="execution",
                )
            identity = json.dumps(item[item_identity_field], sort_keys=True)
            if identity in seen_identities:
                raise ContinuityReplayError(
                    f"page replayed identity {identity} from an earlier page",
                    kind="duplicate_pagination_identity",
                    failure_class="execution",
                )
            seen_identities.add(identity)
            page_identities.append(identity)
        proof.item_identities.extend(page_identities)
        proof.pages.append(
            {
                "page": proof.page_count + 1,
                "offset": offset,
                "item_count": len(items),
                "query_ref": current_query_ref,
                "result_ref": current_result_ref,
                "has_continuation": payload.get("continuation") is not None,
            }
        )
        continuation = payload.get("continuation")
        next_offset = payload.get("next_offset")
        expected_offset += len(items)
        if continuation is None:
            if next_offset is not None:
                raise ContinuityReplayError(
                    f"terminal page unexpectedly carried next_offset {next_offset!r}",
                    kind="terminal_next_offset_present",
                    failure_class="execution",
                )
            break
        if not isinstance(continuation, str) or not continuation:
            raise ContinuityReplayError(
                "invalid continuation token", kind="invalid_continuation", failure_class="execution"
            )
        if not items:
            raise ContinuityReplayError(
                "empty page returned a continuation", kind="empty_progress_page", failure_class="execution"
            )
        if continuation in seen_continuations:
            raise ContinuityReplayError(
                "non-progressing continuation", kind="non_progressing_continuation", failure_class="execution"
            )
        seen_continuations.add(continuation)
        call_arguments = {"continuation": continuation}

    return proof


async def cross_check_exact_count(route: StdioMCPContinuityRoute, expression: str) -> int:
    """Independently verify the enumerated row count via ``<expression> | count``."""

    response_text = await route.invoke("query", {"expression": f"{expression} | count", "limit": 1})
    payload = json.loads(response_text)
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise ContinuityReplayError(
            "exact-count cross-check returned no single aggregate row",
            kind="invalid_exact_count_probe",
            failure_class="execution",
        )
    count = items[0].get("count")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ContinuityReplayError(
            f"exact-count cross-check returned a non-integer count: {count!r}",
            kind="invalid_exact_count_probe",
            failure_class="execution",
        )
    return count


# ── Orchestration ───────────────────────────────────────────────────────


async def run_cold_model_continuity_proof(
    *,
    archive_root: Path | None = None,
    goal: ColdModelGoal | None = None,
    keep_archive: bool = False,
    cancellation_grace_ms: int = 5_000,
) -> JSONDocument:
    """Run the joint cold-model/pagination/cancellation proof as one report.

    Defaults to a fresh, disposable synthetic archive (the deterministic
    lane); pass ``archive_root`` to reuse an existing archive whose
    ``goal.substitutions``/``forced_page_limit`` already match seeded rows.
    """

    resolved_goal = goal or DEFAULT_COLD_MODEL_GOAL
    started_ns = time.perf_counter_ns()
    workdir: TemporaryDirectory[str] | None = None

    if archive_root is None:
        workdir = TemporaryDirectory(prefix="cold-model-continuity-replay-")
        resolved_root = Path(workdir.name) / "archive"
        seed_cold_model_corpus(resolved_root, resolved_goal)
    else:
        resolved_root = archive_root

    diagnostics: list[str] = []
    plan_error: str | None = None
    pagination_error: str | None = None
    catalog_fetch: DiscoveryCatalogFetch | None = None
    plan: ColdModelPlan | None = None
    pagination_proof: PaginationProof | None = None
    cancellation_receipt: CancellationExerciseReceipt | None = None

    try:
        async with StdioMCPContinuityRoute(resolved_root) as route:
            catalog_fetch = await fetch_discovery_catalog_over_wire(route)
            if catalog_fetch.source == "in-process-registry-fallback":
                diagnostics.append(catalog_fetch.diagnostic or "wire catalog fetch failed")

            tools = route.discovery.get("tools")
            query_schema = tools.get("query") if isinstance(tools, Mapping) else None
            input_schema = query_schema.get("input_schema") if isinstance(query_schema, Mapping) else None
            query_properties = input_schema.get("properties") if isinstance(input_schema, Mapping) else None

            try:
                if not isinstance(query_properties, Mapping):
                    raise ColdModelPlanningError("live MCP discovery omitted the query tool's input schema properties")
                plan = select_cold_model_plan(resolved_goal, catalog_fetch.examples, query_properties)
            except ColdModelPlanningError as exc:
                plan_error = str(exc)
                plan = None

            if plan is not None:
                try:
                    pagination_proof = await page_and_verify(
                        route, plan.tool, plan.arguments, item_identity_field="assertion_id"
                    )
                    pagination_proof.exact_count = await cross_check_exact_count(
                        route, str(plan.arguments["expression"])
                    )
                except ContinuityReplayError as exc:
                    pagination_error = str(exc)
                    pagination_proof = None

                cancellation_receipt = await route.exercise_cancellation(
                    plan.tool, plan.arguments, grace_ms=cancellation_grace_ms
                )
    finally:
        if workdir is not None and not keep_archive:
            workdir.cleanup()

    # catalog_fetch is unconditionally assigned before anything that could
    # skip past it without raising (which would propagate out of this
    # function entirely, never reaching this line) -- only plan formulation
    # and pagination have a caught, continue-anyway failure path below.
    assert catalog_fetch is not None
    cold_catalog_status: Literal["pass", "fail"] = "pass" if catalog_fetch.source == "mcp-wire" else "fail"
    plan_status: Literal["pass", "fail"] = "pass" if plan is not None and plan_error is None else "fail"
    pagination_status: Literal["pass", "fail"] = (
        "pass"
        if pagination_proof is not None
        and pagination_proof.exact_enumeration_verified
        and pagination_proof.page_count > 1
        else "fail"
    )
    cancellation_status: Literal["pass", "fail"] = (
        "pass" if cancellation_receipt is not None and cancellation_receipt.confirmed else "fail"
    )
    overall_status: Literal["pass", "fail"] = (
        "pass"
        if plan_status == "pass"
        and pagination_status == "pass"
        and cancellation_status == "pass"
        and cold_catalog_status == "pass"
        else "fail"
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "gap": "polylogue-z9gh gap #3: paged/cancelled/cold-model MCP continuity replay",
        "goal": resolved_goal.to_payload(),
        "archive_root": str(resolved_root.resolve()) if keep_archive or archive_root is not None else None,
        "elapsed_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        "status": overall_status,
        "diagnostics": diagnostics,
        "properties": {
            "cold_model_catalog_wire_fetch": {
                "status": cold_catalog_status,
                "detail": (
                    "explain(subject='result') served the full catalog over the wire with no Python import needed."
                    if cold_catalog_status == "pass"
                    else "wire fetch overflowed the MCP response budget (polylogue-3k30); fell back to the "
                    "in-process QUERY_DISCOVERY_EXAMPLES registry -- the identical data explain is contracted "
                    "to serve -- to keep proving plan formulation, pagination, and cancellation jointly."
                ),
                "catalog_fetch": catalog_fetch.to_payload(),
            },
            "cold_model_plan_formulation": {
                "status": plan_status,
                "error": plan_error,
                "plan": plan.to_payload() if plan is not None else None,
            },
            "lossless_pagination": {
                "status": pagination_status,
                "error": pagination_error,
                "proof": pagination_proof.to_payload() if pagination_proof is not None else None,
            },
            "midflight_cancellation": {
                "status": cancellation_status,
                "receipt": cancellation_receipt.to_payload() if cancellation_receipt is not None else None,
            },
        },
    }
    return require_json_document(report, context="cold-model continuity replay report")


def _cli_goal(goal_text: str | None, topic: str | None) -> ColdModelGoal:
    if goal_text is None and topic is None:
        return DEFAULT_COLD_MODEL_GOAL
    base = DEFAULT_COLD_MODEL_GOAL
    return ColdModelGoal(
        goal_text=goal_text or base.goal_text,
        goal_keywords=base.goal_keywords,
        substitutions={"topic": topic} if topic is not None else base.substitutions,
        forced_page_limit=base.forced_page_limit,
        corpus_row_count=base.corpus_row_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--goal-text", default=None)
    parser.add_argument("--topic", default=None, help="Substituted 'topic' template parameter for the default goal")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    goal = _cli_goal(args.goal_text, args.topic)
    report = asyncio.run(
        run_cold_model_continuity_proof(archive_root=args.archive_root, goal=goal, keep_archive=args.keep_archive)
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ColdModelGoal",
    "ColdModelPlan",
    "ColdModelPlanningError",
    "DEFAULT_COLD_MODEL_GOAL",
    "DiscoveryCatalogFetch",
    "PaginationProof",
    "cross_check_exact_count",
    "fetch_discovery_catalog_over_wire",
    "main",
    "page_and_verify",
    "run_cold_model_continuity_proof",
    "seed_cold_model_corpus",
    "select_cold_model_plan",
]
