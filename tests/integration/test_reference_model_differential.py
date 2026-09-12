"""Cross-surface differential against the executable reference model.

One declared corpus is admitted to a real archive through the production
write choke point and answered independently by the reference model.  The
same generated requests then run on every public read surface, and the answers
are compared as semantic facts: which sessions matched, and how many matched
before the page was cut.

Anti-vacuity: :func:`test_seeded_divergence_is_caught` perturbs the *model's*
corpus by one semantic step and requires every surface that could observe the
step to report it.  A differential that stays green under a dropped message, a
swapped origin, a retag or a reworded message is comparing nothing, and that is
the failure this test exists to prevent.  A second guard asserts the coverage
floor: a surface that answered nothing would otherwise agree by silence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio

from tests.infra.reference_corpus_programs import (
    DIVERGENCE_KINDS,
    divergent_corpus,
    generate_corpus,
    generate_requests,
)
from tests.infra.reference_model import ModelCorpus, ModelRequest
from tests.infra.surface_differential import (
    SURFACE_NAMES,
    Divergence,
    ExpressionSurfaceSet,
    build_expression_surface_set,
    compare_to_model,
    format_divergences,
    known_divergence_for,
)

pytestmark = pytest.mark.asyncio

#: Seeds the differential runs.  Each is a whole reproduction: a failure names
#: its seed, and re-running that seed rebuilds the identical corpus and
#: request set.
CORPUS_SEEDS: tuple[int, ...] = (11, 2027, 90210)

#: Surfaces that carry every request.  MCP's session projection carries named
#: filters instead of the DSL, so it answers the translatable subset only.
FULL_COVERAGE_SURFACES: frozenset[str] = frozenset({"api", "cli", "daemon"})

#: The fewest requests MCP must carry per seed.  Every seed's request set
#: translates well above this; the floor exists so a translation that quietly
#: stopped matching anything would fail instead of passing by answering
#: nothing.
MINIMUM_MCP_REQUESTS: int = 6


@pytest.fixture
def seeded_corpus(request: pytest.FixtureRequest, db_path: Path) -> Iterator[tuple[ModelCorpus, int]]:
    seed = int(request.param)
    corpus = generate_corpus(seed, with_lineage=True)
    written = corpus.seed(db_path)
    assert len(written) == len(corpus)
    yield corpus, seed


@pytest_asyncio.fixture
async def surfaces(db_path: Path, workspace_env: dict[str, Path]) -> AsyncIterator[ExpressionSurfaceSet]:
    surface_set = build_expression_surface_set(
        archive_root=workspace_env["archive_root"],
        db_path=db_path,
        names=SURFACE_NAMES,
    )
    try:
        yield surface_set
    finally:
        await surface_set.close()


def differential_requests(corpus: ModelCorpus, seed: int) -> tuple[ModelRequest, ...]:
    """The generated requests, less the ones a ledger entry already covers."""
    return tuple(query for query in generate_requests(corpus, seed) if known_divergence_for(query) is None)


@pytest.mark.parametrize("seeded_corpus", CORPUS_SEEDS, indirect=True)
async def test_generated_requests_agree_with_the_model_on_every_surface(
    seeded_corpus: tuple[ModelCorpus, int],
    surfaces: ExpressionSurfaceSet,
) -> None:
    """Every surface answers what the declared corpus says, for every request."""
    corpus, seed = seeded_corpus
    model = corpus.reference_archive()
    requests = differential_requests(corpus, seed)
    assert requests

    answered: Counter[str] = Counter()
    divergences: list[Divergence] = []
    for query in requests:
        observed = await surfaces.execute(query)
        answered.update(facts.surface for facts in observed)
        divergences.extend(compare_to_model(model, observed))

    assert not divergences, f"seed {seed} produced {len(divergences)} divergences:\n{format_divergences(divergences)}"
    for name in sorted(FULL_COVERAGE_SURFACES):
        assert answered[name] == len(requests), f"{name} answered {answered[name]} of {len(requests)} requests"
    assert answered["mcp"] >= MINIMUM_MCP_REQUESTS, f"mcp answered only {answered['mcp']} requests"


@pytest.mark.parametrize("seeded_corpus", (CORPUS_SEEDS[0],), indirect=True)
@pytest.mark.parametrize("kind", DIVERGENCE_KINDS)
async def test_seeded_divergence_is_caught(
    seeded_corpus: tuple[ModelCorpus, int],
    surfaces: ExpressionSurfaceSet,
    kind: str,
) -> None:
    """A model one semantic step from the archive must make the comparison red.

    The archive is untouched; only the model's declared corpus is perturbed.
    That isolates the comparison from the writer: if a perturbed model still
    agrees with the surfaces, the comparison is not reading the facts it claims
    to read.

    The claim is exact rather than "something went red": for each request, the
    surfaces whose expected answer the perturbation actually changed are the
    surfaces required to report a divergence.  A perturbation no request can
    observe fails too — the request set, not the comparison, would be the thing
    that had gone vacuous.
    """
    corpus, seed = seeded_corpus
    pristine = corpus.reference_archive()
    perturbed = divergent_corpus(corpus, kind=kind).reference_archive()

    observed_by: set[str] = set()
    for query in differential_requests(corpus, seed):
        answers = await surfaces.execute(query)
        changed = {
            facts.surface
            for facts in answers
            if perturbed.query(facts.request).id_set != pristine.query(facts.request).id_set
        }
        if not changed:
            continue
        reported = {item.surface for item in compare_to_model(perturbed, answers)}
        assert reported >= changed, (
            f"perturbation {kind!r} changed the expected answer to {query.expression!r} for "
            f"{sorted(changed)}, and only {sorted(reported)} reported it"
        )
        observed_by |= changed
        if observed_by >= FULL_COVERAGE_SURFACES:
            break

    assert observed_by >= FULL_COVERAGE_SURFACES, (
        f"perturbation {kind!r} on seed {seed} changed no request's expected answer on "
        f"{sorted(FULL_COVERAGE_SURFACES - observed_by)}: the request set cannot observe it"
    )


@pytest.mark.parametrize("seeded_corpus", (CORPUS_SEEDS[0],), indirect=True)
async def test_predicate_differential_rejects_dropped_predicates_page_counts_and_replays(
    seeded_corpus: tuple[ModelCorpus, int],
    surfaces: ExpressionSurfaceSet,
) -> None:
    """The comparator rejects the three pagination/predicate escape hatches.

    The observations originate at the real API, CLI, MCP and daemon adapters;
    each mutation then models one plausible adapter or product regression.
    This is deliberately more specific than asserting that a perturbed corpus
    differs: a comparator that reduced a page to a set, or compared totals at
    page grain, would otherwise make the production routes look green.
    """
    corpus, _ = seeded_corpus
    model = corpus.reference_archive()
    origins = {session.origin for session in corpus}
    selected_origin = max(origins, key=lambda origin: sum(session.origin == origin for session in corpus))

    predicate_request = ModelRequest(
        name="predicate-control",
        expression=f"sessions where origin:{selected_origin}",
    )
    page_request = ModelRequest(
        name="page-control",
        expression=predicate_request.expression,
        limit=1,
        offset=1,
    )
    assert model.query(predicate_request).total < len(corpus)
    assert model.query(page_request).total > len(model.query(page_request).session_ids)

    predicate_answers = await surfaces.execute(predicate_request)
    page_answers = await surfaces.execute(page_request)
    assert {facts.surface for facts in predicate_answers} == set(SURFACE_NAMES)
    assert {facts.surface for facts in page_answers} == set(SURFACE_NAMES)
    assert all(facts.session_ids for facts in page_answers)

    unfiltered = model.query(ModelRequest(name="without-predicate", expression="sessions"))
    dropped_predicate = tuple(
        replace(facts, session_ids=unfiltered.session_ids, total=unfiltered.total) for facts in predicate_answers
    )
    page_sized_total = tuple(replace(facts, total=len(facts.session_ids)) for facts in page_answers)
    replayed_page = tuple(replace(facts, session_ids=facts.session_ids * 2) for facts in page_answers)

    controls = (
        ("dropped predicate", predicate_request, dropped_predicate, "ids differ"),
        ("page-sized total", page_request, page_sized_total, "model total"),
        ("replayed page", page_request, replayed_page, "more than once"),
    )
    for name, request, mutated, diagnostic in controls:
        divergences = compare_to_model(model, mutated)
        assert {item.surface for item in divergences} == set(SURFACE_NAMES), (
            f"{name} was not rejected by every real surface adapter:\n{format_divergences(divergences)}"
        )
        assert all(item.request == request for item in divergences)
        assert all(
            any(diagnostic in item.detail for item in divergences if item.surface == surface)
            for surface in SURFACE_NAMES
        )


@pytest.mark.parametrize("seeded_corpus", (CORPUS_SEEDS[0],), indirect=True)
async def test_spec_tags_postfilter_still_drops_user_tags(
    seeded_corpus: tuple[ModelCorpus, int],
    surfaces: ExpressionSurfaceSet,
) -> None:
    """Pin the one known divergence the generated differential holds out.

    ``tag:x`` and ``sessions where tag:x`` are the same filter in the DSL and
    answer differently on the same archive.  The compact spelling compiles to
    ``SessionQuerySpec.tags``, which ``list_sessions_for_spec`` postfilters
    against each hydrated ``Session.tags``; hydration carries no user tag
    assertions, so a session tagged through ``polylogue mark --tag`` is
    filtered out.  The Boolean spelling queries the tag union in SQL, and the
    CLI and daemon reach the compact clause through another reader — all three
    agree with the model, which is what makes this a lowering defect rather
    than a disagreement about what a tag is.

    When the postfilter is fixed this test fails, and the ledger entry in
    ``surface_differential.KNOWN_DIVERGENCES`` must be deleted with it.  That
    is what keeps the ledger from becoming a place to hide failures.
    """
    corpus, _ = seeded_corpus
    model = corpus.reference_archive()
    tag = next(tag for session in corpus for tag in session.tags)

    boolean = ModelRequest(name="boolean-tag", expression=f"sessions where tag:{tag}")
    compact = ModelRequest(name="compact-tag", expression=f"tag:{tag}")
    assert model.query(boolean).session_ids, "the corpus carries no session with the tag under test"

    assert not compare_to_model(model, await surfaces.execute(boolean))

    entry = known_divergence_for(compact)
    assert entry is not None
    divergences = compare_to_model(model, await surfaces.execute(compact))
    assert {item.surface for item in divergences} == set(entry.surfaces), (
        f"ledger entry {entry.name!r} claims {sorted(entry.surfaces)} but the compact clause diverged on "
        f"{sorted({item.surface for item in divergences})}:\n{format_divergences(divergences)}"
    )
