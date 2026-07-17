"""ranker:<hash> -- content-addressed aggregation models over judgment sets (rxdo.9.13, mechanism M).

A ranking is a DERIVED object: a ranker definition (engine + judgment set +
dimension + params) applied to a judgment set yields a ranking result -- items,
latent scores, judge provenance. Same content-address discipline as
``query:<hash>``: two rankings citing different ranker hashes are visibly
incomparable; re-running is cheap; the fitted model is never the truth, the
judgment rows are.

Default output is a PARTIAL order: disconnected components, cycles, ties, and
incomparable pairs stay visible. A total rank is emitted only when the
definition declares a completion/tie-break policy.

Bradley-Terry fits via iterative scaling (Zermelo's algorithm) in pure
Python -- no scipy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from polylogue.core.enums import ComparativeVerdict
from polylogue.core.hashing import hash_payload
from polylogue.insights.judgment.types import (
    ComparativeJudgment,
    PairwiseComponent,
    all_items,
    decompose_to_pairwise,
    undirected_pair_kind,
)

RankerEngine: TypeAlias = Literal["bradley_terry_mle", "win_rate", "majority"]

_SUPPORTED_COMPLETION_POLICIES: frozenset[str] = frozenset({"score_desc_stable"})


@dataclass(frozen=True, slots=True)
class RankerDefinition:
    """Content-addressed identity of one aggregation run."""

    engine: RankerEngine
    dimension: str
    judgment_ids: tuple[str, ...]
    completion_policy: str | None = None
    judge_weights: Mapping[str, float] | None = None
    """Optional per-``actor_ref`` weight multiplier (e.g. calibration-derived,
    Dawid-Skene-style) applied to every pairwise edge that judge produced."""

    def __post_init__(self) -> None:
        if self.completion_policy is not None and self.completion_policy not in _SUPPORTED_COMPLETION_POLICIES:
            raise ValueError(f"unsupported completion policy: {self.completion_policy!r}")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "dimension": self.dimension,
            "judgment_ids": sorted(set(self.judgment_ids)),
            "completion_policy": self.completion_policy,
            "judge_weights": dict(sorted((self.judge_weights or {}).items())),
        }

    @property
    def ranker_hash(self) -> str:
        return hash_payload(self.canonical_payload())

    @property
    def ranker_ref(self) -> str:
        return f"ranker:{self.ranker_hash}"


@dataclass(frozen=True, slots=True)
class RankerScore:
    item_ref: str
    score: float | None
    """``None`` means no information: the item had zero observed comparisons."""
    component_id: int
    n_observations: int


@dataclass(frozen=True, slots=True)
class PartialOrderResult:
    ranker_ref: str
    components: tuple[tuple[str, ...], ...]
    cycles: tuple[tuple[str, ...], ...]
    ties: tuple[tuple[str, str], ...]
    incomparable_pairs: tuple[tuple[str, str], ...]
    scores: tuple[RankerScore, ...]
    total_rank: tuple[str, ...] | None
    completion_policy: str | None


def _connected_components(items: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, int]:
    """Union-find over undirected edges; returns item_ref -> component id."""

    parent = {item: item for item in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    roots = sorted({find(item) for item in items})
    component_of_root = {root: index for index, root in enumerate(roots)}
    return {item: component_of_root[find(item)] for item in items}


def _detect_cycles(items: Sequence[str], directed_edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    """Bounded DFS simple-cycle detection over the directed win graph."""

    adjacency: dict[str, list[str]] = defaultdict(list)
    for winner, loser in directed_edges:
        adjacency[winner].append(loser)

    item_set = set(items)
    cycles: list[tuple[str, ...]] = []
    seen_cycle_keys: set[frozenset[str]] = set()

    def dfs(start: str, node: str, path: list[str], visited: set[str]) -> None:
        for nxt in adjacency.get(node, ()):
            if nxt == start and len(path) >= 2:
                key = frozenset(path)
                if key not in seen_cycle_keys:
                    seen_cycle_keys.add(key)
                    cycles.append(tuple(path))
                continue
            if nxt in visited or nxt not in item_set:
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(start, nxt, path, visited)
            path.pop()
            visited.discard(nxt)

    for item in sorted(items):
        dfs(item, item, [item], {item})
    return tuple(cycles)


def bradley_terry_mle(
    items: Sequence[str],
    wins: Mapping[tuple[str, str], float],
    *,
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> dict[str, float | None]:
    """Fit Bradley-Terry strengths via iterative scaling (Zermelo's algorithm).

    ``wins[(a, b)]`` = (weighted) count of times ``a`` beat ``b``. Returns
    natural-log strengths, geometric-mean-normalized to 0 for identifiability.
    Items with zero observed games return ``None`` (no information) rather
    than an arbitrary default score.
    """

    games_between: dict[tuple[str, str], float] = defaultdict(float)
    win_totals: dict[str, float] = dict.fromkeys(items, 0.0)
    for (a, b), n in wins.items():
        if n <= 0:
            continue
        win_totals[a] = win_totals.get(a, 0.0) + n
        key = (a, b) if a <= b else (b, a)
        games_between[key] += n

    pair_games: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), n in games_between.items():
        pair_games[a].append((b, n))
        pair_games[b].append((a, n))

    observed = {item for item, games in pair_games.items() if games}
    strengths: dict[str, float] = dict.fromkeys(observed, 1.0)

    for _ in range(iterations):
        new_strengths: dict[str, float] = {}
        for item in observed:
            numerator = win_totals.get(item, 0.0)
            denominator = sum(
                n_games / (strengths[item] + strengths[opponent]) for opponent, n_games in pair_games[item]
            )
            new_strengths[item] = numerator / denominator if numerator > 0 and denominator > 0 else strengths[item]
        positive = [v for v in new_strengths.values() if v > 0]
        if positive:
            scale = math.exp(sum(math.log(v) for v in positive) / len(positive))
            for item in new_strengths:
                if new_strengths[item] > 0:
                    new_strengths[item] /= scale
        max_delta = max((abs(new_strengths[item] - strengths[item]) for item in observed), default=0.0)
        strengths = new_strengths
        if max_delta < tolerance:
            break

    result: dict[str, float | None] = dict.fromkeys(items)
    for item in observed:
        value = strengths[item]
        result[item] = math.log(value) if value > 0 else None
    return result


def _win_rate_scores(items: Sequence[str], wins: Mapping[tuple[str, str], float]) -> dict[str, float | None]:
    total_wins: dict[str, float] = dict.fromkeys(items, 0.0)
    total_games: dict[str, float] = dict.fromkeys(items, 0.0)
    for (a, b), n in wins.items():
        if n <= 0:
            continue
        total_wins[a] = total_wins.get(a, 0.0) + n
        total_games[a] = total_games.get(a, 0.0) + n
        total_games[b] = total_games.get(b, 0.0) + n
    return {item: (total_wins[item] / total_games[item]) if total_games.get(item, 0.0) > 0 else None for item in items}


def fit_ranker(
    definition: RankerDefinition,
    judgments: Sequence[ComparativeJudgment],
    *,
    require_total_rank: bool = False,
) -> PartialOrderResult:
    """Aggregate a judgment set into a partial order (and optional total rank).

    Filters ``judgments`` to ``definition.judgment_ids`` and ``dimension``
    before fitting, so a definition citing a stale/wrong judgment id fits over
    an empty set rather than silently including unintended rows.

    ``require_total_rank=True`` fails closed (``ValueError``) when
    ``definition.completion_policy`` is unset -- a caller MUST declare and
    justify a tie-break/completion policy to get a fabricated total ordering
    out of what is, by default, only a partial order.
    """

    if require_total_rank and definition.completion_policy is None:
        raise ValueError("a total rank requires a declared completion_policy on the ranker definition")

    selected_ids = set(definition.judgment_ids)
    selected = [j for j in judgments if j.judgment_id in selected_ids and j.dimension == definition.dimension]
    items = all_items(selected)

    directed_edges: list[tuple[str, str]] = []
    weighted_wins: dict[tuple[str, str], float] = defaultdict(float)
    ties: set[tuple[str, str]] = set()
    incomparable: set[tuple[str, str]] = set()
    connectivity_edges: list[tuple[str, str]] = []

    for judgment in selected:
        weight = (definition.judge_weights or {}).get(judgment.judge.actor_ref, 1.0)
        pairwise_components: Sequence[PairwiseComponent] = decompose_to_pairwise(judgment)
        for component in pairwise_components:
            weighted_wins[(component.winner_ref, component.loser_ref)] += component.weight * weight
            directed_edges.append((component.winner_ref, component.loser_ref))
            connectivity_edges.append((component.winner_ref, component.loser_ref))
        pair = undirected_pair_kind(judgment)
        if pair is not None:
            connectivity_edges.append(pair)
            if judgment.verdict is ComparativeVerdict.TIE:
                ties.add(pair)
            elif judgment.verdict is ComparativeVerdict.INCOMPARABLE:
                incomparable.add(pair)
        elif not pairwise_components:
            # abstain / insufficient-evidence (pairwise or n-wise): every item
            # in the attempted comparison connects to every other -- the
            # comparison happened, it just yielded no directed or tie/
            # incomparable signal. Never treated as a component/score edge.
            for i in range(len(judgment.items)):
                for j in range(i + 1, len(judgment.items)):
                    a, b = judgment.items[i], judgment.items[j]
                    connectivity_edges.append((a, b) if a <= b else (b, a))

    component_of = _connected_components(items, connectivity_edges)
    cycles = _detect_cycles(items, directed_edges)

    if definition.engine == "bradley_terry_mle":
        raw_scores = bradley_terry_mle(items, weighted_wins)
    elif definition.engine == "win_rate" or definition.engine == "majority":
        raw_scores = _win_rate_scores(items, weighted_wins)
    else:  # pragma: no cover - exhaustive Literal, mypy-enforced
        raise ValueError(f"unsupported ranker engine: {definition.engine!r}")

    n_observations: dict[str, int] = dict.fromkeys(items, 0)
    for a, b in connectivity_edges:
        n_observations[a] = n_observations.get(a, 0) + 1
        n_observations[b] = n_observations.get(b, 0) + 1

    scores = tuple(
        RankerScore(
            item_ref=item,
            score=raw_scores.get(item),
            component_id=component_of[item],
            n_observations=n_observations.get(item, 0),
        )
        for item in items
    )

    components_grouped: dict[int, list[str]] = defaultdict(list)
    for item, component_id in component_of.items():
        components_grouped[component_id].append(item)
    component_groups = tuple(tuple(sorted(members)) for _, members in sorted(components_grouped.items()))

    total_rank: tuple[str, ...] | None = None
    if definition.completion_policy == "score_desc_stable":
        score_by_item: dict[str, float] = {
            s.item_ref: (s.score if s.score is not None else float("-inf")) for s in scores
        }
        total_rank = tuple(
            sorted(
                items,
                key=lambda item: (component_of[item], -score_by_item[item], item),
            )
        )

    return PartialOrderResult(
        ranker_ref=definition.ranker_ref,
        components=component_groups,
        cycles=cycles,
        ties=tuple(sorted(ties)),
        incomparable_pairs=tuple(sorted(incomparable)),
        scores=scores,
        total_rank=total_rank,
        completion_policy=definition.completion_policy,
    )


__all__ = [
    "PartialOrderResult",
    "RankerDefinition",
    "RankerEngine",
    "RankerScore",
    "bradley_terry_mle",
    "fit_ranker",
]
