"""Session-population selection over the existing package session driver."""

from __future__ import annotations

import random
from collections.abc import Generator, Iterable

from polylogue.schemas.synthetic.core import SyntheticCorpus


class SyntheticComposition:
    """Choose providers by measured main revisions, excluding child files.

    This is a selection adapter. Every chosen corpus still generates through
    its original session driver, including its own children and adjuncts.
    """

    def __init__(self, corpora: Iterable[SyntheticCorpus]) -> None:
        self.corpora = tuple(sorted(corpora, key=lambda corpus: corpus.provider))
        if not self.corpora or len({corpus.provider for corpus in self.corpora}) != len(self.corpora):
            raise ValueError("composition requires distinct measured providers")
        self.populations: dict[str, int] = {}
        self.profile_ids: dict[str, str] = {}
        for corpus in self.corpora:
            profile = corpus.workload_profile
            provenance = profile.get("provenance") if profile is not None else None
            count = provenance.get("main_session_count") if isinstance(provenance, dict) else None
            population = provenance.get("main_session_population") if isinstance(provenance, dict) else None
            profile_id = profile.get("profile_id") if profile is not None else None
            if (
                type(count) is not int
                or count < 0
                or population != "retained-original-main-session-revisions-or-neutral-scopes"
                or not isinstance(profile_id, str)
            ):
                raise ValueError("composition needs measured main-session counts; re-infer and publish the package")
            self.populations[corpus.provider] = count
            self.profile_ids[corpus.provider] = profile_id
        if not sum(self.populations.values()):
            raise ValueError("composition has no measured main sessions")

    @property
    def weights(self) -> dict[str, float]:
        total = sum(self.populations.values())
        return {provider: count / total for provider, count in self.populations.items()}

    def iter_choices(self, *, count: int | None, seed: int) -> Generator[tuple[SyntheticCorpus, int], None, None]:
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("composition session count must be a nonnegative exact integer")
        rng = random.Random(seed)
        weights = [self.populations[corpus.provider] for corpus in self.corpora]
        index = 0
        while count is None or index < count:
            index += 1
            corpus = rng.choices(self.corpora, weights=weights, k=1)[0]
            yield corpus, rng.getrandbits(64)
