"""Stratified rebuild-cost benchmark (polylogue-623q follow-up).

The only way to know whether a rebuild-path change helped used to be running
a full multi-hour rebuild against the live archive. This module builds a
cheap substitute: stratify the raw population by (origin x byte-size decile),
synthesize a small representative sample per stratum, drive it through the
REAL rebuild engine (``polylogue.maintenance.rebuild_index.
rebuild_index_from_source_sync`` -- the same function the offline CLI and
daemon bulk-rebuild call, no reimplementation), measure seconds-per-raw at
each stratum's characteristic size, and extrapolate to the full population
using the population's real per-stratum counts.

Two cost regimes exist in this archive's raw population and neither should be
assumed to dominate a priori:

- byte-bound: a few thousand huge raws (multi-MB Codex rollouts) where cost
  scales with payload bytes (parse/decode + writer throughput).
- count-bound: tens of thousands of small raws where FIXED per-raw overhead
  (transaction bookkeeping, blob open, census receipt, index inserts) matters
  more than their few KB of payload.

Sampling at each stratum's OWN characteristic mean size and then scaling by
seconds-per-raw naturally blends both regimes without needing to fit which
one applies -- a stratum's measured seconds-per-raw already reflects
whatever mix of fixed and byte-proportional cost governs raws of that shape.

``PopulationSnapshot`` below is a captured, aggregate-only (counts and byte
sums, never content) description of the live archive's raw population,
recorded because the live archive is not available to CI/cloud runs and
carries private content this public repo must never expose. Point
``collect_population_strata`` at a real ``source.db`` (read-only) to refresh
it from a live archive -- the snapshot embedded here does not need to be
re-derived to run the model; it is a fixed reference point analogous to
``PopulationSnapshot.captured_at``.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

# ---------------------------------------------------------------------------
# Population strata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stratum:
    """One (origin, byte-size decile) population bucket.

    ``chain_fraction`` and ``ambiguous_fraction`` (polylogue-o56w follow-up)
    are the structural-mix corrections this module was missing: a stratum
    built from ``count``/``total_bytes`` alone synthesizes only
    "first-time, unambiguous" raws, which never exercises the real
    engine's cohort-arbitration machinery (``replay.classify_cohort``,
    ``replay.adoptable_check``, ``membership.candidates/project/classify``)
    -- exactly the untimed-apply "dark matter" this follow-up targets. Both
    default to ``0.0`` so existing hand-built ``Stratum`` instances (tests,
    ad hoc strata) keep the prior singleton-only corpus shape.

    - ``chain_fraction``: fraction of the stratum's sample raws that belong
      to a depth-2 byte-growth revision chain (one older, byte-prefix
      member plus the head that supersedes it) instead of being a
      standalone raw. Mirrors a session transcript re-acquired after it
      grew (Claude Code/Codex resume). Drives
      ``classify_untyped_full_revision_groups`` and the real
      ``revision_replay`` cohort-replay path; the older member ends up
      permanently ``RawRevisionAuthority.QUARANTINED`` by construction,
      exactly like the live archive's non-head chain raws.
    - ``ambiguous_fraction``: fraction of the stratum's sample raws
      organized into duplicate-content pairs -- two raws sharing one
      session identity with genuinely divergent (non-prefix) content, the
      shape ``classify_membership_revisions`` cannot arbitrate. Drives the
      ``membership_replay``/``membership.*`` stages and produces a
      genuine, non-zero ``quarantined_raw_count`` (both members go
      unclaimed), the same outcome a live duplicate/conflicting export
      produces.

    Both fractions were measured against the live archive's
    ``source.db`` (aggregate-only, read-only, no content -- see
    ``_origin_structural_fractions``) and are DOCUMENTED APPROXIMATIONS,
    not an exact replay of the live population's revision graph: real
    chains range up to depth 16+ (this module always uses depth 2, the
    modal case) and ``ambiguous_fraction`` is a residual estimate
    (``revision_authority='quarantined' fraction - chain-involved
    fraction``), not a directly measured ambiguous-arbitration rate --
    the live archive does not persist *why* a raw never got promoted.
    """

    label: str
    provider: Provider
    count: int
    total_bytes: int
    chain_fraction: float = 0.0
    ambiguous_fraction: float = 0.0

    @property
    def mean_bytes(self) -> int:
        return max(1, round(self.total_bytes / self.count)) if self.count else 0


def _origin_structural_fractions(conn: sqlite3.Connection, origin: str) -> tuple[float, float]:
    """Read-only: (chain_fraction, ambiguous_fraction) for one origin.

    ``chain_fraction`` is the fraction of the origin's raws that sit in a
    ``logical_source_key`` group with >=2 members (a proven byte-growth
    revision chain) -- i.e. NOT the lone raw for its logical identity.
    ``ambiguous_fraction`` is a residual: the fraction of raws whose
    *final* persisted ``revision_authority`` is ``'quarantined'`` minus the
    chain-involved fraction, floored at 0. ``raw_sessions.revision_authority``
    does not record *why* a raw was never promoted, so this treats
    "quarantined but not in a multi-member chain" as the closest available
    proxy for genuine cross-acquisition arbitration loss (duplicate/
    conflicting exports) -- see the ``Stratum`` docstring for the caveat.
    Returns ``(0.0, 0.0)`` for an origin with no raws.
    """
    total = int(conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE origin = ?", (origin,)).fetchone()[0])
    if not total:
        return 0.0, 0.0
    quarantined = int(
        conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE origin = ? AND revision_authority = 'quarantined'", (origin,)
        ).fetchone()[0]
    )
    group_sizes = [
        int(r[0])
        for r in conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE origin = ? AND logical_source_key IS NOT NULL "
            "GROUP BY logical_source_key",
            (origin,),
        )
    ]
    chain_involved = sum(size for size in group_sizes if size >= 2)
    chain_fraction = chain_involved / total
    ambiguous_fraction = max(0.0, quarantined / total - chain_fraction)
    return chain_fraction, ambiguous_fraction


def collect_population_strata(
    source_db: Path, *, deciles_by_origin: Sequence[str] = ("codex-session", "claude-code-session")
) -> list[Stratum]:
    """Read-only: stratify a live archive's raw_sessions by origin x byte-weighted decile.

    Never reads payload content -- only ``origin``/``blob_size`` aggregate
    columns, plus (polylogue-o56w follow-up) the aggregate
    ``logical_source_key``/``revision_authority`` columns used to derive each
    origin's ``chain_fraction``/``ambiguous_fraction`` (see
    ``_origin_structural_fractions`` -- still counts only, never raw
    content). Origins listed in ``deciles_by_origin`` (the byte-dominant
    ones) are split into 10 byte-weighted deciles each; every other origin is
    pooled into one "long tail" stratum per origin group, since they are
    small enough in aggregate bytes that within-origin size variance does not
    materially change the wall-clock projection. The long-tail bucket's own
    structural fractions are left at 0.0: those origins showed ~0 measured
    chain involvement live (see module docstring), and their high raw
    ``revision_authority='quarantined'`` rate could not be attributed to a
    verified cost-bearing mechanism within this pass -- a documented gap, not
    a silent omission.
    """
    strata: list[Stratum] = []
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        origins = [str(r[0]) for r in conn.execute("SELECT DISTINCT origin FROM raw_sessions")]
        tail_count = 0
        tail_bytes = 0
        for origin in origins:
            provider = _provider_for_origin(origin)
            sizes = [int(r[0]) for r in conn.execute("SELECT blob_size FROM raw_sessions WHERE origin = ?", (origin,))]
            if not sizes:
                continue
            if origin not in deciles_by_origin:
                tail_count += len(sizes)
                tail_bytes += sum(sizes)
                continue
            chain_fraction, ambiguous_fraction = _origin_structural_fractions(conn, origin)
            sizes.sort(reverse=True)
            total = sum(sizes)
            bucket_target = total / 10 if total else 0
            bucket_counts = [0] * 10
            bucket_bytes = [0] * 10
            acc = 0
            bi = 0
            for size in sizes:
                acc += size
                bucket_counts[bi] += 1
                bucket_bytes[bi] += size
                if bucket_target and acc >= bucket_target * (bi + 1) and bi < 9:
                    bi += 1
            for decile, (count, nbytes) in enumerate(zip(bucket_counts, bucket_bytes, strict=True)):
                if count == 0:
                    continue
                strata.append(
                    Stratum(
                        label=f"{origin}/d{decile}",
                        provider=provider,
                        count=count,
                        total_bytes=nbytes,
                        chain_fraction=chain_fraction,
                        ambiguous_fraction=ambiguous_fraction,
                    )
                )
        if tail_count:
            strata.append(
                Stratum(
                    label="long-tail/other-origins", provider=Provider.CODEX, count=tail_count, total_bytes=tail_bytes
                )
            )
    return strata


def _provider_for_origin(origin: str) -> Provider:
    # Synthesis only supports origins with a validated realistic payload
    # generator below (Codex, Claude Code). Every other origin is folded into
    # the long-tail Codex-shaped stratum by collect_population_strata's
    # deciles_by_origin default -- see module docstring's "documented
    # simplification".
    if origin == "claude-code-session":
        return Provider.CLAUDE_CODE
    return Provider.CODEX


#: Aggregate-only snapshot of the live archive's raw population, captured
#: 2026-07-30 via a read-only ``mode=ro`` connection (counts/bytes only, no
#: content -- see ``.claude/CLAUDE.md`` "repo is PUBLIC" constraint). Refresh
#: with ``collect_population_strata`` against a live ``source.db`` when the
#: archive shape has materially changed.
POPULATION_SNAPSHOT_CAPTURED_AT = datetime(2026, 7, 30, tzinfo=timezone.utc)

#: Per-origin structural fractions (polylogue-o56w follow-up), captured
#: 2026-07-31 via the same read-only connection as the byte-size snapshot
#: above using ``_origin_structural_fractions`` -- see ``Stratum``'s
#: docstring for what each number means and its documented approximation.
#: ``codex-session``: chain_fraction=0.4668, ambiguous_fraction=0.1015.
#: ``claude-code-session``: chain_fraction=0.0909, ambiguous_fraction=0.0985.
#: long-tail origins measured ~0 chain involvement and their high raw
#: ``quarantined`` rate could not be attributed to a verified cost-bearing
#: mechanism in this pass -- left at 0.0/0.0, a documented gap (see module
#: docstring "Residual error").
_CODEX_CHAIN_FRACTION = 0.4668
_CODEX_AMBIGUOUS_FRACTION = 0.1015
_CLAUDE_CODE_CHAIN_FRACTION = 0.0909
_CLAUDE_CODE_AMBIGUOUS_FRACTION = 0.0985

POPULATION_SNAPSHOT: tuple[Stratum, ...] = (
    Stratum(
        "codex-session/d0",
        Provider.CODEX,
        16,
        7_201_200_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d1",
        Provider.CODEX,
        19,
        7_344_300_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d2",
        Provider.CODEX,
        24,
        6_929_000_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d3",
        Provider.CODEX,
        38,
        7_139_300_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d4",
        Provider.CODEX,
        49,
        7_178_300_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d5",
        Provider.CODEX,
        111,
        7_060_600_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d6",
        Provider.CODEX,
        234,
        7_128_400_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d7",
        Provider.CODEX,
        428,
        7_146_500_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d8",
        Provider.CODEX,
        715,
        7_131_400_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "codex-session/d9",
        Provider.CODEX,
        7508,
        7_136_700_000,
        chain_fraction=_CODEX_CHAIN_FRACTION,
        ambiguous_fraction=_CODEX_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d0",
        Provider.CLAUDE_CODE,
        5,
        2_266_100_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d1",
        Provider.CLAUDE_CODE,
        32,
        2_073_200_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d2",
        Provider.CLAUDE_CODE,
        51,
        2_111_600_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d3",
        Provider.CLAUDE_CODE,
        93,
        2_155_900_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d4",
        Provider.CLAUDE_CODE,
        139,
        2_136_200_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d5",
        Provider.CLAUDE_CODE,
        205,
        2_150_000_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d6",
        Provider.CLAUDE_CODE,
        339,
        2_147_700_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d7",
        Provider.CLAUDE_CODE,
        745,
        2_145_500_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d8",
        Provider.CLAUDE_CODE,
        2142,
        2_146_600_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    Stratum(
        "claude-code-session/d9",
        Provider.CLAUDE_CODE,
        15954,
        2_147_700_000,
        chain_fraction=_CLAUDE_CODE_CHAIN_FRACTION,
        ambiguous_fraction=_CLAUDE_CODE_AMBIGUOUS_FRACTION,
    ),
    # Long-tail origins (chatgpt-export, claude-ai-export, aistudio-drive,
    # antigravity-session, hermes-session, ...): measured ~0 chain
    # involvement live; see module-level structural-fraction note above for
    # why ambiguous_fraction is left at 0.0 rather than guessed.
    Stratum("long-tail/other-origins", Provider.CODEX, 12725, 6_439_959_524),
)

#: The one real, measured full-corpus rebuild wall-clock -- the acceptance
#: criterion. A model that cannot reproduce this within a stated margin is
#: not trustworthy for evaluating future changes (operator directive).
CALIBRATION_WALL_S = 4 * 3600 + 20 * 60  # 4h20m
CALIBRATION_RAW_COUNT = 41_363
CALIBRATION_TOTAL_BYTES = round(92.4 * 1024**3)


# ---------------------------------------------------------------------------
# Synthetic per-stratum corpus
# ---------------------------------------------------------------------------

_ENVELOPE_OVERHEAD_BYTES = 220


def _codex_payload(index: int, *, target_bytes: int, variant: str = "") -> bytes:
    session_meta = (
        json.dumps(
            {"type": "session_meta", "payload": {"id": f"cost-model-{index:06d}", "timestamp": "2026-06-01T00:00:00Z"}},
            separators=(",", ":"),
        )
        + "\n"
    )
    marker = f"variant-{variant}-" if variant else ""
    text_len = max(1, target_bytes - len(session_meta) - _ENVELOPE_OVERHEAD_BYTES - len(marker))
    text = f"cost-model-payload-{index:06d}-{marker}" + ("x" * text_len)
    response_item = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "one",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return (session_meta + response_item).encode()


def _claude_code_payload(index: int, *, target_bytes: int, variant: str = "") -> bytes:
    session_id = str(uuid.UUID(int=index, version=4))
    base_ts = 1_700_000_000.0 + index * 60
    marker = f"variant-{variant}-" if variant else ""
    pad_len = max(1, target_bytes - 2 * _ENVELOPE_OVERHEAD_BYTES - len(marker))
    lines: list[str] = []
    for turn, role in enumerate(("user", "assistant")):
        record = {
            "type": role,
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_OID, f"{index}:{variant}:{turn}")),
            "parentUuid": (str(uuid.uuid5(uuid.NAMESPACE_OID, f"{index}:{variant}:0")) if (index or turn) else None),
            "sessionId": session_id,
            "message": {
                "role": role,
                "content": [{"type": "text", "text": (marker + "x" * pad_len) if turn == 0 else "ack"}],
            },
            "timestamp": datetime.fromtimestamp(base_ts + turn, tz=timezone.utc).isoformat(),
        }
        lines.append(json.dumps(record, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


def _payload_for_provider(provider: Provider, index: int, *, target_bytes: int, variant: str = "") -> bytes:
    if provider is Provider.CLAUDE_CODE:
        return _claude_code_payload(index, target_bytes=target_bytes, variant=variant)
    return _codex_payload(index, target_bytes=target_bytes, variant=variant)


def _chain_member_payloads(provider: Provider, index: int, *, target_bytes: int) -> tuple[bytes, bytes]:
    """(older, head) byte-growth chain pair -- older is a strict byte-prefix of head.

    Mirrors a growing session transcript file re-acquired across two
    captures (Claude Code/Codex resume): the same physical shape
    ``classify_untyped_full_revision_groups`` -> ``classify_historical_full_
    revision_streams`` proves as a ``byte_proven`` chain without ever
    independently parsing the older member. The older member is deliberately
    NOT required to be independently valid JSON/JSONL -- the real engine
    never parses it either, exactly like a live truncated-mid-line partial
    capture.
    """
    head = _payload_for_provider(provider, index, target_bytes=target_bytes)
    older_size = max(1, len(head) // 2)
    return head[:older_size], head


def _ambiguous_pair_payloads(provider: Provider, index: int, *, target_bytes: int) -> tuple[bytes, bytes]:
    """Two raws sharing one session identity with genuinely divergent content.

    Neither is a byte-prefix of the other, so ``classify_membership_
    revisions`` cannot arbitrate a winner -- both members go unclaimed
    (``decision='ambiguous'``), exactly the shape a duplicate/conflicting
    export of one session produces live.
    """
    return (
        _payload_for_provider(provider, index, target_bytes=target_bytes, variant="a"),
        _payload_for_provider(provider, index, target_bytes=target_bytes, variant="b"),
    )


def build_stratum_sample_corpus(archive_root: Path, stratum: Stratum, *, sample_n: int) -> list[str]:
    """Write ``sample_n`` synthetic raws representative of ``stratum``.

    Fresh archive root; the caller owns cleanup. Every raw is still sized at
    ``stratum.mean_bytes`` (or a byte-prefix of it, for an older chain
    member), preserving the existing byte-size-driven timing behavior. On
    top of that, ``stratum.ambiguous_fraction`` and ``stratum.chain_fraction``
    (polylogue-o56w follow-up) carve out a structurally representative
    share of the sample into ambiguous-duplicate pairs and byte-growth
    chains instead of every raw being a standalone, unambiguous first-time
    session -- see ``Stratum``'s docstring for why this matters (it is what
    makes ``replay.classify_cohort``/``replay.adoptable_check``/
    ``membership.candidates``/``membership.project``/``membership.classify``
    fire at all in this harness). Allocation order: ambiguous pairs first
    (2 raws each), then chains (2 raws each) from what remains, then plain
    singleton raws for the rest -- all deterministic, no randomness, so a
    given ``(stratum, sample_n)`` always yields byte-identical corpora.
    """
    initialize_active_archive_root(archive_root)
    raw_ids: list[str] = []
    index = 0

    def next_index() -> int:
        nonlocal index
        index += 1
        return index

    n_ambiguous_raws = 2 * (round(sample_n * stratum.ambiguous_fraction / 2) if stratum.ambiguous_fraction else 0)
    n_ambiguous_raws = min(n_ambiguous_raws, sample_n - (sample_n % 2))
    pool = sample_n - n_ambiguous_raws
    n_chain_raws = 2 * (round(pool * stratum.chain_fraction / 2) if stratum.chain_fraction else 0)
    n_chain_raws = min(n_chain_raws, pool - (pool % 2))
    n_single_raws = pool - n_chain_raws

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for _ in range(n_ambiguous_raws // 2):
            unit = next_index()
            payload_a, payload_b = _ambiguous_pair_payloads(stratum.provider, unit, target_bytes=stratum.mean_bytes)
            for suffix, payload in (("a", payload_a), ("b", payload_b)):
                raw_ids.append(
                    archive.write_raw_payload(
                        provider=stratum.provider,
                        payload=payload,
                        source_path=f"cost-model/{stratum.label}/amb-{unit:06d}-{suffix}.jsonl",
                        acquired_at_ms=len(raw_ids) + 1,
                    )
                )
        for _ in range(n_chain_raws // 2):
            unit = next_index()
            older_payload, head_payload = _chain_member_payloads(
                stratum.provider, unit, target_bytes=stratum.mean_bytes
            )
            # Same source_path for both members -- classify_untyped_full_
            # revision_groups groups candidates by source_path (see its
            # docstring); the older (smaller) acquisition must be written
            # first so acquired_at_ms orders them chronologically.
            source_path = f"cost-model/{stratum.label}/chain-{unit:06d}.jsonl"
            raw_ids.append(
                archive.write_raw_payload(
                    provider=stratum.provider,
                    payload=older_payload,
                    source_path=source_path,
                    acquired_at_ms=len(raw_ids) + 1,
                )
            )
            raw_ids.append(
                archive.write_raw_payload(
                    provider=stratum.provider,
                    payload=head_payload,
                    source_path=source_path,
                    acquired_at_ms=len(raw_ids) + 1,
                )
            )
        for _ in range(n_single_raws):
            unit = next_index()
            payload = _payload_for_provider(stratum.provider, unit, target_bytes=stratum.mean_bytes)
            raw_ids.append(
                archive.write_raw_payload(
                    provider=stratum.provider,
                    payload=payload,
                    source_path=f"cost-model/{stratum.label}/{unit:06d}.jsonl",
                    acquired_at_ms=len(raw_ids) + 1,
                )
            )
    return raw_ids


def default_sample_n(stratum: Stratum, *, target_sample_bytes: int = 2_000_000, min_n: int = 3, max_n: int = 40) -> int:
    """More samples for small/count-bound strata, fewer for byte-bound whales."""
    if stratum.mean_bytes <= 0:
        return min(stratum.count, min_n)
    raw_target = max(min_n, round(target_sample_bytes / stratum.mean_bytes))
    return max(1, min(stratum.count, min(max_n, raw_target)))


# ---------------------------------------------------------------------------
# Stage-proportion fidelity (polylogue-o56w follow-up)
# ---------------------------------------------------------------------------
#
# The population-size/byte projection above (``predicted_wall_s``,
# ``calibration_ratio``) answers "how long will this take". It says nothing
# about WHETHER the harness is spending its time the way the real engine
# does -- a model that matches total wall-clock by getting one stage 3x too
# slow and another 3x too fast is not trustworthy for evaluating a change
# that touches only one of those stages. This section makes that mix
# checkable: it buckets a pass's ``stage_timings_s`` into the same four
# broad categories the real 4h22m rebuild's receipt decomposes into, and
# compares the harness's own proportions against that receipt's.

#: The one real, measured full-corpus rebuild's stage decomposition -- the
#: fidelity acceptance criterion (operator directive: a harness is
#: trustworthy when its per-stage PROPORTIONS match this within a stated
#: tolerance, not when absolute times match). Computed directly from
#: ``/realm/db/polylogue/.index-rebuild-transactions/857984cb-b4cc-4537-
#: b0fb-eae89ca3fa96.receipts/pass-000000.json`` (private-archive path, not
#: readable from this public repo -- these are the derived aggregate
#: numbers only, no raw content):
#:
#: - ``total_wall_s`` = transaction.updated_at_ms - transaction.created_at_ms
#:   = 15724.569s (4h22m04.6s, matches the operator's "4h22m" figure).
#: - ``parse_s`` = stage_timings_s["census"] + ["spill_load"] = 4032.184s.
#: - ``timed_apply_s`` = stage_timings_s["revision_replay.index_parsed_write"]
#:   + ["membership_replay.index_parsed_write"] = 2453.585 + 979.514 =
#:   3433.099s -- the only apply-phase cost the PRE-#3469 harness could see
#:   at all (its corpus never produced a membership_replay.* key).
#: - ``untimed_apply_s`` = apply_s - timed_apply_s = 8601.253 - 3433.099 =
#:   5168.154s -- the "33% dark matter" this follow-up targets: cohort
#:   classification, membership candidate/project/classify, and the commit
#:   storm PR #3469 already fixed. This receipt predates PR #3469's
#:   ``replay.*``/``membership.*`` instrumentation, so none of it is
#:   individually named here -- it is exactly the gap that instrumentation
#:   now decomposes on the NEXT real rebuild (bead polylogue-o56w).
#: - ``terminal_s`` = total_wall_s - stage_timings_s["total"] = 15724.569 -
#:   12633.437 = 3091.132s -- post-pass repopulate/insights/readiness/
#:   promote, paid ONCE for the whole archive (not per-stratum -- see the
#:   "Residual error" note on ``PredictedRun.compare_to_real_run``).
REAL_RUN_TOTAL_WALL_S = 15724.569
REAL_RUN_STAGE_SECONDS: dict[str, float] = {
    "parse": 4032.184306,
    "timed_apply": 3433.098599,
    "untimed_apply": 5168.154311,
    "terminal": 3091.132,
}
REAL_RUN_STAGE_PROPORTIONS: dict[str, float] = {
    name: seconds / REAL_RUN_TOTAL_WALL_S for name, seconds in REAL_RUN_STAGE_SECONDS.items()
}

#: Order used everywhere a stage-proportion table is printed.
_STAGE_BUCKET_ORDER: tuple[str, ...] = ("parse", "timed_apply", "untimed_apply", "terminal")


def bucket_stage_seconds(
    replay: Mapping[str, object], terminal_timings_s: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Bucket one pass's stage timings into (parse, timed_apply, untimed_apply, terminal).

    Mirrors exactly how ``REAL_RUN_STAGE_SECONDS`` above was derived from the
    real receipt, so the two are directly comparable:

    - ``parse`` = ``parse_s`` (``census`` + ``spill_load``), read straight off
      the replay dict (already computed by ``split_parse_and_apply_seconds``).
    - ``timed_apply`` = sum of every ``*.index_parsed_write`` stage key --
      the only apply-phase cost a corpus with no revision chains or
      membership cohorts can ever produce (pre-#3469-follow-up harness).
    - ``untimed_apply`` = ``apply_s`` - ``timed_apply`` (never negative) --
      everything else charged against the pass's own wall-clock ``total``:
      cohort classification, membership arbitration, census receipt commits.
    - ``terminal`` = sum of a REBUILD RECEIPT's (not the backfill replay
      dict's) ``timings_s`` -- the post-pass repopulate/insights/readiness/
      promote stages instrumented by PR #3469. ``0.0`` if not supplied.
    """
    stage_timings_s = cast("Mapping[str, float]", replay.get("stage_timings_s", {}))
    parse_s = float(cast("float", replay.get("parse_s", 0.0)))
    apply_s = float(cast("float", replay.get("apply_s", 0.0)))
    timed_apply_s = sum(v for k, v in stage_timings_s.items() if k.endswith(".index_parsed_write"))
    untimed_apply_s = max(0.0, apply_s - timed_apply_s)
    terminal_s = sum(terminal_timings_s.values()) if terminal_timings_s else 0.0
    return {
        "parse": parse_s,
        "timed_apply": timed_apply_s,
        "untimed_apply": untimed_apply_s,
        "terminal": terminal_s,
    }


def stage_proportions(stage_seconds: Mapping[str, float]) -> dict[str, float]:
    """Normalize a bucket-seconds dict into fractions of its own total."""
    total = sum(stage_seconds.get(name, 0.0) for name in _STAGE_BUCKET_ORDER)
    if total <= 0:
        return dict.fromkeys(_STAGE_BUCKET_ORDER, 0.0)
    return {name: stage_seconds.get(name, 0.0) / total for name in _STAGE_BUCKET_ORDER}


def stage_proportion_table(
    measured: Mapping[str, float], reference: Mapping[str, float] = REAL_RUN_STAGE_PROPORTIONS
) -> str:
    """Render a measured-vs-real-run proportion table, one row per bucket."""
    lines = [f"{'stage':14s} {'measured':>10s} {'real_run':>10s} {'delta_pp':>9s}"]
    for name in _STAGE_BUCKET_ORDER:
        m = measured.get(name, 0.0)
        r = reference.get(name, 0.0)
        lines.append(f"{name:14s} {m * 100:9.1f}% {r * 100:9.1f}% {(m - r) * 100:+8.1f}pp")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StratumMeasurement:
    """Two-point regression result for one stratum.

    A single-sample measurement conflates two different costs: the FIXED
    one-time overhead ``rebuild_index_from_source_sync`` pays per PASS
    (generation bootstrap, embeddings/FTS bulk-repopulate, promote,
    readiness checks -- paid once no matter how many raws the pass
    replays) and the MARGINAL cost per raw actually replayed. Dividing a
    single sample's wall-clock by its (small) sample_n and multiplying by
    the population count multiplies the fixed cost too, which the real
    full rebuild pays only ONCE across the whole population -- this was
    the source of a measured 1.78x over-prediction in the first version of
    this model (see polylogue-623q notes).

    Fitting two points (``n1`` < ``n2`` raws, same stratum, fresh archive
    each) separates the two: ``marginal_s_per_raw`` is the slope
    ``(wall_s2 - wall_s1) / (n2 - n1)``; ``fixed_s`` is the intercept.
    Population extrapolation then multiplies ONLY the marginal term
    (``predicted_wall_s``) -- ``fixed_s`` is reported for transparency but
    deliberately not added into any per-stratum or population total: the
    real full rebuild runs as ONE pass, so its one-time fixed cost should
    be counted once across the whole run, not once per stratum sample (and
    this model does not attempt that separate accounting -- see the
    module-level report footer).
    """

    stratum: Stratum
    n1: int
    n2: int
    wall_s1: float
    wall_s2: float
    fixed_s: float
    marginal_s_per_raw: float
    sample_bytes: int
    parse_s: float
    apply_s: float
    regression_valid: bool
    #: Stage-bucket seconds (``bucket_stage_seconds``) from the LARGER
    #: sample's pass (``n2``, or ``n1`` when the regression couldn't split)
    #: -- used for the stage-proportion fidelity table, never for the
    #: wall-clock population projection above.
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def predicted_wall_s(self) -> float:
        """Population extrapolation using ONLY the marginal per-raw term."""
        return max(0.0, self.marginal_s_per_raw) * self.stratum.count


def _run_one_rebuild_pass(
    archive_root: Path, stratum: Stratum, n: int
) -> tuple[float, dict[str, object], dict[str, float]]:
    build_stratum_sample_corpus(archive_root, stratum, sample_n=n)
    prior_env = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    try:
        started = time.perf_counter()
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=archive_root, promote=True, raw_batch_size=max(n, 1))
        )
        wall_s = time.perf_counter() - started
    finally:
        if prior_env is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = prior_env
    assert receipt.status == "replayed", f"stratum {stratum.label}: unexpected receipt status {receipt.status!r}"
    return wall_s, receipt.replay, dict(receipt.timings_s)


def two_point_sample_sizes(stratum: Stratum) -> tuple[int, int]:
    """(n1, n2) sample sizes for the regression -- n2 > n1 when population allows."""
    n1 = default_sample_n(stratum)
    n2 = min(stratum.count, max(n1 + 3, n1 * 4))
    return n1, n2


def measure_stratum(
    archive_root: Path, stratum: Stratum, *, sample_sizes: tuple[int, int] | None = None
) -> StratumMeasurement:
    """Two-point regression: replay the same stratum at two sample sizes to
    separate fixed per-pass overhead from marginal per-raw cost.

    Each sample gets a fresh scratch archive (``archive_root / "n1"`` /
    ``"n2"``) so neither pass's index/generation state leaks into the other.
    """
    n1, n2 = sample_sizes if sample_sizes is not None else two_point_sample_sizes(stratum)
    wall_s1, replay1, terminal1 = _run_one_rebuild_pass(archive_root / "n1", stratum, n1)
    regression_valid = n2 > n1
    if regression_valid:
        wall_s2, replay2, terminal2 = _run_one_rebuild_pass(archive_root / "n2", stratum, n2)
        marginal_s_per_raw = (wall_s2 - wall_s1) / (n2 - n1)
        fixed_s = wall_s1 - marginal_s_per_raw * n1
        replay, terminal = replay2, terminal2
    else:
        # Population too small to split (n2 == n1): fall back to treating the
        # whole single-sample wall-clock as marginal cost, matching the prior
        # (known-biased) behavior for this stratum only. These strata are a
        # tiny fraction of the population by construction (see
        # POPULATION_SNAPSHOT) so the bias this reintroduces is bounded.
        wall_s2 = wall_s1
        marginal_s_per_raw = wall_s1 / n1 if n1 else 0.0
        fixed_s = 0.0
        replay, terminal = replay1, terminal1

    parse_s = float(cast("float", replay.get("parse_s", 0.0)))
    apply_s = float(cast("float", replay.get("apply_s", 0.0)))
    return StratumMeasurement(
        stratum=stratum,
        n1=n1,
        n2=n2,
        wall_s1=wall_s1,
        wall_s2=wall_s2,
        fixed_s=fixed_s,
        marginal_s_per_raw=marginal_s_per_raw,
        sample_bytes=n2 * stratum.mean_bytes,
        parse_s=parse_s,
        apply_s=apply_s,
        regression_valid=regression_valid,
        stage_seconds=bucket_stage_seconds(replay, terminal),
    )


@dataclass(slots=True)
class PredictedRun:
    measurements: list[StratumMeasurement] = field(default_factory=list)

    @property
    def total_predicted_wall_s(self) -> float:
        return sum(m.predicted_wall_s for m in self.measurements)

    @property
    def total_raws(self) -> int:
        return sum(m.stratum.count for m in self.measurements)

    @property
    def total_bytes(self) -> int:
        return sum(m.stratum.total_bytes for m in self.measurements)

    def calibration_ratio(self, calibration_wall_s: float = CALIBRATION_WALL_S) -> float:
        """predicted / actual -- 1.0 is a perfect match, >1 over-predicts."""
        return self.total_predicted_wall_s / calibration_wall_s if calibration_wall_s else float("nan")

    @property
    def total_fixed_s_measured(self) -> list[float]:
        """Every stratum's measured one-time pass overhead (not summed into
        any total -- the real full rebuild pays this ONCE, not once per
        stratum; see StratumMeasurement's docstring)."""
        return [m.fixed_s for m in self.measurements if m.regression_valid]

    @property
    def population_stage_proportions(self) -> dict[str, float]:
        """Population-count-weighted average of every stratum's OWN stage mix.

        Each stratum's measured pass already reports what fraction of ITS
        OWN wall-clock went to parse/timed_apply/untimed_apply/terminal
        (``stage_proportions(m.stage_seconds)``); this weights those
        per-stratum mixes by the stratum's real population share
        (``stratum.count / total_raws``) to estimate the mix a full rebuild
        would show, without re-deriving absolute seconds (which the
        two-point fixed/marginal split already handles separately). See
        ``compare_to_real_run`` for why ``terminal`` specifically must be
        read as a HARNESS ARCHITECTURE artifact, not a fidelity number.
        """
        total_count = self.total_raws
        if not total_count:
            return dict.fromkeys(_STAGE_BUCKET_ORDER, 0.0)
        weighted: dict[str, float] = dict.fromkeys(_STAGE_BUCKET_ORDER, 0.0)
        for m in self.measurements:
            weight = m.stratum.count / total_count
            for name, fraction in stage_proportions(m.stage_seconds).items():
                weighted[name] += fraction * weight
        return weighted

    def compare_to_real_run(self) -> str:
        """The fidelity deliverable: measured stage mix vs. the real 4h22m run's.

        Two tables: the full 4-bucket mix (comparable to
        ``REAL_RUN_STAGE_PROPORTIONS`` as-is) and an apply-phase-only mix
        (parse/timed_apply/untimed_apply renormalized to 100%, terminal
        excluded). The apply-only table is the one that actually answers
        "is the harness representative of the 33% dark-matter gap" --
        ``terminal`` is structurally incomparable 1:1: every stratum pass
        here pays the FULL terminal-stage cost (repopulate/insights/
        readiness/promote run at whatever scale this pass's tiny index.db
        is), while the real archive pays it exactly ONCE across the whole
        41k-raw population. A multi-stratum harness that runs N separate
        passes cannot reproduce a cost that is by design paid once per
        FULL rebuild -- inflating ``terminal``'s measured share here is
        expected harness architecture, not something a corpus change can
        fix. That is this model's stated residual error on ``terminal``;
        the parse/timed_apply/untimed_apply mix is the part this follow-up
        claims fidelity on.
        """
        measured = self.population_stage_proportions
        full_table = stage_proportion_table(measured)
        apply_only_names = ("parse", "timed_apply", "untimed_apply")
        measured_apply_total = sum(measured.get(name, 0.0) for name in apply_only_names)
        real_apply_total = sum(REAL_RUN_STAGE_PROPORTIONS.get(name, 0.0) for name in apply_only_names)
        measured_apply_only = (
            {name: measured.get(name, 0.0) / measured_apply_total for name in apply_only_names}
            if measured_apply_total > 0
            else dict.fromkeys(apply_only_names, 0.0)
        )
        real_apply_only = {
            name: REAL_RUN_STAGE_PROPORTIONS.get(name, 0.0) / real_apply_total for name in apply_only_names
        }
        apply_lines = [f"{'stage':14s} {'measured':>10s} {'real_run':>10s} {'delta_pp':>9s}"]
        for name in apply_only_names:
            m = measured_apply_only.get(name, 0.0)
            r = real_apply_only.get(name, 0.0)
            apply_lines.append(f"{name:14s} {m * 100:9.1f}% {r * 100:9.1f}% {(m - r) * 100:+8.1f}pp")
        return (
            "full mix (parse/timed_apply/untimed_apply/terminal, population-weighted):\n"
            f"{full_table}\n\n"
            "apply-phase-only mix (terminal excluded, renormalized -- the fidelity claim this follow-up makes):\n"
            f"{chr(10).join(apply_lines)}\n\n"
            "residual error: terminal is structurally over-represented per-stratum-pass "
            "(each pass pays it once for its own tiny index.db; the real run pays it once "
            "for the whole archive) -- not comparable via this table; see "
            "compare_to_real_run's docstring."
        )

    def to_report(self) -> str:
        lines = [
            f"{'stratum':28s} {'n_pop':>8s} {'n1':>4s} {'n2':>4s} {'fixed_s':>8s} {'marg_s/raw':>11s} {'pred_min':>9s}",
        ]
        for m in sorted(self.measurements, key=lambda m: -m.predicted_wall_s):
            flag = "" if m.regression_valid else "*"
            lines.append(
                f"{m.stratum.label:28s} {m.stratum.count:8d} {m.n1:4d} {m.n2:4d} "
                f"{m.fixed_s:8.3f} {m.marginal_s_per_raw:11.4f} {m.predicted_wall_s / 60:9.2f}{flag}"
            )
        total_min = self.total_predicted_wall_s / 60
        lines.append(
            f"\npredicted total (marginal term only): {total_min:.1f} min over "
            f"{self.total_raws} raws, {self.total_bytes / 1024**3:.2f} GiB"
        )
        fixed_values = self.total_fixed_s_measured
        if fixed_values:
            lines.append(
                f"measured one-time per-pass fixed overhead across strata: "
                f"min={min(fixed_values):.2f}s max={max(fixed_values):.2f}s "
                f"median={sorted(fixed_values)[len(fixed_values) // 2]:.2f}s "
                "-- NOT added into the predicted total (the real run pays this once, "
                "not once per stratum sample; see module docstring)."
            )
        starred = [m.stratum.label for m in self.measurements if not m.regression_valid]
        if starred:
            lines.append(f"* regression not possible (population too small to split): {', '.join(starred)}")
        lines.append(
            f"calibration: actual={CALIBRATION_WALL_S / 60:.1f} min "
            f"({CALIBRATION_RAW_COUNT} raws, {CALIBRATION_TOTAL_BYTES / 1024**3:.2f} GiB) "
            f"-> ratio(predicted/actual)={self.calibration_ratio():.2f}"
        )
        lines.append("\nstage-proportion fidelity (polylogue-o56w follow-up):")
        lines.append(self.compare_to_real_run())
        return "\n".join(lines)


def run_cost_model(
    workdir: Path,
    strata: Sequence[Stratum] = POPULATION_SNAPSHOT,
    *,
    sample_sizes_override: tuple[int, int] | None = None,
) -> PredictedRun:
    """Measure every stratum (each gets a fresh scratch archive) and extrapolate."""
    predicted = PredictedRun()
    for i, stratum in enumerate(strata):
        stratum_root = workdir / f"stratum-{i:03d}"
        try:
            measurement = measure_stratum(stratum_root, stratum, sample_sizes=sample_sizes_override)
            predicted.measurements.append(measurement)
        finally:
            shutil.rmtree(stratum_root, ignore_errors=True)
    return predicted


__all__ = [
    "CALIBRATION_RAW_COUNT",
    "CALIBRATION_TOTAL_BYTES",
    "CALIBRATION_WALL_S",
    "POPULATION_SNAPSHOT",
    "POPULATION_SNAPSHOT_CAPTURED_AT",
    "REAL_RUN_STAGE_PROPORTIONS",
    "REAL_RUN_STAGE_SECONDS",
    "REAL_RUN_TOTAL_WALL_S",
    "PredictedRun",
    "Stratum",
    "StratumMeasurement",
    "bucket_stage_seconds",
    "build_stratum_sample_corpus",
    "collect_population_strata",
    "default_sample_n",
    "measure_stratum",
    "run_cost_model",
    "stage_proportion_table",
    "stage_proportions",
    "two_point_sample_sizes",
]
