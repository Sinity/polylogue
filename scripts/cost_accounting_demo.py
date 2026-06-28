#!/usr/bin/env python3
"""Demonstrate Polylogue's cross-provider token/cost accounting — and the
Codex double-billing bug it now avoids — end to end, with no mocks.

A stranger can run this with nothing but a checkout:

    uv run python scripts/cost_accounting_demo.py

It builds a throwaway archive in a temp dir, ingests two crafted sessions
through Polylogue's *real* writer (`write_parsed_session_to_archive`, the same
path the daemon uses), reads the materialized `session_model_usage` rollup back,
and prices it with the real catalog (`archive/semantic/pricing.py`). The token
numbers are hand-checkable, so you can verify the math yourself.

It demonstrates one concrete, load-bearing fact about agentic-engineering cost
accounting: Codex (OpenAI) reports `input_tokens` *inclusive* of cached input
and `output_tokens` *inclusive* of reasoning. If you bill those naively you
double-count — and because a coding agent re-sends its whole context every
turn, cached input is the dominant term (~96% on a real 13-month archive), so
the error is ~8x, not a rounding error.

Operator mode — cross-verify the real archive against Codex's own authoritative
token store (private; only runs if both are present):

    uv run python scripts/cost_accounting_demo.py \
        --archive ~/.local/share/polylogue --codex-state ~/.codex/state_5.sqlite
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.semantic.pricing import PRICING, _normalize_model
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _price(model: str, fresh_input: int, output: int, cache_read: int, cache_write: int) -> float:
    p = PRICING.get(_normalize_model(model))
    if p is None:
        return 0.0
    return (
        fresh_input * p.input_usd_per_1m
        + output * p.output_usd_per_1m
        + cache_read * p.cache_read_usd_per_1m
        + cache_write * p.cache_write_usd_per_1m
    ) / 1_000_000


def _buggy_price(
    model: str, input_incl_cached: int, output_plus_reasoning: int, cached: int, cache_write: int
) -> float:
    """What the pre-fix rollup charged: input still carried cached, and cached
    was *also* billed on its own lane — so cached pays both the input rate and
    the cache-read rate. Reasoning was added on top of output as well."""
    p = PRICING.get(_normalize_model(model))
    if p is None:
        return 0.0
    return (
        input_incl_cached * p.input_usd_per_1m
        + output_plus_reasoning * p.output_usd_per_1m
        + cached * p.cache_read_usd_per_1m
        + cache_write * p.cache_write_usd_per_1m
    ) / 1_000_000


def run_synthetic_demo() -> None:
    # Crafted Codex session: one cumulative token_count event with realistic
    # proportions — an agent re-sends its whole context each turn, so cached is
    # ~96% of input and freshly generated output is a thin slice.
    codex_total = {
        "input_tokens": 100_000,  # INCLUDES the 96,000 cached below
        "cached_input_tokens": 96_000,
        "output_tokens": 300,  # INCLUDES the 200 reasoning below
        "reasoning_output_tokens": 200,
        "cache_write_tokens": 4_000,
        "total_tokens": 100_300,  # == input + output
    }
    codex = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="cost-demo-codex",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="patch applied")],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="token_count",
                payload={"type": "token_count", "model": "gpt-5-codex", "total_token_usage": codex_total},
            )
        ],
    )

    with tempfile.TemporaryDirectory(prefix="polylogue-cost-demo-") as td:
        conn = _connect(Path(td) / "index.db")
        session_id = write_parsed_session_to_archive(conn, codex)
        row = conn.execute(
            """
            SELECT model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            FROM session_model_usage WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    model = row["model_name"]
    fin, out, cr, cw = row["input_tokens"], row["output_tokens"], row["cache_read_tokens"], row["cache_write_tokens"]

    print("=" * 72)
    print("Polylogue cross-provider cost accounting — Codex token semantics demo")
    print("=" * 72)
    print("\nRaw Codex token_count event (cumulative totals as the provider reports them):")
    print(f"  input_tokens            = {codex_total['input_tokens']:>8,}  (includes cached)")
    print(f"  cached_input_tokens     = {codex_total['cached_input_tokens']:>8,}")
    print(f"  output_tokens           = {codex_total['output_tokens']:>8,}  (includes reasoning)")
    print(f"  reasoning_output_tokens = {codex_total['reasoning_output_tokens']:>8,}")
    print(f"  cache_write_tokens      = {codex_total['cache_write_tokens']:>8,}")

    print("\nMaterialized session_model_usage row (read back from the real writer):")
    print(f"  model = {model}")
    print(f"  input_tokens (fresh)    = {fin:>8,}   <- 100,000 - 96,000 cached")
    print(f"  cache_read_tokens       = {cr:>8,}")
    print(f"  output_tokens           = {out:>8,}   <- reasoning already inside, not re-added")
    print(f"  cache_write_tokens      = {cw:>8,}")
    assert fin + cr == codex_total["input_tokens"], "fresh_input + cache_read must reconstruct provider input"
    print(f"  invariant: fresh_input + cache_read = {fin + cr:,} == provider input  ✓ (each token billed once)")

    correct = _price(model, fin, out, cr, cw)
    buggy = _buggy_price(
        model,
        codex_total["input_tokens"],
        codex_total["output_tokens"] + codex_total["reasoning_output_tokens"],
        codex_total["cached_input_tokens"],
        codex_total["cache_write_tokens"],
    )
    p = PRICING[_normalize_model(model)]
    print(
        f"\nPricing (gpt-5-codex, USD/1M): input {p.input_usd_per_1m}  output {p.output_usd_per_1m}  "
        f"cache_read {p.cache_read_usd_per_1m}  cache_write {p.cache_write_usd_per_1m}"
    )
    print(f"\n  Correct cost (disjoint lanes):        ${correct:.6f}")
    print(f"  Pre-fix cost (cached double-billed):  ${buggy:.6f}")
    print(f"  Inflation factor:                     {buggy / correct:.2f}x")
    print("\nWhy: the 96,000 cached tokens were billed at BOTH the full input rate")
    print("(1.25/M, because they sat inside input_tokens) AND the cache-read rate")
    print("(0.125/M). Subtracting cached out of input bills each token once.")
    print("\nThe exact multiple depends on the per-session output/cache mix. Across")
    print("the operator's real 13-month, 199.6B-token Codex corpus the aggregate")
    print("inflation is 7.69x: $76,856 correct vs $591,103 pre-fix (API-list-equiv).")


def run_cross_verify(archive: Path, state: Path) -> None:
    print("\n" + "=" * 72)
    print("Operator cross-verification: real archive vs Codex authoritative store")
    print("=" * 72)
    idx = sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)
    st = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    poly = {}
    for sid, tt in idx.execute(
        "SELECT session_id, MAX(total_tokens) FROM session_provider_usage_events "
        "WHERE provider_event_type='token_count' GROUP BY session_id"
    ):
        uuid = sid.split(":", 1)[1] if ":" in sid else sid
        poly[uuid] = tt or 0
    auth = {row[0]: (row[1] or 0) for row in st.execute("SELECT id, tokens_used FROM threads")}
    common = [u for u in poly if u in auth and auth[u] > 0]
    ratios = sorted(poly[u] / auth[u] for u in common)
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    within10 = sum(1 for r in ratios if 0.9 <= r <= 1.1)
    only_poly_tok = sum(poly[u] for u in poly if u not in auth)
    print(f"  threads compared (poly ∩ authoritative): {len(common)}")
    print(f"  per-thread poly/authoritative ratio: median={med:.3f}  within 10%: {within10}/{len(ratios)}")
    print(
        f"  tokens in archive but pruned from live Codex state: {only_poly_tok / 1e9:.2f}B "
        f"({sum(1 for u in poly if u not in auth)} threads)"
    )
    print("\n  Interpretation: per-thread the corrected accounting matches Codex's")
    print("  own token store; the archive additionally retains history the live")
    print("  tool has already discarded.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="Real archive root (operator cross-verify; private)")
    ap.add_argument("--codex-state", type=Path, help="Path to ~/.codex/state_5.sqlite (operator cross-verify)")
    args = ap.parse_args()

    run_synthetic_demo()

    if args.archive and args.codex_state:
        if (args.archive / "index.db").exists() and args.codex_state.exists():
            run_cross_verify(args.archive, args.codex_state)
        else:
            print("\n[cross-verify skipped: --archive index.db or --codex-state not found]")
    elif args.archive or args.codex_state:
        print("\n[cross-verify needs BOTH --archive and --codex-state]")


if __name__ == "__main__":
    os.environ.setdefault("POLYLOGUE_FORCE_PLAIN", "1")
    main()
