"""Real-pipeline corpus and manifest for the reindex campaign tranche.

The campaign deliberately starts with provider-shaped wire artifacts.  The
archive is populated by ``parse_sources_archive`` and converged by the same
daemon stages used by the product.  The parser-failure fixture is then written
through ``ArchiveStore`` and finalized through its typed parse-state route.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

REINDEX_CAMPAIGN_MINIMUMS: dict[str, int] = {
    "lineage_edges": 1,
    "attachment_refs": 1,
    "attachment_blob_refs": 1,
    "attachment_blob_bytes": 1,
    "structured_tool_failures": 1,
    "materialized_insight_sessions": 1,
    "fts_documents": 1,
    "parser_failure_residuals": 1,
    "duplicate_raw_rows": 1,
    "restart_debt_sessions": 1,
}


@dataclass(frozen=True, slots=True)
class ReindexCampaignManifest:
    """Stable campaign evidence, including positive class denominators."""

    specs: tuple[CorpusSpec, ...]
    session_ids: tuple[str, ...]
    lineage_session_ids: tuple[str, ...]
    attachment_session_ids: tuple[str, ...]
    restart_session_ids: tuple[str, ...]
    parser_failure_raw_ids: tuple[str, ...]
    duplicate_raw_ids: tuple[str, ...]
    fts_queries: tuple[str, ...]
    denominators: tuple[tuple[str, int], ...]

    def denominator(self, name: str) -> int:
        try:
            return dict(self.denominators)[name]
        except KeyError as exc:
            raise AssertionError(f"campaign manifest has no denominator {name!r}") from exc

    def assert_positive(self) -> None:
        for name, minimum in REINDEX_CAMPAIGN_MINIMUMS.items():
            actual = self.denominator(name)
            if actual < minimum:
                raise AssertionError(f"campaign denominator {name!r} is {actual}, expected at least {minimum}")


@dataclass(frozen=True, slots=True)
class ReindexCampaignCorpus:
    root: Path
    manifest: ReindexCampaignManifest


def reindex_campaign_corpus_specs() -> tuple[CorpusSpec, ...]:
    """Return authored provider specs used by the campaign corpus."""

    return (
        CorpusSpec.for_provider(
            "codex",
            count=3,
            messages_min=4,
            messages_max=4,
            seed=801,
            style="tool-heavy",
            session_native_ids=("campaign-codex-a", "campaign-codex-b", "campaign-codex-c"),
            origin="test.reindex-campaign",
            tags=("reindex", "campaign", "tool-results"),
        ),
        CorpusSpec.for_provider(
            "claude-code",
            count=4,
            messages_min=4,
            messages_max=4,
            seed=802,
            style="demo-tool-heavy",
            session_native_ids=(
                "campaign-claude-failure-a",
                "campaign-claude-failure-b",
                "campaign-claude-stable-a",
                "campaign-claude-stable-b",
            ),
            origin="test.reindex-campaign",
            tags=("reindex", "campaign", "structured-failures"),
        ),
        CorpusSpec.for_provider(
            "gemini",
            count=1,
            messages_min=3,
            messages_max=3,
            seed=803,
            style="demo-attachments",
            session_native_ids=("campaign-attachment",),
            origin="test.reindex-campaign",
            tags=("reindex", "campaign", "attachments"),
        ),
    )


def _codex_records(
    session_id: str, texts: tuple[str, ...], *, parent: str | None = None
) -> tuple[dict[str, object], ...]:
    meta: dict[str, object] = {"id": session_id, "timestamp": "2026-08-05T00:00:00Z"}
    if parent is not None:
        meta["forked_from_id"] = parent
    records: list[dict[str, object]] = [{"type": "session_meta", "payload": meta}]
    for position, text in enumerate(texts):
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{session_id}-m{position}",
                    "role": "user" if position % 2 == 0 else "assistant",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    return tuple(records)


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    return path


def _campaign_session_ids(root: Path) -> tuple[str, ...]:
    with sqlite3.connect(root / "index.db") as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))


def _campaign_manifest(
    root: Path,
    *,
    specs: tuple[CorpusSpec, ...],
    parser_failure_raw_ids: tuple[str, ...],
    duplicate_raw_ids: tuple[str, ...],
) -> ReindexCampaignManifest:
    with sqlite3.connect(root / "index.db") as index_conn:
        session_ids = _campaign_session_ids(root)
        lineage_session_ids = tuple(
            str(row[0])
            for row in index_conn.execute(
                "SELECT DISTINCT src_session_id FROM session_links WHERE link_type = 'branch' ORDER BY src_session_id"
            )
        )
        attachment_session_ids = tuple(
            str(row[0])
            for row in index_conn.execute(
                "SELECT DISTINCT session_id FROM attachment_refs WHERE session_id IS NOT NULL ORDER BY session_id"
            )
        )
        structured_tool_failures = int(
            index_conn.execute(
                """
                SELECT COUNT(*) FROM blocks
                WHERE block_type = 'tool_result'
                  AND (tool_result_is_error = 1 OR tool_result_exit_code != 0)
                """
            ).fetchone()[0]
        )
        materialized_insight_sessions = int(
            index_conn.execute("SELECT COUNT(DISTINCT session_id) FROM insight_materialization").fetchone()[0]
        )
        fts_documents = int(index_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        attachment_refs = int(index_conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0])
        attachment_blob_bytes = int(
            index_conn.execute(
                "SELECT COALESCE(SUM(byte_count), 0) FROM attachments WHERE blob_hash IS NOT NULL"
            ).fetchone()[0]
        )
        restart_session_ids = tuple(
            str(row[0])
            for row in index_conn.execute(
                "SELECT session_id FROM sessions WHERE origin = 'claude-code-session' ORDER BY session_id LIMIT 1"
            )
        )
    with sqlite3.connect(root / "source.db") as source_conn:
        attachment_blob_refs = int(
            source_conn.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone()[0]
        )
        duplicate_rows = int(
            source_conn.execute(
                """
                SELECT COUNT(*) FROM raw_sessions
                WHERE blob_hash IN (
                    SELECT blob_hash FROM raw_sessions GROUP BY blob_hash HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        parser_failure_residuals = int(
            source_conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NULL AND parse_error IS NOT NULL"
            ).fetchone()[0]
        )
    denominators = (
        ("lineage_edges", len(lineage_session_ids)),
        ("attachment_refs", attachment_refs),
        ("attachment_blob_refs", attachment_blob_refs),
        ("attachment_blob_bytes", attachment_blob_bytes),
        ("structured_tool_failures", structured_tool_failures),
        ("materialized_insight_sessions", materialized_insight_sessions),
        ("fts_documents", fts_documents),
        ("parser_failure_residuals", parser_failure_residuals),
        ("duplicate_raw_rows", duplicate_rows),
        ("restart_debt_sessions", len(restart_session_ids)),
    )
    manifest = ReindexCampaignManifest(
        specs=specs,
        session_ids=session_ids,
        lineage_session_ids=lineage_session_ids,
        attachment_session_ids=attachment_session_ids,
        restart_session_ids=restart_session_ids,
        parser_failure_raw_ids=parser_failure_raw_ids,
        duplicate_raw_ids=duplicate_raw_ids,
        fts_queries=("generated", "fixture", "failed"),
        denominators=denominators,
    )
    manifest.assert_positive()
    if parser_failure_residuals < len(parser_failure_raw_ids):
        raise AssertionError("campaign parser-failure manifest exceeds source residuals")
    if duplicate_rows < len(duplicate_raw_ids):
        raise AssertionError("campaign duplicate manifest exceeds duplicate source rows")
    return manifest


def build_reindex_campaign_corpus(root: Path) -> ReindexCampaignCorpus:
    """Build and converge the campaign corpus through production routes."""

    root = Path(root)
    initialize_active_archive_root(root)
    specs = reindex_campaign_corpus_specs()
    wire_root = root / "wire"
    sources: list[Source] = []
    first_payload: bytes | None = None
    for index, spec in enumerate(specs):
        written = SyntheticCorpus.write_spec_artifacts(spec, wire_root / spec.provider, prefix=f"campaign-{index:02d}")
        if first_payload is None:
            first_payload = written.files[0].read_bytes()
        sources.extend(Source(name=spec.provider, path=path) for path in written.files)

    lineage_dir = wire_root / "lineage"
    sources.append(
        Source(
            name="codex",
            path=_write_jsonl(
                lineage_dir / "campaign-lineage-parent.jsonl",
                _codex_records("campaign-lineage-parent", ("shared campaign prefix", "parent tail")),
            ),
        )
    )
    sources.append(
        Source(
            name="codex",
            path=_write_jsonl(
                lineage_dir / "campaign-lineage-child.jsonl",
                _codex_records(
                    "campaign-lineage-child",
                    ("child tail",),
                    parent="campaign-lineage-parent",
                ),
            ),
        )
    )
    assert first_payload is not None
    parse_result = asyncio.run(parse_sources_archive(root, sources, parse_workers=1))
    if parse_result.parse_failures != 0:
        raise AssertionError(f"campaign provider ingest unexpectedly failed: {parse_result.parse_failures}")

    with sqlite3.connect(root / "source.db") as source_conn:
        restart_native_id = source_conn.execute(
            """
            SELECT m.provider_session_id
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.origin = 'claude-code-session'
            ORDER BY m.provider_session_id
            LIMIT 1
            """
        ).fetchone()
        restart_native_id = str(restart_native_id[0])

    duplicate_paths = tuple(wire_root / "duplicates" / f"campaign-duplicate-{suffix}.jsonl" for suffix in ("a", "b"))
    duplicate_sources = [Source(name="codex", path=path) for path in duplicate_paths]
    for path in duplicate_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(first_payload)
    duplicate_parse_result = asyncio.run(parse_sources_archive(root, duplicate_sources, parse_workers=1))
    if duplicate_parse_result.parse_failures != 0:
        raise AssertionError(f"campaign duplicate ingest unexpectedly failed: {duplicate_parse_result.parse_failures}")
    with sqlite3.connect(root / "source.db") as source_conn:
        duplicate_raw_ids = tuple(
            str(row[0])
            for row in source_conn.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path IN (?, ?) ORDER BY source_path",
                tuple(str(path) for path in duplicate_paths),
            )
        )
    if len(duplicate_raw_ids) != len(duplicate_paths):
        raise AssertionError("campaign duplicate ingest did not retain both raw acquisitions")

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        parser_failure_raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=first_payload,
            source_path="campaign-parser-failure.jsonl",
            native_id=restart_native_id,
            acquired_at_ms=3,
        )
        from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind

        archive.record_raw_failure_evidence(
            parser_failure_raw_id,
            provider=Provider.CLAUDE_CODE,
            source_path="campaign-parser-failure.jsonl",
            source_index=0,
            acquired_at_ms=3,
            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        )
        archive.mark_raw_parse_failed(
            parser_failure_raw_id,
            provider=Provider.CLAUDE_CODE,
            error=ValueError("campaign parser failure"),
        )

    session_ids = _campaign_session_ids(root)
    states, _timings = DaemonConverger(
        (make_fts_stage(root / "index.db"), make_insights_stage(root / "index.db"))
    ).converge_sessions(session_ids)
    not_converged = {session_id: state.last_error for session_id, state in states.items() if not state.converged}
    if not_converged:
        raise AssertionError(f"campaign corpus did not converge: {not_converged}")

    manifest = _campaign_manifest(
        root,
        specs=specs,
        parser_failure_raw_ids=(parser_failure_raw_id,),
        duplicate_raw_ids=duplicate_raw_ids,
    )
    return ReindexCampaignCorpus(root=root, manifest=manifest)


__all__ = [
    "REINDEX_CAMPAIGN_MINIMUMS",
    "ReindexCampaignCorpus",
    "ReindexCampaignManifest",
    "build_reindex_campaign_corpus",
    "reindex_campaign_corpus_specs",
]
