"""Tag-rollup builders and aggregate reducers for archive products."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from polylogue.archive_products import (
    SessionTagRollupProduct,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.lib.repo_identity import normalize_repo_names
from polylogue.lib.session_profile import SessionProfile
from polylogue.storage.store import SessionTagRollupRecord


def build_session_tag_rollup_records(
    profiles: Iterable[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[SessionTagRollupRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for profile in profiles:
        bucket_day = profile_bucket_day(profile)
        if bucket_day is None:
            continue
        explicit_tags = {tag for tag in profile.tags if tag}
        auto_tags = {tag for tag in profile.auto_tags if tag}
        all_tags = explicit_tags | auto_tags
        if not all_tags:
            continue
        iso_timestamps, sort_keys = profile_timestamp_values(profile)
        for tag in all_tags:
            key = (profile.provider, bucket_day.isoformat(), tag)
            bucket = grouped.setdefault(
                key,
                {
                    "conversation_count": 0,
                    "explicit_count": 0,
                    "auto_count": 0,
                    "repos": Counter(),
                    "source_updated_at": [],
                    "source_sort_key": [],
                },
            )
            bucket["conversation_count"] = int(bucket["conversation_count"]) + 1
            if tag in explicit_tags:
                bucket["explicit_count"] = int(bucket["explicit_count"]) + 1
            if tag in auto_tags:
                bucket["auto_count"] = int(bucket["auto_count"]) + 1
            cast_repos = bucket["repos"]
            assert isinstance(cast_repos, Counter)
            cast_repos.update(profile.repo_names or normalize_repo_names(repo_paths=profile.repo_paths))
            cast_updates = bucket["source_updated_at"]
            cast_sorts = bucket["source_sort_key"]
            assert isinstance(cast_updates, list)
            assert isinstance(cast_sorts, list)
            cast_updates.extend(iso_timestamps)
            cast_sorts.extend(sort_keys)

    rows: list[SessionTagRollupRecord] = []
    for (provider_name, bucket_day, tag), bucket in sorted(grouped.items()):
        repos = bucket["repos"]
        source_updates = bucket["source_updated_at"]
        source_sorts = bucket["source_sort_key"]
        assert isinstance(repos, Counter)
        assert isinstance(source_updates, list)
        assert isinstance(source_sorts, list)
        search_text = " \n".join(part for part in (tag, provider_name, *sorted(repos.keys())) if part)
        rows.append(
            SessionTagRollupRecord(
                tag=tag,
                bucket_day=bucket_day,
                provider_name=provider_name,
                materialized_at=built_at,
                source_updated_at=max(source_updates) if source_updates else None,
                source_sort_key=max(source_sorts) if source_sorts else None,
                conversation_count=int(bucket["conversation_count"]),
                explicit_count=int(bucket["explicit_count"]),
                auto_count=int(bucket["auto_count"]),
                repo_breakdown=dict(repos),
                search_text=search_text or tag,
            )
        )
    return rows


def aggregate_session_tag_rollup_products(
    rows: Sequence[SessionTagRollupRecord],
) -> list[SessionTagRollupProduct]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        bucket = grouped.setdefault(
            row.tag,
            {
                "conversation_count": 0,
                "explicit_count": 0,
                "auto_count": 0,
                "provider_breakdown": Counter(),
                "repo_breakdown": Counter(),
                "rows": [],
            },
        )
        bucket["conversation_count"] = int(bucket["conversation_count"]) + row.conversation_count
        bucket["explicit_count"] = int(bucket["explicit_count"]) + row.explicit_count
        bucket["auto_count"] = int(bucket["auto_count"]) + row.auto_count
        provider_breakdown = bucket["provider_breakdown"]
        repo_breakdown = bucket["repo_breakdown"]
        record_rows = bucket["rows"]
        assert isinstance(provider_breakdown, Counter)
        assert isinstance(repo_breakdown, Counter)
        assert isinstance(record_rows, list)
        provider_breakdown[row.provider_name] += row.conversation_count
        repo_breakdown.update(row.repo_breakdown)
        record_rows.append(row)

    products: list[SessionTagRollupProduct] = []
    for tag, bucket in sorted(grouped.items(), key=lambda item: (-int(item[1]["conversation_count"]), item[0])):
        provider_breakdown = bucket["provider_breakdown"]
        repo_breakdown = bucket["repo_breakdown"]
        record_rows = bucket["rows"]
        assert isinstance(provider_breakdown, Counter)
        assert isinstance(repo_breakdown, Counter)
        assert isinstance(record_rows, list)
        products.append(
            SessionTagRollupProduct(
                tag=tag,
                conversation_count=int(bucket["conversation_count"]),
                explicit_count=int(bucket["explicit_count"]),
                auto_count=int(bucket["auto_count"]),
                provider_breakdown=dict(provider_breakdown),
                repo_breakdown=dict(repo_breakdown),
                provenance=records_provenance(record_rows),
            )
        )
    return products


__all__ = [
    "aggregate_session_tag_rollup_products",
    "build_session_tag_rollup_records",
]
