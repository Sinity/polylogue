"""Tag-rollup builders and aggregate reducers for archive products."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polylogue.archive_products import (
    SessionTagRollupProduct,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.lib.repo_identity import normalize_repo_names
from polylogue.lib.session_profile import SessionProfile
from polylogue.storage.runtime import SessionTagRollupRecord


@dataclass(slots=True)
class _TagRollupBucket:
    conversation_count: int = 0
    explicit_count: int = 0
    auto_count: int = 0
    repos: Counter[str] = field(default_factory=Counter)
    source_updated_at: list[str] = field(default_factory=list)
    source_sort_key: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _TagAggregateBucket:
    conversation_count: int = 0
    explicit_count: int = 0
    auto_count: int = 0
    provider_breakdown: Counter[str] = field(default_factory=Counter)
    repo_breakdown: Counter[str] = field(default_factory=Counter)
    rows: list[SessionTagRollupRecord] = field(default_factory=list)


def build_session_tag_rollup_records(
    profiles: Iterable[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[SessionTagRollupRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    grouped: dict[tuple[str, str, str], _TagRollupBucket] = {}
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
        repo_names = profile.repo_names or normalize_repo_names(repo_paths=profile.repo_paths)
        bucket_day_text = bucket_day.isoformat()
        for tag in all_tags:
            key = (profile.provider, bucket_day_text, tag)
            bucket = grouped.setdefault(key, _TagRollupBucket())
            bucket.conversation_count += 1
            if tag in explicit_tags:
                bucket.explicit_count += 1
            if tag in auto_tags:
                bucket.auto_count += 1
            bucket.repos.update(repo_names)
            bucket.source_updated_at.extend(iso_timestamps)
            bucket.source_sort_key.extend(sort_keys)

    rows: list[SessionTagRollupRecord] = []
    for (provider_name, bucket_day_text, tag), bucket in sorted(grouped.items()):
        search_text = " \n".join(part for part in (tag, provider_name, *sorted(bucket.repos.keys())) if part)
        rows.append(
            SessionTagRollupRecord(
                tag=tag,
                bucket_day=bucket_day_text,
                provider_name=provider_name,
                materialized_at=built_at,
                source_updated_at=max(bucket.source_updated_at) if bucket.source_updated_at else None,
                source_sort_key=max(bucket.source_sort_key) if bucket.source_sort_key else None,
                conversation_count=bucket.conversation_count,
                explicit_count=bucket.explicit_count,
                auto_count=bucket.auto_count,
                repo_breakdown=dict(bucket.repos),
                search_text=search_text or tag,
            )
        )
    return rows


def aggregate_session_tag_rollup_products(
    rows: Sequence[SessionTagRollupRecord],
) -> list[SessionTagRollupProduct]:
    grouped: dict[str, _TagAggregateBucket] = {}
    for row in rows:
        bucket = grouped.setdefault(row.tag, _TagAggregateBucket())
        bucket.conversation_count += row.conversation_count
        bucket.explicit_count += row.explicit_count
        bucket.auto_count += row.auto_count
        bucket.provider_breakdown[row.provider_name] += row.conversation_count
        bucket.repo_breakdown.update(row.repo_breakdown)
        bucket.rows.append(row)

    products: list[SessionTagRollupProduct] = []
    for tag, bucket in sorted(grouped.items(), key=lambda item: (-item[1].conversation_count, item[0])):
        products.append(
            SessionTagRollupProduct(
                tag=tag,
                conversation_count=bucket.conversation_count,
                explicit_count=bucket.explicit_count,
                auto_count=bucket.auto_count,
                provider_breakdown=dict(bucket.provider_breakdown),
                repo_breakdown=dict(bucket.repo_breakdown),
                provenance=records_provenance(bucket.rows),
            )
        )
    return products


__all__ = [
    "aggregate_session_tag_rollup_products",
    "build_session_tag_rollup_records",
]
