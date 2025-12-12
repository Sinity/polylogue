from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

from ..commands import CommandEnv, search_command
from ..options import SearchHit, SearchOptions
from ..schema import stamp_payload


def _compare_hits(provider: str, hits: List[SearchHit], fields: List[str]) -> Dict[str, Any]:
    total_attachments = sum(hit.attachment_count or 0 for hit in hits)
    models = sorted({hit.model for hit in hits if hit.model})
    payload_hits = []
    for hit in hits:
        row = {
            "provider": hit.provider,
            "slug": hit.slug,
            "branchId": hit.branch_id,
            "messageId": hit.message_id,
            "score": hit.score,
            "snippet": hit.snippet,
            "model": hit.model,
            "path": str(hit.branch_path or hit.conversation_path) if (hit.branch_path or hit.conversation_path) else None,
        }
        payload_hits.append({k: row.get(k) for k in fields})
    return {
        "provider": provider,
        "count": len(hits),
        "attachments": total_attachments,
        "models": models,
        "hits": payload_hits,
    }


def run_compare_cli(args: object, env: CommandEnv) -> None:
    ui = env.ui
    fields = [f.strip() for f in (getattr(args, "fields", "") or "").split(",") if f.strip()]
    limit = max(1, int(getattr(args, "limit", 20)))
    query = getattr(args, "query", None)
    provider_a = getattr(args, "provider_a", None)
    provider_b = getattr(args, "provider_b", None)

    def _search(provider: str) -> List[SearchHit]:
        options = SearchOptions(
            query=query,
            limit=limit,
            provider=provider,
            slug=None,
            conversation_id=None,
            branch_id=None,
            model=None,
            since=None,
            until=None,
            has_attachments=None,
        )
        return search_command(options, env).hits

    hits_a = _search(provider_a)
    hits_b = _search(provider_b)

    if getattr(args, "json", False):
        payload = stamp_payload(
            {
                "query": query,
                "limit": limit,
                "providers": [
                    _compare_hits(provider_a, hits_a, fields),
                    _compare_hits(provider_b, hits_b, fields),
                ],
            }
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    lines = [f"Query: {query}", f"Limit: {limit}"]
    for name, hits in ((provider_a, hits_a), (provider_b, hits_b)):
        attachments_total = sum(hit.attachment_count or 0 for hit in hits)
        models = sorted({hit.model for hit in hits if hit.model})
        lines.append(
            f"{name}: {len(hits)} hit(s), attachments={attachments_total}, models={', '.join(models) if models else 'n/a'}"
        )
        for hit in hits[: min(3, len(hits))]:
            path_val = hit.branch_path or hit.conversation_path
            snippet = (hit.snippet or "").replace("\n", " ")
            snippet = textwrap.shorten(snippet, width=96, placeholder="…")
            lines.append(f"  - {hit.slug} [{hit.branch_id}] score={hit.score:.3f} {snippet} ({path_val})")
    ui.summary("Provider Compare", lines)


__all__ = ["run_compare_cli"]

