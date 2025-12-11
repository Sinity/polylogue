from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

from .options import SearchHit, SearchOptions, SearchResult
from .services.conversation_service import ConversationService, create_conversation_service
from .util import parse_input_time_to_epoch, parse_rfc3339_to_epoch


def execute_search(
    options: SearchOptions,
    service: Optional[ConversationService] = None,
) -> SearchResult:
    query = (options.query or "").strip()
    if not query:
        return SearchResult(hits=[])

    service = service or create_conversation_service()
    if options.in_attachments:
        return _search_attachments(options, service)
    return _search_messages(options, service)


def _search_messages(options: SearchOptions, service: ConversationService) -> SearchResult:
    query = (options.query or "").strip()
    sql = [
        """
        SELECT
            m.provider,
            m.conversation_id,
            c.slug,
            c.title,
            m.branch_id,
            m.message_id,
            m.position,
            m.timestamp,
            m.attachment_count,
            bm25(messages_fts) AS score,
            snippet(messages_fts, '[', ']', ' … ', 4, 64) AS snippet,
            m.rendered_text AS body
        FROM messages_fts
        JOIN messages AS m
          ON messages_fts.provider = m.provider
         AND messages_fts.conversation_id = m.conversation_id
         AND messages_fts.branch_id = m.branch_id
         AND messages_fts.message_id = m.message_id
        JOIN conversations AS c
          ON m.provider = c.provider
         AND m.conversation_id = c.conversation_id
        WHERE messages_fts MATCH ?
        """
    ]
    params: List[object] = [query]

    if options.provider:
        sql.append("AND m.provider = ?")
        params.append(options.provider)
    if options.slug:
        sql.append("AND c.slug = ?")
        params.append(options.slug)
    if options.conversation_id:
        sql.append("AND m.conversation_id = ?")
        params.append(options.conversation_id)
    if options.branch_id:
        sql.append("AND m.branch_id = ?")
        params.append(options.branch_id)
    if options.has_attachments is True:
        sql.append("AND m.attachment_count > 0")
    elif options.has_attachments is False:
        sql.append("AND m.attachment_count = 0")
    if getattr(options, "attachment_name", None):
        sql.append("AND m.attachment_names LIKE ?")
        params.append(f"%{options.attachment_name}%")

    limit = max(1, options.limit)
    fetch_limit = max(limit * 4, limit)
    sql.append("ORDER BY score ASC")
    sql.append("LIMIT ?")
    params.append(fetch_limit)

    since_epoch = parse_input_time_to_epoch(options.since)
    until_epoch = parse_input_time_to_epoch(options.until)

    hits: List[SearchHit] = []

    try:
        rows = list(service.database.query("\n".join(sql), params))
    except sqlite3.OperationalError:
        return SearchResult(hits=[])

    for row in rows:
        if len(hits) >= limit:
            break
        timestamp = row["timestamp"]
        ts_epoch = parse_rfc3339_to_epoch(timestamp) if timestamp else None
        if since_epoch is not None and (ts_epoch is None or ts_epoch < since_epoch):
            continue
        if until_epoch is not None and (ts_epoch is None or ts_epoch > until_epoch):
            continue

        state = service.get_state(row["provider"], row["conversation_id"]) or {}
        model = _extract_model(state)
        if options.model and model:
            if model.lower() != options.model.lower():
                continue
        if options.model and not model:
            continue

        conversation_path = _safe_path(state.get("outputPath"))
        branch_path = _branch_path(conversation_path, row["branch_id"])

        score = row["score"]
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            score_val = 0.0

        snippet = (row["snippet"] or "").replace("\n", " ").strip()
        body = row["body"] or ""

        hits.append(
            SearchHit(
                provider=row["provider"],
                conversation_id=row["conversation_id"],
                slug=row["slug"],
                title=row["title"],
                branch_id=row["branch_id"],
                message_id=row["message_id"],
                position=row["position"],
                timestamp=timestamp,
                attachment_count=row["attachment_count"] or 0,
                score=score_val,
                snippet=snippet,
                body=body,
                conversation_path=conversation_path,
                branch_path=branch_path,
                model=model,
                kind="message",
            )
        )

    return SearchResult(hits=hits)


def _search_attachments(options: SearchOptions, service: ConversationService) -> SearchResult:
    query = (options.query or "").strip()
    sql = [
        """
        SELECT
            a.provider,
            a.conversation_id,
            c.slug,
            c.title,
            a.branch_id,
            a.message_id,
            m.position,
            m.timestamp,
            COALESCE(m.attachment_count, 0) AS attachment_count,
            a.size_bytes,
            a.mime_type,
            a.text_bytes,
            a.ocr_used,
            a.attachment_name,
            a.attachment_path,
            a.text_content,
            bm25(attachments_fts) AS score,
            snippet(attachments_fts, '[', ']', ' … ', 4, 64) AS snippet
        FROM attachments_fts
        JOIN attachments AS a
          ON attachments_fts.provider = a.provider
         AND attachments_fts.conversation_id = a.conversation_id
         AND ifnull(attachments_fts.branch_id, '') = ifnull(a.branch_id, '')
         AND ifnull(attachments_fts.message_id, '') = ifnull(a.message_id, '')
        JOIN conversations AS c
          ON a.provider = c.provider
         AND a.conversation_id = c.conversation_id
        LEFT JOIN messages AS m
          ON a.provider = m.provider
         AND a.conversation_id = m.conversation_id
         AND ifnull(a.branch_id, '') = ifnull(m.branch_id, '')
         AND ifnull(a.message_id, '') = ifnull(m.message_id, '')
        WHERE attachments_fts MATCH ?
        """
    ]
    params: List[object] = [query]

    if options.provider:
        sql.append("AND a.provider = ?")
        params.append(options.provider)
    if options.slug:
        sql.append("AND c.slug = ?")
        params.append(options.slug)
    if options.conversation_id:
        sql.append("AND a.conversation_id = ?")
        params.append(options.conversation_id)
    if options.branch_id:
        sql.append("AND a.branch_id = ?")
        params.append(options.branch_id)
    if getattr(options, "attachment_name", None):
        sql.append("AND a.attachment_name LIKE ?")
        params.append(f"%{options.attachment_name}%")

    limit = max(1, options.limit)
    fetch_limit = max(limit * 4, limit)
    sql.append("ORDER BY score ASC")
    sql.append("LIMIT ?")
    params.append(fetch_limit)

    since_epoch = parse_input_time_to_epoch(options.since)
    until_epoch = parse_input_time_to_epoch(options.until)

    hits: List[SearchHit] = []

    try:
        rows = list(service.database.query("\n".join(sql), params))
    except sqlite3.OperationalError:
        return SearchResult(hits=[])

    for row in rows:
        if len(hits) >= limit:
            break

        timestamp = row["timestamp"]
        ts_epoch = parse_rfc3339_to_epoch(timestamp) if timestamp else None
        if since_epoch is not None and (ts_epoch is None or ts_epoch < since_epoch):
            continue
        if until_epoch is not None and (ts_epoch is None or ts_epoch > until_epoch):
            continue

        state = service.get_state(row["provider"], row["conversation_id"]) or {}
        model = _extract_model(state)
        if options.model and model:
            if model.lower() != options.model.lower():
                continue
        if options.model and not model:
            continue

        conversation_path = _safe_path(state.get("outputPath"))
        branch_path = _branch_path(conversation_path, row["branch_id"])

        try:
            score_val = float(row["score"])
        except (TypeError, ValueError):
            score_val = 0.0

        snippet = (row["snippet"] or "").replace("\n", " ").strip()
        text_body = row["text_content"] or ""

        hits.append(
            SearchHit(
                provider=row["provider"],
                conversation_id=row["conversation_id"],
                slug=row["slug"],
                title=row["title"],
                branch_id=row["branch_id"],
                message_id=row["message_id"],
                position=row["position"],
                timestamp=timestamp,
                attachment_count=row["attachment_count"] or 0,
                score=score_val,
                snippet=snippet,
                body=text_body,
                conversation_path=conversation_path,
                branch_path=branch_path,
                model=model,
                kind="attachment",
                attachment_name=row["attachment_name"],
                attachment_path=_safe_path(row["attachment_path"]),
                attachment_bytes=row["size_bytes"],
                attachment_mime=row["mime_type"],
                attachment_text_bytes=row["text_bytes"],
                ocr_used=bool(row["ocr_used"]),
            )
        )

    return SearchResult(hits=hits)


def _extract_model(state: dict) -> Optional[str]:
    model = state.get("sourceModel")
    if isinstance(model, str) and model:
        return model
    run_settings = state.get("runSettings")
    if isinstance(run_settings, dict):
        candidate = run_settings.get("model")
        if isinstance(candidate, str):
            return candidate
    return None


def _safe_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _branch_path(conversation_path: Optional[Path], branch_id: str) -> Optional[Path]:
    if not conversation_path or not branch_id:
        return None
    branch_dir = conversation_path.parent / "branches" / branch_id
    candidate = branch_dir / f"{branch_id}.md"
    if candidate.exists():
        return candidate
    return None

