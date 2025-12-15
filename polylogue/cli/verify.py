from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from ..archive import Archive
from ..commands import CommandEnv
from ..document_store import read_existing_document
from ..schema import stamp_payload


@dataclass
class VerifyIssue:
    provider: str
    conversation_id: str
    path: Optional[Path]
    message: str
    severity: str = "error"  # error|warning|info

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "conversationId": self.conversation_id,
            "path": str(self.path) if self.path else None,
            "message": self.message,
            "severity": self.severity,
        }


_PROVIDER_ALIASES = {
    "drive-sync": "drive",
    "claude.ai": "claude",
}


def _parse_csv_set(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    values = {chunk.strip().lower() for chunk in str(raw).split(",") if chunk.strip()}
    return values or None


def _safe_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_archive_root(archive: Archive, provider: str) -> Optional[Path]:
    canonical = _PROVIDER_ALIASES.get(provider, provider)
    try:
        return archive.provider_root(canonical)
    except Exception:
        return None


def _resolve_conversation_paths(
    *,
    archive: Archive,
    provider: str,
    slug: str,
    state_entry: Dict[str, Any],
) -> Tuple[Optional[Path], Optional[Path]]:
    output_path = state_entry.get("outputPath")
    if isinstance(output_path, str) and output_path.strip():
        path = Path(output_path)
        return path, path.parent
    root = _resolve_archive_root(archive, provider)
    if root is None:
        return None, None
    md_path = root / slug / "conversation.md"
    return md_path, md_path.parent


def run_verify_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    provider_filter = _parse_csv_set(getattr(args, "provider", None))
    slug_filter = str(getattr(args, "slug", "") or "").strip() or None
    conversation_ids = [cid.strip() for cid in getattr(args, "conversation_ids", []) if cid and str(cid).strip()]
    conversation_id_filter = set(conversation_ids) if conversation_ids else None
    limit = getattr(args, "limit", None)
    json_mode = bool(getattr(args, "json", False))

    rows_raw = env.database.query(
        "SELECT provider, conversation_id, slug, content_hash, metadata_json FROM conversations"
    )
    rows: List[Dict[str, Any]] = [dict(row) for row in rows_raw]
    if provider_filter:
        rows = [row for row in rows if str(row.get("provider") or "").lower() in provider_filter]
    if slug_filter:
        rows = [row for row in rows if str(row.get("slug") or "") == slug_filter]
    if conversation_id_filter:
        rows = [row for row in rows if str(row.get("conversation_id") or "") in conversation_id_filter]
    if isinstance(limit, int) and limit > 0:
        rows = rows[:limit]

    issues: List[VerifyIssue] = []
    verified = 0
    for row in rows:
        provider = str(row.get("provider") or "")
        conversation_id = str(row.get("conversation_id") or "")
        slug = str(row.get("slug") or "")
        stored_content_hash = row.get("content_hash")
        state_entry = _safe_json(row.get("metadata_json"))
        md_path, conversation_dir = _resolve_conversation_paths(
            archive=env.archive,
            provider=provider,
            slug=slug,
            state_entry=state_entry,
        )
        if md_path is None:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=None,
                    message="Unable to resolve conversation path (missing state entry outputPath and archive root).",
                )
            )
            continue
        if not md_path.exists():
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message="Missing conversation.md",
                )
            )
            continue

        existing = read_existing_document(md_path)
        if existing is None:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message="Failed to parse front matter.",
                )
            )
            continue

        polylogue_meta = existing.metadata.get("polylogue")
        if not isinstance(polylogue_meta, dict):
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message="Missing polylogue front matter metadata.",
                )
            )
            continue

        fm_provider = polylogue_meta.get("provider")
        fm_conversation_id = polylogue_meta.get("conversationId")
        fm_slug = polylogue_meta.get("slug")
        fm_content_hash = polylogue_meta.get("contentHash")
        computed_hash = existing.content_hash

        if fm_provider != provider:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message=f"Front matter provider mismatch: {fm_provider!r} != {provider!r}",
                )
            )
        if fm_conversation_id != conversation_id:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message=f"Front matter conversationId mismatch: {fm_conversation_id!r} != {conversation_id!r}",
                )
            )
        if fm_slug != slug:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message=f"Front matter slug mismatch: {fm_slug!r} != {slug!r}",
                    severity="warning",
                )
            )
        if fm_content_hash != computed_hash:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message="Front matter contentHash mismatch (re-render recommended).",
                )
            )
        if stored_content_hash and stored_content_hash != computed_hash:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message="DB content_hash mismatch (re-render or repair DB).",
                )
            )
        state_hash = state_entry.get("contentHash")
        if state_hash and state_hash != computed_hash:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message="State entry contentHash mismatch (re-render recommended).",
                    severity="warning",
                )
            )
        state_output = state_entry.get("outputPath")
        if isinstance(state_output, str) and state_output and str(md_path) != state_output:
            issues.append(
                VerifyIssue(
                    provider=provider,
                    conversation_id=conversation_id,
                    path=md_path,
                    message=f"State entry outputPath mismatch: {state_output!r}",
                    severity="warning",
                )
            )

        conv_dir = conversation_dir or md_path.parent
        # Verify attachments referenced by the index.
        att_rows = env.database.query(
            """
            SELECT attachment_path
              FROM attachments
             WHERE provider = ?
               AND conversation_id = ?
               AND attachment_path IS NOT NULL
            """,
            (provider, conversation_id),
        )
        for att in att_rows:
            raw = att.get("attachment_path") if isinstance(att, dict) else att["attachment_path"]
            if not isinstance(raw, str) or not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = conv_dir / path
            if not path.exists():
                issues.append(
                    VerifyIssue(
                        provider=provider,
                        conversation_id=conversation_id,
                        path=path,
                        message="Missing attachment file referenced by DB.",
                    )
                )

        # Verify branch documents.
        branch_rows = env.database.query(
            "SELECT branch_id FROM branches WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        )
        for branch_row in branch_rows:
            branch_id = branch_row.get("branch_id") if isinstance(branch_row, dict) else branch_row["branch_id"]
            branch_id = str(branch_id or "")
            if not branch_id:
                continue
            branch_dir = conv_dir / "branches" / branch_id
            branch_path = branch_dir / f"{branch_id}.md"
            overlay_path = branch_dir / "overlay.md"
            if not branch_path.exists():
                issues.append(
                    VerifyIssue(
                        provider=provider,
                        conversation_id=conversation_id,
                        path=branch_path,
                        message="Missing branch markdown file.",
                    )
                )
            if not overlay_path.exists():
                issues.append(
                    VerifyIssue(
                        provider=provider,
                        conversation_id=conversation_id,
                        path=overlay_path,
                        message="Missing branch overlay.md file.",
                        severity="warning",
                    )
                )

        verified += 1

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    info_count = sum(1 for issue in issues if issue.severity == "info")

    if json_mode:
        payload = stamp_payload(
            {
                "verified": verified,
                "candidateCount": len(rows),
                "errors": error_count,
                "warnings": warning_count,
                "info": info_count,
                "issues": [issue.as_dict() for issue in issues],
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        ui.console.print(f"Verified {verified} conversation(s).")
        if issues:
            ui.console.print(f"[red]Errors:[/red] {error_count}  [yellow]Warnings:[/yellow] {warning_count}")
            for issue in issues[:50]:
                prefix = "ERROR" if issue.severity == "error" else "WARN" if issue.severity == "warning" else "INFO"
                location = f" ({issue.path})" if issue.path else ""
                ui.console.print(f"[{prefix}] {issue.provider}/{issue.conversation_id}: {issue.message}{location}")
            if len(issues) > 50:
                ui.console.print(f"... and {len(issues) - 50} more issue(s)")

    if error_count:
        raise SystemExit(1)
