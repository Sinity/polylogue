from __future__ import annotations

import html
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from jinja2 import Environment, BaseLoader, select_autoescape

from .db import open_connection
from .util import colorize
from .persistence.state import ConversationStateRepository


def _state_repo() -> ConversationStateRepository:
    return ConversationStateRepository()


@dataclass
class BranchNodeSummary:
    branch_id: str
    parent_branch_id: Optional[str]
    is_canonical: bool
    depth: int
    message_count: int
    token_count: int
    word_count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    divergence_index: int
    divergence_role: Optional[str]
    divergence_snippet: Optional[str]
    attachment_count: int = 0
    branch_path: Optional[Path] = None
    overlay_path: Optional[Path] = None


@dataclass
class BranchConversationSummary:
    provider: str
    conversation_id: str
    slug: str
    title: Optional[str]
    current_branch: Optional[str]
    last_updated: Optional[str]
    branch_count: int
    canonical_branch_id: Optional[str]
    conversation_path: Optional[Path]
    conversation_dir: Optional[Path]
    nodes: Dict[str, BranchNodeSummary] = field(default_factory=dict)

    def ordered_children(self, branch_id: str) -> List[BranchNodeSummary]:
        children = [
            node
            for node in self.nodes.values()
            if node.parent_branch_id == branch_id and node.branch_id != branch_id
        ]
        children.sort(key=lambda n: (n.divergence_index, n.branch_id))
        return children


def _short_snippet(text: Optional[str], *, limit: int = 80) -> Optional[str]:
    if not text:
        return None
    clean = " ".join(text.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _safe_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _branch_paths(conversation_path: Optional[Path], branch_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    if not conversation_path:
        return None, None
    convo_dir = conversation_path.parent
    branch_dir = convo_dir / "branches" / branch_id
    branch_path = branch_dir / f"{branch_id}.md"
    overlay_path = branch_dir / "overlay.md"
    if not branch_path.exists():
        branch_path = None
    if not overlay_path.exists():
        overlay_path = None
    return branch_path, overlay_path


def _load_divergence_metadata(raw: Optional[str]) -> int:
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
        value = payload.get("divergence_index")
        if isinstance(value, int):
            return max(0, value)
    except Exception:
        return 0
    return 0


def _load_branch_overview(
    provider: str,
    conversation_id: str,
    slug: str,
    title: Optional[str],
    current_branch: Optional[str],
    last_updated: Optional[str],
    conversation_path: Optional[Path],
    db_path: Optional[Path] = None,
) -> BranchConversationSummary:
    with open_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                b.branch_id,
                b.parent_branch_id,
                b.is_current,
                b.label,
                b.depth,
                b.metadata_json,
                COALESCE(m.attachment_count, 0) AS attachment_count
            FROM branches AS b
            LEFT JOIN (
                SELECT provider, conversation_id, branch_id, SUM(attachment_count) AS attachment_count
                  FROM messages
                 WHERE provider = ? AND conversation_id = ?
                 GROUP BY provider, conversation_id, branch_id
            ) AS m
              ON b.provider = m.provider
             AND b.conversation_id = m.conversation_id
             AND b.branch_id = m.branch_id
            WHERE b.provider = ? AND b.conversation_id = ?
            ORDER BY b.is_current DESC, b.branch_id
            """,
            (provider, conversation_id, provider, conversation_id),
        ).fetchall()

        divergence_roles: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        if rows:
            branch_ids = [row["branch_id"] for row in rows]
            placeholders = ",".join("?" for _ in branch_ids)
            divergence_map: Dict[str, int] = {}
            for row in rows:
                metadata_json = row["metadata_json"]
                metadata = json.loads(metadata_json) if metadata_json else {}
                divergence_map[row["branch_id"]] = int(metadata.get("divergence_index") or 0)
            if branch_ids:
                sql = f"""
                    SELECT branch_id, role, rendered_text
                      FROM messages
                     WHERE provider = ?
                       AND conversation_id = ?
                       AND branch_id IN ({placeholders})
                     ORDER BY branch_id, position
                """
                cursor = conn.execute(sql, [provider, conversation_id, *branch_ids])
                # Build a minimal cache for the divergence message per branch
                current_branch = None
                position_counter = -1
                for row_msg in cursor:
                    branch_id = row_msg["branch_id"]
                    if branch_id != current_branch:
                        current_branch = branch_id
                        position_counter = 0
                    else:
                        position_counter += 1
                    target_idx = divergence_map.get(branch_id, 0)
                    if position_counter == target_idx:
                        divergence_roles[branch_id] = (
                            row_msg["role"],
                            row_msg["rendered_text"] or "",
                        )
                # Ensure all branches have a key even if missing messages
                for bid in branch_ids:
                    divergence_roles.setdefault(bid, (None, None))

        if not rows:
            return BranchConversationSummary(
                provider=provider,
                conversation_id=conversation_id,
                slug=slug,
                title=title,
                current_branch=current_branch,
                last_updated=last_updated,
                branch_count=0,
                canonical_branch_id=current_branch,
                conversation_path=conversation_path,
                conversation_dir=conversation_path.parent if conversation_path else None,
                nodes={},
            )

        nodes: Dict[str, BranchNodeSummary] = {}

        for row in rows:
            branch_id = row["branch_id"]
            metadata_json = row["metadata_json"]
            metadata = json.loads(metadata_json) if metadata_json else {}
            divergence_index = int(metadata.get("divergence_index") or 0)
            role, snippet = divergence_roles.get(branch_id, (None, None))
            branch_path, overlay_path = _branch_paths(conversation_path, branch_id)
            nodes[branch_id] = BranchNodeSummary(
                branch_id=branch_id,
                parent_branch_id=row["parent_branch_id"],
                is_canonical=bool(row["is_current"]),
                depth=row["depth"] or 0,
                message_count=int(metadata.get("message_count") or 0),
                token_count=int(metadata.get("token_count") or 0),
                word_count=int(metadata.get("word_count") or 0),
                first_timestamp=metadata.get("first_timestamp"),
                last_timestamp=metadata.get("last_timestamp"),
                divergence_index=divergence_index,
                divergence_role=role,
                divergence_snippet=_short_snippet(snippet),
                attachment_count=row["attachment_count"] or 0,
                branch_path=branch_path,
                overlay_path=overlay_path,
            )

    canonical_branch_id = None
    for node in nodes.values():
        if node.is_canonical:
            canonical_branch_id = node.branch_id
            break
    if canonical_branch_id is None and nodes:
        canonical_branch_id = next(iter(nodes))

    return BranchConversationSummary(
        provider=provider,
        conversation_id=conversation_id,
        slug=slug,
        title=title,
        current_branch=current_branch or canonical_branch_id,
        last_updated=last_updated,
        branch_count=len(nodes),
        canonical_branch_id=canonical_branch_id,
        conversation_path=conversation_path,
        conversation_dir=conversation_path.parent if conversation_path else None,
        nodes=nodes,
    )


def _fetch_divergence_message(
    conn,
    *,
    provider: str,
    conversation_id: str,
    branch_id: str,
    divergence_index: int,
) -> Tuple[Optional[str], Optional[str]]:
    row = conn.execute(
        """
        SELECT role, rendered_text
          FROM messages
         WHERE provider = ?
           AND conversation_id = ?
           AND branch_id = ?
         ORDER BY position
         LIMIT 1 OFFSET ?
        """,
        (provider, conversation_id, branch_id, max(divergence_index, 0)),
    ).fetchone()
    if not row:
        return None, None
    role = row["role"]
    body = row["rendered_text"] or ""
    return role, body


def list_branch_conversations(
    *,
    provider: Optional[str] = None,
    slug: Optional[str] = None,
    conversation_id: Optional[str] = None,
    min_branches: int = 1,
) -> List[BranchConversationSummary]:
    results: List[BranchConversationSummary] = []
    with open_connection() as conn:
        clauses: List[str] = []
        params: List[object] = []
        if provider:
            clauses.append("c.provider = ?")
            params.append(provider)
        if slug:
            clauses.append("c.slug = ?")
            params.append(slug)
        if conversation_id:
            clauses.append("c.conversation_id = ?")
            params.append(conversation_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_clause = "ORDER BY (c.last_updated IS NULL), c.last_updated DESC, c.title ASC"
        query = f"""
            SELECT
                c.provider,
                c.conversation_id,
                c.slug,
                c.title,
                c.current_branch,
                c.last_updated,
                COUNT(b.branch_id) AS branch_count
            FROM conversations AS c
            LEFT JOIN branches AS b
              ON c.provider = b.provider AND c.conversation_id = b.conversation_id
            {where}
            GROUP BY c.provider, c.conversation_id
            HAVING branch_count >= ?
            {order_clause}
        """
        params.append(max(0, min_branches))
        rows = conn.execute(query, params).fetchall()

    for row in rows:
        state = _state_repo().get(row["provider"], row["conversation_id"]) or {}
        output_path = _safe_path(state.get("outputPath"))
        summary = _load_branch_overview(
            provider=row["provider"],
            conversation_id=row["conversation_id"],
            slug=row["slug"],
            title=row["title"],
            current_branch=row["current_branch"],
            last_updated=row["last_updated"],
            conversation_path=output_path,
        )
        if summary.nodes:
            results.append(summary)
    return results


def format_branch_tree(conversation: BranchConversationSummary, *, use_color: bool = True) -> str:
    if not conversation.nodes:
        title = conversation.title or conversation.slug
        return f"No branch data recorded for {title} ({conversation.provider})."

    canonical_id = conversation.canonical_branch_id or conversation.current_branch
    if canonical_id not in conversation.nodes:
        canonical_id = next(iter(conversation.nodes))

    def render_label(node: BranchNodeSummary) -> str:
        parts: List[str] = []
        base = node.branch_id
        if node.is_canonical:
            base = f"{base} [canonical]"
        parts.append(base)
        parts.append(f"messages={node.message_count}")
        if node.token_count:
            parts.append(f"tokens={node.token_count}")
        if node.attachment_count:
            parts.append(f"attachments={node.attachment_count}")
        if node.divergence_index and not node.is_canonical:
            delta = node.divergence_index + 1
            prefix = ""
            if node.divergence_role:
                prefix = f"{node.divergence_role}: "
            snippet = node.divergence_snippet or ""
            parts.append(f"delta#{delta} {prefix}{snippet}")
        label = " — ".join(parts)
        if not use_color or node.is_canonical:
            return label if not node.is_canonical else colorize(label, "green") if use_color else label
        return colorize(label, "cyan")

    lines: List[str] = []

    def walk(node: BranchNodeSummary, prefix: str = "", is_last: bool = True) -> None:
        connector = "└─ " if is_last else "├─ "
        if not lines:
            lines.append(render_label(node))
        else:
            lines.append(f"{prefix}{connector}{render_label(node)}")
        children = conversation.ordered_children(node.branch_id)
        if not children:
            return
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        for idx, child in enumerate(children):
            walk(child, child_prefix, idx == len(children) - 1)

    root_node = conversation.nodes.get(canonical_id)
    if root_node is None:
        root_node = next(iter(conversation.nodes.values()))
    walk(root_node, "", True)
    return "\n".join(lines)


def branch_diff(conversation: BranchConversationSummary, branch_id: str) -> Optional[str]:
    canonical_id = conversation.canonical_branch_id or conversation.current_branch
    if not canonical_id:
        return None
    if branch_id == canonical_id:
        return None
    canonical_node = conversation.nodes.get(canonical_id)
    target_node = conversation.nodes.get(branch_id)
    if not canonical_node or not target_node:
        return None
    canonical_path = canonical_node.branch_path or conversation.conversation_path
    branch_path = target_node.branch_path
    if not canonical_path or not canonical_path.exists():
        return None
    if not branch_path or not branch_path.exists():
        return None
    canonical_lines = canonical_path.read_text(encoding="utf-8").splitlines()
    branch_lines = branch_path.read_text(encoding="utf-8").splitlines()
    from difflib import unified_diff

    diff_lines = unified_diff(
        canonical_lines,
        branch_lines,
        fromfile=f"{canonical_id}.md",
        tofile=f"{branch_id}.md",
        lineterm="",
    )
    return "\n".join(diff_lines)


_BRANCH_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Branch graph · {{ title }}</title>
    <style>
      :root {
        color-scheme: {{ 'dark' if theme == 'dark' else 'light' }};
      }
      body {
        background: {{ palette.bg }};
        color: {{ palette.fg }};
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        margin: 2rem auto;
        max-width: 960px;
        line-height: 1.6;
      }
      h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        color: {{ palette.muted }};
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
      }
      .branch-card {
        border: 1px solid {{ palette.border }};
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        background: {{ palette.alternate_bg }};
      }
      .branch-card.canonical {
        background: {{ palette.canonical_bg }};
      }
      .branch-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
      }
      .branch-header h3 {
        margin: 0;
      }
      .badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }
      .badge {
        background: {{ palette.border }};
        color: {{ palette.fg }};
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
      }
      .badge.canonical {
        background: {{ palette.accent }};
        color: #fff;
      }
      .divergence {
        margin-top: 0.75rem;
        font-size: 0.95rem;
      }
      .links {
        margin-top: 0.75rem;
        font-size: 0.9rem;
      }
      .links a {
        color: {{ palette.accent }};
      }
      ul {
        list-style: none;
        padding-left: 1.5rem;
      }
      ul > li {
        position: relative;
      }
      ul > li::before {
        content: "";
        position: absolute;
        left: -1rem;
        top: 0;
        bottom: 0;
        width: 1px;
        background: {{ palette.border }};
      }
    </style>
  </head>
  <body>
    <h1>Branch graph · {{ title }}</h1>
    <div class="meta">
      {% for item in conversation_meta %}
      <span>{{ item }}</span>
      {% endfor %}
    </div>
    {{ render_node(canonical_node) }}
    <ul>
      {% for child in children %}
        {{ render_child(child) }}
      {% endfor %}
    </ul>
  </body>
</html>
"""


def _branch_template() -> Environment:
    env = Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(enabled_extensions=("html",)),
    )
    env.filters["href"] = lambda path: quote(path.as_posix(), safe="/:") if path else None
    env.filters["badge_list"] = lambda badges: "".join(badges)
    env.filters["safe"] = lambda x: x
    return env


def build_branch_html(conversation: BranchConversationSummary, *, theme: str = "light") -> str:
    palettes = {
        "light": {
            "bg": "#ffffff",
            "fg": "#1f2933",
            "accent": "#2563eb",
            "muted": "#6b7280",
            "border": "#d1d5db",
            "canonical_bg": "#d1fae5",
            "alternate_bg": "#dbeafe",
        },
        "dark": {
            "bg": "#111827",
            "fg": "#e5e7eb",
            "accent": "#93c5fd",
            "muted": "#9ca3af",
            "border": "#374151",
            "canonical_bg": "#065f461a",
            "alternate_bg": "#1e3a8a1f",
        },
    }
    palette = palettes.get(theme, palettes["light"])

    env = _branch_template()
    template = env.from_string(_BRANCH_TEMPLATE)

    canonical_id = conversation.canonical_branch_id or conversation.current_branch
    if canonical_id not in conversation.nodes:
        canonical_id = next(iter(conversation.nodes))
    canonical_node = conversation.nodes[canonical_id]
    children = conversation.ordered_children(canonical_id)

    def render_node(node: BranchNodeSummary) -> str:
        badges: List[str] = []
        if node.is_canonical:
            badges.append('<span class="badge canonical">canonical</span>')
        else:
            parent_label = html.escape(node.parent_branch_id or "—")
            badges.append(f'<span class="badge">parent: {parent_label}</span>')
        badges.append(f'<span class="badge">messages: {html.escape(str(node.message_count))}</span>')
        if node.token_count:
            badges.append(f'<span class="badge">tokens: {html.escape(str(node.token_count))}</span>')
        if node.attachment_count:
            badges.append(
                f'<span class="badge">attachments: {html.escape(str(node.attachment_count))}</span>'
            )
        divergence_html = ""
        if node.divergence_index and not node.is_canonical:
            delta = node.divergence_index + 1
            role = html.escape(node.divergence_role or "")
            snippet = html.escape(node.divergence_snippet or "")
            divergence_html = f'<div class="divergence"><strong>delta message #{delta}</strong> {role}: {snippet}</div>'
        links: List[str] = []
        if node.branch_path:
            links.append(f'<a href="{quote(node.branch_path.as_posix(), safe="/:")}">branch markdown</a>')
        if node.overlay_path:
            links.append(f'<a href="{quote(node.overlay_path.as_posix(), safe="/:")}">overlay</a>')
        link_html = ""
        if links:
            link_html = f'<div class="links">{" • ".join(links)}</div>'
        child_html = ""
        sub_children = conversation.ordered_children(node.branch_id)
        if sub_children:
            rendered = "".join(render_child(child) for child in sub_children)
            child_html = f"<ul>{rendered}</ul>"
        card_class = "branch-card canonical" if node.is_canonical else "branch-card alternate"
        return (
            f'<div class="{card_class}">'
            f'<div class="branch-header"><h3>{html.escape(node.branch_id)}</h3><div class="badges">{"".join(badges)}</div></div>'
            f"{divergence_html}"
            f"{link_html}"
            f"{child_html}"
            f"</div>"
        )

    def render_child(node: BranchNodeSummary) -> str:
        return f"<li>{render_node(node)}</li>"

    conversation_meta: List[str] = []
    conversation_meta.append(f"<strong>Provider:</strong> {html.escape(conversation.provider)}")
    conversation_meta.append(f"<strong>Slug:</strong> {html.escape(conversation.slug)}")
    conversation_meta.append(
        f"<strong>Conversation ID:</strong> {html.escape(conversation.conversation_id)}"
    )
    conversation_meta.append(
        f"<strong>Last updated:</strong> {html.escape(conversation.last_updated or 'unknown')}"
    )
    if conversation.conversation_path:
        conversation_meta.append(
            f"<strong>Canonical file:</strong> {html.escape(str(conversation.conversation_path))}"
        )

    return template.render(
        title=conversation.title or conversation.slug,
        palette=palette,
        theme=theme,
        conversation_meta=conversation_meta,
        canonical_node=canonical_node,
        children=children,
        render_node=render_node,
        render_child=render_child,
    )
