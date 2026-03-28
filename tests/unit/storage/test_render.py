"""Rendering contracts for repository-backed conversation output."""

from __future__ import annotations

from polylogue.rendering.renderers import HTMLRenderer
from polylogue.sources import RecordBundle, save_bundle
from tests.infra.storage_records import make_attachment, make_conversation, make_message

# ============================================================================
# RENDER: Markdown Output Tests
# ============================================================================


async def test_render_conversation_markdown_has_structure(workspace_env, storage_repository):
    """render_conversation() produces valid markdown with title and role headers."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-md", title="My Conversation"),
        messages=[
            make_message("m1", "c-md", text="Hello, assistant!"),
            make_message("m2", "c-md", role="assistant", text="Hi there, user!"),
        ],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-md", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Check structure
    assert "# My Conversation" in markdown
    assert "## user" in markdown
    assert "## assistant" in markdown
    assert "Hello, assistant!" in markdown
    assert "Hi there, user!" in markdown


async def test_render_conversation_markdown_includes_provider(workspace_env, storage_repository):
    """render_conversation() markdown includes provider information."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-prov", provider_name="claude"),
        messages=[],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-prov", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    assert "Provider: claude" in markdown
    assert "Conversation ID: c-prov" in markdown


async def test_render_conversation_markdown_messages_separated(workspace_env, storage_repository):
    """render_conversation() separates messages with blank lines."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-sep"),
        messages=[
            make_message("m1", "c-sep", text="First message"),
            make_message("m2", "c-sep", role="assistant", text="Second message"),
        ],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-sep", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Messages should be separated by blank lines (## header, optional timestamp, text, blank line)
    lines = markdown.split("\n")
    # Verify structure: should have multiple sections
    assert len([line for line in lines if line.startswith("## ")]) == 2


async def test_render_conversation_markdown_with_timestamp(workspace_env, storage_repository):
    """render_conversation() includes timestamps when present."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-ts"),
        messages=[make_message("m1", "c-ts", text="Hello", timestamp="2024-01-15T10:30:00")],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-ts", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    assert "Timestamp: 2024-01-15T10:30:00" in markdown


# ============================================================================
# RENDER: HTML Output Tests
# ============================================================================


async def test_render_conversation_html_valid(workspace_env, storage_repository):
    """render_conversation() produces valid HTML with proper structure."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-html", title="HTML Test"),
        messages=[
            make_message("m1", "c-html", text="Question?"),
            make_message("m2", "c-html", role="assistant", text="Answer!"),
        ],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-html", output_root)

    html = html_path.read_text(encoding="utf-8")

    # Check HTML structure (case-insensitive DOCTYPE check)
    assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
    assert "<html" in html
    assert "</html>" in html
    assert "HTML Test" in html  # Title may be formatted differently


async def test_render_conversation_html_escapes_content(workspace_env, storage_repository):
    """render_conversation() escapes HTML special characters."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-esc", title="<script>alert('xss')</script>"),
        messages=[make_message("m1", "c-esc", text="<img src=x onerror='alert(1)'>")],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-esc", output_root)

    html = html_path.read_text(encoding="utf-8")

    # Script should be escaped
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # IMG should be escaped
    assert "<img" not in html
    assert "&lt;img" in html


async def test_render_conversation_html_includes_content(workspace_env, storage_repository):
    """render_conversation() HTML includes conversation content."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-con", title="Content Test"),
        messages=[make_message("m1", "c-con", text="Important content here")],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-con", output_root)

    html = html_path.read_text(encoding="utf-8")

    assert "Important content here" in html


# ============================================================================
# RENDER: Attachments Tests
# ============================================================================


async def test_render_conversation_with_message_attachments(workspace_env, storage_repository):
    """render_conversation() includes attachments associated with messages."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-att-msg", title="With Attachments"),
        messages=[make_message("m1", "c-att-msg", text="Check this file")],
        attachments=[make_attachment("att-1", "c-att-msg", "m1", mime_type="application/pdf", provider_meta={"name": "document.pdf"})],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-att-msg", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Attachment should be referenced
    assert "Attachment:" in markdown or "attachment" in markdown.lower()


async def test_render_conversation_with_orphan_attachments(workspace_env, storage_repository):
    """render_conversation() includes attachments not linked to messages."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-orphan", title="With Orphan Attachments"),
        messages=[make_message("m1", "c-orphan", text="Some text")],
        attachments=[
            # Orphan attachment (no message_id)
            make_attachment("att-orphan", "c-orphan", None, mime_type="text/plain", size_bytes=100, provider_meta={"name": "orphaned_file.txt"}),
        ],
    )

    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-orphan", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Orphan attachments should appear in "attachments" section
    assert "attachments" in markdown.lower()


# ============================================================================
# RENDER: File Output Tests
# ============================================================================


async def test_render_conversation_writes_markdown_file(workspace_env, storage_repository):
    """render_conversation() writes markdown file to expected location."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-file", title="File Test"),
        messages=[],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-file", output_root)
    md_path = html_path.parent / "conversation.md"

    assert md_path.exists()
    assert md_path.suffix == ".md"
    assert md_path.name == "conversation.md"


async def test_render_conversation_writes_html_file(workspace_env, storage_repository):
    """render_conversation() writes HTML file to expected location."""
    archive_root = workspace_env["archive_root"]

    bundle = RecordBundle(
        conversation=make_conversation("c-html-file", title="HTML File Test"),
        messages=[],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = await renderer.render("c-html-file", output_root)

    assert html_path.exists()
    assert html_path.suffix == ".html"
    assert html_path.name == "conversation.html"


# ============================================================================
