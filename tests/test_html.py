from pathlib import Path

from polylogue.branch_explorer import BranchConversationSummary, BranchNodeSummary, build_branch_html
from polylogue.html import HtmlRenderOptions, render_html, _transform_callouts
from polylogue.render import AttachmentInfo, MarkdownDocument


def test_transform_callouts_wraps_details():
    html = (
        "<blockquote>\n"
        "<p>[!INFO]- Collapsed Block</p>\n"
        "<p>First line</p>\n"
        "<p>Second line</p>\n"
        "</blockquote>"
    )
    transformed = _transform_callouts(html)
    assert '<details class="callout" data-kind="info">' in transformed
    assert '<summary>Collapsed Block</summary>' in transformed
    assert 'First line' in transformed
    assert 'Second line' in transformed
    assert ' open' not in transformed


def test_transform_callouts_preserves_plain_blockquotes():
    html = "<blockquote><p>Regular quote</p></blockquote>"
    transformed = _transform_callouts(html)
    assert transformed == html


def test_transform_callouts_handles_open_callout():
    html = (
        "<blockquote>\n"
        "<p>[!TIP]+ Helpful</p>\n"
        "<p><strong>Bold</strong> text</p>\n"
        "</blockquote>"
    )
    transformed = _transform_callouts(html)
    assert 'open' in transformed
    assert 'data-kind="tip"' in transformed
    assert '<strong>Bold</strong>' in transformed


def test_render_html_metadata_escaped(tmp_path: Path):
    doc = MarkdownDocument(
        body="Just text",
        metadata={"title": "Safe", "author": "<script>alert('x')</script>"},
        attachments=[AttachmentInfo(name="a", link="http://example.com", local_path=None, size_bytes=None, remote=True)],
        stats={},
    )
    output = render_html(doc, HtmlRenderOptions())
    assert "<script>alert" not in output
    assert "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;" in output


def test_branch_html_escapes_untrusted_content(tmp_path: Path):
    branch_path = Path("branch file.md")
    node = BranchNodeSummary(
        branch_id="branch<script>",
        parent_branch_id="parent<script>",
        is_canonical=True,
        depth=0,
        message_count=1,
        token_count=10,
        word_count=5,
        first_timestamp=None,
        last_timestamp=None,
        divergence_index=0,
        divergence_role="assistant<script>",
        divergence_snippet="<script>snip</script>",
        attachment_count=2,
        branch_path=branch_path,
        overlay_path=Path("overlay file.md"),
    )
    summary = BranchConversationSummary(
        provider="provider<script>",
        conversation_id="conv<script>",
        slug="slug<script>",
        title="Title<script>",
        current_branch="branch-000",
        last_updated="2024-01-01",
        branch_count=1,
        canonical_branch_id="branch-000",
        conversation_path=None,
        conversation_dir=None,
        nodes={"branch-000": node},
    )
    html_output = build_branch_html(summary)
    assert "<script>" not in html_output
    assert "branch%20file.md" in html_output


def test_branch_html_handles_empty_conversation():
    summary = BranchConversationSummary(
        provider="chatgpt",
        conversation_id="conv-empty",
        slug="empty",
        title="Empty",
        current_branch=None,
        last_updated=None,
        branch_count=0,
        canonical_branch_id=None,
        conversation_path=None,
        conversation_dir=None,
        nodes={},
    )

    html_output = build_branch_html(summary, theme="light")

    assert "No branch data" in html_output


def test_branch_html_includes_inline_diff_when_paths_exist(tmp_path: Path):
    canonical_path = tmp_path / "conversation.md"
    branch_path = tmp_path / "branch-001.md"
    canonical_path.write_text("---\n---\n\nHello\n", encoding="utf-8")
    branch_path.write_text("---\n---\n\nHello world\n", encoding="utf-8")

    canonical = BranchNodeSummary(
        branch_id="branch-000",
        parent_branch_id=None,
        is_canonical=True,
        depth=0,
        message_count=1,
        token_count=0,
        word_count=1,
        first_timestamp=None,
        last_timestamp=None,
        divergence_index=0,
        divergence_role=None,
        divergence_snippet=None,
        attachment_count=0,
        branch_path=canonical_path,
        overlay_path=None,
    )
    alt = BranchNodeSummary(
        branch_id="branch-001",
        parent_branch_id="branch-000",
        is_canonical=False,
        depth=0,
        message_count=1,
        token_count=0,
        word_count=2,
        first_timestamp=None,
        last_timestamp=None,
        divergence_index=1,
        divergence_role="model",
        divergence_snippet="Hello world",
        attachment_count=0,
        branch_path=branch_path,
        overlay_path=None,
    )
    summary = BranchConversationSummary(
        provider="chatgpt",
        conversation_id="conv-1",
        slug="conv-1",
        title="Title",
        current_branch="branch-000",
        last_updated="2024-01-01",
        branch_count=2,
        canonical_branch_id="branch-000",
        conversation_path=canonical_path,
        conversation_dir=tmp_path,
        nodes={"branch-000": canonical, "branch-001": alt},
    )
    html_output = build_branch_html(summary, theme="light")
    assert "Diff vs canonical" in html_output
    assert "-Hello" in html_output
    assert "+Hello world" in html_output
