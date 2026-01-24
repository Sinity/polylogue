"""Tests for server template filters."""


from polylogue.server.web import _render_markdown, templates


class TestRenderMarkdownFilter:
    """Test the render_markdown Jinja2 filter."""

    def test_render_markdown_simple_text(self):
        """Render plain text as paragraph."""
        result = _render_markdown("Hello world")
        assert "<p>Hello world</p>" in result

    def test_render_markdown_with_formatting(self):
        """Render markdown with bold, italic, code."""
        text = "This is **bold** and *italic* and `code`"
        result = _render_markdown(text)
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
        assert "<code>code</code>" in result

    def test_render_markdown_with_links(self):
        """Render markdown links (explicit syntax)."""
        text = "Visit [example](https://example.com) for more"
        result = _render_markdown(text)
        assert '<a href="https://example.com">example</a>' in result

    def test_render_markdown_with_code_block(self):
        """Render fenced code blocks."""
        text = "```python\nprint('hello')\n```"
        result = _render_markdown(text)
        assert "<pre>" in result or "<code>" in result
        assert "print('hello')" in result

    def test_render_markdown_with_table(self):
        """Render markdown tables (table extension enabled)."""
        text = """| Col1 | Col2 |
|------|------|
| A    | B    |"""
        result = _render_markdown(text)
        assert "<table>" in result
        assert "<th>Col1</th>" in result
        assert "<td>A</td>" in result

    def test_render_markdown_empty_string(self):
        """Render empty string returns empty."""
        result = _render_markdown("")
        assert result == ""

    def test_render_markdown_none(self):
        """Render None returns empty string."""
        result = _render_markdown(None)
        assert result == ""

    def test_render_markdown_multiline(self):
        """Render multiline markdown with paragraphs."""
        text = "First paragraph\n\nSecond paragraph"
        result = _render_markdown(text)
        # Should have two separate paragraphs
        assert result.count("<p>") == 2

    def test_render_markdown_no_html_passthrough(self):
        """HTML in markdown should be escaped (html: False)."""
        text = "Text with <script>alert('xss')</script> tag"
        result = _render_markdown(text)
        # HTML should be escaped or not rendered as raw HTML
        assert "<script>" not in result or "&lt;script&gt;" in result

    def test_filter_registered_in_jinja_env(self):
        """Verify render_markdown is registered as Jinja2 filter."""
        assert "render_markdown" in templates.env.filters
        assert templates.env.filters["render_markdown"] == _render_markdown

    def test_filter_usage_in_template_context(self):
        """Test filter can be called through Jinja2 environment."""
        filter_func = templates.env.filters["render_markdown"]
        result = filter_func("**bold text**")
        assert "<strong>bold text</strong>" in result

    def test_render_markdown_with_lists(self):
        """Render unordered and ordered lists."""
        text = "- Item 1\n- Item 2\n\n1. First\n2. Second"
        result = _render_markdown(text)
        assert "<ul>" in result
        assert "<ol>" in result
        assert "<li>" in result

    def test_render_markdown_with_headers(self):
        """Render headers (h1-h6)."""
        text = "# H1\n## H2\n### H3"
        result = _render_markdown(text)
        assert "<h1>" in result
        assert "<h2>" in result
        assert "<h3>" in result

    def test_render_markdown_with_blockquote(self):
        """Render blockquotes."""
        text = "> This is a quote\n> Multi-line"
        result = _render_markdown(text)
        assert "<blockquote>" in result
        assert "This is a quote" in result
