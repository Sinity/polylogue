from polylogue.html import _transform_callouts


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
