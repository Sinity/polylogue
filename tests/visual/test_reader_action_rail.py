"""Current React session reader shell contract."""

from __future__ import annotations

from tests.visual.conftest import READER_C1, ReaderWorkspace, get_text, parse_dom, running_reader_server


def test_reader_session_shell_contract(reader_workspace: ReaderWorkspace) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, f"/s/{READER_C1}")

    assert status == 200
    assert "text/html" in content_type
    dom = parse_dom(body)
    assert "message-flow" in dom.ids
    assert "message-flow-more" in dom.ids
    assert f"msg-{READER_C1}:n:reader-c1-m1" in dom.ids
    assert "/assets/session-read-" in body
