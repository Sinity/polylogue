"""Semantic transcript card web wiring (polylogue-ap7).

Pins the daemon web shell's contract for the shared CLI/web renderer
registry: the daemon session-detail routes (``daemon/http.py``) attach
``semantic_entries`` plus compatibility ``semantic_cards`` to every message
from the exact ``SemanticTranscript`` registry the CLI renders to Markdown
(``rendering/semantic_cards.py``, exercised by
``tests/unit/rendering/test_semantic_cards.py``), and the web reader
(``daemon/webui.py``) renders that same ordered entry JSON
to DOM without reclassifying raw roles or prose.

Structure parity is proven directly against both public schemas on the actual
paginated message route used by the browser, rather than inferred from the
full-detail compatibility route.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from urllib.parse import quote

import jsonschema
import pytest

from tests.visual.conftest import (
    READER_SEM1,
    READER_SEM1_EDIT_RESULT,
    READER_SEM1_EDIT_USE,
    READER_SEM1_SHELL_RESULT,
    READER_SEM1_SHELL_USE,
    READER_SEM1_TASK_USE,
    ReaderWorkspace,
    assert_no_private_paths,
    get_json,
    get_text,
    running_reader_server,
    seed_reader_semantic_cards,
    write_evidence_manifest,
)

SEMANTIC_CARD_SCHEMA_PATH = Path("docs/schemas/semantic-card-v1.schema.json")
SEMANTIC_TRANSCRIPT_SCHEMA_PATH = Path("docs/schemas/semantic-transcript-v1.schema.json")


def _message(messages: list[dict[str, object]], message_id: str) -> dict[str, object]:
    return next(m for m in messages if m["id"] == message_id)


def _field(card: dict[str, object], label: str) -> str | None:
    for field in cast("list[dict[str, object]]", card["fields"]):
        if field["label"] == label:
            return cast(str, field["value"])
    return None


def _preview(card: dict[str, object], kind: str) -> dict[str, object] | None:
    for preview in cast("list[dict[str, object]]", card["previews"]):
        if preview["kind"] == kind:
            return preview
    return None


def _source(card: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", card["source"])


def _outcome(card: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", card["outcome"])


@pytest.fixture
def _semantic_card_session(reader_workspace: ReaderWorkspace) -> tuple[str, dict[str, object]]:
    seed_reader_semantic_cards(reader_workspace)
    with running_reader_server(reader_workspace) as (_, base_url):
        payload = get_json(base_url, f"/api/sessions/{READER_SEM1}/messages?limit=100&offset=0")
    assert isinstance(payload, dict)
    return base_url, payload


def test_semantic_card_web_json_contract(_semantic_card_session: tuple[str, dict[str, object]]) -> None:
    _, payload = _semantic_card_session
    messages = cast("list[dict[str, object]]", payload["messages"])

    schema = json.loads(SEMANTIC_CARD_SCHEMA_PATH.read_text(encoding="utf-8"))
    transcript_schema = json.loads(SEMANTIC_TRANSCRIPT_SCHEMA_PATH.read_text(encoding="utf-8"))
    ordered_entries = cast("list[dict[str, object]]", payload.get("semantic_entries", [])) + [
        entry for message in messages for entry in cast("list[dict[str, object]]", message["semantic_entries"])
    ]
    jsonschema.validate(
        instance={
            "schema_version": "semantic-transcript.v1",
            "session_id": READER_SEM1,
            "entries": ordered_entries,
        },
        schema=transcript_schema,
    )
    all_cards: list[dict[str, object]] = []
    for message in messages:
        cards = cast("list[dict[str, object]]", message["semantic_cards"])
        all_cards.extend(cards)
    assert all_cards, "expected at least one semantic card in the seeded session"
    for card in all_cards:
        jsonschema.validate(instance=card, schema=schema)
        # The literal that only ``SemanticCard.to_document()`` produces —
        # proves the web payload came from the shared pure card model, not a
        # parallel web-only tool-classification shortcut.
        assert card["schema_version"] == "semantic-card.v1"

    # --- Bash: paired, succeeded, exit 0 --------------------------------
    shell_use = _message(messages, READER_SEM1_SHELL_USE)
    shell_cards = cast("list[dict[str, object]]", shell_use["semantic_cards"])
    assert len(shell_cards) == 1
    shell_card = shell_cards[0]
    assert shell_card["kind"] == "shell"
    assert _source(shell_card)["tool_name"] == "Bash"
    assert _outcome(shell_card)["state"] == "succeeded"
    assert _outcome(shell_card)["exit_code"] == 0
    assert _field(shell_card, "command") == "pytest -q tests/unit/rendering"
    output_preview = _preview(shell_card, "output")
    assert output_preview is not None
    assert "5 passed in 0.42s" in cast(str, output_preview["text"])

    shell_result = _message(messages, READER_SEM1_SHELL_RESULT)
    assert shell_result["semantic_card_suppressed"] is True
    assert shell_result["semantic_cards"] == []

    # --- Edit: paired, failed, diff preview -----------------------------
    edit_use = _message(messages, READER_SEM1_EDIT_USE)
    edit_cards = cast("list[dict[str, object]]", edit_use["semantic_cards"])
    assert len(edit_cards) == 1
    edit_card = edit_cards[0]
    assert edit_card["kind"] == "file_edit"
    assert _outcome(edit_card)["state"] == "failed"
    assert _field(edit_card, "path") == "src/app.py"
    diff_preview = _preview(edit_card, "diff")
    assert diff_preview is not None
    diff_text = cast(str, diff_preview["text"])
    assert "-return 1" in diff_text
    assert "+return 2" in diff_text

    edit_result = _message(messages, READER_SEM1_EDIT_RESULT)
    assert edit_result["semantic_card_suppressed"] is True
    assert edit_result["semantic_cards"] == []

    # --- Task: unpaired, outcome unknown ---------------------------------
    task_use = _message(messages, READER_SEM1_TASK_USE)
    task_cards = cast("list[dict[str, object]]", task_use["semantic_cards"])
    assert len(task_cards) == 1
    task_card = task_cards[0]
    assert task_card["kind"] == "task"
    assert _outcome(task_card)["state"] == "unknown"
    assert _field(task_card, "agent") == "general-purpose"
    assert _field(task_card, "request") == "Investigate the flaky concurrency test"


def test_semantic_card_web_dom_shape_contract(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Pin the served semantic-card DOM against the typed WebUI renderer.

    This is the DOM-level half of the contract the JSON-envelope tests above
    pin at the data level: it proves the session read route actually delegates
    to ``polylogue/daemon/webui.py``'s ``_render_semantic_card`` rather than
    re-deriving tool semantics from raw ``has_tool_use`` flags.

    Anti-vacuity: it goes red if the session read route stops rendering cards
    (the page falls back to ``_render_message_flow_fallback_body``), or if the
    card wrapper, header, outcome, or field class names move without this
    contract moving with them.
    """
    seed_reader_semantic_cards(reader_workspace)
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, f"/sessions/{quote(READER_SEM1, safe='')}")

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(body, context="reader session read HTML")

    # Structural wiring emitted by ``_render_semantic_card`` /
    # ``_render_semantic_outcome`` in polylogue/daemon/webui.py.
    for phrase in (
        "data-card-kind=",
        "card__header",
        "card__kind",
        "card__title",
        "card__field",
        "card__outcome",
        "data-outcome-state=",
    ):
        assert phrase in body, f"semantic card wiring missing {phrase!r}"

    # The seeded session covers three card kinds; each must reach the DOM as
    # its own typed card rather than a generic fallback.
    for kind in ("shell", "file_edit", "task"):
        assert f'data-card-kind="{kind}"' in body, f"semantic card kind missing {kind!r}"

    write_evidence_manifest(
        tmp_path / "reader-semantic-cards-evidence.json",
        artifact_id="polylogue.local_reader.semantic_cards",
        route="/sessions/{id}",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "status": status,
            "content_type": content_type,
            "semantic_transcript_renderer_wired": True,
            "card_kinds_covered": ["shell", "file_edit", "task"],
            "suppression_wired": True,
            "private_path_safe": True,
        },
    )
