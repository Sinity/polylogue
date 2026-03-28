"""Law-based contracts for Drive client payload parsing and file filtering."""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.drive_client import (
    GEMINI_PROMPT_MIME_TYPE,
    DriveClient,
    _parse_downloaded_json_payload,
)
from tests.infra.drive_mocks import MockDriveService, mock_drive_file
from tests.infra.strategies import json_document_strategy


@given(
    st.lists(json_document_strategy(), min_size=0, max_size=8),
    st.sampled_from(("session.jsonl", "session.jsonl.txt", "session.ndjson", "SESSION.JSONL")),
)
@settings(max_examples=35)
def test_parse_downloaded_json_payload_preserves_newline_delimited_documents(
    documents: list[dict[str, object]],
    name: str,
) -> None:
    raw = b"\n\n".join(json.dumps(document).encode("utf-8") for document in documents)
    assert _parse_downloaded_json_payload(raw, name=name) == documents


@given(
    st.one_of(
        json_document_strategy(),
        st.lists(json_document_strategy(), min_size=0, max_size=6),
    )
)
@settings(max_examples=35)
def test_parse_downloaded_json_payload_round_trips_standard_json(payload: object) -> None:
    raw = json.dumps(payload).encode("utf-8")
    assert _parse_downloaded_json_payload(raw, name="payload.json") == payload


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.sampled_from((".json", ".jsonl", ".jsonl.txt", ".ndjson", ".txt", ".md", "")),
            st.sampled_from((
                "application/json",
                "text/plain",
                "application/octet-stream",
                GEMINI_PROMPT_MIME_TYPE,
            )),
            st.booleans(),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda item: item[0],
    )
)
@settings(max_examples=30, deadline=None)
def test_iter_json_files_filters_supported_entries(
    file_specs: list[tuple[int, str, str, bool]],
) -> None:
    folder_id = "folder-law"
    client = DriveClient()
    service = MockDriveService()
    client._service = service
    service._files_resource.files.clear()

    expected_ids: list[str] = []
    for file_num, suffix, mime_type, in_folder in file_specs:
        file_id = f"file-{file_num}"
        name = f"payload-{file_num}{suffix}"
        parents = [folder_id] if in_folder else ["other-folder"]
        service._files_resource.files[file_id] = mock_drive_file(
            file_id=file_id,
            name=name,
            mime_type=mime_type,
            parents=parents,
        )
        if in_folder and (
            name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson"))
            or mime_type == GEMINI_PROMPT_MIME_TYPE
        ):
            expected_ids.append(file_id)

    files = list(client.iter_json_files(folder_id))

    assert [file.file_id for file in files] == expected_ids
    assert list(client._meta_cache) == expected_ids
