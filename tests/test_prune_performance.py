"""Test performance fix for _prune_attachment_refs N+1 query issue."""

from polylogue.ingestion import IngestBundle, ingest_bundle
from polylogue.storage.backends.sqlite import open_connection
from tests.helpers import make_attachment, make_conversation, make_message


def _conversation_record():
    return make_conversation("conv:perf", provider_name="codex", title="Perf Test", created_at=None, updated_at=None, content_hash="hash-perf", provider_meta=None)


def test_prune_multiple_attachments_correctly(workspace_env, storage_repository):
    """Verify that pruning multiple attachments works correctly.

    This exercises the N+1 query fix in _prune_attachment_refs which now
    uses a single UPDATE with IN clause instead of individual UPDATEs per attachment.
    """
    # Create initial conversation with 10 attachments
    attachments = [
        make_attachment(f"att-{i}", "conv:perf", "msg:perf", mime_type="text/plain", size_bytes=10, provider_meta=None)
        for i in range(10)
    ]

    bundle = IngestBundle(
        conversation=_conversation_record(),
        messages=[make_message("msg:perf", "conv:perf", text="hello", timestamp="1", content_hash="msg:perf", provider_meta=None)],
        attachments=attachments,
    )
    ingest_bundle(bundle, repository=storage_repository)

    # Verify all 10 attachments were created
    with open_connection(None) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM attachments WHERE attachment_id LIKE 'att-%'"
        ).fetchone()[0]
        assert count == 10, f"Expected 10 attachments, got {count}"

        # Check ref_count is correct
        refs = conn.execute(
            "SELECT attachment_id, ref_count FROM attachments WHERE attachment_id LIKE 'att-%' ORDER BY attachment_id"
        ).fetchall()
        for ref in refs:
            assert ref["ref_count"] == 1, f"Expected ref_count=1 for {ref['attachment_id']}, got {ref['ref_count']}"

    # Now re-ingest with only 2 attachments, which should prune 8
    new_attachments = [
        make_attachment("att-0", "conv:perf", "msg:perf", mime_type="text/plain", size_bytes=10, provider_meta=None),
        make_attachment("att-1", "conv:perf", "msg:perf", mime_type="text/plain", size_bytes=10, provider_meta=None),
    ]

    ingest_bundle(
        IngestBundle(
            conversation=_conversation_record(),
            messages=[make_message("msg:perf", "conv:perf", text="hello", timestamp="1", content_hash="msg:perf", provider_meta=None)],
            attachments=new_attachments,
        ),
        repository=storage_repository,
    )

    # Verify only 2 attachments remain (the 8 others should have been pruned)
    with open_connection(None) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM attachments WHERE attachment_id LIKE 'att-%'"
        ).fetchone()[0]
        assert count == 2, f"Expected 2 attachments after pruning, got {count}"

        remaining = conn.execute(
            "SELECT attachment_id FROM attachments WHERE attachment_id LIKE 'att-%' ORDER BY attachment_id"
        ).fetchall()
        remaining_ids = [row["attachment_id"] for row in remaining]
        assert remaining_ids == ["att-0", "att-1"], f"Expected att-0 and att-1, got {remaining_ids}"
