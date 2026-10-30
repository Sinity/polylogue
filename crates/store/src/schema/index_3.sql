ALTER TABLE message_projection ADD COLUMN admitted INTEGER CHECK(admitted > 0);
ALTER TABLE session_record ADD COLUMN admitted INTEGER CHECK(admitted > 0);
ALTER TABLE document_source ADD COLUMN admitted INTEGER CHECK(admitted > 0);
ALTER TABLE event_source ADD COLUMN admitted INTEGER CHECK(admitted > 0);
ALTER TABLE link_source ADD COLUMN admitted INTEGER CHECK(admitted > 0);
ALTER TABLE parse_state ADD COLUMN commit_seq INTEGER CHECK(commit_seq > 0);
