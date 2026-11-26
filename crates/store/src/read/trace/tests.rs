use super::*;
use crate::fixture::writer::{WriterArchive, bash, prose};

#[test]
fn trace_maps_a_record_across_every_stored_extent() {
    let mut archive = WriterArchive::new();
    archive.write("trace", prose("trace", 1, 1, "first"));
    archive.write("trace", prose("trace", 2, 2, "second"));
    let index = Connection::open(archive.archive.index_db()).unwrap();
    index.execute("UPDATE message_parse SET source_start=1,source_length=2 WHERE message_id='claude-code:trace-p1'",[]).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let card = reader
        .record_chain(&Reference::parse("message:claude-code:trace-p1").unwrap())
        .unwrap()
        .unwrap();
    assert!(
        matches!(&card.chain[1],ChainStep::InputRecord{input,chain:1,start:1,end:3,route:Route::Live,shown:true} if input=="input-trace")
    );
    let extents: Vec<_> = card
        .chain
        .iter()
        .filter_map(|step| match step {
            ChainStep::StoredBytes {
                blob,
                size,
                offset,
                length,
                blob_offset,
            } => Some((*blob, *size, *offset, *length, *blob_offset)),
            _ => None,
        })
        .collect();
    let hash = polylogue_model::BlobHash::of(b"r\n");
    assert_eq!(extents, vec![(hash, 2, 0, 2, 0), (hash, 2, 2, 2, 0)]);
    assert!(card.chain.iter().any(|step|matches!(step,ChainStep::ReaderVersion{parse_version,..} if parse_version=="synthetic=1")));
    assert_eq!(
        reader
            .stored_range(
                "input-trace",
                1,
                1,
                3,
                &archive.archive.blob_reader().unwrap()
            )
            .unwrap(),
        b"\nr"
    );
}

#[test]
fn trace_states_a_record_beyond_the_reader_checkpoint() {
    let mut archive = WriterArchive::new();
    archive.write("trace", prose("trace", 1, 1, "first"));
    archive.write("trace", prose("trace", 2, 2, "second"));
    let index = Connection::open(archive.archive.index_db()).unwrap();
    index.execute("UPDATE message_parse SET source_start=1,source_length=2 WHERE message_id='claude-code:trace-p1'",[]).unwrap();
    for (parsed_upto, covered) in [(3_i64, true), (2, false)] {
        index.execute("UPDATE parse_state SET parsed_to=?1 WHERE input_pk=(SELECT pk FROM input_ref WHERE id='input-trace')",[parsed_upto]).unwrap();
        let pool = crate::ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        let card = reader
            .record_chain(&Reference::parse("message:claude-code:trace-p1").unwrap())
            .unwrap()
            .unwrap();
        let flagged = card.chain.iter().any(|step| matches!(step, ChainStep::Unavailable { reason } if reason.contains("does not cover this record")));
        assert_eq!(flagged, !covered, "parsed_upto {parsed_upto}");
        assert!(card.chain.iter().any(|step| matches!(step, ChainStep::ReaderVersion { parsed_upto: at, .. } if *at == parsed_upto as u64)));
    }
}

#[test]
fn trace_preserves_missing_range_and_checks_block_span_and_call_identity() {
    let mut archive = WriterArchive::new();
    archive.write("trace", prose("trace", 1, 1, "hello"));
    archive.write("trace", bash("trace", 2, 2, "printf hi", true, "hi"));
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    {
        let reader = pool.reader().unwrap();
        let tool = reader
            .record_chain(&Reference::parse("tool:claude-code:trace#t2").unwrap())
            .unwrap()
            .unwrap();
        assert!(tool.chain.iter().any(|step|matches!(step,ChainStep::Record{reference:Reference::Message(id)} if id.to_string()=="claude-code:trace-c2")));
        assert!(
            matches!(reader.record_chain(&Reference::parse("block:claude-code:trace-p1#99").unwrap()),Err(StoreError::ReferenceUnavailable{reason,..}) if reason=="the original block is unavailable")
        );
        assert!(
            matches!(reader.record_chain(&Reference::parse("span:message:claude-code:trace-p1#c=0-99").unwrap()),Err(StoreError::Refused(polylogue_model::Reason::BadSpan { problem })) if problem=="span bounds exceed the source text")
        );
        assert!(
            reader
                .record_chain(&Reference::parse("message:claude-code:absent").unwrap())
                .unwrap()
                .is_none()
        );
    }
    let index = Connection::open(archive.archive.index_db()).unwrap();
    index
        .execute(
            "UPDATE message_parse SET source_length=0 WHERE message_id='claude-code:trace-p1'",
            [],
        )
        .unwrap();
    index.execute("DELETE FROM parse_state WHERE input_pk=(SELECT pk FROM input_ref WHERE id='input-trace')",[]).unwrap();
    let reader = pool.reader().unwrap();
    let card = reader
        .record_chain(&Reference::parse("message:claude-code:trace-p1").unwrap())
        .unwrap()
        .unwrap();
    assert!(
        !card
            .chain
            .iter()
            .any(|step| matches!(step, ChainStep::StoredBytes { .. }))
    );
    assert!(card.chain.iter().any(|step|matches!(step,ChainStep::Unavailable{reason} if reason=="record byte end is not recorded for input-trace chain 1")));
    assert!(
        !card
            .chain
            .iter()
            .any(|step| matches!(step, ChainStep::ReaderVersion { .. }))
    );
    assert!(card.chain.iter().any(|step| matches!(step,ChainStep::Unavailable{reason} if reason=="reader version is unavailable for input-trace chain 1")));
    assert_eq!(
        card.missing,
        ["quote origins", "paste origins", "file-line lineage"]
    );
}

/// Two tool calls that share a session and call ID are refused as ambiguous,
/// naming both messages (docs/references.md: ambiguity is an error).
#[test]
fn trace_refuses_a_call_id_two_messages_share() {
    let mut archive = WriterArchive::new();
    archive.write("trace", bash("trace", 2, 2, "printf one", true, "one"));
    archive.write("trace", bash("trace", 3, 3, "printf two", true, "two"));
    let index = Connection::open(archive.archive.index_db()).unwrap();
    index
        .execute("UPDATE tool_call SET call_id='t2' WHERE call_id='t3'", [])
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let error = reader
        .record_chain(&Reference::parse("tool:claude-code:trace#t2").unwrap())
        .unwrap_err();
    assert!(
        matches!(&error, StoreError::Ambiguous { candidates, .. }
            if candidates == &["message:claude-code:trace-c2", "message:claude-code:trace-c3"]),
        "{error:?}"
    );
}

#[test]
fn trace_span_does_not_require_unknown_text_after_its_end() {
    use polylogue_model::{Block, Role};
    let mut archive = WriterArchive::new();
    let original = crate::fixture::writer::message(
        "prefix",
        "trace",
        Role::Assistant,
        1,
        vec![
            Block::Text {
                text: "hello".into(),
            },
            Block::Thinking {
                text: None,
                signature: Some("unavailable".into()),
                redacted: true,
            },
            Block::Text {
                text: "outside the selected span".into(),
            },
        ],
    );
    archive.write("trace", vec![original]);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let span = Reference::parse("span:message:claude-code:trace-prefix#c=0-5").unwrap();
    let card = reader.record_chain(&span).unwrap().unwrap();
    assert_eq!(card.item, span);
    assert!(
        card.chain
            .iter()
            .any(|step| matches!(step, ChainStep::StoredBytes { .. }))
    );
}

#[test]
fn trace_states_when_the_reader_checkpoint_stops_before_the_record() {
    let mut archive = WriterArchive::new();
    archive.write("trace", prose("trace", 1, 1, "first"));
    archive.write("trace", prose("trace", 2, 2, "second"));
    let index = Connection::open(archive.archive.index_db()).unwrap();
    index.execute("UPDATE message_parse SET source_start=1,source_length=2 WHERE message_id='claude-code:trace-p1'",[]).unwrap();
    index.execute("UPDATE parse_state SET parsed_to=0 WHERE input_pk=(SELECT pk FROM input_ref WHERE id='input-trace')",[]).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let card = reader
        .record_chain(&Reference::parse("message:claude-code:trace-p1").unwrap())
        .unwrap()
        .unwrap();
    assert!(card.chain.iter().any(|step| matches!(step,ChainStep::Unavailable{reason} if reason=="the current reader checkpoint does not cover this record; its original reader version is unavailable")));
    assert!(card.chain.iter().any(|step| matches!(step,ChainStep::ReaderVersion{parse_version,parsed_upto:0,..} if parse_version=="synthetic=1")));
}
