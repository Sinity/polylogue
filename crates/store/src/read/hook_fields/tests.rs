use super::*;
use crate::fixture::writer::{WriterArchive, message};
use polylogue_model::{Block, MessageRow, Output, ParsedHookField, Role, Session, SessionKind};

fn output() -> Output {
    let message = message(
        "hook",
        "session",
        Role::System,
        1,
        vec![
            Block::Text {
                text: "diagnostic".into(),
            },
            Block::Text {
                text: "hook context".into(),
            },
        ],
    );
    Output {
        sessions: vec![Session::new(message.session.clone(), SessionKind::Main)],
        messages: vec![MessageRow {
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 0,
            end: 0,
            message,
        }],
        ..Output::default()
    }
}

fn capture(
    w: &mut WriterArchive,
    input: &str,
    mut output: Output,
    with_hook_field: bool,
) -> (crate::InputKey, i64, u64, crate::ParsePart) {
    let mut bytes=br#"{"type":"system","subtype":"stop_hook_summary","hookAdditionalContext":[null,null,null,"hook context"]}"#.to_vec();
    let end = bytes.len() as u64;
    bytes.push(b'\n');
    output.messages[0].end = end;
    if with_hook_field {
        output.hook_fields.insert(
            0,
            vec![ParsedHookField {
                block: 1,
                start: 0,
                end,
                pointer: "/hookAdditionalContext/3".into(),
            }],
        );
    }
    let mut batch =
        crate::fixture::writer::step(input, "claude-code", "/p/hook.jsonl", output, 0, 1);
    batch.length = bytes.len() as u64;
    batch.parse[0].parsed_upto = batch.length;
    // Even a kept capture-read record of the harness file cannot authenticate
    // who sent the field.
    batch.capture_read = Some(crate::capture_read::CaptureRead {
        version: crate::capture_read::CaptureRead::VERSION,
        route: crate::capture_read::ReadRoute::NativeCapture,
        source: crate::capture_read::SourceSelector {
            origin: "claude-code".into(),
            machine: "m".into(),
            path_bytes: b"/p/hook.jsonl".to_vec(),
        },
        range: crate::capture_read::RawRange {
            start: 0,
            end: batch.length,
        },
        hash: BlobHash::of(&bytes),
        observed_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        generation: None,
        raw_total: None,
        previous_revision: None,
        chunks: Vec::new(),
    });
    let mut raw = batch.clone();
    raw.parse.clear();
    let blob = w.archive.prepare_bytes(bytes.as_slice()).unwrap();
    let stored = w.writer.store_bytes(blob, raw).unwrap();
    batch.change = crate::Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    let replay = (
        batch.input.clone(),
        stored.chain,
        batch.length,
        batch.parse[0].clone(),
    );
    w.writer.send(batch).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    replay
}

#[test]
fn unauthenticated_hook_field_stays_unknown() {
    let mut w = WriterArchive::new();
    let output = output();
    let message = output.messages[0].message.clone();
    let version = message.content_hash().unwrap();
    let (input, chain, length, mut step) = capture(&mut w, "hook-input", output, true);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let expected = {
        let reader = pool.reader().unwrap();
        let injections = reader.injection_evidence(&message.id, version).unwrap();
        assert_eq!(injections.len(), 1);
        assert!(injections[0].observed.is_none());
        let field = &injections[0].declared[0];
        assert_eq!(field.input, "hook-input");
        assert_eq!(field.chain, chain);
        assert_eq!((field.start, field.end), (0, step.output.messages[0].end));
        assert_eq!(field.pointer, "/hookAdditionalContext/3");
        assert_eq!(
            reader
                .message_with_block_selection(&message.id)
                .unwrap()
                .unwrap()
                .0
                .content_hash()
                .unwrap(),
            version
        );
        injections
    };
    let mut reading = crate::writer::Replacement::new().unwrap();
    reading.push(&mut step).unwrap();
    w.writer
        .replace_chain(input, chain, length, reading)
        .unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .injection_evidence(&message.id, version)
            .unwrap(),
        expected
    );
    crate::archive::open_write(&w.archive.archive_db(), 1)
        .unwrap()
        .execute("DELETE FROM extent WHERE input='hook-input'", [])
        .unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .injection_evidence(&message.id, version)
            .unwrap(),
        expected
    );
}

#[test]
fn forged_imported_observation_stays_outside_local_membership() {
    let mut w = WriterArchive::new();
    let mut output = output();
    let message = output.messages[0].message.clone();
    let version = message.content_hash().unwrap();
    output.declared_injections.insert(
        0,
        vec![InjectionEvidence {
            message: message.id.clone(),
            source_version: version,
            block: 1,
            observed: Some(StatedHookField {
                input: "other-input".into(),
                chain: 77,
                start: 900,
                end: 1000,
                pointer: "/hookAdditionalContext/0".into(),
            }),
            declared: vec![],
        }],
    );
    capture(&mut w, "local-bundle", output, false);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let injections = reader.injection_evidence(&message.id, version).unwrap();
    assert_eq!(injections.len(), 1);
    assert!(injections[0].observed.is_none());
    assert_eq!(injections[0].declared[0].input, "other-input");
    assert_eq!(injections[0].declared[0].chain, 77);
    let local: String = reader
        .conn()
        .unwrap()
        .query_row(
            "SELECT i.id FROM message_injection_field f JOIN input_ref i ON i.pk=f.input_pk",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(local, "local-bundle");
}

#[test]
fn sparse_injection_block_ordinals_keep_the_full_u64_domain() {
    let mut w = WriterArchive::new();
    let mut output = output();
    output.messages[0].message.blocks.remove(0);
    let message = output.messages[0].message.clone();
    let version = BlobHash::of(b"unavailable full value");
    let block = u64::MAX - 1;
    output.block_selections.insert(
        0,
        polylogue_model::bundle::ShownBlocks {
            source_version: version,
            block_count: u64::MAX,
            block_ordinals: vec![block],
        },
    );
    output.declared_injections.insert(
        0,
        vec![InjectionEvidence {
            message: message.id.clone(),
            source_version: version,
            block,
            observed: None,
            declared: vec![StatedHookField {
                input: "original-input".into(),
                chain: 1,
                start: 1,
                end: 2,
                pointer: "/hookAdditionalContext/0".into(),
            }],
        }],
    );
    capture(&mut w, "sparse-bundle", output, false);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let injections = reader.injection_evidence(&message.id, version).unwrap();
    assert_eq!(injections[0].block, block);
    assert!(injections[0].observed.is_none());
}
