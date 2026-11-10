use super::*;
use crate::fixture::writer::{WriterArchive, message};
use polylogue_model::{Block, HookFieldClaim, MessageRow, Output, Role, Session, SessionKind};

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
                text: "actual context".into(),
            },
        ],
    );
    Output {
        sessions: vec![Session::new(message.session.clone(), SessionKind::Main)],
        messages: vec![MessageRow {
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
    native_claim: bool,
) -> (crate::InputKey, i64, u64, crate::ParseStep) {
    let mut bytes=br#"{"type":"system","subtype":"stop_hook_summary","hookAdditionalContext":[null,null,null,"actual context"]}"#.to_vec();
    let end = bytes.len() as u64;
    bytes.push(b'\n');
    output.messages[0].end = end;
    if native_claim {
        output.native_hook_fields.insert(
            0,
            vec![HookFieldClaim {
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
    // Even a retained native acquisition payload cannot authenticate its owner.
    batch.acquisition = Some(crate::primary::AcquisitionReceipt {
        version: crate::primary::AcquisitionReceipt::VERSION,
        route: crate::primary::AcquisitionRoute::NativeCapture,
        source: crate::primary::SourceSelector {
            origin: "claude-code".into(),
            machine: "m".into(),
            path_bytes: b"/p/hook.jsonl".to_vec(),
        },
        range: crate::primary::RawRange {
            start: 0,
            end: batch.length,
        },
        hash: BlobHash::of(&bytes),
        observed_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        generation: None,
        raw_total: None,
        raw_predecessor: None,
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
fn exact_native_field_is_retained_as_unknown_without_authenticated_owner_and_survives_replacement()
{
    let mut w = WriterArchive::new();
    let output = output();
    let message = output.messages[0].message.clone();
    let version = message.content_hash().unwrap();
    let (input, chain, length, mut step) = capture(&mut w, "native-hook", output, true);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let expected = {
        let reader = pool.reader().unwrap();
        let evidence = reader.injection_evidence(&message.id, version).unwrap();
        assert_eq!(evidence.len(), 1);
        assert!(evidence[0].observed.is_none());
        let field = &evidence[0].declared[0];
        assert_eq!(field.input, "native-hook");
        assert_eq!(field.chain, chain);
        assert_eq!((field.start, field.end), (0, step.output.messages[0].end));
        assert_eq!(field.pointer, "/hookAdditionalContext/3");
        assert_eq!(
            reader
                .message_with_projection(&message.id)
                .unwrap()
                .unwrap()
                .0
                .content_hash()
                .unwrap(),
            version
        );
        evidence
    };
    let mut reading = crate::writer::ChainReading::new().unwrap();
    reading.push(&mut step).unwrap();
    w.writer
        .sender()
        .unwrap()
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
        .execute("DELETE FROM extent WHERE input='native-hook'", [])
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
fn forged_imported_observation_remains_a_foreign_declaration_under_actual_local_membership() {
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
            observed: Some(NativeHookField {
                input: "foreign-claim".into(),
                chain: 77,
                start: 900,
                end: 1000,
                pointer: "/hookAdditionalContext/0".into(),
            }),
            declared: vec![],
        }],
    );
    capture(&mut w, "actual-local-bundle", output, false);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let evidence = reader.injection_evidence(&message.id, version).unwrap();
    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].observed.is_none());
    assert_eq!(evidence[0].declared[0].input, "foreign-claim");
    assert_eq!(evidence[0].declared[0].chain, 77);
    let local: String = reader
        .conn()
        .unwrap()
        .query_row(
            "SELECT i.id FROM message_injection_field f JOIN input_ref i ON i.pk=f.input_pk",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(local, "actual-local-bundle");
}

#[test]
fn sparse_injection_block_ordinals_keep_the_full_u64_domain() {
    let mut w = WriterArchive::new();
    let mut output = output();
    output.messages[0].message.blocks.remove(0);
    let message = output.messages[0].message.clone();
    let version = BlobHash::of(b"unavailable full canonical value");
    let block = u64::MAX - 1;
    output.message_projections.insert(
        0,
        polylogue_model::bundle::MessageProjection {
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
            declared: vec![NativeHookField {
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
    let evidence = reader.injection_evidence(&message.id, version).unwrap();
    assert_eq!(evidence[0].block, block);
    assert!(evidence[0].observed.is_none());
}
