use crate::{Authentication, Config, Mode, Sinex};
use polylogue_model::{BlobHash, Output, RecordCounts};
use polylogue_store::primary::{
    Capture, CapturedInput, CapturedReading, Product, ProductWrite, SourceLocation,
};
use serde_json::json;

fn client(root: &std::path::Path) -> Sinex {
    Sinex::open(
        &Config {
            mode: Mode::Primary,
            nats: "nats://127.0.0.1:1".into(),
            password_file: None,
            subject_root: "offline".into(),
            producer: "polylogue-test".into(),
            store: root.into(),
        },
        &Authentication::Anonymous,
        "fixture-host",
    )
    .unwrap()
}
fn capture(bytes: &[u8], parsed: Vec<CapturedReading>) -> CapturedInput {
    serde_json::from_value(json!({
        "input":{"origin":"codex","path":"/home/op/session.jsonl","host":"fixture-host"},
        "rows":{"input":"input-x","format":"codex","revision":{"number":1,"produced":"file","chain":1,"length":bytes.len()},"extent":{"chain":1,"offset":0,"length":bytes.len(),"blob":BlobHash::of(bytes),"blob_offset":0}},
        "index":0,"previous":null,"hash":BlobHash::of(bytes),"length":bytes.len(),"group":null,"arrival_rank":0,"parsed":parsed
    })).unwrap()
}
fn extent(root: &std::path::Path, bytes: &[u8]) -> hashed_blobs::Extent {
    let prepared = hashed_blobs::Preparation::new(root.join("scratch"))
        .unwrap()
        .plain(bytes, 1)
        .unwrap();
    let mut writer =
        hashed_blobs::SegmentWriter::open(root, "segments/test.zst", 0, &[], 0o600).unwrap();
    let group = writer
        .append_group(vec![prepared], |_| Ok(None))
        .unwrap()
        .wait()
        .unwrap();
    writer.acknowledge(&group).unwrap();
    group.extents[0].clone()
}

#[test]
fn offline_capture_keeps_complete_parser_facts_through_restart() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"synthetic source";
    let mut output = Output::default();
    output.messages.push(serde_json::from_value(json!({"offset":0,"end":bytes.len(),"message":{"id":"codex:message","session":"codex:session","role":"assistant","attrs":{"native":{"answer":false}},"blocks":[{"type":"text","text":"Synthetic answer"}]}})).unwrap());
    let mut step = polylogue_store::ParseStep::new(
        output,
        RecordCounts::default(),
        0..bytes.len() as u64,
        true,
    );
    step.prepare();
    let captured = capture(bytes, vec![CapturedReading::from_step(&step).unwrap()]);
    let sinex = client(root.path());
    let prepared = sinex
        .prepare_input(&captured, &mut bytes.as_slice())
        .unwrap();
    assert!(matches!(
        prepared,
        polylogue_store::primary::PreparedInput::Pending { .. }
    ));
    let pending = sinex.pending_captures().unwrap();
    assert_eq!(pending.len(), 1);
    assert!(pending[0].anchor.is_none());
    assert_eq!(
        sinex.capture(&pending[0].identity).unwrap(),
        Capture::Input(Box::new(captured.clone()))
    );
    drop(sinex);
    let sinex = client(root.path());
    let pending = sinex.pending_captures().unwrap();
    assert_eq!(
        sinex.capture(&pending[0].identity).unwrap(),
        Capture::Input(Box::new(captured))
    );
    assert_eq!(sinex.status().unwrap().prepared_materials, 1);
}

#[test]
fn capture_retries_keep_identity_but_a_b_a_is_three_observations() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    for bytes in [
        b"A".as_slice(),
        b"B".as_slice(),
        b"A".as_slice(),
        b"A".as_slice(),
    ] {
        sinex
            .prepare_input(&capture(bytes, Vec::new()), &mut &bytes[..])
            .unwrap();
    }
    let pending = sinex.pending_captures().unwrap();
    assert_eq!(pending.len(), 3);
    assert_ne!(pending[0].identity, pending[2].identity);
    let hashes: Vec<_> = pending
        .iter()
        .map(|entry| match &entry.captured {
            Capture::Input(input) => input.hash,
            _ => panic!(),
        })
        .collect();
    assert_eq!(
        hashes,
        [BlobHash::of(b"A"), BlobHash::of(b"B"), BlobHash::of(b"A")]
    );
}

#[test]
fn grouped_capture_remains_pending_until_its_confirmed_barrier() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let bytes = b"attachment";
    let raw = sinex
        .prepare_blob(
            BlobHash::of(bytes),
            bytes.len() as u64,
            &mut bytes.as_slice(),
        )
        .unwrap();
    let physical = extent(root.path(), bytes);
    let receipt=serde_json::from_value(json!({"number":1,"frame":raw.identity,"material":crate::material_id(&raw.identity),"material_extents":[physical.encode()],"admitted":[],"outcome":{},"refusal":null,"commit_seq":null})).unwrap();
    sinex.spool.lock().unwrap().acknowledge(receipt).unwrap();
    let pending = sinex.pending_captures().unwrap();
    let material = pending[0].material.clone().unwrap();
    let write = ProductWrite {
        product: Product::Blob(material.clone()),
        location: Some(SourceLocation {
            material: material.id,
            anchor: 0,
        }),
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    };
    let grouped = sinex
        .complete_captures(&[raw.identity.clone()], &[write])
        .unwrap();
    assert_eq!(sinex.pending_captures().unwrap()[0].identity, raw.identity);
    sinex.spool.lock().unwrap().acknowledge(serde_json::from_value(json!({"number":2,"frame":grouped.identity,"material":null,"admitted":[],"outcome":{},"refusal":null,"commit_seq":10})).unwrap()).unwrap();
    sinex.spool.lock().unwrap().retire(9).unwrap();
    assert_eq!(sinex.pending_captures().unwrap().len(), 1);
    sinex.spool.lock().unwrap().retire(10).unwrap();
    assert!(sinex.pending_captures().unwrap().is_empty());
}
