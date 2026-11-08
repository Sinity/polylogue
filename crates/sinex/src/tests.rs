use crate::{Authentication, Config, Mode, Sinex};
use polylogue_model::{BlobHash, Output, RecordCounts};
use polylogue_store::primary::{
    Capture, CapturedInput, CapturedReading, Product, ProductWrite, SourceLocation,
};
use serde_json::json;

/// The complete Core outcome for an accepted frame admitting no new events.
pub(crate) fn no_event_receipt(frame: &sinex_protocol::Ingress) -> sinex_protocol::Acknowledgment {
    let mut event_types = match frame {
        sinex_protocol::Ingress::Events { events, .. } => events
            .iter()
            .map(|event| event.kind.name.clone())
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    event_types.sort();
    event_types.dedup();
    let source = match frame {
        sinex_protocol::Ingress::Events { source, .. } => {
            source.as_ref().map(|source| source.name.as_str())
        }
        sinex_protocol::Ingress::ReplaySequence { sequence, .. } => {
            Some(sequence.source.name.as_str())
        }
        _ => None,
    };
    serde_json::from_value(json!({
        "number":1,"frame":frame.frame(),"material":null,"admitted":[],
        "outcome":{
            "frame_content_hash":sinex_transport::producer::content_hash(frame).unwrap(),
            "registered_kinds":[],"commit_range":null,"superseded_input":0,
            "replay_archived":0,"removed_at_source":0,"dirty_keys":0,
            "admitted":0,"repeats":0,"replaced":0,"older":0,"superseded":0,
            "failures":0,"refusal_kind":null,"source":source,"event_types":event_types
        },"refusal":null,"commit_seq":null
    }))
    .unwrap()
}

pub(crate) fn applied(position: i64) -> polylogue_store::primary::ReplayState {
    polylogue_store::primary::ReplayState {
        applied_position: Some(position),
        head: Some(position),
        scope: Some("fixture-scope".into()),
        staged_delivery: None,
    }
}

pub(crate) fn body(root: &std::path::Path, bytes: &[u8]) -> polylogue_store::primary::PreparedBody {
    let prepared = hashed_blobs::Preparation::new(root.join("source-staging"))
        .unwrap()
        .plain(bytes, 3)
        .unwrap();
    polylogue_store::primary::PreparedBody {
        descriptor: prepared.descriptor(),
        file: prepared.into_file(),
        path: None,
    }
}

pub(crate) fn client(root: &std::path::Path) -> Sinex {
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
pub(crate) fn capture(bytes: &[u8], parsed: Vec<CapturedReading>) -> CapturedInput {
    serde_json::from_value(json!({
        "input":{"origin":"codex","path":"/home/op/session.jsonl","host":"fixture-host"},
        "rows":{"input":"input-x","format":"codex","revision":{"number":1,"produced":"file","chain":1,"length":bytes.len()},"extent":{"chain":1,"offset":0,"length":bytes.len(),"blob":BlobHash::of(bytes),"blob_offset":0}},
        "index":0,"previous":null,"hash":BlobHash::of(bytes),"length":bytes.len(),"raw_range":{"start":0,"end":bytes.len()},"group":null,"arrival_rank":0,"parsed":parsed
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
        .append_group([Ok(prepared)], |_| Ok(None))
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
    let captured = capture(
        bytes,
        vec![CapturedReading::from_step(&step, None).unwrap()],
    );
    let sinex = client(root.path());
    let prepared = sinex
        .prepare_input(&captured, body(root.path(), bytes))
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
            .prepare_input(&capture(bytes, Vec::new()), body(root.path(), bytes))
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
            body(root.path(), bytes),
        )
        .unwrap();
    let physical = extent(root.path(), bytes);
    let receipt=serde_json::from_value(json!({"number":1,"frame":raw.identity,"material":crate::material_id("polylogue-test", &raw.identity),"material_extents":[physical.encode()],"admitted":[],"outcome":{},"refusal":null,"commit_seq":null})).unwrap();
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
        .complete_captures(std::slice::from_ref(&raw.identity), &[write])
        .unwrap();
    assert_eq!(sinex.pending_captures().unwrap()[0].identity, raw.identity);
    sinex.spool.lock().unwrap().acknowledge(serde_json::from_value(json!({"number":2,"frame":grouped.identity,"material":null,"admitted":[],"outcome":{},"refusal":null,"commit_seq":10})).unwrap()).unwrap();
    sinex
        .spool
        .lock()
        .unwrap()
        .retire(&crate::tests::applied(9))
        .unwrap();
    assert_eq!(sinex.pending_captures().unwrap().len(), 1);
    sinex
        .spool
        .lock()
        .unwrap()
        .retire(&crate::tests::applied(10))
        .unwrap();
    assert!(sinex.pending_captures().unwrap().is_empty());
}

#[test]
fn complete_reading_custody_uses_the_encoded_plain_identity_and_survives_restart() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let write = ProductWrite {
        product: Product::Session {
            reading: serde_json::from_value(json!({"input":{"id":"input-x","format":"codex","arrival_rank":0},"chain":1,"reading_version":"reading-2","order_from":null,"repeats_whole_conversations":false,"start":0,"parsed_upto":12,"progress":"at_end"})).unwrap(),
            facts: polylogue_model::Session::new(polylogue_model::SessionId::parse("codex:session").unwrap(), polylogue_model::SessionKind::Main),
        },
        location: Some(SourceLocation {material:"00000000-0000-0000-0000-000000000001".into(),anchor:0}),
        parents: Vec::new(), machine: Some("fixture-host".into()), output_index:0,
    };
    let prepare = || {
        sinex
            .prepare_reading(
                "00000000-0000-0000-0000-000000000001",
                &polylogue_store::primary::ExportReadingKey {
                    input: polylogue_store::InputKey {
                        id: "input-x".into(),
                        format: "codex".into(),
                        arrival_rank: 0,
                    },
                    chain: 1,
                    reading_version: "reading-2".into(),
                    length: 12,
                },
                &mut |emit| emit(write.clone()),
            )
            .unwrap()
    };
    let first = prepare();
    assert_eq!(prepare().identity, first.identity);
    drop(sinex);
    let reopened = client(root.path());
    let queue = reopened.queue().unwrap();
    let sinex_protocol::Ingress::Material { material, .. } =
        queue.decode_frame(&first.identity).unwrap()
    else {
        panic!("reading material expected")
    };
    let body = queue.prepared_body(&first.identity).unwrap();
    assert_eq!(material.blob, body.hash().to_string());
    assert_eq!(material.length, body.decoded_len());
    let mut reader = std::io::BufReader::new(
        body.reader(None, hashed_blobs::ReadLimits::default())
            .unwrap(),
    );
    let round_trip: ProductWrite = serde_json::from_reader(&mut reader).unwrap();
    assert_eq!(round_trip, write);
}

pub(crate) mod named;
mod signals;

mod native_capture;

mod custody_rebuild;

mod export_delivery;

#[test]
fn advisory_readiness_is_nonblocking_and_retains_pending_advice_across_restart() {
    use polylogue_model::authored::{
        AuthoredEffect, DeliveryChange, DeliveryRecord, PreparedRequest,
    };
    use polylogue_model::user::{Delivery, UserOutcome, UserWrite};
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    assert!(sinex.advisory_history_ready());
    let queue = sinex.queue().unwrap();
    assert!(
        !sinex.advisory_history_ready(),
        "a busy owner means uncertainty, not empty history"
    );
    drop(queue);
    let delivery = Delivery {
        recipient_kind: "session".into(),
        recipient_id: Some("codex:recipient".into()),
        sender: "hook".into(),
        event: "advice".into(),
        cwd: Some("/synthetic/repo".into()),
        text: "one exact file pointer".into(),
        refs: vec![
            "assertion:root@revision".into(),
            "file:src/parser.rs@commit".into(),
        ],
        left_out: vec![],
    };
    let request = polylogue_store::fixture::authored::operator_request(
        "pending-advice".into(),
        UserWrite::Deliver(delivery.clone()),
    );
    let at = request.at;
    let prepared = PreparedRequest {
        request,
        outcome: UserOutcome {
            id: "pending-advice".into(),
            repeat: false,
            pending: false,
        },
        effect: AuthoredEffect {
            deliveries: vec![DeliveryChange {
                delivery: DeliveryRecord {
                    signal: None,
                    id: "pending-advice".into(),
                    prepared: delivery,
                    at,
                    events: vec![],
                },
                added_events: vec![],
            }],
            ..Default::default()
        },
    };
    assert!(sinex.prepare_request(&prepared).unwrap().pending);
    assert!(!sinex.advisory_history_ready());
    let identity = "request:pending-advice";
    let queue = sinex.queue().unwrap();
    let receipt = serde_json::from_value(json!({
        "number":1,"frame":identity,"material":crate::material_id("polylogue-test",identity),"material_extents":[],
        "admitted":[],"outcome":{},"refusal":null,"commit_seq":18
    })).unwrap();
    queue.acknowledge(receipt).unwrap();
    drop(queue);
    assert!(
        !sinex.advisory_history_ready(),
        "publication ACK is not confirmed application"
    );
    drop(sinex);
    let reopened = client(root.path());
    assert!(
        !reopened.advisory_history_ready(),
        "restart must not offer the pair again from empty projected delivery history"
    );
}
