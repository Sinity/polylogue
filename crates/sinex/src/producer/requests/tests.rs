use super::*;
use polylogue_model::{
    BlobHash,
    historical::{AcceptedHashEvidence, ConservedAccepted, SourceSchema},
};
use serde_json::json;
use sinex_model::{InputRef, Manifest, MaterialRole};

fn old_body() -> serde_json::Value {
    json!({"request": {"id":"retained-request","machine":"fixture-host",
        "at":"2026-01-02T03:04:05Z","authorship":{"author":"operator","via":"cli"},
        "write":{"write":"add_note","note":{"kind":"note","value":"retained assertion","targets":["repo:synthetic"]}}},
        "outcome":{"id":"retained-note","repeat":false,"pending":false},
        "effect":{"variants":{"variants":[],"variant_parts":[]},"notes":[{
            "note":{"id":"retained-note","first_note":"retained-note","kind":"note",
                "value":"retained assertion","author":"operator","created_at":"2026-01-02T03:04:05Z","targets":["repo:synthetic"]},
            "machine":"fixture-host","revises":[]}],"withdrawals":[],"settings":[],
            "kept_results":[],"deliveries":[],"chats":[],"backfills":[]}})
}

fn prepare(queue: &Spool, parser: &str, body: &[u8]) -> (Prepared, QueuedPublicationCustody) {
    let source: sinex_model::SourceDeclaration = serde_json::from_str(include_str!(
        "../../client/confirmed/source3/declaration.json"
    ))
    .unwrap();
    let identity = "request:retained-request";
    let material = MaterialFrame {
        source: source.source.clone(),
        id: crate::material_id("polylogue-test", identity),
        input: InputRef {
            kind: "polylogue".into(),
            locator: "requests".into(),
            host: "fixture-host".into(),
        },
        blob: format!("blake3:{}", BlobHash::of(body)),
        length: body.len() as u64,
        parser: parser.into(),
        parser_version: 3,
        part_index: 0,
        previous_part: None,
        manifest: Manifest {
            role: MaterialRole::Records,
            media_type: Some("application/json".into()),
            ..Default::default()
        },
    };
    let frame = Ingress::Material {
        frame: identity.into(),
        material: Box::new(material.clone()),
    };
    let prepared = queue.prepare_material(frame, body).unwrap();
    let custody = QueuedPublicationCustody {
        producer_family: "polylogue".into(),
        producer_identity: "polylogue-test".into(),
        receipt_number: 17,
        frame: identity.into(),
        frame_content_hash: BlobHash::parse(prepared.frame_hash.strip_prefix("blake3:").unwrap())
            .unwrap(),
        material: material.id.to_string(),
        blob: BlobHash::of(body),
        length: material.length,
        source: source.source.name.clone(),
        source_version: source.source.version,
        source_fingerprint: source.fingerprint(),
        parser: parser.into(),
        parser_version: 3,
    };
    (prepared, custody)
}

#[test]
fn retained_source3_body_reopens_as_queued_history_without_a_fetched_event_or_new_spool_entry() {
    let directory = tempfile::tempdir().unwrap();
    let body = serde_json::to_vec(&old_body()).unwrap();
    let custody = {
        let client = crate::tests::client(directory.path());
        let queue = client.queue().unwrap();
        let (_, custody) = prepare(&queue, "polylogue-request", &body);
        custody
    };
    let client = crate::tests::client(directory.path());
    let queue = client.queue().unwrap();
    let entry = queue.get(&custody.frame).unwrap().unwrap();
    let before = entry.frame_hash.clone();
    let mut count = 0;
    visit(
        &queue,
        &entry,
        "polylogue-test",
        Some(&custody),
        &mut |pending| {
            let PendingAuthored::ConservedQueued {
                accepted,
                claims,
                custody: actual,
            } = pending
            else {
                panic!("queued only")
            };
            assert_eq!(actual, &custody);
            assert!(claims.is_empty());
            assert!(pending.prepared().is_none());
            let ConservedAccepted::Historical(history) = accepted else {
                panic!("retired note")
            };
            assert!(matches!(
                &history.source,
                SourceSchema::RetainedPublication {
                    source_version: 3,
                    ..
                }
            ));
            assert!(matches!(
                &history.hash_evidence,
                AcceptedHashEvidence::Derived { .. }
            ));
            assert!(history.native.is_empty());
            assert_eq!(
                history.original.as_ref().unwrap().request.as_deref(),
                Some(
                    serde_json::to_vec(&old_body()["request"])
                        .unwrap()
                        .as_slice()
                )
            );
            count += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(count, 1);
    assert_eq!(
        queue.get(&custody.frame).unwrap().unwrap().frame_hash,
        before
    );
    assert_eq!(queue.pending_after(None, 64).unwrap().len(), 1);
}

#[test]
fn missing_or_cross_import_binding_never_enters_the_pending_overlay() {
    let directory = tempfile::tempdir().unwrap();
    let client = crate::tests::client(directory.path());
    let queue = client.queue().unwrap();
    let (entry, custody) = prepare(
        &queue,
        "polylogue-request",
        &serde_json::to_vec(&old_body()).unwrap(),
    );
    let mut count = 0;
    assert!(
        visit(&queue, &entry, "polylogue-test", None, &mut |_| {
            count += 1;
            Ok(())
        })
        .is_err()
    );
    for field in [
        "producer",
        "fingerprint",
        "frame",
        "frame_hash",
        "material",
        "blob",
        "length",
        "receipt",
        "parser",
        "version",
    ] {
        let mut other = custody.clone();
        match field {
            "producer" => other.producer_identity = "another-producer".into(),
            "fingerprint" => other.source_fingerprint = "changed".into(),
            "frame" => other.frame = "another-frame".into(),
            "frame_hash" => other.frame_content_hash = BlobHash::of(b"another-frame"),
            "material" => other.material = uuid::Uuid::from_bytes([9; 16]).to_string(),
            "blob" => other.blob = BlobHash::of(b"changed"),
            "length" => other.length += 1,
            "receipt" => other.receipt_number = 0,
            "parser" => other.parser = "polylogue-products".into(),
            _ => other.source_version = 4,
        }
        assert!(
            visit(&queue, &entry, "polylogue-test", Some(&other), &mut |_| {
                count += 1;
                Ok(())
            })
            .is_err(),
            "{field}"
        );
    }
    assert_eq!(count, 0);
    assert_eq!(queue.pending_after(None, 64).unwrap().len(), 1);
}

#[test]
fn retained_restore_stream_keeps_actual_carried_imports_declared_and_full_body_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let client = crate::tests::client(directory.path());
    let queue = client.queue().unwrap();
    let claim = json!({"machine":"old-importer","input":"old-bundle","chain":2,"actor":"operator","at":"2026-01-02T03:04:06Z"});
    let body=serde_json::to_vec(&json!([{"product":{"product":"restored_request","prepared":old_body(),"claims":[],"current":claim},"location":null,"parents":[],"machine":null,"output_index":0}])).unwrap();
    let (entry, custody) = prepare(&queue, "polylogue-request-restore", &body);
    let mut count = 0;
    visit(
        &queue,
        &entry,
        "polylogue-test",
        Some(&custody),
        &mut |pending| {
            let PendingAuthored::ConservedQueued {
                claims, accepted, ..
            } = pending
            else {
                panic!("queued only")
            };
            assert_eq!(claims.len(), 1);
            assert_eq!(claims[0].machine, "old-importer");
            let ConservedAccepted::Historical(history) = accepted else {
                panic!("retired note")
            };
            assert!(history.imports.observed.is_empty());
            assert!(history.native.is_empty());
            count += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(count, 1);
    let raw: Box<serde_json::value::RawValue> =
        crate::spool::decode_json(queue.body(&entry.identity).unwrap()).unwrap();
    assert_eq!(raw.get().as_bytes(), body);
}

#[test]
fn retry_routing_holds_only_source3_request_frames_and_streams_deep_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let client = crate::tests::client(directory.path());
    let queue = client.queue().unwrap();
    let (material, _) = prepare(
        &queue,
        "polylogue-request",
        &serde_json::to_vec(&old_body()).unwrap(),
    );
    assert!(requires_retained(&queue, &material).unwrap());
    let mut deep = json!("retained");
    for _ in 0..200 {
        deep = json!({"nested":deep});
    }
    for (name, source, kind, version, expected) in [
        ("old-request", 3, "polylogue.note", 4, true),
        ("old-raw", 3, "polylogue.blob", 3, false),
        ("unsupported-request", 2, "polylogue.note", 4, true),
        ("current-request", 4, "polylogue.assertion", 1, false),
    ] {
        let frame = queue
            .prepare(Ingress::Events {
                frame: name.into(),
                source: Some(sinex_protocol::SourceRef {
                    name: "polylogue".into(),
                    version: source,
                }),
                coverage: None,
                events: vec![sinex_protocol::ProducerEvent {
                    kind: sinex_model::KindRef {
                        name: kind.into(),
                        version,
                    },
                    payload: deep.clone(),
                    location: None,
                    output_index: 0,
                    parents: Vec::new(),
                    reading_order: None,
                    class: sinex_model::Class::Open,
                }],
                failures: Vec::new(),
            })
            .unwrap();
        assert_eq!(
            requires_retained(&queue, &frame).unwrap(),
            expected,
            "{name}"
        );
    }
}

#[test]
fn malformed_later_retained_restore_cannot_partially_apply_the_pending_overlay() {
    let directory = tempfile::tempdir().unwrap();
    let client = crate::tests::client(directory.path());
    let queue = client.queue().unwrap();
    let claim = json!({"machine":"old-importer","input":"old-bundle","chain":2,"actor":"operator","at":"2026-01-02T03:04:06Z"});
    let body=serde_json::to_vec(&json!([
        {"product":{"product":"restored_request","prepared":old_body(),"claims":[],"current":claim},"location":null,"parents":[],"machine":null,"output_index":0},
        {"product":{"product":"restored_assertion"}}
    ])).unwrap();
    let (entry, custody) = prepare(&queue, "polylogue-request-restore", &body);
    let mut count = 0;
    assert!(
        visit(
            &queue,
            &entry,
            "polylogue-test",
            Some(&custody),
            &mut |_| {
                count += 1;
                Ok(())
            }
        )
        .is_err()
    );
    assert_eq!(count, 0);
    assert_eq!(queue.pending_after(None, 64).unwrap().len(), 1);
}
