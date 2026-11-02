use super::*;
use polylogue_model::batch::StepProgress;
use polylogue_store::primary::{Product, Reading};
use serde_json::{Value, json};
use sinex_model::{Class, EventId, RunId, Timestamp, Validation};
use sinex_protocol::{ConfirmedEvent, ProjectionIdentity};

fn reading(progress: StepProgress) -> Reading {
    Reading {
        delivery: None,
        source_length: None,
        input: polylogue_store::InputKey {
            id: "earlier-input".into(),
            format: "codex".into(),
            arrival_rank: 0,
        },
        chain: 1,
        reading_version: "reader-earlier".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start: 0,
        parsed_upto: 8,
        progress,
    }
}

fn document(progress: StepProgress) -> Product {
    Product::Document {
        reading: reading(progress),
        facts: serde_json::from_value(json!({"offset":0,"end":8,"document":{"id":"codex:earlier-document","purpose":"context","blocks":[{"type":"text","text":"synthetic"}]}})).unwrap(),
    }
}

fn fragment(progress: StepProgress) -> Product {
    use polylogue_model::bundle::fragments::{
        TextCoverage, TextFragment, TextFragmentProvenance, TextFragmentRecord, TextSourceVersion,
        TextSpanOrigin,
    };
    use polylogue_model::{BlobHash, Reference};
    let container =
        Reference::Message(polylogue_model::MessageId::parse("codex:earlier-fragment").unwrap());
    let mut fragment = TextFragment {
        container: container.clone(),
        source_version: TextSourceVersion::ContentHash {
            record: container,
            hash: BlobHash::of(b"original source value"),
        },
        coverage: TextCoverage {
            start: 0,
            end: 3,
            total_bytes: None,
        },
        text: "abc".into(),
        text_hash: BlobHash::of(b"abc"),
        provenance: TextFragmentProvenance {
            spans: vec![],
            sources: vec![],
            imports: vec![],
        },
    };
    fragment.provenance.spans.push(TextSpanOrigin {
        reference: fragment.reference(),
        start: 0,
        end: 3,
    });
    let record = TextFragmentRecord {
        reference: fragment.reference().to_string(),
        commit: 9000,
        fragment,
    };
    record.validate().unwrap();
    Product::Fragment {
        reading: reading(progress),
        facts: Box::new(polylogue_model::batch::FragmentRow {
            offset: 0,
            end: 8,
            record,
        }),
    }
}

/// The payload an earlier build wrote: the reading's progress as two flags.
fn earlier_payload(product: &Product, at_end: bool, no_parser_yet: bool) -> Value {
    let mut payload = crate::products::payload(product, 0).unwrap();
    let reading = payload["record"]["reading"].as_object_mut().unwrap();
    reading.remove("progress").unwrap();
    reading.insert("at_end".into(), json!(at_end));
    reading.insert("no_parser_yet".into(), json!(no_parser_yet));
    payload
}

fn event(name: &str, version: u64, hash: String, payload: Value) -> ConfirmedEvent {
    ConfirmedEvent {
        source_ref: Some(crate::source_reference()),
        source_fingerprint: Some(crate::kinds::source_declaration().fingerprint()),
        producer_family: Some("polylogue".into()),
        producer_identity: Some("earlier-build".into()),
        kind_version: version,
        kind_definition_hash: hash,
        validation: Validation::Validated,
        commit_seq: 17,
        id: EventId(uuid::Uuid::from_bytes([1; 16])),
        event_type: name.into(),
        source: "polylogue".into(),
        run: RunId(uuid::Uuid::from_bytes([2; 16])),
        location: Some(sinex_model::RecordLocation {
            material: sinex_model::MaterialId(uuid::Uuid::from_bytes([3; 16])),
            anchor: 9,
        }),
        output_index: 2,
        parents: vec![],
        ts_orig: None,
        ts_quality: "unknown".into(),
        ts_coided: None,
        ts_persisted: Timestamp::from_unix_secs(1).unwrap(),
        record_state: "present".into(),
        class: Class::Open,
        equivalence_key: None,
        reading_order: 17,
        record_hash: "5".repeat(64),
        projection: ProjectionIdentity::for_payload(7, &payload),
        payload,
    }
}

#[test]
fn readings_stored_as_two_flags_decode_with_their_progress_value() {
    for (product, at_end, no_parser_yet, progress) in [
        (
            document(StepProgress::NoParserYet { at_end: false }),
            false,
            true,
            StepProgress::NoParserYet { at_end: false },
        ),
        (
            fragment(StepProgress::Partway),
            false,
            false,
            StepProgress::Partway,
        ),
        (
            fragment(StepProgress::AtEnd),
            true,
            false,
            StepProgress::AtEnd,
        ),
    ] {
        let name = crate::products::kind(&product).unwrap();
        let version = match name {
            "polylogue.document" => 4,
            _ => 2,
        };
        let declaration = source3::reading_declaration(name, version)
            .unwrap()
            .unwrap();
        let payload = earlier_payload(&product, at_end, no_parser_yet);
        jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
            .unwrap()
            .validate(&payload)
            .unwrap();
        let event = event(
            name,
            declaration.kind.version,
            declaration.definition_hash(),
            payload,
        );
        let confirmed = decode_event(&event).unwrap();
        let ConfirmedChange::Live(actual) = &confirmed.change else {
            panic!("{name} did not decode as a live product");
        };
        let (Product::Document { reading, .. } | Product::Fragment { reading, .. }) =
            actual.as_ref()
        else {
            panic!("{name} decoded as another product");
        };
        assert_eq!(reading.progress, progress, "{name}");
        assert_eq!(actual.as_ref(), &product, "{name}");
        let native = confirmed.native.unwrap();
        assert_eq!(native.kind.version, declaration.kind.version);
        assert_eq!(native.kind_definition_hash, declaration.definition_hash());
    }
}

#[test]
fn a_version_that_never_existed_or_a_changed_frozen_definition_is_refused() {
    let product = document(StepProgress::AtEnd);
    let never = event(
        "polylogue.document",
        3,
        "0".repeat(64),
        earlier_payload(&product, true, false),
    );
    assert!(matches!(
        decode_event(&never),
        Err(Error::Invalid(message)) if message == "unsupported indexed kind polylogue.document@3"
    ));
    let changed = event(
        "polylogue.document",
        4,
        "0".repeat(64),
        earlier_payload(&product, true, false),
    );
    assert!(matches!(
        decode_event(&changed),
        Err(Error::Invalid(message))
            if message == "fetched indexed kind differs from the supported immutable schema"
    ));
}

#[test]
fn every_reading_kind_reads_its_last_two_flag_declaration() {
    // Definition hashes the build before the single progress value
    // registered (integrate-20261101 at d2b15c76f0).
    for (name, version, hash) in [
        (
            "polylogue.session",
            4,
            "66880ad11a2e6b55a1035ad54b96fc93071144b9e92a5fb8ea2d6b512396a262",
        ),
        (
            "polylogue.message",
            5,
            "08ef9196fe3188ee554e535f5bb0cbfb698e29f5ad98331f58709fe7622a3c6a",
        ),
        (
            "polylogue.document",
            4,
            "93425dedc73838b370fc143d5b6de425c999be6a3a7ff0307a26e35123d0e622",
        ),
        (
            "polylogue.fragment",
            2,
            "e55c7b9984280828e9d78630c1de31234b6acb6353ddee128cc67d970c3e1474",
        ),
        (
            "polylogue.event",
            4,
            "93f3194ed903bc208cda4c71bb2728c48c1405f1660986c612993b2ad3eeea07",
        ),
        (
            "polylogue.link",
            4,
            "3f019211050260f5c61d9663293add3cf61c7d76e8aad82bb6f5c3ef77677cdf",
        ),
        (
            "polylogue.background",
            4,
            "f9c9b590c604a9d3f1117de286c26086daf04c440100bc1bb10948c5f80d58fd",
        ),
        (
            "polylogue.repository",
            4,
            "034c20a82ffc1dd96c0d6d1a3b730b25a13da4f41914ec8ac53a86d6d0a7cd32",
        ),
        (
            "polylogue.parse_progress",
            4,
            "d58c43d61a51cd632eaa4784011216be22c25fcd885e7c5243230d2137934bc4",
        ),
    ] {
        let declaration = source3::reading_declaration(name, version)
            .unwrap()
            .unwrap();
        assert_eq!(declaration.definition_hash(), hash, "{name}");
        let reading = &declaration.payload_schema.as_ref().unwrap()["$defs"]["Reading"];
        assert!(reading["properties"].get("at_end").is_some(), "{name}");
        assert!(
            reading["properties"].get("no_parser_yet").is_some(),
            "{name}"
        );
        assert!(reading["properties"].get("progress").is_none(), "{name}");
        let current = crate::kinds::declarations("polylogue")
            .into_iter()
            .find(|declaration| declaration.kind.name == name)
            .unwrap();
        assert_eq!(current.kind.version, version + 1, "{name}");
    }
    assert!(
        source3::reading_declaration("polylogue.note", 4)
            .unwrap()
            .is_none()
    );
    assert!(
        source3::reading_declaration("polylogue.message", 6)
            .unwrap()
            .is_none()
    );
}
