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
            route: polylogue_model::Route::Live,
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

pub(super) fn document(progress: StepProgress) -> Product {
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
        projection: ProjectionIdentity::for_payload(&payload),
        payload,
    }
}

pub(super) fn current_event(product: &Product) -> ConfirmedEvent {
    let name = crate::products::kind(product);
    let declaration = crate::kinds::declarations("polylogue")
        .into_iter()
        .find(|declaration| declaration.kind.name == name)
        .unwrap();
    event(
        name,
        declaration.kind.version,
        declaration.definition_hash(),
        crate::products::payload(product, 0).unwrap(),
    )
}

#[test]
fn current_readings_keep_their_progress_and_authenticated_owner_context() {
    for product in [
        document(StepProgress::AtEnd),
        fragment(StepProgress::Partway),
    ] {
        let event = current_event(&product);
        let confirmed = decode_event(&event).unwrap();
        let ConfirmedChange::Live(actual) = confirmed.change else {
            panic!("current reading did not decode as a live product");
        };
        assert_eq!(*actual, product);
        let native = confirmed.native.unwrap();
        assert_eq!(native.kind.name, event.event_type);
        assert_eq!(native.kind.version, event.kind_version);
        assert_eq!(native.kind_definition_hash, event.kind_definition_hash);
        assert_eq!(
            native.projection.payload_hash,
            event.projection.payload_hash
        );
        assert_eq!(native.origin.producer_identity, event.producer_identity);
        assert_eq!(native.origin.source_fingerprint, event.source_fingerprint);
        assert_eq!(
            serde_json::to_value(native.projection).unwrap(),
            json!({"payload_hash": event.projection.payload_hash})
        );
    }
}

#[test]
fn unsupported_versions_and_changed_kind_definitions_are_refused() {
    let mut unsupported = current_event(&document(StepProgress::AtEnd));
    unsupported.kind_version += 1000;
    assert!(
        matches!(decode_event(&unsupported), Err(Error::Invalid(message))
        if message.starts_with("unsupported Polylogue kind polylogue.document@"))
    );
    let mut changed = current_event(&document(StepProgress::AtEnd));
    changed.kind_definition_hash = "0".repeat(64);
    assert!(
        matches!(decode_event(&changed), Err(Error::Invalid(message))
        if message == "a confirmed event's kind definition differs from Polylogue's")
    );
}

#[test]
fn no_position_confirmed_removal_requires_post_ack_head_in_the_applied_scope() {
    let state = crate::tests::applied(42);
    let before = std::time::Instant::now();
    let after = before + std::time::Duration::from_secs(1);
    let mut health = crate::status::Health {
        owner_position: state.head,
        owner_scope: state.scope.clone(),
        position_observed: Some(before),
        last_ack: Some(after),
        ..Default::default()
    };
    assert_eq!(
        confirmed_removal_state(&state, &health).head,
        None,
        "a caught-up presend head cannot cover a new acknowledgment"
    );
    health.position_observed = Some(after);
    assert_eq!(
        confirmed_removal_state(&state, &health).head,
        Some(42),
        "a newly verified head need not advance numerically"
    );
    health.owner_scope = Some("replacement-scope".into());
    assert_eq!(
        confirmed_removal_state(&state, &health).head,
        None,
        "the same number in another scope is not the applied prefix"
    );
    let mut reset = state;
    reset.scope = health.owner_scope.clone();
    reset.applied_position = None;
    assert_eq!(
        confirmed_removal_state(&reset, &health).applied_position,
        None
    );
    health.owner_position = Some(43);
    assert_eq!(
        confirmed_removal_state(&reset, &health).head,
        None,
        "a newer observed head must itself be saved before confirmed_removal"
    );
}

mod following;
