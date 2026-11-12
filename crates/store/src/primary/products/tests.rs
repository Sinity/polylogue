use super::*;
use polylogue_model::batch::{ExportDelivery, ExportDeliverySource, FragmentRow};
use polylogue_model::bundle::fragments::{
    TextCoverage, TextFragment, TextFragmentProvenance, TextFragmentRecord, TextSourceVersion,
    TextSpanOrigin,
};
use polylogue_model::{
    Block, Document, DocumentPurpose, LinkKind, Output, RecordOutcome, Reference, Role,
    SessionKind, SpanUnit,
};

#[test]
fn complete_callback_product_roundtrip_preserves_all_semantic_maps_and_generated_event_ids() {
    let message = crate::fixture::writer::message(
        "export-message",
        "export-session",
        Role::Assistant,
        1,
        vec![Block::Text {
            text: "original body".into(),
        }],
    );
    let version = message.content_hash().unwrap();
    let document = Document::new(
        polylogue_model::DocumentId::parse("claude-code:export-document").unwrap(),
        DocumentPurpose::Context,
    );
    let event: Event = serde_json::from_value(
        serde_json::json!({"kind":"native","range":[22,29],"data":{"nested":{"exact":1.25}}}),
    )
    .unwrap();
    let container = Reference::Message(message.id.clone());
    let fragment = TextFragment {
        container: container.clone(),
        source_version: TextSourceVersion::ContentHash {
            record: container.clone(),
            hash: version,
        },
        coverage: TextCoverage {
            start: 0,
            end: 1,
            total_bytes: None,
        },
        text: "x".into(),
        text_hash: BlobHash::of(b"x"),
        provenance: TextFragmentProvenance {
            spans: vec![TextSpanOrigin {
                reference: Reference::Span {
                    container: Box::new(container),
                    unit: SpanUnit::Bytes,
                    start: 0,
                    end: 1,
                },
                start: 0,
                end: 1,
            }],
            sources: vec![],
            imports: vec![],
        },
    };
    let mut output = Output {
        sessions: vec![Session::new(message.session.clone(), SessionKind::Main)],
        messages: vec![MessageRow {
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 7,
            end: 19,
            message: message.clone(),
        }],
        documents: vec![DocumentRow {
            offset: 3,
            end: 9,
            document,
        }],
        fragments: vec![FragmentRow {
            offset: 30,
            end: 31,
            record: TextFragmentRecord {
                reference: fragment.reference().to_string(),
                commit: 999,
                fragment,
            },
        }],
        events: vec![event.clone(), event],
        links: vec![Link {
            kind: LinkKind::Spawn,
            copies_prefix: false,
            how: None,
            parent: message.session.clone(),
            child: polylogue_model::SessionId::parse("claude-code:export-child").unwrap(),
            at: None,
            call_id: None,
            evidence: "original marker".into(),
        }],
        prefix_copies: vec![polylogue_model::PrefixCopy {
            explicit_members: false,
            parent: message.session.clone(),
            child: polylogue_model::SessionId::parse("claude-code:export-child").unwrap(),
        }],
        prefix_members: vec![polylogue_model::PrefixMember {
            parent: message.session.clone(),
            child: polylogue_model::SessionId::parse("claude-code:export-child").unwrap(),
            message: message.id.clone(),
            source: Some(polylogue_model::PrefixSource {
                input: "original-copy".into(),
                chain: 4,
                reading_version: "provider-reading".into(),
            }),
            position: [u64::MAX, 2],
        }],
        repo_lookups: vec![RepoLookup {
            path: "/synthetic/project".into(),
            remote: None,
            root_commits: Vec::new(),
        }],
        ..Default::default()
    };
    output.message_order.insert(0, 51);
    output.message_projections.insert(
        0,
        polylogue_model::bundle::MessageProjection {
            source_version: version,
            block_count: 1,
            block_ordinals: vec![0],
        },
    );
    output.native_hook_fields.insert(0, Vec::new());
    output.declared_injections.insert(0, Vec::new());
    output.declared_attachment_bindings.insert(0, Vec::new());
    output.event_ids.insert(0, "explicit-event".into());
    output.imported_events.insert(
        0,
        polylogue_model::bundle::ImportedEventIdentity {
            id: "explicit-event".into(),
            sources: vec![polylogue_model::bundle::SourceClaim {
                input: "original-input".into(),
                source: "provider".into(),
                source_chain: 4,
                source_start: Some(22),
                source_end: Some(29),
            }],
        },
    );
    output.message_receipts.insert(0, 10);
    output.session_receipts.insert(0, 11);
    output.document_receipts.insert(0, 12);
    output.fragment_receipts.insert(0, 13);
    output.event_receipts.insert(0, 14);
    output.link_receipts.insert(0, 15);
    output.prefix_copy_receipts.insert(0, 16);
    output.prefix_member_receipts.insert(0, 17);
    let mut counts = RecordCounts {
        records: 1,
        ..Default::default()
    };
    counts.add("item", &RecordOutcome::Used);
    let mut step = crate::ParseStep::new(output, counts, 0..0, false);
    step.reading_version = "crate::export::VERSION=2".into();
    step.order_from = Some(40);
    step.reset = true;
    step.repeats_whole_conversations = true;
    step.seal_export(
        0,
        ExportDeliverySource::Item { start: 0, end: 64 },
        ExportDelivery::seed(),
    )
    .unwrap();
    let key = ExportReadingKey {
        input: InputKey {
            id: "source".into(),
            format: "fixture".into(),
            route: polylogue_model::Route::Live,
        },
        chain: 1,
        reading_version: step.reading_version.clone(),
        length: 64,
    };
    step.prepare();
    let captured = CapturedReading::from_step(&step, Some(key.clone())).unwrap();
    let products = parse(&key.input, key.chain, Some(key.length), &step).unwrap();
    assert!(products.iter().any(|product|matches!(product,Product::Message{native_fields:Some(fields),declared_injections:Some(injections),declared_attachment_bindings:Some(bindings),..} if fields.is_empty()&&injections.is_empty()&&bindings.is_empty())));
    let restored = CapturedReading::from_product_group(&key, products.iter()).unwrap();
    restored.clone().into_step().validate_export().unwrap();
    let mut expected = captured.output.clone();
    expected.message_receipts.clear();
    expected.session_receipts.clear();
    expected.document_receipts.clear();
    expected.fragment_receipts.clear();
    expected.event_receipts.clear();
    expected.link_receipts.clear();
    expected.prefix_copy_receipts.clear();
    expected.prefix_member_receipts.clear();
    assert_eq!(restored.output, expected);
    assert_eq!(restored.delivery, captured.delivery);
    assert_eq!(restored.counts, captured.counts);
    let mut replay = restored.into_step();
    replay.prepare();
    assert_eq!(
        replay.event_identities(&key.input.id, key.chain).unwrap(),
        step.event_identities(&key.input.id, key.chain).unwrap()
    );
    let mut reranked = key.clone();
    reranked.input.route = polylogue_model::Route::Export;
    assert_eq!(key, reranked);
    assert_eq!(key.identity(), reranked.identity());
}
