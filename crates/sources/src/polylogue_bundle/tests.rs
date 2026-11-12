use super::*;
use polylogue_model::{BlobHash, Message, Reference, SpanUnit, bundle::*, injection::*};
use serde_json::json;

fn message_record(id: &str, order: u64) -> MessageRecord {
    let message: Message = serde_json::from_value(json!({
        "id":format!("codex:{id}"), "session":"codex:session", "role":"system",
        "prompt_kind":"harness", "blocks":[{"type":"text","text":"context"}]
    }))
    .unwrap();
    let source_version = message.content_hash().unwrap();
    MessageRecord {
        reference: format!("message:codex:{id}"),
        commit: order,
        order,
        source_version,
        block_count: 1,
        block_ordinals: vec![0],
        bindings: vec![],
        copied_for: Vec::new(),
        injections: vec![InjectionEvidence {
            message: message.id.clone(),
            source_version,
            block: 0,
            observed: Some(NativeHookField {
                input: "foreign-input".into(),
                chain: 7,
                start: 10,
                end: 40,
                pointer: "/hookAdditionalContext/0".into(),
            }),
            declared: vec![],
        }],
        message,
    }
}

fn fragment_record() -> TextFragmentRecord {
    let container = Reference::parse("message:codex:source").unwrap();
    let fragment = TextFragment {
        container: container.clone(),
        source_version: TextSourceVersion::ContentHash {
            record: container.clone(),
            hash: BlobHash::of(b"complete source value"),
        },
        coverage: TextCoverage {
            start: 5,
            end: 8,
            total_bytes: Some(20),
        },
        text: "λZ".into(),
        text_hash: BlobHash::of("λZ".as_bytes()),
        provenance: TextFragmentProvenance {
            spans: vec![TextSpanOrigin {
                reference: Reference::Span {
                    container: Box::new(container),
                    unit: SpanUnit::Bytes,
                    start: 5,
                    end: 8,
                },
                start: 5,
                end: 8,
            }],
            sources: vec![SourceClaim {
                input: "foreign-input".into(),
                source: "codex".into(),
                source_chain: 7,
                source_start: Some(10),
                source_end: Some(40),
            }],
            imports: vec![],
        },
    };
    TextFragmentRecord {
        reference: fragment.reference().to_string(),
        commit: 999,
        fragment,
    }
}

#[test]
fn bundle_import_keeps_hook_claims_declared_and_offsets_their_message_keys() {
    let records = [message_record("first", 21), message_record("second", 22)];
    let mut bytes = Vec::new();
    for record in &records {
        bytes.extend(serde_json::to_vec(&Record::Message(Box::new(record.clone()))).unwrap());
        bytes.push(b'\n');
    }
    let mut parser = open(std::path::Path::new("bundle.jsonl")).unwrap();
    let step = crate::export::fixture::parse(&mut parser, &bytes).unwrap();
    assert_eq!(step.output.messages.len(), 2);
    assert_eq!(step.output.declared_injections.len(), 2);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(step.output.messages[index].message, record.message);
        assert_eq!(
            step.output.message_projections[&index].source_version,
            record.source_version
        );
        assert_eq!(step.output.message_order[&index], record.order);
        let [claim] = &step.output.declared_injections[&index][..] else {
            panic!("one claim")
        };
        assert_eq!(*claim, record.injections[0].clone().into_declared());
        assert!(claim.observed.is_none());
    }
    assert!(step.output.native_hook_fields.is_empty());
    assert!(step.output.message_receipts.is_empty());
}

#[test]
fn imported_text_fragment_keeps_foreign_claims_and_exact_receiving_record_range() {
    let record = fragment_record();
    record.validate().unwrap();
    let bytes = serde_json::to_vec(&Record::TextFragment(Box::new(record.clone()))).unwrap();
    let range = 23..23 + bytes.len() as u64;
    let step = crate::export::parse_item(
        &mut Bundle,
        &bytes,
        range.clone(),
        &mut crate::export::MemoryState::default(),
    )
    .unwrap();
    let [row] = &step.output.fragments[..] else {
        panic!("one fragment")
    };
    assert_eq!((row.offset, row.end), (range.start, range.end));
    assert_eq!(row.record, record);
    assert!(step.output.messages.is_empty());
    assert!(step.output.documents.is_empty());
    assert!(step.output.fragment_receipts.is_empty());
    assert!(step.output.native_hook_fields.is_empty());
}

#[test]
fn imported_copy_members_keep_original_order_and_do_not_claim_other_parent_messages() {
    let mut copied = message_record("copied", 1);
    copied.copied_for.push(CopiedFor {
        child: polylogue_model::SessionId::parse("codex:child").unwrap(),
        source: polylogue_model::PrefixSource {
            input: "original-child-input".into(),
            chain: 4,
            reading_version: "original-reading".into(),
        },
        position: [u64::MAX, u64::MAX - 1],
    });
    let later = message_record("later", 2);
    let mut bytes = serde_json::to_vec(&Record::Message(Box::new(copied))).unwrap();
    bytes.push(b'\n');
    bytes.extend(serde_json::to_vec(&Record::Message(Box::new(later))).unwrap());
    let mut parser = open(Path::new("bundle.jsonl")).unwrap();
    let step = crate::export::fixture::parse(&mut parser, &bytes).unwrap();
    assert_eq!(step.output.messages.len(), 2);
    let [member] = &step.output.prefix_members[..] else {
        panic!("only the observed copy")
    };
    assert_eq!(member.parent.to_string(), "codex:session");
    assert_eq!(member.child.to_string(), "codex:child");
    assert_eq!(member.message.to_string(), "codex:copied");
    assert_eq!(member.position, [u64::MAX, u64::MAX - 1]);
    let source = member.source.as_ref().unwrap();
    assert_eq!(
        (&*source.input, source.chain, &*source.reading_version),
        ("original-child-input", 4, "original-reading")
    );
}
