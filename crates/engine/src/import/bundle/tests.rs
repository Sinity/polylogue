use super::*;
use polylogue_model::bundle::{
    MessageRecord, SourceClaim, TextCoverage, TextFragment, TextFragmentProvenance,
    TextFragmentRecord, TextSourceVersion, TextSpanOrigin,
};
use polylogue_model::{BlobHash, Message, Reference, SpanUnit};

fn fragment(version: BlobHash, text: &str, total: u64, source: &str) -> Record {
    let container = Reference::parse("message:codex:source").unwrap();
    let end = text.len() as u64;
    let fragment = TextFragment {
        container: container.clone(),
        source_version: TextSourceVersion::ContentHash {
            record: container.clone(),
            hash: version,
        },
        coverage: TextCoverage {
            start: 0,
            end,
            total_bytes: Some(total),
        },
        text: text.into(),
        text_hash: BlobHash::of(text.as_bytes()),
        provenance: TextFragmentProvenance {
            spans: vec![TextSpanOrigin {
                reference: Reference::Span {
                    container: Box::new(container),
                    unit: SpanUnit::Bytes,
                    start: 0,
                    end,
                },
                start: 0,
                end,
            }],
            sources: vec![SourceClaim {
                input: source.into(),
                source: "foreign-provider".into(),
                source_chain: 1,
                source_start: Some(0),
                source_end: Some(100),
            }],
            imports: vec![],
        },
    };
    Record::TextFragment(Box::new(TextFragmentRecord {
        reference: fragment.reference().to_string(),
        commit: 17,
        fragment,
    }))
}

#[test]
fn fragment_staging_conserves_versions_conflicting_claims_and_complete_canonical_input() {
    let message: Message = serde_json::from_value(serde_json::json!({
        "id":"codex:source", "session":"codex:session", "role":"user",
        "blocks":[{"type":"text","text":"AB canonical full source"}]
    }))
    .unwrap();
    let version = message.content_hash().unwrap();
    let canonical = Record::Message(Box::new(MessageRecord {
        reference: "message:codex:source".into(),
        commit: 17,
        order: 0,
        source_version: version,
        block_count: 1,
        block_ordinals: vec![0],
        injections: vec![],
        bindings: vec![],
        copied_for: Vec::new(),
        message,
    }));
    let declarations = vec![
        fragment(version, "AB", 8, "claim-a"),
        fragment(BlobHash::of(b"another source version"), "AB", 8, "claim-a"),
        fragment(version, "CD", 8, "claim-a"),
        fragment(version, "AB", 9, "claim-a"),
        fragment(version, "AB", 8, "claim-b"),
    ];
    let mut staged = polylogue_store::export_items::ExportItems::new().unwrap();
    for (ordinal, record) in declarations
        .iter()
        .chain(std::iter::once(&canonical))
        .enumerate()
    {
        record.validate().unwrap();
        stage_record(
            &mut staged,
            ordinal as u64..ordinal as u64 + 1,
            record.clone(),
        )
        .unwrap();
    }
    stage_record(&mut staged, 100..101, declarations[0].clone()).unwrap();
    let mut counts = Counts::default();
    let mut retained = Vec::new();
    while let Some(item) = staged.next_item().unwrap() {
        let record: Record = serde_json::from_reader(item.reader).unwrap();
        count(&mut counts, &record);
        retained.push(record);
    }
    assert_eq!(counts.messages, 1);
    assert_eq!(counts.text_fragments, 5);
    assert_eq!(retained.len(), 6);
    assert!(retained.contains(&canonical));
    for declaration in declarations {
        assert!(retained.contains(&declaration));
    }
}
