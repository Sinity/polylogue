use super::*;
use crate::fixture::writer::{WriterArchive, message};
use polylogue_model::Role;

fn source(id: &str, blocks: Vec<Block>) -> polylogue_model::Message {
    let mut value = message(id, "s", Role::User, 1, blocks);
    value.at = None;
    value
}
fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
fn text(reader: &Reader<'_>, reference: &str) -> Option<String> {
    let mut text = String::new();
    reader
        .visit_reference_text(&r(reference), |chunk| {
            text.push_str(chunk);
            Ok(())
        })
        .unwrap()
        .map(|_| text)
}
#[test]
fn reference_text_streams_original_blocks_and_hashes_the_declared_plain_text() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![
            source(
                "m1",
                vec![
                    Block::Text {
                        text: "a💡ż".into(),
                    },
                    Block::ToolCall {
                        call_id: Some("call".into()),
                        tool: Some("tool".into()),
                        action: Default::default(),
                        input: serde_json::json!({"hidden":"not source text"}),
                        command: None,
                        path: None,
                        run_by_script: false,
                    },
                    Block::Text {
                        text: "beta".into(),
                    },
                ],
            ),
            source(
                "m2",
                vec![Block::Text {
                    text: "gamma".into(),
                }],
            ),
        ],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        text(&reader, "session:claude-code:s").as_deref(),
        Some("a💡ż\n\nbeta\n\ngamma")
    );
    assert_eq!(
        text(&reader, "message:claude-code:s-m1").as_deref(),
        Some("a💡ż\n\nbeta")
    );
    assert_eq!(
        text(&reader, "block:claude-code:s-m1#2").as_deref(),
        Some("beta")
    );
    assert_eq!(
        reader.reference_hash(&r("session:claude-code:s")).unwrap(),
        Some(BlobHash(
            *blake3::hash("a💡ż\n\nbeta\n\ngamma".as_bytes()).as_bytes()
        ))
    );
    assert_eq!(
        text(&reader, "message:claude-code:s-m1..claude-code:s-m2").as_deref(),
        Some("a💡ż\n\nbeta\n\ngamma")
    );
    assert!(matches!(
        reader.reference_hash(&r("message:claude-code:s-m2..claude-code:s-m1")),
        Err(StoreError::Invalid(_))
    ));
}
#[test]
fn spans_use_source_offsets_and_refuse_half_characters_before_emitting() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![source(
            "m1",
            vec![Block::Text {
                text: "a💡ż".into(),
            }],
        )],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    for reference in [
        "span:message:claude-code:s-m1#b=1-5",
        "span:message:claude-code:s-m1#c=1-2",
        "span:message:claude-code:s-m1#u16=1-3",
    ] {
        assert_eq!(text(&reader, reference).as_deref(), Some("💡"));
    }
    for reference in [
        "span:message:claude-code:s-m1#b=2-5",
        "span:message:claude-code:s-m1#u16=1-2",
        "span:message:claude-code:s-m1#c=0-4",
    ] {
        let mut emitted = false;
        assert!(matches!(
            reader.visit_reference_text(&r(reference), |_| {
                emitted = true;
                Ok(())
            }),
            Err(StoreError::Invalid(_))
        ));
        assert!(!emitted);
    }
}
#[test]
fn missing_and_unrecorded_text_never_become_an_accessible_empty_hash() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![
            source(
                "empty",
                vec![Block::Text {
                    text: String::new(),
                }],
            ),
            source(
                "unknown",
                vec![Block::Thinking {
                    text: None,
                    signature: Some("opaque".into()),
                    redacted: true,
                }],
            ),
        ],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .reference_hash(&r("message:claude-code:s-empty"))
            .unwrap(),
        Some(BlobHash(*blake3::hash(b"").as_bytes()))
    );
    assert_eq!(
        reader
            .reference_hash(&r("message:claude-code:missing"))
            .unwrap(),
        None
    );
    assert!(matches!(
        reader.reference_hash(&r("message:claude-code:s-unknown")),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
}
