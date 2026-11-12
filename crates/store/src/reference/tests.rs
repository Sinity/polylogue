use super::*;
use crate::fixture::writer::{WriterArchive, message};
use polylogue_model::Role;

fn source(id: &str, blocks: Vec<Block>) -> polylogue_model::Message {
    message(id, "s", Role::User, 1, blocks)
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
                        written_text: None,
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

#[test]
fn sparse_source_blocks_keep_original_addresses_and_never_hash_as_complete_text() {
    use polylogue_model::{MessageRow, Output, Session, SessionKind, bundle::MessageProjection};
    let mut archive = WriterArchive::new();
    let original = source(
        "sparse",
        vec![
            Block::Text {
                text: "first".into(),
            },
            Block::Text {
                text: "middle".into(),
            },
            Block::Text {
                text: "last".into(),
            },
        ],
    );
    let output = |ordinals: &[u64]| {
        let mut selected = original.clone();
        selected.blocks = ordinals
            .iter()
            .map(|n| original.blocks[*n as usize].clone())
            .collect();
        let mut output = Output {
            sessions: vec![Session::new(original.session.clone(), SessionKind::Main)],
            messages: vec![MessageRow {
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 0,
                end: 1,
                message: selected,
            }],
            ..Default::default()
        };
        output.message_projections.insert(
            0,
            MessageProjection {
                source_version: original.content_hash().unwrap(),
                block_count: 3,
                block_ordinals: ordinals.to_vec(),
            },
        );
        output
    };
    archive.write_output(
        "first-sparse",
        "synthetic",
        "/synthetic/first",
        output(&[2]),
        1,
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        text(&reader, "block:claude-code:s-sparse#2").as_deref(),
        Some("last")
    );
    assert_eq!(text(&reader, "block:claude-code:s-sparse#3"), None);
    assert_eq!(
        text(&reader, "span:block:claude-code:s-sparse#2#c=0-4").as_deref(),
        Some("last")
    );
    for reference in [
        "block:claude-code:s-sparse#0",
        "message:claude-code:s-sparse",
        "session:claude-code:s",
        "message:claude-code:s-sparse..claude-code:s-sparse",
    ] {
        assert!(
            matches!(
                reader.reference_hash(&r(reference)),
                Err(StoreError::ReferenceUnavailable { .. })
            ),
            "{reference}"
        );
    }
    drop(reader);
    archive.write_output(
        "second-sparse",
        "synthetic",
        "/synthetic/second",
        output(&[0, 1]),
        1,
    );
    let reader = pool.reader().unwrap();
    assert_eq!(
        text(&reader, "message:claude-code:s-sparse").as_deref(),
        Some("first\n\nmiddle\n\nlast")
    );
    assert_eq!(
        reader.reference_hash(&r("session:claude-code:s")).unwrap(),
        Some(BlobHash::of(b"first\n\nmiddle\n\nlast"))
    );
}

#[test]
fn document_spans_share_original_body_offsets_without_loading_structured_results() {
    use polylogue_model::{Document, DocumentId, DocumentPurpose, DocumentRow, Output};
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let mut archive = WriterArchive::new();
    let mut document = Document::new(
        DocumentId::parse("claude-ai:document").unwrap(),
        DocumentPurpose::Context,
    );
    document.blocks = [
        serde_json::json!({"type":"text","text":"a💡"}),
        serde_json::json!({"type":"thinking","text":"ż"}),
        serde_json::json!({"type":"tool_result","output":"尾","outcome":{"kind":"ok"},"data":{"structured":"not source text"}}),
    ].into_iter().map(|value| serde_json::from_value(value).unwrap()).collect();
    archive.write_output(
        "document-source",
        "synthetic",
        "/synthetic/document",
        Output {
            documents: vec![DocumentRow {
                offset: 0,
                end: 1,
                document,
            }],
            ..Default::default()
        },
        1,
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| {
            if matches!(
                context.action,
                AuthAction::Read {
                    table_name: "document_block",
                    column_name: "data"
                }
            ) {
                Authorization::Deny
            } else {
                Authorization::Allow
            }
        }))
        .unwrap();
    assert_eq!(
        text(&reader, "document:claude-ai:document").as_deref(),
        Some("a💡\n\nż\n\n尾")
    );
    for reference in [
        "span:document:claude-ai:document#u16=1-6",
        "span:document:claude-ai:document#b=1-9",
    ] {
        assert_eq!(text(&reader, reference).as_deref(), Some("💡\n\nż"));
    }
    let mut emitted = false;
    assert!(matches!(
        reader.visit_reference_text(&r("span:document:claude-ai:document#u16=2-3"), |_| {
            emitted = true;
            Ok(())
        }),
        Err(StoreError::Invalid(_))
    ));
    assert!(!emitted);
}

#[test]
fn preparation_hashes_native_text_from_original_and_authored_text_from_private_overlay() {
    use polylogue_model::{
        Author,
        assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole},
    };
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![source(
            "m1",
            vec![Block::Text {
                text: "original native text".into(),
            }],
        )],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let scratch = rusqlite::Connection::open_in_memory().unwrap();
    scratch.execute_batch(crate::schema::ARCHIVE).unwrap();

    let fact = AssertionFact {
        id: "pending".into(),
        root: "pending".into(),
        kind: AssertionKind::Note,
        key: None,
        value: Some(serde_json::json!("pending exact authored text")),
        author: Author::Unattributed,
        via: polylogue_model::authorship::Channel::Import,
        machine: "synthetic-host".into(),
        created_at: Some(polylogue_model::Timestamp::from_unix_ms(1).unwrap()),
        valid_until: None,
        links: vec![AssertionLink {
            role: LinkRole::About,
            reference: r("message:claude-code:s-m1"),
            seen_hash: None,
            quote: None,
        }],
    };
    crate::assertions::apply(&scratch, &[fact], 0).unwrap();
    let native = r("message:claude-code:s-m1");
    let authored = r("assertion:pending@pending");
    let original = reader.conn().unwrap();
    assert_eq!(
        hash(original, &scratch, &native).unwrap(),
        reader.reference_hash(&native).unwrap()
    );
    assert_eq!(
        hash(original, &scratch, &authored).unwrap(),
        Some(BlobHash(
            *blake3::hash(b"pending exact authored text").as_bytes()
        ))
    );
    assert_eq!(
        hash(original, &scratch, &r("assertion:pending")).unwrap(),
        hash(original, &scratch, &authored).unwrap()
    );
    assert_eq!(
        hash(
            original,
            &scratch,
            &r("span:message:claude-code:s-m1#b=0-8")
        )
        .unwrap(),
        Some(BlobHash(*blake3::hash(b"original").as_bytes()))
    );
    assert_eq!(
        hash(
            original,
            &scratch,
            &r("span:assertion:pending@pending#b=0-7")
        )
        .unwrap(),
        Some(BlobHash(*blake3::hash(b"pending").as_bytes()))
    );
    assert!(
        scratch.prepare("SELECT 1 FROM message").is_err(),
        "native tables are absent from the authored overlay"
    );
}

#[test]
fn structured_assertions_remain_readable_with_unavailable_text_and_not_applicable_links() {
    use polylogue_model::{
        RevisionReference,
        assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink, values},
        user::UserWrite,
    };
    let archive = WriterArchive::new();
    archive
        .writer
        .user(crate::fixture::authored::operator_request(
            "structured".into(),
            UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: "on".into(),
                machine: None,
            },
        ))
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    {
        let reader = pool.reader().unwrap();
        let exact = RevisionReference {
            root: "structured".into(),
            revision: Some("structured".into()),
        };
        let record = reader.assertion(&exact).unwrap().unwrap();
        assert!(
            matches!(record.fact.validate().unwrap(),values::Value::Setting(value) if value.fields.value==polylogue_model::settings::Value::Boolean(true))
        );
        let reference = Reference::Assertion(exact);
        assert!(matches!(
            reader.reference_hash(&reference),
            Err(StoreError::ReferenceUnavailable { .. })
        ));
        let mut emitted = false;
        assert!(matches!(
            reader.visit_reference_text(&r("span:assertion:structured@structured#b=0-0"), |_| {
                emitted = true;
                Ok(())
            }),
            Err(StoreError::ReferenceUnavailable { .. })
        ));
        assert!(!emitted);
        let conn = reader.conn().unwrap();
        assert_eq!(
            link_hash(conn, conn, &reference).unwrap(),
            LinkHash::NotApplicable
        );
        assert_eq!(
            link_hash(conn, conn, &r("assertion:missing@missing")).unwrap(),
            LinkHash::Missing
        );
    }
    archive
        .writer
        .user(crate::fixture::authored::operator_request(
            "empty".into(),
            UserWrite::Assert {
                assertions: vec![NewAssertion {
                    kind: AssertionKind::Note,
                    key: None,
                    value: Some("".into()),
                    valid_until: None,
                    links: vec![NewAssertionLink {
                        role: LinkRole::About,
                        reference: r("assertion:structured@structured"),
                    }],
                }],
            },
        ))
        .unwrap();
    let reader = pool.reader().unwrap();
    let empty = reader
        .assertion(&RevisionReference {
            root: "empty".into(),
            revision: Some("empty".into()),
        })
        .unwrap()
        .unwrap();
    assert_eq!(empty.fact.links[0].seen_hash, None);
    assert_eq!(
        reader.reference_hash(&r("assertion:empty@empty")).unwrap(),
        Some(BlobHash::of(b""))
    );
}
