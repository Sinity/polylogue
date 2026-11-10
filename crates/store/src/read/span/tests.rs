use super::*;
use crate::fixture::writer::{WriterArchive, message};
use polylogue_model::{Block, Role};

fn reference(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
#[test]
fn selected_prefix_is_portable_without_reading_or_hashing_unknown_suffix() {
    let mut archive = WriterArchive::new();
    let original = message(
        "prefix",
        "s",
        Role::Assistant,
        1,
        vec![
            Block::Text {
                text: "a💡e\u{301}".into(),
            },
            Block::Thinking {
                text: None,
                signature: Some("unavailable".into()),
                redacted: true,
            },
            Block::Text {
                text: "private outside selection".into(),
            },
        ],
    );
    let expected_version = original.content_hash().unwrap();
    archive.write("s", vec![original]);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    for coordinate in ["b=1-5", "c=1-2", "u16=1-3"] {
        let span = reference(&format!("span:message:claude-code:s-prefix#{coordinate}"));
        let selected = reader.select_span_source(&span).unwrap().unwrap();
        assert_eq!(
            selected.coverage(),
            TextCoverage {
                start: 1,
                end: 5,
                total_bytes: None
            }
        );
        assert_eq!(
            selected.version(),
            &TextSourceVersion::ContentHash {
                record: reference("message:claude-code:s-prefix"),
                hash: expected_version
            }
        );
        assert!(selected.admitted().is_some());
        let mut text = String::new();
        assert!(
            selected
                .visit_text(|piece| {
                    text.push_str(piece);
                    Ok(ControlFlow::Continue(()))
                })
                .unwrap()
                .is_continue()
        );
        assert_eq!(text, "💡");
    }
    assert!(matches!(
        reader.reference_hash(&reference("message:claude-code:s-prefix")),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    assert!(matches!(
        reader.select_span_source(&reference("span:message:claude-code:s-prefix#c=0-20")),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    assert!(matches!(
        reader.select_span_source(&reference("span:message:claude-code:s-prefix#u16=1-2")),
        Err(StoreError::Invalid(_))
    ));
}
#[test]
fn source_cancellation_stops_before_unavailable_next_body() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![message(
            "stop",
            "s",
            Role::User,
            1,
            vec![
                Block::Text {
                    text: "first".into(),
                },
                Block::Thinking {
                    text: None,
                    signature: None,
                    redacted: true,
                },
            ],
        )],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut visits = 0;
    assert_eq!(
        reader
            .visit_reference_text_while(&reference("message:claude-code:s-stop"), |_| {
                visits += 1;
                Ok(ControlFlow::Break(()))
            })
            .unwrap(),
        Some(ControlFlow::Break(5))
    );
    assert_eq!(visits, 1);
}
#[test]
fn accessible_empty_absent_and_unavailable_text_remain_distinct() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![
            message("empty", "s", Role::User, 1, vec![]),
            message(
                "unknown",
                "s",
                Role::User,
                2,
                vec![Block::Thinking {
                    text: None,
                    signature: None,
                    redacted: true,
                }],
            ),
        ],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let empty = reader
        .select_span_source(&reference("span:message:claude-code:s-empty#b=0-0"))
        .unwrap()
        .unwrap();
    assert_eq!(empty.coverage().total_bytes, Some(0));
    let mut emitted = false;
    let flow = empty
        .visit_text(|_| {
            emitted = true;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert!(!emitted);
    assert!(
        reader
            .select_span_source(&reference("span:message:claude-code:missing#b=0-0"))
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        reader.select_span_source(&reference("span:message:claude-code:s-unknown#b=0-0")),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
}

#[test]
fn sparse_known_prefix_preserves_version_and_never_bridges_unknown_blocks() {
    use polylogue_model::{MessageRow, Output, Session, SessionKind, bundle::MessageProjection};
    let mut archive = WriterArchive::new();
    let original = message(
        "sparse",
        "s",
        Role::User,
        1,
        vec![
            Block::Text {
                text: "prefix".into(),
            },
            Block::Text {
                text: "missing".into(),
            },
            Block::Text {
                text: "tail".into(),
            },
        ],
    );
    let mut selected = original.clone();
    selected.blocks = vec![original.blocks[0].clone(), original.blocks[2].clone()];
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
            block_ordinals: vec![0, 2],
        },
    );
    archive.write_output("sparse", "synthetic", "/synthetic/sparse", output, 1);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let source = reader
        .select_span_source(&reference("span:message:claude-code:s-sparse#b=0-6"))
        .unwrap()
        .unwrap();
    let mut text = String::new();
    let flow = source
        .visit_text(|piece| {
            text.push_str(piece);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(text, "prefix");
    assert_eq!(source.coverage().total_bytes, None);
    assert!(matches!(
        reader.select_span_source(&reference("span:message:claude-code:s-sparse#b=6-8")),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    assert!(
        reader
            .reference_hash(&reference("message:claude-code:s-sparse"))
            .is_err()
    );
    let tail = reader
        .select_span_source(&reference("span:block:claude-code:s-sparse#2#b=0-4"))
        .unwrap()
        .unwrap();
    let mut text = String::new();
    let flow = tail
        .visit_text(|piece| {
            text.push_str(piece);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(text, "tail");
}

#[test]
fn replay_uses_native_field_seek_without_hydrating_a_poisoned_earlier_body() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![message(
            "resume",
            "s",
            Role::User,
            1,
            vec![
                Block::Text {
                    text: "earlier".into(),
                },
                Block::Text {
                    text: "a💡target".into(),
                },
            ],
        )],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let original = reference("span:message:claude-code:s-resume#u16=10-12");
    let (origin, version, seek) = {
        let reader = pool.reader().unwrap();
        let source = reader.select_span_source(&original).unwrap().unwrap();
        assert_eq!(
            source.seek().start,
            NativeTextPosition {
                piece: NativeTextPiece::Body {
                    body: NativeTextBody {
                        block: 1,
                        field: TextField::Text
                    }
                },
                byte: 1
            }
        );
        (
            TextSpanOrigin {
                reference: original.clone(),
                start: source.coverage().start,
                end: source.coverage().end,
            },
            source.version().clone(),
            source.seek(),
        )
    };
    let conn =
        crate::archive::open_write(&archive.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute("UPDATE block SET body=?1 WHERE message_pk=(SELECT pk FROM message WHERE id='claude-code:s-resume') AND idx=0", [vec![255u8]]).unwrap();
    drop(conn);
    let reader = pool.reader().unwrap();
    assert!(reader.select_span_source(&original).is_err());
    let source = reader.resume_span_source(&origin, &version, seek).unwrap();
    let mut selected = String::new();
    let flow = source
        .visit_text(|piece| {
            selected.push_str(piece);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(selected, "💡");
    assert_eq!(source.version(), &version);
    let wrong = TextSourceVersion::ContentHash {
        record: reference("message:claude-code:s-resume"),
        hash: BlobHash([7; 32]),
    };
    assert!(matches!(
        reader.resume_span_source(&origin, &wrong, seek),
        Err(StoreError::Stale { .. })
    ));
    archive.archive.read_scope.begin_reset().unwrap();
    assert!(matches!(
        source.visit_text(|_| Ok(ControlFlow::Continue(()))),
        Err(StoreError::ScopeChanged)
    ));
    archive.archive.read_scope.finish_reset();
}
#[test]
fn a_recorded_separator_can_resume_without_decoding_either_unselected_body() {
    let mut archive = WriterArchive::new();
    archive.write(
        "s",
        vec![message(
            "separator",
            "s",
            Role::User,
            1,
            vec![
                Block::Text {
                    text: "before".into(),
                },
                Block::Text {
                    text: "after".into(),
                },
            ],
        )],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let (origin, version, seek) = {
        let reader = pool.reader().unwrap();
        let original = reference("span:message:claude-code:s-separator#b=6-7");
        let source = reader.select_span_source(&original).unwrap().unwrap();
        assert!(matches!(
            source.seek().start.piece,
            NativeTextPiece::Separator { .. }
        ));
        (
            TextSpanOrigin {
                reference: original,
                start: 6,
                end: 7,
            },
            source.version().clone(),
            source.seek(),
        )
    };
    let conn =
        crate::archive::open_write(&archive.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute("UPDATE block SET body=?1 WHERE message_pk=(SELECT pk FROM message WHERE id='claude-code:s-separator')", [vec![255u8]]).unwrap();
    drop(conn);
    let reader = pool.reader().unwrap();
    let source = reader.resume_span_source(&origin, &version, seek).unwrap();
    let mut selected = String::new();
    let flow = source
        .visit_text(|piece| {
            selected.push_str(piece);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(selected, "\n");
}

#[test]
fn document_replay_seeks_the_canonical_document_block_ordinal() {
    use polylogue_model::{Document, DocumentId, DocumentPurpose, DocumentRow, Output};
    let mut archive = WriterArchive::new();
    let mut document = Document::new(
        DocumentId::parse("claude-ai:resume").unwrap(),
        DocumentPurpose::Context,
    );
    document.blocks = vec![
        Block::Text {
            text: "earlier".into(),
        },
        Block::Text {
            text: "selected".into(),
        },
    ];
    archive.write_output(
        "document",
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
    let (origin, version, seek) = {
        let reader = pool.reader().unwrap();
        let original = reference("span:document:claude-ai:resume#b=9-17");
        let source = reader.select_span_source(&original).unwrap().unwrap();
        assert_eq!(
            source.seek().start,
            NativeTextPosition {
                piece: NativeTextPiece::Body {
                    body: NativeTextBody {
                        block: 1,
                        field: TextField::Text
                    }
                },
                byte: 0
            }
        );
        (
            TextSpanOrigin {
                reference: original,
                start: 9,
                end: 17,
            },
            source.version().clone(),
            source.seek(),
        )
    };
    let conn =
        crate::archive::open_write(&archive.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute("UPDATE document_block SET body=?1 WHERE document_pk=(SELECT pk FROM document WHERE id='claude-ai:resume') AND idx=0", [vec![255u8]]).unwrap();
    drop(conn);
    let reader = pool.reader().unwrap();
    let source = reader.resume_span_source(&origin, &version, seek).unwrap();
    let mut selected = String::new();
    let flow = source
        .visit_text(|piece| {
            selected.push_str(piece);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(selected, "selected");
}
