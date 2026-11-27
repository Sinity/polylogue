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
        assert!(selected.committed().is_some());
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
        Err(StoreError::Refused(polylogue_model::Reason::BadSpan { .. }))
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
            .visit_reference_text(&reference("message:claude-code:s-stop"), |_| {
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
    use polylogue_model::{MessageRow, Output, Session, SessionKind, bundle::ShownBlocks};
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
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 0,
            end: 1,
            message: selected,
        }],
        ..Default::default()
    };
    output.block_selections.insert(
        0,
        ShownBlocks {
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
fn replay_seeks_by_field_past_a_bad_earlier_body() {
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
            SourceTextPosition {
                piece: SourceTextPiece::Body {
                    body: SourceTextBody {
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
    crate::content::poison_bodies(
        &conn,
        "block",
        "message_pk=(SELECT pk FROM message WHERE id='claude-code:s-resume') AND position=0",
        [],
    )
    .unwrap();
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
            SourceTextPiece::Separator { .. }
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
    crate::content::poison_bodies(
        &conn,
        "block",
        "message_pk=(SELECT pk FROM message WHERE id='claude-code:s-separator')",
        [],
    )
    .unwrap();
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
fn document_replay_seeks_the_source_block_ordinal() {
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
            SourceTextPosition {
                piece: SourceTextPiece::Body {
                    body: SourceTextBody {
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
    conn.execute("UPDATE document_block SET body=?1 WHERE document_pk=(SELECT pk FROM document WHERE id='claude-ai:resume') AND position=0", [vec![255u8]]).unwrap();
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

#[test]
fn tool_span_replays_the_original_result_block() {
    use crate::fixture::writer::call;
    use polylogue_model::{
        bundle::{TextFragment, TextFragmentSource},
        query_objects::NormalizedReadSpan,
    };

    let mut archive = WriterArchive::new();
    let mut messages = call(
        "s",
        1,
        1,
        "synthetic",
        None,
        serde_json::json!({}),
        Ok("a💡z"),
    );
    messages[1].blocks.insert(
        0,
        Block::Text {
            text: "separate body".into(),
        },
    );
    let result = messages[1].id.clone();
    let hash = messages[1].content_hash().unwrap();
    archive.write("s", messages);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let requested = reference("span:tool:claude-code:s#t1#u16=1-3");
    let text_of = |source: &SpanSource<'_, '_>| {
        let mut selected = String::new();
        source
            .visit_text(|piece| {
                selected.push_str(piece);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        selected
    };
    let bound = {
        let reader = pool.reader().unwrap();
        let source = reader.select_span_source(&requested).unwrap().unwrap();
        assert_eq!(
            source.origin(),
            &TextSpanOrigin {
                reference: reference("span:block:claude-code:s-r1#1#u16=1-3"),
                start: 1,
                end: 5,
            }
        );
        assert_eq!(
            source.version(),
            &TextSourceVersion::ContentHash {
                record: Reference::Message(result),
                hash,
            }
        );
        assert_eq!(text_of(&source), "💡");
        let mut reference_text = String::new();
        assert_eq!(
            reader
                .visit_reference_text(&requested, |piece| {
                    reference_text.push_str(piece);
                    Ok(ControlFlow::Continue(()))
                })
                .unwrap(),
            Some(ControlFlow::Continue(4))
        );
        assert_eq!(reference_text, "💡");

        let fragment = TextFragment {
            container: source.container().clone(),
            source_version: source.version().clone(),
            coverage: source.coverage(),
            text: text_of(&source),
            text_hash: BlobHash::of("💡".as_bytes()),
            provenance: TextFragmentSource {
                spans: vec![source.origin().clone()],
                sources: vec![],
                imports: vec![],
            },
        };
        assert_eq!(
            fragment.reference(),
            reference("span:block:claude-code:s-r1#1#b=1-5")
        );
        fragment.validate().unwrap();
        NormalizedReadSpan::new(
            source.origin().clone(),
            source.version().clone(),
            source.seek(),
        )
        .unwrap()
    };

    let conn =
        crate::archive::open_write(&archive.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute(
        "UPDATE tool_call SET result_message_id=NULL, result_position=NULL WHERE call_id='t1'",
        [],
    )
    .unwrap();
    let reader = pool.reader().unwrap();
    assert!(matches!(
        reader.select_span_source(&requested),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    assert!(matches!(
        reader.visit_reference_text(&requested, |_| Ok(ControlFlow::Continue(()))),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    let replay = reader
        .resume_span_source(&bound.origin, &bound.version, bound.seek)
        .unwrap();
    assert_eq!(text_of(&replay), "💡");
}

#[test]
fn tool_span_refuses_ambiguous_or_unavailable_output() {
    use crate::fixture::writer::call;

    let mut archive = WriterArchive::new();
    let mut messages = call(
        "s",
        1,
        1,
        "synthetic",
        None,
        serde_json::json!({}),
        Ok("ignored"),
    );
    let Block::ToolResult { output, .. } = &mut messages[1].blocks[0] else {
        panic!("result fixture");
    };
    *output = None;
    messages.extend(call(
        "s",
        2,
        3,
        "synthetic",
        None,
        serde_json::json!({}),
        Ok("first"),
    ));
    messages.extend(call(
        "s",
        3,
        5,
        "synthetic",
        None,
        serde_json::json!({}),
        Ok("second"),
    ));
    archive.write("s", messages);
    let conn =
        crate::archive::open_write(&archive.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute("UPDATE tool_call SET call_id='t2' WHERE call_id='t3'", [])
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let missing = reference("span:tool:claude-code:s#missing#b=0-0");
    assert!(reader.select_span_source(&missing).unwrap().is_none());
    let without_text = reference("span:tool:claude-code:s#t1#b=0-0");
    assert!(matches!(
        reader.select_span_source(&without_text),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    let ambiguous = reference("span:tool:claude-code:s#t2#b=0-0");
    assert!(matches!(
        reader.select_span_source(&ambiguous),
        Err(StoreError::Ambiguous { reference, candidates })
            if reference == "tool:claude-code:s#t2"
                && candidates == ["message:claude-code:s-c2", "message:claude-code:s-c3"]
    ));
}
