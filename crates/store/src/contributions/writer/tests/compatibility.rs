use super::*;
/// Stores a new input and parses it to its end. A non-empty context makes it a
/// Claude Code input.
fn add_input(f: &mut Fixture, raw: &[u8], context: &[&str]) -> String {
    let format = if context.is_empty() {
        InputFormat::CodexRollout
    } else {
        InputFormat::ClaudeCodeJsonl
    };
    let id = f.add_input(raw, format, context);
    f.advance_parser_to(f.head.acquired_upto);
    id
}
fn compatible_body(
    f: &mut Fixture,
    text: &[u8],
    parent: &[u8],
    time: i64,
    origin: &[u8],
) -> SourceProjection {
    compatible_body_sequence(f, text, parent, time, origin, 0)
}
fn compatible_body_sequence(
    f: &mut Fixture,
    text: &[u8],
    parent: &[u8],
    time: i64,
    origin: &[u8],
    sequence: u64,
) -> SourceProjection {
    let build = f.prepare_with(|i, _| i.sequence = sequence).unwrap().build;
    f.value(&build, text);
    f.writer
        .write_block(
            &build,
            0,
            0,
            BlockKind::Text,
            presence(true, !text.is_empty()),
            &step(0, b"block"),
            4096,
        )
        .unwrap();
    f.json(&build, ValuePurpose::Normalized, parent);
    f.json(&build, ValuePurpose::Origin, origin);
    f.writer
        .write_message_field(
            &build,
            &Slot::Message(MessageField::Parent),
            Some(time),
            true,
            &step(1, b"parent"),
            4096,
        )
        .unwrap();
    f.writer
        .add_message_source(
            &build,
            SourceRange {
                start: 0,
                end: f.head.parsed_upto,
            },
            &step(2, b"source"),
            4096,
        )
        .unwrap();
    let done = f
        .writer
        .finish_message_projection(
            &build,
            &step(3, b"done"),
            BlobHash::of(b"identical raw version evidence"),
            FinishState::Unknown,
            4096,
        )
        .unwrap();
    f.writer
        .publish_source_projection(&done, None, b"same source head", 4096)
        .unwrap()
}
fn compatible_ready(
    f: &mut Fixture,
    source: &SourceProjection,
) -> Result<Option<CompatibleSelection>> {
    let mut work = f.writer.begin_compatible_source_selection(source, 4096)?;
    let reader = f.writer.read_handle()?;
    for _ in 0..10000 {
        work = match work.advance(&reader, 4096, &std::sync::atomic::AtomicBool::new(false))? {
            CompatibilityTurn::Progress(w) => w,
            CompatibilityTurn::Certified(w, c) => {
                f.writer.certify_body_compatibility(c)?;
                w
            }
            CompatibilityTurn::Ready(w) => return Ok(Some(w)),
            CompatibilityTurn::Contested => return Ok(None),
        };
    }
    panic!("compatibility failed to make bounded progress")
}
#[test]
fn compatible_snapshots_share_one_body_in_any_order() {
    let mut outcomes = Vec::new();
    for order in [[0, 1], [1, 0]] {
        let mut f = Fixture::new();
        let mut ids = Vec::new();
        let mut sources = Vec::new();
        for i in order {
            let id = add_input(
                &mut f,
                if i == 0 { b"snapshot-a" } else { b"snapshot-b" },
                &[],
            );
            let origin = if i == 0 {
                br#"{"witness":"a"}"#.as_slice()
            } else {
                br#"{"witness":"b"}"#.as_slice()
            };
            let source = compatible_body(&mut f, &vec![b'x'; 131073], b"null", 1, origin);
            let proof = compatible_ready(&mut f, &source).unwrap().unwrap();
            f.writer
                .install_compatible_source_selection(proof, 4096)
                .unwrap();
            ids.push(id);
            sources.push((f.head.clone(), source));
        }
        let reader = f.writer.read_handle().unwrap();
        let view = reader.selected_message(&f.message).unwrap().unwrap();
        let FieldRead::Value(field) = view
            .field(&Slot::Message(MessageField::Parent), 4096)
            .unwrap()
            .unwrap()
        else {
            panic!("conflict")
        };
        let origin = field.origin_range(0, 4096).unwrap().unwrap().bytes;
        assert_eq!(
            view.block(BlockSlot::Item(0))
                .unwrap()
                .unwrap()
                .text_range(131072, 1)
                .unwrap()
                .unwrap()
                .bytes,
            b"x"
        );
        let mut cursor = None;
        let mut citations = Vec::new();
        let mut preferred = Vec::new();
        loop {
            match view.next_source(cursor.as_ref(), 4096).unwrap() {
                SourceStep::Continue { cursor: c } => cursor = Some(c),
                SourceStep::Span { cursor: c, source } => {
                    if source.preferred {
                        preferred.push(source.input.clone())
                    }
                    citations.push((source.input, source.chain, source.start, source.end));
                    cursor = Some(c)
                }
                SourceStep::End => break,
            }
        }
        assert_eq!(preferred, vec![ids.iter().min().unwrap().clone()]);
        assert_eq!(citations.len(), 2);
        assert!(
            citations
                .iter()
                .all(|(_, chain, start, end)| (*chain, *start, *end) == (1, 0, 10))
        );
        for (head, source) in sources {
            let snapshot = f.writer.parser_snapshot(&head.token).unwrap();
            let own = snapshot
                .selected_source_projection(&[3; 32], 4096)
                .unwrap()
                .unwrap();
            assert_eq!(own.body(), source.body());
            assert_eq!(own.evidence_guard(), source.evidence_guard());
        }
        citations.sort();
        outcomes.push((origin, citations, preferred));
    }
    assert_eq!(outcomes[0], outcomes[1]);
}
#[test]
fn equal_evidence_hash_does_not_hide_content_differences() {
    for variant in 0..5 {
        let mut f = Fixture::new();
        add_input(&mut f, b"first snapshot", &[]);
        let first = compatible_body(&mut f, b"same text", b"null", 1, b"{}");
        let proof = compatible_ready(&mut f, &first).unwrap().unwrap();
        f.writer
            .install_compatible_source_selection(proof, 4096)
            .unwrap();
        add_input(
            &mut f,
            b"second snapshot",
            if variant == 3 {
                &["claude-main", "context-session"]
            } else {
                &[]
            },
        );
        let second = compatible_body_sequence(
            &mut f,
            if variant == 0 {
                b"different text"
            } else {
                b"same text"
            },
            if variant == 1 {
                b"\"other-parent\""
            } else {
                b"null"
            },
            if variant == 2 { 2 } else { 1 },
            b"{}",
            if variant == 4 { 1 } else { 0 },
        );
        assert_eq!(first.version_hash(), second.version_hash());
        assert!(
            compatible_ready(&mut f, &second).unwrap().is_none(),
            "variant {variant}"
        );
        assert_eq!(
            f.scalar("SELECT selected_body_pk FROM message"),
            first.body().pk
        );
    }
}
#[test]
fn compatible_selection_detects_changes_and_cancellation() {
    let mut f = Fixture::new();
    add_input(&mut f, b"snapshot-one", &[]);
    let source = compatible_body(&mut f, b"text", b"null", 1, b"{}");
    let reader = f.writer.read_handle().unwrap();
    let work = f
        .writer
        .begin_compatible_source_selection(&source, 4096)
        .unwrap();
    assert!(matches!(
        work.advance(&reader, 4096, &std::sync::atomic::AtomicBool::new(true)),
        Err(Error::Cancelled)
    ));
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_body_compatibility"),
        0
    );
    add_input(&mut f, b"snapshot-two", &[]);
    let other = compatible_body(&mut f, b"text", b"null", 1, b"{}");
    let work = f
        .writer
        .begin_compatible_source_selection(&other, 4096)
        .unwrap();
    assert!(matches!(
        work.advance(&reader, 0, &std::sync::atomic::AtomicBool::new(false)),
        Err(Error::ParserAdmission { .. })
    ));
    let proof = compatible_ready(&mut f, &source).unwrap().unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE message_candidate SET eligible=0 WHERE pk=?1",
            [source.candidate],
        )
        .unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE message_candidate SET eligible=1 WHERE pk=?1",
            [source.candidate],
        )
        .unwrap();
    assert!(matches!(
        f.writer.install_compatible_source_selection(proof, 4096),
        Err(Error::StaleMessage)
    ));
    let proof = compatible_ready(&mut f, &source).unwrap().unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE message SET removal_epoch=removal_epoch+1,hidden=1",
            [],
        )
        .unwrap();
    assert!(
        f.writer
            .install_compatible_source_selection(proof, 4096)
            .is_err()
    );
    assert!(matches!(
        f.writer.begin_compatible_source_selection(&source, 4096),
        Err(Error::Invariant(_))
    ));
    // A cancelled or stale attempt selects nothing.
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    let _ = std::mem::size_of::<CompatibilityTurn>();
}
#[test]
fn reparse_withdraws_compatibility_and_retirement_drops_it() {
    let mut f = Fixture::new();
    add_input(&mut f, b"reparse snapshot", &[]);
    let old = compatible_body(&mut f, b"old interpretation", b"null", 1, b"{}");
    let proof = compatible_ready(&mut f, &old).unwrap().unwrap();
    f.writer
        .install_compatible_source_selection(proof, 4096)
        .unwrap();
    let old_head = f.head.clone();
    add_input(&mut f, b"second reparse snapshot", &[]);
    let old_other = compatible_body(&mut f, b"old interpretation", b"null", 1, b"{}");
    let old_other_head = f.head.clone();
    let both = compatible_ready(&mut f, &old_other).unwrap().unwrap();
    f.writer
        .install_compatible_source_selection(both, 4096)
        .unwrap();
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_body_compatibility"),
        2
    );
    let stale = compatible_ready(&mut f, &old).unwrap().unwrap();
    f.head = old_head;
    let reader = f.writer.read_handle().unwrap();
    let pinned = reader.selected_message(&f.message).unwrap().unwrap();
    f.restart_parser(4, 8);
    f.advance_parser_to(f.head.acquired_upto);
    let new = compatible_body(&mut f, b"new interpretation", b"null", 1, b"{}");
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_candidate WHERE eligible=1"),
        2
    );
    assert!(
        f.writer
            .install_compatible_source_selection(stale, 4096)
            .is_err()
    );
    // Until the other input is reparsed too, the two bodies differ and the
    // selection stays contested.
    assert!(compatible_ready(&mut f, &new).unwrap().is_none());
    f.head = old_other_head;
    f.restart_parser(4, 8);
    f.advance_parser_to(f.head.acquired_upto);
    let other_new = compatible_body(&mut f, b"new interpretation", b"null", 1, b"{}");
    let proof = compatible_ready(&mut f, &other_new).unwrap().unwrap();
    f.writer
        .install_compatible_source_selection(proof, 4096)
        .unwrap();
    assert_eq!(
        pinned
            .block(polylogue_model::BlockSlot::Item(0))
            .unwrap()
            .unwrap()
            .text_range(0, 4096)
            .unwrap()
            .unwrap()
            .bytes,
        b"old interpretation"
    );
    drop(pinned);
    drop(reader);
    assert_eq!(
        f.writer.retire_message_body(old.body()).unwrap(),
        BodyRetirement::DetachedCandidate
    );
    for _ in 0..10 {
        if f.writer.retire_message_body(old.body()).unwrap() == BodyRetirement::Missing {
            break;
        }
    }
    assert_eq!(
        f.scalar(&format!(
            "SELECT count(*) FROM message_body_compatibility WHERE body_pk={}",
            old.body().pk
        )),
        0
    );
    assert!(
        f.writer
            .current_source_projection(&old.token, &[3; 32], 4096)
            .is_err()
    );
}
