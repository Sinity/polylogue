//! Canonical message associations survive confirmation before their target arrives.
use super::*;
use polylogue_model::{
    Event, Link, LinkKind, Message, MessageId, MessageRow, Output, RecordCounts, RecordOutcome,
    Session, SessionId, SessionKind,
};

fn admit_step(
    index: &mut Connection,
    resolver: &Arc<crate::writer::resolve::Resolver>,
    input: &str,
    position: &mut i64,
    mut step: crate::ParseStep,
) {
    step.reading_version = "association-v1".into();
    step.prepare();
    let key = crate::InputKey {
        id: input.into(),
        format: "synthetic".into(),
        route: polylogue_model::Route::Live,
    };
    for original in crate::primary::products::parse(&key, 1, None, &step).unwrap() {
        // Use the canonical product's actual serialization boundary before the
        // primary plan builds its disk-backed reading and resolves row choices.
        let product: Product =
            serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
        assert_eq!(product, original);
        *position += 1;
        apply(
            index,
            resolver,
            &[
                Confirmed {
                    native: None,
                    position: *position,
                    output_index: 0,
                    event: format!("{input}-fact-{position}"),
                    change: ConfirmedChange::Live(Box::new(product)),
                },
                barrier(*position),
            ],
        )
        .unwrap();
    }
}

fn step(output: Output, from: u64) -> crate::ParseStep {
    let mut counts = RecordCounts {
        records: 1,
        ..Default::default()
    };
    counts.add("synthetic", &RecordOutcome::Used);
    crate::ParseStep::new(output, counts, from..from + 1, true)
}

#[test]
fn confirmed_forward_message_associations_survive_weaker_donors_and_target_arrival() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) =
        crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes")).unwrap();
    let mut index = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &crate::text_rows::TextIds::new(&index, false).unwrap(),
        &mut index,
        &resolver,
        &[
            input(1, "parent-source"),
            input(2, "child-source"),
            barrier(2),
        ],
    )
    .unwrap();
    let parent = SessionId::parse("claude-code:parent").unwrap();
    let child = SessionId::parse("claude-code:child").unwrap();
    let anchor = MessageId::parse("claude-code:anchor").unwrap();
    let event = Event {
        session: Some(parent.clone()),
        kind: "marker".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: Some(anchor.clone()),
        range: Some((0, 1)),
        data: serde_json::json!({"synthetic":true}),
    };
    let link = Link {
        kind: LinkKind::Spawn,
        copies_prefix: false,
        how: None,
        parent: parent.clone(),
        child: child.clone(),
        at: Some(anchor.clone()),
        call_id: Some("spawn-call".into()),
        evidence: "explicit parent progress".into(),
    };
    let mut position = 2;
    admit_step(
        &mut index,
        &resolver,
        "parent-source",
        &mut position,
        step(
            Output {
                sessions: vec![
                    Session::new(parent.clone(), SessionKind::Main),
                    Session::new(child.clone(), SessionKind::Subagent),
                ],
                events: vec![event.clone()],
                links: vec![link.clone()],
                ..Default::default()
            },
            0,
        ),
    );
    let verify = || {
        let pool = crate::ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        let exported = reader.session_export(&parent).unwrap().unwrap();
        assert_eq!(exported.events, vec![event.clone()]);
        assert_eq!(exported.links, vec![link.clone()]);
    };
    verify();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    // A child's metadata can name the call but omit its parent's message.
    let mut weaker = link.clone();
    weaker.at = None;
    weaker.evidence = "child metadata".into();
    admit_step(
        &mut index,
        &resolver,
        "child-source",
        &mut position,
        step(
            Output {
                links: vec![weaker],
                ..Default::default()
            },
            0,
        ),
    );
    // Evidence wording is independently selected; canonical associations must remain exact.
    let verify_associations = || {
        let pool = crate::ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        let exported = reader.session_export(&parent).unwrap().unwrap();
        assert_eq!(exported.events[0].after, Some(anchor.clone()));
        assert_eq!(exported.links[0].at, Some(anchor.clone()));
    };
    verify_associations();
    let message: Message = serde_json::from_value(serde_json::json!({
        "id":anchor,"session":parent,"role":"assistant","blocks":[{"type":"text","text":"synthetic target"}],"usage":{}
    })).unwrap();
    admit_step(
        &mut index,
        &resolver,
        "parent-source",
        &mut position,
        step(
            Output {
                messages: vec![MessageRow {
                    page_text: false,
                    id_kind: polylogue_model::IdKind::Native,
                    offset: 1,
                    end: 2,
                    message,
                }],
                ..Default::default()
            },
            1,
        ),
    );
    verify_associations();
    let stored: (Option<String>, Option<i64>) = index
        .query_row(
            "SELECT after_message_id,after_message_pk FROM event",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(stored.0, Some(anchor.to_string()));
    assert!(
        stored.1.is_some(),
        "target arrival resolves the cached pointer without changing its canonical reference"
    );
}

#[test]
fn confirmed_explicit_copied_prefix_members_wait_for_message_and_keep_original_coordinates() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) =
        crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes")).unwrap();
    let mut index = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &crate::text_rows::TextIds::new(&index, false).unwrap(),
        &mut index,
        &resolver,
        &[
            input(1, "copy-source"),
            input(2, "parent-source"),
            barrier(2),
        ],
    )
    .unwrap();
    let parent = SessionId::parse("claude-code:parent").unwrap();
    let child = SessionId::parse("claude-code:child").unwrap();
    let copied = crate::fixture::writer::message(
        "copied",
        "parent",
        polylogue_model::Role::User,
        0,
        vec![polylogue_model::Block::Text {
            text: "copied".into(),
        }],
    );
    let own = crate::fixture::writer::message(
        "own",
        "child",
        polylogue_model::Role::User,
        0,
        vec![polylogue_model::Block::Text { text: "own".into() }],
    );
    let mut position = 2;
    let row = |message| MessageRow {
        page_text: false,
        id_kind: polylogue_model::IdKind::Native,
        offset: 0,
        end: 1,
        message,
    };
    admit_step(
        &mut index,
        &resolver,
        "copy-source",
        &mut position,
        step(
            Output {
                sessions: vec![
                    Session::new(parent.clone(), SessionKind::Main),
                    Session::new(child.clone(), SessionKind::Main),
                ],
                messages: vec![row(own)],
                prefix_members: vec![polylogue_model::PrefixMember {
                    parent: parent.clone(),
                    child: child.clone(),
                    message: copied.id.clone(),
                    source: Some(polylogue_model::PrefixSource {
                        input: "original-source".into(),
                        chain: 3,
                        reading_version: "original-reading".into(),
                    }),
                    position: [u64::MAX, 7],
                }],
                ..Default::default()
            },
            0,
        ),
    );
    {
        let pool = crate::ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        assert!(
            matches!(reader.select_messages(&child,true).unwrap().parents().issues.as_slice(), [polylogue_model::read::ParentIssue::MissingCopiedMessage {message,..}] if message==&copied.id)
        );
    }
    let uncopied = crate::fixture::writer::message(
        "uncopied",
        "parent",
        polylogue_model::Role::User,
        0,
        vec![polylogue_model::Block::Text {
            text: "uncopied".into(),
        }],
    );
    admit_step(
        &mut index,
        &resolver,
        "parent-source",
        &mut position,
        step(
            Output {
                sessions: vec![Session::new(parent, SessionKind::Main)],
                messages: vec![row(copied.clone()), row(uncopied)],
                ..Default::default()
            },
            0,
        ),
    );
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader.select_messages(&child, true).unwrap();
    selection.require_complete().unwrap();
    let mut texts = Vec::new();
    selection
        .visit(|message| {
            texts.push(message.text());
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(texts, ["copied", "own"]);
    assert!(reader.links(&child).unwrap().is_empty());
    let copies = reader.copied_for(&copied.id).unwrap();
    assert_eq!(copies.len(), 1);
    assert_eq!(copies[0].source.input, "original-source");
    assert_eq!(copies[0].position, [u64::MAX, 7]);
}
