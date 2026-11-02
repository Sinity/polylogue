use super::*;
use crate::fixture::{MessageFacts, SessionFacts};
use std::ops::ControlFlow;

fn session(f: &mut Fixture, name: &str, texts: &[&str]) {
    let id = format!("claude-code:{name}");
    let pk = f
        .session(SessionFacts {
            id: &id,
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for (i, text) in texts.iter().enumerate() {
        f.message(
            pk,
            MessageFacts {
                id: &format!("claude-code:{name}-{i}"),
                role: "user",
                written_by: "human",
                text: Some(text),
                ..Default::default()
            },
        )
        .unwrap();
    }
}

fn fork(f: &mut Fixture, parent: &str, child: &str, anchor: Option<&str>) {
    let parent = format!("claude-code:{parent}");
    let child = format!("claude-code:{child}");
    f.link(&parent, &child, "continuation", "fork").unwrap();
    crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute(
            "UPDATE link SET branch_point=?3 WHERE parent_id=?1 AND child_id=?2",
            params![parent, child, anchor],
        )
        .unwrap();
}

fn recorded_messages(
    writer: &mut crate::fixture::writer::WriterArchive,
    session: &str,
    bodies: &[&str],
    copied: &[polylogue_model::Message],
) -> Vec<polylogue_model::Message> {
    let messages: Vec<_> = bodies
        .iter()
        .enumerate()
        .map(|(ordinal, body)| {
            crate::fixture::writer::message(
                &ordinal.to_string(),
                session,
                polylogue_model::Role::User,
                0,
                vec![polylogue_model::Block::Text {
                    text: (*body).into(),
                }],
            )
        })
        .collect();
    writer.write(session, copied.iter().chain(&messages).cloned().collect());
    messages
}

fn recorded_link(
    kind: polylogue_model::LinkKind,
    parent: &str,
    child: &str,
) -> polylogue_model::Link {
    polylogue_model::Link {
        kind,
        parent: crate::fixture::writer::sid(parent),
        child: crate::fixture::writer::sid(child),
        at: None,
        call_id: None,
        evidence: "recorded source relation".into(),
    }
}

fn copied_source(
    writer: &mut crate::fixture::writer::WriterArchive,
    child: &str,
    parents: &[&str],
    messages: Vec<polylogue_model::Message>,
) {
    writer.links.extend(
        parents
            .iter()
            .map(|parent| recorded_link(polylogue_model::LinkKind::Copy, parent, child)),
    );
    writer.write_from(
        &format!("copied-{child}"),
        polylogue_model::Session::new(
            crate::fixture::writer::sid(child),
            polylogue_model::SessionKind::Main,
        ),
        messages,
    );
}

fn texts(selected: &MessageSelection<'_, '_>) -> Vec<String> {
    let mut out = Vec::new();
    selected
        .visit(|m| {
            out.push(m.text());
            ControlFlow::Continue(())
        })
        .unwrap();
    out
}

#[test]
fn a_1030_session_chain_has_no_depth_or_call_stack_cutoff() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    for i in 0..1030 {
        session(&mut f, &format!("s{i}"), &[&format!("message {i}")]);
        if i > 0 {
            fork(
                &mut f,
                &format!("s{}", i - 1),
                &format!("s{i}"),
                Some(&format!("claude-code:s{}-0", i - 1)),
            );
        }
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let id = SessionId::parse("claude-code:s1029").unwrap();
    let selected = r.select_messages(&id, true).unwrap();
    assert!(selected.parents().complete);
    assert_eq!(
        texts(&selected),
        (0..1030)
            .map(|i| format!("message {i}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(r.message_keys_with_parents(&id).unwrap().len(), 1030);
}

#[test]
fn an_anchor_in_the_parents_inherited_prefix_excludes_its_own_tail() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut f, "a", &["grandparent first", "grandparent later"]);
    session(&mut f, "b", &["parent tail"]);
    session(&mut f, "c", &["child tail"]);
    fork(&mut f, "a", "b", Some("claude-code:a-1"));
    fork(&mut f, "b", "c", Some("claude-code:a-0"));
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let selected = r
        .select_messages(&SessionId::parse("claude-code:c").unwrap(), true)
        .unwrap();
    assert!(selected.parents().complete);
    assert_eq!(texts(&selected), ["grandparent first", "child tail"]);
}

#[test]
fn unknown_or_absent_fork_anchors_never_substitute_the_full_parent() {
    for anchor in [None, Some("claude-code:absent")] {
        let dir = tempfile::tempdir().unwrap();
        let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
        session(&mut f, "a", &["unproven parent content"]);
        session(&mut f, "b", &["available child"]);
        fork(&mut f, "a", "b", anchor);
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        let id = SessionId::parse("claude-code:b").unwrap();
        let selected = r.select_messages(&id, true).unwrap();
        assert!(!selected.parents().complete);
        assert!(
            matches!(&selected.parents().issues[..], [ParentIssue::MissingAnchor { anchor: got, .. }] if got.as_deref() == anchor)
        );
        assert_eq!(texts(&selected), ["available child"]);
        assert!(matches!(
            r.messages(&id, true),
            Err(StoreError::IncompleteParents(_))
        ));
        assert!(matches!(
            r.message_keys_with_parents(&id),
            Err(StoreError::IncompleteParents(_))
        ));
        assert_eq!(r.messages(&id, false).unwrap()[0].text(), "available child");
    }
}

#[test]
fn missing_parents_and_session_cycles_report_their_actual_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut f, "a", &["a"]);
    session(&mut f, "b", &["b"]);
    session(&mut f, "orphan", &["orphan"]);
    fork(&mut f, "a", "b", Some("claude-code:a-0"));
    fork(&mut f, "b", "a", Some("claude-code:b-0"));
    fork(&mut f, "missing", "orphan", Some("claude-code:missing-0"));
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let cycle = r
        .select_messages(&SessionId::parse("claude-code:a").unwrap(), true)
        .unwrap();
    assert_eq!(texts(&cycle), ["b", "a"]);
    assert!(
        matches!(&cycle.parents().issues[..], [ParentIssue::Cycle { session }] if session.to_string() == "claude-code:a")
    );
    let orphan = r
        .select_messages(&SessionId::parse("claude-code:orphan").unwrap(), true)
        .unwrap();
    assert_eq!(texts(&orphan), ["orphan"]);
    assert!(matches!(
        &orphan.parents().issues[..],
        [ParentIssue::MissingParent { .. }]
    ));
}

#[test]
fn a_shared_ancestor_is_deduplicated_without_becoming_a_session_cycle() {
    let mut writer = crate::fixture::writer::WriterArchive::new();
    let p = recorded_messages(&mut writer, "p", &["p"], &[]);
    writer
        .links
        .push(recorded_link(polylogue_model::LinkKind::Copy, "p", "a"));
    let a = recorded_messages(&mut writer, "a", &["a"], &p);
    writer
        .links
        .push(recorded_link(polylogue_model::LinkKind::Copy, "p", "b"));
    let b = recorded_messages(&mut writer, "b", &["b"], &p);
    let child = crate::fixture::writer::message(
        "0",
        "child",
        polylogue_model::Role::User,
        0,
        vec![polylogue_model::Block::Text {
            text: "child".into(),
        }],
    );
    // The child really repeats the common ancestor and both branches in this
    // source order. Links alone do not supply an ordering between the branches.
    copied_source(
        &mut writer,
        "child",
        &["p", "a", "b"],
        vec![p[0].clone(), a[0].clone(), b[0].clone(), child],
    );
    let pool = ReadPool::open(&writer.archive).unwrap();
    let r = pool.reader().unwrap();
    let selected = r
        .select_messages(&SessionId::parse("claude-code:child").unwrap(), true)
        .unwrap();
    assert!(selected.parents().complete);
    assert_eq!(texts(&selected), ["p", "a", "b", "child"]);
    assert_eq!(
        r.query(
            "SELECT COUNT(*) AS n FROM message WHERE id='claude-code:p-0'",
            &[]
        )
        .unwrap()[0]["n"],
        1
    );
}

#[test]
fn source_checkpoint_skips_prefix_hydration_in_physical_and_logical_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut fixture, "prefix", &["prefix one", "prefix two"]);
    session(&mut fixture, "tail", &["tail visible"]);
    fork(&mut fixture, "prefix", "tail", Some("claude-code:prefix-1"));
    let conn =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute("UPDATE block SET body=X'00' WHERE message_pk=(SELECT pk FROM message WHERE id='claude-code:prefix-0')", []).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (id, with_parents, from, expected) in [
        ("claude-code:prefix", false, 1, vec!["prefix two"]),
        (
            "claude-code:tail",
            true,
            1,
            vec!["prefix two", "tail visible"],
        ),
        ("claude-code:tail", true, 2, vec!["tail visible"]),
    ] {
        let selected = reader
            .select_messages(&SessionId::parse(id).unwrap(), with_parents)
            .unwrap();
        assert!(
            selected.visit(|_| ControlFlow::Continue(())).is_err(),
            "the reference case must try the corrupt prefix body"
        );
        let mut actual = Vec::new();
        selected
            .visit_from(
                VisitStart {
                    message: from,
                    key: None,
                },
                |message| {
                    actual.push(message.text());
                    ControlFlow::Continue(())
                },
            )
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn previous_indices_include_hidden_messages_in_own_and_parent_selections() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut f, "a", &["first", "hidden", "last"]);
    session(&mut f, "b", &["child"]);
    fork(&mut f, "a", "b", Some("claude-code:a-2"));
    let conn =
        crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    conn.execute(
        "UPDATE message SET role='unparseable',written_by='unknown' WHERE id='claude-code:a-1'",
        [],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    for (id, parents) in [("claude-code:a", false), ("claude-code:b", true)] {
        let selected = r
            .select_messages(&SessionId::parse(id).unwrap(), parents)
            .unwrap();
        let at = selected
            .locate(Some(&MessageId::parse("claude-code:a-2").unwrap()))
            .unwrap()
            .unwrap();
        let previous = selected
            .previous(at, &[polylogue_model::BlockKind::Text], true)
            .unwrap()
            .unwrap();
        assert_eq!(previous.message, 0);
        let mut found = Vec::new();
        selected
            .for_blocks(&[polylogue_model::BlockKind::Text], true)
            .visit_from(previous, |m| {
                found.push(m.text());
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(found[0], "first");
        assert_eq!(found[1], "");
    }
}

#[test]
fn incompatible_recorded_forks_report_both_exact_references_without_choosing_a_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut f, "a", &["left parent"]);
    session(&mut f, "b", &["right parent"]);
    session(&mut f, "child", &["own child"]);
    fork(&mut f, "a", "child", Some("claude-code:a-0"));
    fork(&mut f, "b", "child", Some("claude-code:b-0"));
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader
        .select_messages(&SessionId::parse("claude-code:child").unwrap(), true)
        .unwrap();
    assert!(!selected.parents().complete);
    assert!(
        matches!(&selected.parents().issues[..], [ParentIssue::ConflictingParents { child, first, first_anchor, second, second_anchor }]
        if child.to_string() == "claude-code:child" && first.to_string() == "claude-code:a" && first_anchor == "claude-code:a-0" && second.to_string() == "claude-code:b" && second_anchor == "claude-code:b-0")
    );
    assert_eq!(texts(&selected), ["own child"]);
}
