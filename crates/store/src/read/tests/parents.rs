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
    assert_eq!(
        selected.fingerprint().unwrap(),
        r.message_fingerprint(&id, true).unwrap()
    );
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
fn changed_missing_anchor_changes_fingerprint_even_when_available_messages_do_not() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut f, "a", &["a"]);
    session(&mut f, "b", &["b"]);
    fork(&mut f, "a", "b", Some("claude-code:missing-one"));
    let pool = f.reads().unwrap();
    let id = SessionId::parse("claude-code:b").unwrap();
    let before = {
        let r = pool.reader().unwrap();
        r.select_messages(&id, true).unwrap().fingerprint().unwrap()
    };
    crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute("UPDATE link SET branch_point='claude-code:missing-two'", [])
        .unwrap();
    let r = pool.reader().unwrap();
    let selected = r.select_messages(&id, true).unwrap();
    assert_eq!(texts(&selected), ["b"]);
    assert_ne!(before, selected.fingerprint().unwrap());
}

#[test]
fn a_shared_ancestor_is_deduplicated_without_becoming_a_session_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    for name in ["p", "a", "b", "child"] {
        session(&mut f, name, &[name]);
    }
    fork(&mut f, "p", "a", Some("claude-code:p-0"));
    fork(&mut f, "p", "b", Some("claude-code:p-0"));
    fork(&mut f, "a", "child", Some("claude-code:a-0"));
    fork(&mut f, "b", "child", Some("claude-code:b-0"));
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let selected = r
        .select_messages(&SessionId::parse("claude-code:child").unwrap(), true)
        .unwrap();
    assert!(selected.parents().complete);
    assert_eq!(texts(&selected), ["p", "a", "b", "child"]);
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
fn outline_and_anchors_keep_shared_ancestry_in_selected_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut fixture, "a", &["shared first", "shared second"]);
    session(&mut fixture, "b", &["branch b"]);
    session(&mut fixture, "c", &["branch c"]);
    session(&mut fixture, "d", &["joint session"]);
    fork(&mut fixture, "a", "b", Some("claude-code:a-0"));
    fork(&mut fixture, "a", "c", Some("claude-code:a-1"));
    fork(&mut fixture, "b", "d", Some("claude-code:b-0"));
    fork(&mut fixture, "c", "d", Some("claude-code:c-0"));
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let id = SessionId::parse("claude-code:d").unwrap();
    let selected = reader.select_messages(&id, true).unwrap();
    let expected = [
        "claude-code:a-0",
        "claude-code:b-0",
        "claude-code:a-1",
        "claude-code:c-0",
        "claude-code:d-0",
    ];
    let entries = selected.outline(None, 100).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.message.to_string())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        selected
            .outline(Some(entries[1].key), 2)
            .unwrap()
            .iter()
            .map(|entry| entry.message.to_string())
            .collect::<Vec<_>>(),
        expected[2..4]
    );
    let anchor = polylogue_model::MessageId::parse("claude-code:a-1").unwrap();
    assert_eq!(selected.locate(Some(&anchor)).unwrap().unwrap().message, 2);
    assert!(
        reader
            .select_messages(&id, false)
            .unwrap()
            .locate(Some(&anchor))
            .unwrap()
            .is_none()
    );
    let last = selected
        .latest(&[polylogue_model::BlockKind::Text], true)
        .unwrap()
        .unwrap();
    assert_eq!(last.message, 4);
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
fn outline_includes_microcompactions_and_seeks_indexed_branch_points() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(&mut f, "a", &["prompt", "context change"]);
    let conn =
        crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    conn.execute(
        "UPDATE message SET written_by='model' WHERE id='claude-code:a-1'",
        [],
    )
    .unwrap();
    let pk = conn
        .query_row(
            "SELECT pk FROM session WHERE id='claude-code:a'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap();
    f.event(pk, "microcompaction", None).unwrap();
    conn.execute(
        "UPDATE event SET after_message_pk=(SELECT pk FROM message WHERE id='claude-code:a-1')",
        [],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let selected = r
        .select_messages(&SessionId::parse("claude-code:a").unwrap(), false)
        .unwrap();
    let entries = selected.outline(None, 10).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].events.len(), 1);
    assert!(entries[1].kinds.iter().any(|kind| kind == "compaction"));
    let plan: String = conn
        .query_row(
            "EXPLAIN QUERY PLAN SELECT 1 FROM link WHERE branch_point='claude-code:a-1'",
            [],
            |row| row.get(3),
        )
        .unwrap();
    assert!(plan.contains("link_branch_point"), "{plan}");
}
