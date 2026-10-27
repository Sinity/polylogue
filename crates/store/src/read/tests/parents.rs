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
fn a_read_starts_anywhere_with_the_masking_state_masking_from_the_start_reaches() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    session(
        &mut f,
        "keyed",
        &[
            "ordinary",
            "closed -----BEGIN RSA PRIVATE KEY-----x-----END RSA PRIVATE KEY----- block",
            "left open -----BEGIN RSA PRIVATE KEY-----",
            "inside the open key",
            "still inside",
            "closes -----END RSA PRIVATE KEY----- shown",
            "after the key",
            "a second -----BEGIN EC PRIVATE KEY----- never closed",
            "hidden to the end",
        ],
    );
    session(&mut f, "plain", &["one", "two", "three"]);
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let plain = reader
        .select_messages(&SessionId::parse("claude-code:plain").unwrap(), false)
        .unwrap();
    assert!(plain.masking_before(2).unwrap().is_none());
    let keyed = reader
        .select_messages(&SessionId::parse("claude-code:keyed").unwrap(), false)
        .unwrap();
    let mut messages = Vec::new();
    keyed
        .visit(|message| {
            messages.push(message);
            ControlFlow::Continue(())
        })
        .unwrap();
    let shown = |mut masker: polylogue_model::masking::MessageMasker, from: usize| {
        messages[from..]
            .iter()
            .map(|message| {
                let mut message = message.clone();
                masker.message_text(&mut message);
                message.text()
            })
            .collect::<Vec<_>>()
    };
    for at in 0..messages.len() {
        let mut from_start = polylogue_model::masking::MessageMasker::default();
        for message in &messages[..at] {
            from_start.message_text(&mut message.clone());
        }
        let resumed = keyed.masking_before(at).unwrap();
        assert_eq!(resumed.is_none(), at <= 2, "at {at}");
        assert_eq!(
            shown(resumed.unwrap_or_default(), at),
            shown(from_start, at),
            "at {at}"
        );
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
    let entries = selected.outline(0, 100).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.message.as_str())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        selected
            .outline(2, 2)
            .unwrap()
            .iter()
            .map(|entry| entry.message.as_str())
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
