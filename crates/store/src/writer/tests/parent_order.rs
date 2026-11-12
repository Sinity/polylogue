use super::*;

fn copy(parent: &str) -> Link {
    Link {
        kind: LinkKind::Resume,
        copies_prefix: true,
        how: None,
        parent: sid(parent),
        child: sid("c1"),
        at: None,
        call_id: None,
        evidence: "recorded copied prefix".into(),
    }
}

fn available_texts(selected: &crate::read::MessageSelection<'_, '_>) -> Vec<String> {
    let mut texts = Vec::new();
    selected
        .visit(|message| {
            texts.push(message.text());
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    texts
}

fn archive() -> TestArchive {
    let w = test_archive();
    for (session, id) in [("p0", "p0-0"), ("c0", "c0-0")] {
        w.store_reading(
            batch(
                &format!("{session}-file"),
                Route::Live,
                0,
                vec![row(0, text_message(id, session, Role::User, id, None))],
                &[session],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
    }
    w
}

fn child(w: &TestArchive, input: &str, order: &[(&str, &str)], links: Vec<Link>) {
    let rows = order
        .iter()
        .enumerate()
        .map(|(ordinal, (session, id))| {
            row(
                (ordinal * 25) as u64,
                text_message(id, session, Role::User, id, None),
            )
        })
        .collect();
    let mut b = batch(input, Route::Live, 0, rows, &["p0", "c0", "c1"]);
    b.parse[0].output.prefix_copies = order
        .iter()
        .map(|(session, _)| *session)
        .filter(|session| *session != "c1")
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|parent| polylogue_model::PrefixCopy {
            parent: sid(parent),
            child: sid("c1"),
            explicit_members: false,
        })
        .collect();
    b.parse[0].output.links = links;
    w.store_reading(b, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
}

#[test]
fn copied_parent_families_follow_actual_child_coordinates_instead_of_link_insertion() {
    let w = archive();
    child(
        &w,
        "child-file",
        &[("p0", "p0-0"), ("c0", "c0-0"), ("c1", "c1-0")],
        vec![copy("c0")],
    );
    // The p0 link arrives later, independently of its earlier physical record.
    child(&w, "later-link", &[], vec![copy("p0")]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let links = reader
        .query(
            "SELECT parent_id FROM link WHERE child_id='claude-code:c1' ORDER BY pk",
            &[],
        )
        .unwrap();
    assert_eq!(
        links
            .iter()
            .map(|row| row["parent_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["claude-code:c0", "claude-code:p0"]
    );
    let coordinates = reader.query("SELECT m.id,s.source_start FROM message_source s JOIN input_ref i ON i.pk=s.input_pk JOIN message m ON m.pk=s.message_pk WHERE i.id='child-file' ORDER BY s.seq,s.record_position", &[]).unwrap();
    assert_eq!(
        coordinates
            .iter()
            .map(|row| (
                row["id"].as_str().unwrap(),
                row["source_start"].as_i64().unwrap()
            ))
            .collect::<Vec<_>>(),
        [
            ("claude-code:p0-0", 0),
            ("claude-code:c0-0", 25),
            ("claude-code:c1-0", 50)
        ]
    );
    let selected = reader.select_messages(&sid("c1"), true).unwrap();
    assert!(selected.parents().complete);
    let actual = reader.messages(&sid("c1"), true).unwrap();
    assert_eq!(
        actual.iter().map(Message::text).collect::<Vec<_>>(),
        ["p0-0", "c0-0", "c1-0"]
    );
}

#[test]
fn conflicting_child_orders_remain_explicit_without_a_chosen_parent_prefix() {
    let w = archive();
    child(
        &w,
        "child-first",
        &[("p0", "p0-0"), ("c0", "c0-0"), ("c1", "c1-0")],
        vec![copy("c0"), copy("p0")],
    );
    child(
        &w,
        "child-other",
        &[("c0", "c0-0"), ("p0", "p0-0"), ("c1", "c1-0")],
        vec![],
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader.select_messages(&sid("c1"), true).unwrap();
    assert!(!selected.parents().complete);
    assert!(
        matches!(&selected.parents().issues[..], [crate::read::ParentIssue::ConflictingOrder { child, .. }] if *child == sid("c1"))
    );
    assert!(matches!(
        reader.messages(&sid("c1"), true),
        Err(StoreError::IncompleteParents(_))
    ));
    assert_eq!(available_texts(&selected), ["c1-0"]);
}

#[test]
fn tied_historical_child_coordinates_never_gain_order_from_link_or_message_keys() {
    for parents in [["p0", "c0"], ["c0", "p0"]] {
        let w = archive();
        child(
            &w,
            "child-file",
            &[("p0", "p0-0"), ("c0", "c0-0"), ("c1", "c1-0")],
            vec![copy(parents[0])],
        );
        child(&w, "later-link", &[], vec![copy(parents[1])]);
        // Historical rows with the original default coordinates retain their
        // actual bytes and identity, but supply no relative display order.
        crate::archive::open_write(&w.archive.index_db(), crate::archive::cache::SETUP).unwrap()
            .execute("UPDATE message_source SET seq=0,record_position=0,source_start=0,source_end=0 WHERE input_pk=(SELECT pk FROM input_ref WHERE id='child-file')", []).unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        let selected = reader.select_messages(&sid("c1"), true).unwrap();
        assert!(!selected.parents().complete);
        let [
            crate::read::ParentIssue::UnknownOrder {
                child,
                first,
                second,
            },
        ] = &selected.parents().issues[..]
        else {
            panic!("{:?}", selected.parents())
        };
        assert_eq!(*child, sid("c1"));
        let expected = [
            MessageId::parse("claude-code:p0-0").unwrap(),
            MessageId::parse("claude-code:c0-0").unwrap(),
        ];
        assert!(
            (first == &expected[0] && second == &expected[1])
                || (first == &expected[1] && second == &expected[0])
        );
        assert!(matches!(
            reader.messages(&sid("c1"), true),
            Err(crate::StoreError::IncompleteParents(issue))
                if matches!(*issue, crate::read::ParentIssue::UnknownOrder { .. })
        ));
        let mut visible = Vec::new();
        selected
            .visit(|message| {
                visible.push(message.text());
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(visible, ["c1-0"]);
        assert_eq!(
            reader
                .query("SELECT COUNT(*) AS n FROM message", &[])
                .unwrap()[0]["n"],
            3
        );
    }
}
