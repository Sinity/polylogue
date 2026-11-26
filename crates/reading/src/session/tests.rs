use super::*;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};

fn child_with(parent_link: impl FnOnce(&mut Fixture)) -> (tempfile::TempDir, Fixture) {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    let child = fixture
        .session(SessionFacts {
            id: "test:child",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            child,
            MessageFacts {
                id: "test:own",
                role: "user",
                text: Some("own words"),
                ..Default::default()
            },
        )
        .unwrap();
    parent_link(&mut fixture);
    (root, fixture)
}

fn composed() -> Composed {
    Composed {
        parts: Vec::new(),
        indexes: Vec::new(),
        issues: Vec::new(),
        unstored_copy: false,
        before: Default::default(),
        after: Vec::new(),
    }
}

#[test]
fn a_copy_from_an_unstored_session_reads_as_own_with_its_marker() {
    let (_root, fixture) = child_with(|fixture| {
        fixture
            .copying_link("test:missing", "test:child", "fork")
            .unwrap();
    });
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut found = composed();
    prefix(
        &reader,
        &SessionId::parse("test:child").unwrap(),
        &mut found,
    )
    .unwrap();
    assert!(found.unstored_copy);
    assert!(found.issues.is_empty());
    assert!(found.parts.is_empty());
}

#[test]
fn a_link_that_copies_nothing_is_never_the_unstored_copy_marker() {
    let (_root, fixture) = child_with(|fixture| {
        fixture
            .link("test:missing", "test:child", "continuation", None)
            .unwrap();
    });
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut found = composed();
    prefix(
        &reader,
        &SessionId::parse("test:child").unwrap(),
        &mut found,
    )
    .unwrap();
    assert!(!found.unstored_copy);
    assert!(matches!(
        &found.issues[..],
        [ParentIssue::MissingBranchPoint { .. }]
    ));
}

#[test]
fn transcript_and_store_compose_the_same_copied_prefix() {
    use polylogue_testkit::{Morrow, Set, session};
    let archive = Morrow::build(Set::Main);
    let pool = archive.reads();
    let reader = pool.reader().unwrap();
    let id = session(4);
    let mut expected = Vec::new();
    let selection = reader.select_messages(&id, true).unwrap();
    selection
        .visit(Default::default(), |message| {
            expected.push(message.id.clone());
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let (mut parts, parents, unstored) = reader.parent_parts(&id).unwrap();
    assert!(parents.issues.is_empty());
    assert!(!unstored);
    parts.push(Part {
        session: id.clone(),
        start: 0,
        end: reader.item_state(&id).unwrap().unwrap().entries as usize,
        inherited: false,
        joined_by: None,
    });
    let mut actual = Vec::new();
    for part in parts {
        let items = reader.item_index(&part.session).unwrap().unwrap();
        for entry in &items.entries[part.start..part.end] {
            if entry.kind == EntryKind::Header {
                actual.push(entry.key.message().unwrap().clone());
            }
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 29);
}
