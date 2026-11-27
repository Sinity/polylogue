use super::compose::*;
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
