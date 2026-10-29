use super::*;
use crate::{
    ReadPool,
    fixture::{Fixture, SessionFacts},
};
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

#[test]
fn physical_family_counts_match_membership_without_reading_payload_columns() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    let target = fixture
        .session(SessionFacts {
            id: "test:target",
            ..Default::default()
        })
        .unwrap();
    let other = fixture
        .session(SessionFacts {
            id: "test:other",
            ..Default::default()
        })
        .unwrap();
    fixture
        .session(SessionFacts {
            id: "test:child",
            ..Default::default()
        })
        .unwrap();
    fixture
        .session(SessionFacts {
            id: "test:empty",
            ..Default::default()
        })
        .unwrap();
    fixture.event(target, "notice", None).unwrap();
    fixture.event(target, "notice", None).unwrap();
    fixture.event(other, "notice", None).unwrap();
    fixture
        .link("test:target", "test:child", "continuation", "fork")
        .unwrap();
    fixture
        .link("test:other", "test:target", "continuation", "fork")
        .unwrap();
    fixture
        .link("test:other", "test:child", "continuation", "fork")
        .unwrap();
    let pool = ReadPool::open(&fixture.archive).unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| {
            if matches!(
                context.action,
                AuthAction::Read {
                    table_name: "event",
                    column_name: "data" | "fields"
                } | AuthAction::Read {
                    table_name: "link",
                    column_name: "evidence"
                }
            ) {
                Authorization::Deny
            } else {
                Authorization::Allow
            }
        }))
        .unwrap();
    assert_eq!(
        reader
            .physical_record_counts(&SessionId::parse("test:target").unwrap())
            .unwrap(),
        Some(PhysicalRecordCounts {
            events: 2,
            links: 2
        })
    );
    assert_eq!(
        reader
            .physical_record_counts(&SessionId::parse("test:empty").unwrap())
            .unwrap(),
        Some(PhysicalRecordCounts {
            events: 0,
            links: 0
        })
    );
    assert_eq!(
        reader
            .physical_record_counts(&SessionId::parse("test:absent").unwrap())
            .unwrap(),
        None
    );
    assert!(
        reader
            .visit_links_from_ordinal(&SessionId::parse("test:target").unwrap(), 0, |_| {
                std::ops::ControlFlow::Break(())
            })
            .is_err()
    );
    fixture.archive.read_scope.begin_reset().unwrap();
    assert!(matches!(
        reader.physical_record_counts(&SessionId::parse("test:target").unwrap()),
        Err(StoreError::ScopeChanged)
    ));
    fixture.archive.read_scope.finish_reset();
}
