use crate::{
    ReadPool,
    fixture::{Fixture, SessionFacts},
};
use polylogue_model::SessionId;

/// A link row whose stored kind this build does not know makes the export
/// fail as a store failure, not as something wrong with the request.
#[test]
fn export_over_a_corrupt_stored_link_kind_is_a_store_failure() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    for id in ["test:parent", "test:child"] {
        fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .link("test:parent", "test:child", "continuation", "fork")
        .unwrap();
    let child = SessionId::parse("test:child").unwrap();
    let export = |fixture: &Fixture| {
        let pool = ReadPool::open(&fixture.archive).unwrap();
        pool.reader().unwrap().session_export(&child)
    };
    assert_eq!(export(&fixture).unwrap().unwrap().links.len(), 1);
    fixture
        .link("test:parent", "test:child", "unheard-of", "fork")
        .unwrap();
    let error = export(&fixture).unwrap_err();
    assert!(
        matches!(error, crate::StoreError::StoredValue { column: 0, .. }),
        "{error:?}"
    );
    assert_eq!(error.code(), "store");
}
