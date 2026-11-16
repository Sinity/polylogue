use crate::{
    ReadPool, StoreError,
    fixture::{Fixture, SessionFacts},
};
use polylogue_model::Code;

#[test]
fn unreadable_event_coordinates_are_stored_value_failures() {
    for (change, column) in [
        ("UPDATE event SET source_end=-1", 5),
        ("UPDATE event_source SET source_start=-1,source_end=1", 3),
        ("UPDATE event_source SET committed=-1", 6),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(root.path()).unwrap();
        let session = fixture
            .session(SessionFacts {
                id: "test:events",
                ..Default::default()
            })
            .unwrap();
        fixture.event(session, "marker", None).unwrap();
        let conn = crate::archive::open_write(&fixture.archive.index_db(), 1).unwrap();
        conn.execute("UPDATE event_source SET committed=1", [])
            .unwrap();
        let pool = ReadPool::open(&fixture.archive).unwrap();
        {
            let reader = pool.reader().unwrap();
            assert!(reader.event_record("e1").unwrap().is_some());
        }
        conn.execute_batch("PRAGMA ignore_check_constraints=ON")
            .unwrap();
        conn.execute(change, []).unwrap();
        let error = pool.reader().unwrap().event_record("e1").unwrap_err();
        assert!(
            matches!(error, StoreError::StoredValue { column: found, .. } if found == column),
            "{error:?}"
        );
        assert_eq!(error.code(), Code::Store);
    }
}
