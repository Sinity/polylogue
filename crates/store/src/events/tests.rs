use crate::{
    ReadPool, StoreError,
    fixture::{Fixture, SessionFacts},
};
use polylogue_model::Code;
use std::ops::ControlFlow;

#[test]
fn an_event_kind_can_be_read_in_reverse_time_without_other_events() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:events",
            ..Default::default()
        })
        .unwrap();
    for (kind, at) in [
        ("chosen", 1000),
        ("other", 4000),
        ("chosen", 3000),
        ("chosen", 2000),
    ] {
        fixture.event(session, kind, Some(at)).unwrap();
    }
    let pool = ReadPool::open(&fixture.archive).unwrap();
    let mut times = Vec::new();
    pool.reader()
        .unwrap()
        .visit_events_by_kind("chosen", |event| {
            assert_eq!(event.kind, "chosen");
            times.push(event.at.unwrap().unix_ms());
            if times.len() == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert_eq!(times, [3000, 2000]);
}

#[test]
fn unreadable_event_coordinates_are_stored_value_failures() {
    for (change, column, reason) in [
        (
            "UPDATE event SET source_end=-1",
            5,
            Some("negative event source coordinate"),
        ),
        (
            "UPDATE event_parse SET source_start=-1,source_length=2",
            3,
            None,
        ),
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
        // The event's byte range, and the archive extent its commit is read from.
        rusqlite::Connection::open(fixture.archive.archive_db())
            .unwrap()
            .execute_batch(
                "PRAGMA foreign_keys=OFF;
                 INSERT INTO input VALUES('fixture','fixture');
                 INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES('fixture',1,'first',1,1,1);
                 INSERT INTO extent(input,chain,offset,length,blob,blob_offset,commit_seq) VALUES('fixture',1,0,1,x'00',0,1);",
            )
            .unwrap();
        let conn = crate::archive::open_write(&fixture.archive.index_db(), 1).unwrap();
        conn.execute("UPDATE event SET source_end=1", []).unwrap();
        let pool = ReadPool::open(&fixture.archive).unwrap();
        {
            let reader = pool.reader().unwrap();
            assert!(reader.event_record("e1").unwrap().is_some());
        }
        conn.execute_batch("PRAGMA ignore_check_constraints=ON")
            .unwrap();
        conn.execute(change, []).unwrap();
        let error = pool.reader().unwrap().event_record("e1").unwrap_err();
        match &error {
            StoreError::StoredValue {
                column: found,
                reason: found_reason,
            } => {
                assert_eq!(*found, Some(column));
                if let Some(reason) = reason {
                    assert_eq!(found_reason, reason);
                }
            }
            other => panic!("expected a stored value failure, got {other:?}"),
        }
        assert_eq!(error.code(), Code::Store);
    }
}
