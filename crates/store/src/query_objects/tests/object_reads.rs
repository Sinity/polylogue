use super::*;
use crate::read::ObjectRecord;
use polylogue_model::Reference;

#[test]
fn run_objects_distinguish_missing_pending_and_complete_records() {
    let archive = WriterArchive::new();
    let object = question("sessions");
    archive
        .writer
        .archive()
        .query_object(object.clone(), None)
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    let beginning = run(&object, "object-run", 1, position);
    let target = Reference::Run(beginning.id.clone());
    assert!(matches!(
        pool.reader().unwrap().prepare_object_read(&target),
        Err(StoreError::NotFound(_))
    ));
    archive
        .writer
        .archive()
        .begin_query_run(beginning.clone(), None)
        .unwrap();
    let pending_hash = {
        let reader = pool.reader().unwrap();
        let prepared = reader.prepare_object_read(&target).unwrap();
        let ObjectRecord::Run(record) = prepared.record().unwrap() else {
            panic!("run object")
        };
        assert_eq!(record.run, beginning);
        assert_eq!(record.outcome, None);
        prepared.content_hash()
    };
    let completed = outcome(&beginning, 0);
    archive
        .writer
        .archive()
        .finish_query_run(completed.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader.prepare_object_read(&target).unwrap();
    assert_ne!(prepared.content_hash(), pending_hash);
    let ObjectRecord::Run(record) = prepared.into_record().unwrap() else {
        panic!("run object")
    };
    assert_eq!(record.run, beginning);
    assert_eq!(record.outcome, Some(completed));
}

#[test]
fn run_completion_without_its_beginning_is_incomplete() {
    let archive = WriterArchive::new();
    let object = question("sessions");
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    let beginning = run(&object, "completion-before-begin", 1, position);
    archive
        .writer
        .archive()
        .finish_query_run(outcome(&beginning, 7))
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let reader = pool.reader().unwrap();
    assert!(
        matches!(reader.prepare_object_read(&Reference::Run(beginning.id)), Err(StoreError::ReferenceUnavailable {reason,..}) if reason == "the run completion is recorded but its beginning is unavailable")
    );
}

#[test]
fn run_object_read_checks_the_completion_identity() {
    let archive = WriterArchive::new();
    let object = question("sessions");
    archive
        .writer
        .archive()
        .query_object(object.clone(), None)
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    let beginning = run(&object, "checked-object", 1, position);
    archive
        .writer
        .archive()
        .begin_query_run(beginning.clone(), None)
        .unwrap();
    let mut completion = outcome(&beginning, 1);
    archive
        .writer
        .archive()
        .finish_query_run(completion.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    completion.run = run_id("another-object");
    let conn = crate::archive::open_write(&archive.archive.archive_db(), 1).unwrap();
    conn.execute(
        "UPDATE query_run SET completion=?1 WHERE id=?2",
        rusqlite::params![
            crate::encoding::pack_json(&completion).unwrap(),
            beginning.id.as_str()
        ],
    )
    .unwrap();
    drop(conn);
    let reader = pool.reader().unwrap();
    assert!(
        matches!(reader.prepare_object_read(&Reference::Run(beginning.id)), Err(StoreError::StoredValue { column: None, reason }) if reason == "stored run completion names another run")
    );
}
