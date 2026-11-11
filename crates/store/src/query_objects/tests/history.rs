use super::*;

#[test]
fn completed_history_returns_before_the_writer_and_commits_as_one_record() {
    let archive = WriterArchive::new();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions");
    let expected = run(
        &object,
        "asynchronous-history",
        123,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let completed = outcome(&expected, 3);
    let sender = archive.writer.sender().unwrap();
    let barrier = sender.pause_user_dispatch_for_test(
        expected.id.as_str(),
        crate::writer::DispatchPhase::BeforeAdmission,
    );
    sender
        .reserve_history()
        .unwrap()
        .record(object.clone(), expected.clone(), completed.clone())
        .unwrap();
    barrier.wait().unwrap();
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .query_object(&object.id.to_string(), None)
            .unwrap()
            .is_none()
    );
    assert!(reader.query_run(&expected.id, None).unwrap().is_none());
    assert!(
        reader
            .query_run_outcome(&expected.id, None)
            .unwrap()
            .is_none()
    );
    drop(reader);
    barrier.release();
    assert!(archive.writer.flush_rows().unwrap().errors.is_empty());
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query_object(&object.id.to_string(), None)
            .unwrap()
            .unwrap()
            .value,
        object
    );
    assert_eq!(
        reader.query_run(&expected.id, None).unwrap().unwrap().value,
        expected
    );
    assert_eq!(
        reader
            .query_run_outcome(&expected.id, None)
            .unwrap()
            .unwrap()
            .value,
        completed
    );
}

#[test]
fn conflicting_history_rolls_back_its_question_and_run() {
    let archive = WriterArchive::new();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions where since:7d");
    let expected = run(
        &object,
        "conflicting-history",
        123,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let completed = outcome(&expected, 3);
    let mut previous = completed.clone();
    previous.binding_hash = polylogue_model::BlobHash::of(b"a different accepted binding");
    archive
        .writer
        .finish_query_run(previous.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    archive
        .writer
        .sender()
        .unwrap()
        .reserve_history()
        .unwrap()
        .record(object.clone(), expected.clone(), completed)
        .unwrap();
    let report = archive.writer.flush_rows().unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.contains("query history")),
        "{:?}",
        report.errors
    );
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .query_object(&object.id.to_string(), None)
            .unwrap()
            .is_none()
    );
    assert!(reader.query_run(&expected.id, None).unwrap().is_none());
    assert_eq!(
        reader
            .query_run_outcome(&expected.id, None)
            .unwrap()
            .unwrap()
            .value,
        previous
    );
}
