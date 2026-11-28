use super::*;

#[test]
fn completed_history_acknowledges_one_committed_record() {
    let archive = WriterArchive::new();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions");
    let expected = run(
        &object,
        "acknowledged-history",
        123,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let completed = outcome(&expected, 3);
    let sender = archive.writer.sender();
    let pause = sender.pause_user_dispatch_for_test(
        expected.id.as_str(),
        crate::writer::DispatchPhase::BeforeParentCheck,
    );
    let slot = sender.archive().reserve_history().unwrap();
    let recorded = std::thread::scope(|scope| {
        let (reply, result) = std::sync::mpsc::channel();
        let (write_object, write_run, write_outcome) =
            (object.clone(), expected.clone(), completed.clone());
        scope.spawn(move || {
            reply
                .send(slot.record(write_object, write_run, write_outcome))
                .unwrap();
        });
        pause.wait().unwrap();
        assert!(matches!(
            result.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        let reader = pool.reader().unwrap();
        assert!(
            reader
                .query_object(&object.id.to_string())
                .unwrap()
                .is_none()
        );
        assert!(reader.query_run(&expected.id).unwrap().is_none());
        assert!(reader.query_run_outcome(&expected.id).unwrap().is_none());
        drop(reader);
        pause.release();
        result.recv().unwrap()
    });
    recorded.unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query_object(&object.id.to_string())
            .unwrap()
            .unwrap(),
        object
    );
    assert_eq!(reader.query_run(&expected.id).unwrap().unwrap(), expected);
    assert_eq!(
        reader.query_run_outcome(&expected.id).unwrap().unwrap(),
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
        .archive()
        .finish_query_run(previous.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    assert!(
        archive
            .writer
            .archive()
            .reserve_history()
            .unwrap()
            .record(object.clone(), expected.clone(), completed)
            .is_err()
    );
    let report = archive.writer.flush_rows().unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.subject.starts_with("query history")),
        "{:?}",
        report.errors
    );
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .query_object(&object.id.to_string())
            .unwrap()
            .is_none()
    );
    assert!(reader.query_run(&expected.id).unwrap().is_none());
    assert_eq!(
        reader.query_run_outcome(&expected.id).unwrap().unwrap(),
        previous
    );
}
