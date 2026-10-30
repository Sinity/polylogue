use super::*;
use crate::writer::{DispatchPhase, ParentLifetime};
use std::time::Instant;

#[test]
fn a_parent_cancelled_after_queueing_cannot_create_question_custody_or_begin() {
    let archive = WriterArchive::new();
    let sender = archive.writer.sender().unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions where title:guarded");
    let cancellation = crate::read::limit::ReadCancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let barrier =
        sender.pause_user_dispatch_for_test(&object.id.to_string(), DispatchPhase::BeforeAdmission);
    let queued = sender.clone();
    let supplied = object.clone();
    let task = std::thread::spawn(move || queued.query_object_with_parent(supplied, parent));
    barrier.wait().unwrap();
    cancellation.cancel();
    barrier.release();
    assert!(matches!(
        task.join().unwrap(),
        Err(StoreError::ReadCancelled)
    ));
    assert!(
        pool.reader()
            .unwrap()
            .query_object(&object.id.to_string(), None)
            .unwrap()
            .is_none()
    );
    // Establish question custody independently, then exercise the closed Begin variant.
    sender.query_object(object.clone()).unwrap();
    let expected = run(
        &object,
        "parent-queued-begin",
        123_456,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let cancellation = crate::read::limit::ReadCancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let barrier =
        sender.pause_user_dispatch_for_test(expected.id.as_str(), DispatchPhase::BeforeAdmission);
    let queued = sender.clone();
    let supplied = expected.clone();
    let task = std::thread::spawn(move || queued.begin_query_run_with_parent(supplied, parent));
    barrier.wait().unwrap();
    cancellation.cancel();
    barrier.release();
    assert!(matches!(
        task.join().unwrap(),
        Err(StoreError::ReadCancelled)
    ));
    let reader = pool.reader().unwrap();
    assert!(reader.query_run(&expected.id, None).unwrap().is_none());
    assert!(
        reader
            .query_run_outcome(&expected.id, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn an_admitted_begin_finishes_after_parent_cancellation_and_keeps_owned_outcome_cleanup() {
    let archive = WriterArchive::new();
    let sender = archive.writer.sender().unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions where title:owned");
    sender.query_object(object.clone()).unwrap();
    let expected = run(
        &object,
        "parent-owned-begin",
        123_456,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let cancellation = crate::read::limit::ReadCancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let barrier = sender.pause_user_dispatch_for_test(expected.id.as_str(), DispatchPhase::Owned);
    let queued = sender.clone();
    let supplied = expected.clone();
    let task = std::thread::spawn(move || queued.begin_query_run_with_parent(supplied, parent));
    barrier.wait().unwrap();
    cancellation.cancel();
    barrier.release();
    assert_eq!(task.join().unwrap().unwrap(), QueryCustody::Canonical);
    let completed = outcome(&expected, 3);
    // Finish has no producing-parent argument and validates its actual prior Begin.
    assert_eq!(
        sender
            .finish_query_run(completed.clone())
            .unwrap()
            .recv()
            .unwrap()
            .unwrap(),
        QueryCustody::Canonical
    );
    let reader = pool.reader().unwrap();
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
fn a_preexpired_parent_cannot_even_replay_existing_question_custody() {
    let archive = WriterArchive::new();
    let object = question("sessions");
    archive.writer.query_object(object.clone()).unwrap();
    let parent = ParentLifetime::capture(Some(Instant::now()), Default::default());
    assert!(matches!(
        archive
            .writer
            .sender()
            .unwrap()
            .query_object_with_parent(object, parent),
        Err(StoreError::ReadDeadlineExceeded)
    ));
}
