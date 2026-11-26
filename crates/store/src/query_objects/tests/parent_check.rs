use super::*;
use crate::writer::{DispatchPhase, ParentLifetime};
use std::time::Instant;

#[test]
fn a_parent_cancelled_after_queueing_cannot_store_the_question_or_begin() {
    let archive = WriterArchive::new();
    let sender = archive.writer.sender();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions where title:guarded");
    let cancellation = crate::cancellation::Cancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let pause = sender
        .pause_user_dispatch_for_test(&object.id.to_string(), DispatchPhase::BeforeParentCheck);
    let queued = sender.clone();
    let supplied = object.clone();
    let task = std::thread::spawn(move || queued.archive().query_object(supplied, Some(parent)));
    pause.wait().unwrap();
    cancellation.cancel();
    pause.release();
    assert!(matches!(
        task.join().unwrap(),
        Err(StoreError::ReadCancelled)
    ));
    assert!(
        pool.reader()
            .unwrap()
            .query_object(&object.id.to_string())
            .unwrap()
            .is_none()
    );
    // Store the question independently, then exercise the closed Begin alternate.
    sender.archive().query_object(object.clone(), None).unwrap();
    let expected = run(
        &object,
        "parent-queued-begin",
        123_456,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let cancellation = crate::cancellation::Cancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let pause =
        sender.pause_user_dispatch_for_test(expected.id.as_str(), DispatchPhase::BeforeParentCheck);
    let queued = sender.clone();
    let supplied = expected.clone();
    let task = std::thread::spawn(move || queued.archive().begin_query_run(supplied, Some(parent)));
    pause.wait().unwrap();
    cancellation.cancel();
    pause.release();
    assert!(matches!(
        task.join().unwrap(),
        Err(StoreError::ReadCancelled)
    ));
    let reader = pool.reader().unwrap();
    assert!(reader.query_run(&expected.id).unwrap().is_none());
    assert!(reader.query_run_outcome(&expected.id).unwrap().is_none());
}

#[test]
fn checked_begin_finishes_after_parent_cancellation() {
    let archive = WriterArchive::new();
    let sender = archive.writer.sender();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let object = question("sessions where title:owned");
    sender.archive().query_object(object.clone(), None).unwrap();
    let expected = run(
        &object,
        "parent-owned-begin",
        123_456,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let cancellation = crate::cancellation::Cancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let pause = sender.pause_user_dispatch_for_test(expected.id.as_str(), DispatchPhase::Owned);
    let queued = sender.clone();
    let supplied = expected.clone();
    let task = std::thread::spawn(move || queued.archive().begin_query_run(supplied, Some(parent)));
    pause.wait().unwrap();
    cancellation.cancel();
    pause.release();
    task.join().unwrap().unwrap();
    let completed = outcome(&expected, 3);
    // Finish has no producing-parent argument and validates its prior Begin.
    sender
        .archive()
        .finish_query_run(completed.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(reader.query_run(&expected.id).unwrap().unwrap(), expected);
    assert_eq!(
        reader.query_run_outcome(&expected.id).unwrap().unwrap(),
        completed
    );
}

#[test]
fn a_preexpired_parent_cannot_even_replay_an_existing_question() {
    let archive = WriterArchive::new();
    let object = question("sessions");
    archive
        .writer
        .archive()
        .query_object(object.clone(), None)
        .unwrap();
    let parent = ParentLifetime::capture(Some(Instant::now()), Default::default());
    assert!(matches!(
        archive.writer.archive().query_object(object, Some(parent)),
        Err(StoreError::ReadDeadlineExceeded)
    ));
}
