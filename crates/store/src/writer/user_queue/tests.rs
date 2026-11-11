use super::*;
use crate::read::limit::ReadCancellation;
use std::time::Duration;

fn message() -> UserMsg {
    let (reply, _) = std::sync::mpsc::sync_channel(1);
    UserMsg::Request(
        Box::new(crate::fixture::authored::operator_request(
            crate::user_writes::new_request_id(),
            polylogue_model::user::UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: "on".into(),
                machine: None,
            },
        )),
        None,
        None,
        None,
        reply,
    )
}

fn wait_for_admission(queue: &UserQueue) {
    let mut state = queue.state.lock().unwrap();
    while state.waiters == 0 {
        state = queue.blocked.wait(state).unwrap();
    }
}

#[test]
fn unfinished_reads_allow_user_writes_and_their_completed_handoff_has_room() {
    let queue = Arc::new(UserQueue::default());
    let mut reservations = (0..USER_QUEUE)
        .map(|_| queue.reserve(None).unwrap())
        .collect::<Vec<_>>();
    for _ in 0..USER_QUEUE {
        queue
            .try_push(message())
            .unwrap_or_else(|_| panic!("ordinary write capacity"));
    }
    assert!(matches!(queue.try_push(message()), Err(Some(_))));
    let history = reservations.pop().unwrap();
    history.push(message()).unwrap();
    assert_eq!(queue.waiting(), USER_QUEUE + 1);
}

#[test]
fn cancelled_admission_wakes_without_a_deadline() {
    let queue = Arc::new(UserQueue::default());
    let _reservations = (0..USER_QUEUE)
        .map(|_| queue.reserve(None).unwrap())
        .collect::<Vec<_>>();
    let cancellation = ReadCancellation::default();
    let limit = ReadLimit::without_deadline(cancellation.clone());
    let waiting = Arc::clone(&queue);
    let thread = std::thread::spawn(move || waiting.reserve(Some(&limit)));
    wait_for_admission(&queue);
    cancellation.cancel();
    assert!(matches!(
        thread.join().unwrap(),
        Err(StoreError::ReadCancelled)
    ));
}

#[test]
fn admission_uses_the_original_deadline() {
    let queue = Arc::new(UserQueue::default());
    let _reservations = (0..USER_QUEUE)
        .map(|_| queue.reserve(None).unwrap())
        .collect::<Vec<_>>();
    let limit = ReadLimit::new(Duration::from_millis(1), ReadCancellation::default());
    assert!(matches!(
        queue.reserve(Some(&limit)),
        Err(StoreError::ReadDeadlineExceeded)
    ));
}

#[test]
fn closing_wakes_admission_and_retains_queued_writes() {
    let queue = Arc::new(UserQueue::default());
    queue
        .try_push(message())
        .unwrap_or_else(|_| panic!("initial capacity"));
    let _reservations = (1..USER_QUEUE)
        .map(|_| queue.reserve(None).unwrap())
        .collect::<Vec<_>>();
    let waiting = Arc::clone(&queue);
    let thread = std::thread::spawn(move || waiting.reserve(None));
    wait_for_admission(&queue);
    queue.close();
    assert!(matches!(
        thread.join().unwrap(),
        Err(StoreError::WriterGone)
    ));
    assert!(queue.pop().is_some());
    assert!(queue.pop().is_none());
}

#[test]
fn receiver_exit_drops_replies_and_refuses_new_admission() {
    let queue = Arc::new(UserQueue::default());
    let (reply, received) = std::sync::mpsc::sync_channel(1);
    let object =
        polylogue_query::plan::prepare("sessions", &polylogue_query::Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    queue
        .try_push(UserMsg::QueryObject(Box::new(object), None, reply))
        .unwrap_or_else(|_| panic!("initial capacity"));
    queue.disconnect();
    assert!(received.recv().is_err());
    assert!(matches!(queue.reserve(None), Err(StoreError::WriterGone)));
    assert!(queue.pop().is_none());
}
