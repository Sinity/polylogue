use super::*;

#[test]
fn child_cancellation_does_not_cancel_another_request() {
    let connection = Cancellation::default();
    let a = connection.child();
    let b = connection.child();
    a.cancel();
    assert!(a.is_cancelled());
    assert!(!b.is_cancelled());
    connection.cancel();
    assert!(b.is_cancelled());
}

#[test]
fn a_withdrawn_request_is_cancelled_and_a_stopped_process_is_stopped() {
    let stop = Cancellation::stop_request();
    let request = stop.child();
    let read = request.child();
    assert!(read.check().is_ok());
    stop.cancel();
    assert!(matches!(read.check(), Err(StoreError::Stopped)));
    request.cancel();
    assert!(matches!(read.check(), Err(StoreError::ReadCancelled)));
}

#[test]
fn a_waker_registered_after_the_cancel_runs_at_once() {
    let stop = Cancellation::stop_request();
    stop.child().cancel();
    stop.cancel();
    let (woken, wait) = std::sync::mpsc::channel();
    let _wake = stop.child().on_cancel(move || woken.send(()).unwrap());
    assert_eq!(wait.try_recv(), Ok(()));
}
