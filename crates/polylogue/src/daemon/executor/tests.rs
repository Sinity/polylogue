use super::*;
use std::sync::mpsc::{Receiver, Sender, channel};

const WAIT: Duration = Duration::from_secs(10);

/// Holds a place until `release` is sent.
fn hold(executor: &Executor, class: Class) -> (Receiver<()>, Sender<()>) {
    let (started, on_start) = channel();
    let (release, released) = channel::<()>();
    assert!(executor.submit(class, None, move || {
        started.send(()).unwrap();
        let _ = released.recv();
    }));
    on_start.recv_timeout(WAIT).expect("held work started");
    (on_start, release)
}

#[test]
fn interactive_work_runs_while_every_ordinary_and_background_place_is_held() {
    let executor = Executor::new(2);
    let held: Vec<_> = (0..2).map(|_| hold(&executor, Class::Ordinary)).collect();
    let background = hold(&executor, Class::Background);
    let (done, finished) = channel();
    assert!(executor.submit(Class::Interactive, None, move || done.send(()).unwrap()));
    finished
        .recv_timeout(WAIT)
        .expect("interactive work took the reserved place");
    // Ordinary work waits: the reserved place is not an ordinary one.
    let (ordinary, ran) = channel();
    assert!(executor.submit(Class::Ordinary, None, move || ordinary.send(()).unwrap()));
    assert_eq!(executor.queued(), 1);
    drop(held);
    drop(background);
    ran.recv_timeout(WAIT)
        .expect("ordinary work ran once a place freed");
}

#[test]
fn queued_work_starts_in_deadline_order() {
    let executor = Executor::new(1);
    let (_started, release) = hold(&executor, Class::Ordinary);
    let (order, ran) = channel();
    let now = Instant::now();
    for (name, deadline) in [
        ("none", None),
        ("late", Some(now + Duration::from_secs(5))),
        ("soon", Some(now + Duration::from_millis(500))),
    ] {
        let order = order.clone();
        assert!(executor.submit(Class::Ordinary, deadline, move || order.send(name).unwrap()));
    }
    release.send(()).unwrap();
    let ran: Vec<_> = (0..3).map(|_| ran.recv_timeout(WAIT).unwrap()).collect();
    assert_eq!(ran, ["soon", "late", "none"]);
}

#[test]
fn work_waiting_outside_gives_its_place_to_queued_work() {
    let executor = Executor::new(1);
    let (waiting, is_waiting) = channel();
    let (wake, woken) = channel::<()>();
    let (resumed, has_resumed) = channel();
    assert!(executor.submit(Class::Ordinary, None, move || {
        outside(|| {
            waiting.send(()).unwrap();
            woken.recv().unwrap();
        });
        resumed.send(()).unwrap();
    }));
    is_waiting.recv_timeout(WAIT).unwrap();
    let (other, ran) = channel();
    assert!(executor.submit(Class::Ordinary, None, move || other.send(()).unwrap()));
    ran.recv_timeout(WAIT)
        .expect("queued work ran in the place given up");
    wake.send(()).unwrap();
    has_resumed.recv_timeout(WAIT).unwrap();
}

#[test]
fn stopping_drops_queued_work_and_refuses_new_work() {
    let executor = Executor::new(1);
    let (_started, release) = hold(&executor, Class::Ordinary);
    let (sender, receiver) = tokio::sync::oneshot::channel::<()>();
    assert!(executor.submit(Class::Ordinary, None, move || {
        let _ = sender.send(());
    }));
    let stopper = std::thread::spawn({
        let shared = Arc::clone(&executor.shared);
        move || Executor { shared }.stop(WAIT)
    });
    // The queued job is dropped unrun: its answer channel closes.
    assert!(receiver.blocking_recv().is_err());
    assert!(!executor.submit(Class::Ordinary, None, || {}));
    release.send(()).unwrap();
    stopper.join().unwrap();
    assert_eq!(executor.threads(), 0);
}

#[test]
fn work_that_ignores_its_cancellation_does_not_hold_up_a_stop() {
    let executor = Executor::new(1);
    let (_started, _never_released) = hold(&executor, Class::Ordinary);
    let started = Instant::now();
    assert_eq!(executor.stop(Duration::from_millis(200)), 1);
    assert!(started.elapsed() < WAIT, "the stop waited past its bound");
}
