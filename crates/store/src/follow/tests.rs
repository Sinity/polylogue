use super::*;
use crate::completion::{CommitEvents, Committed};
use std::sync::{Arc, mpsc};
use std::time::Duration;

fn now() -> Timestamp {
    Timestamp::from_unix_ms(1_700_000_000_000).unwrap()
}

#[test]
fn repeated_failure_logs_once_and_recovery_once() {
    let events = CommitEvents::default();
    let subscription = events.subscribe();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let wake = subscription.wake_handle();
    let (sent, attempts) = mpsc::channel();
    let (changed, changes) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let mut attempt = 0;
        follow(
            "five failures test",
            Some(&subscription),
            &stopped,
            now,
            || changed.send(()).unwrap(),
            || {
                attempt += 1;
                sent.send(attempt).unwrap();
                if attempt <= 5 {
                    Err(StoreError::Failed(Failure::stated(
                        "",
                        Cause::Transient,
                        polylogue_model::Code::Busy,
                        "database busy",
                    )))
                } else {
                    Ok(())
                }
            },
        )
    });
    for attempt in 1..=5 {
        assert_eq!(attempts.recv().unwrap(), attempt);
        changes.recv().unwrap();
        events.committed(Committed::default());
    }
    assert_eq!(attempts.recv().unwrap(), 6);
    changes.recv().unwrap();
    stop.store(true, Ordering::Release);
    let _ = wake.try_send(());
    assert!(thread.join().unwrap());
    let lines: Vec<_> = crate::log::recent(crate::log::KEPT)
        .into_iter()
        .filter(|line| line.text.starts_with("five failures test:"))
        .map(|line| (line.priority, line.text))
        .collect();
    assert_eq!(
        lines,
        [
            (
                crate::log::Priority::Error,
                "five failures test: database busy".into()
            ),
            (
                crate::log::Priority::Info,
                "five failures test: recovered".into()
            ),
        ]
    );
}

#[test]
fn a_failing_worker_waits_for_a_commit() {
    let events = CommitEvents::default();
    let subscription = events.subscribe();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let wake = subscription.wake_handle();
    let (sent, attempts) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        follow(
            "waiting test",
            Some(&subscription),
            &stopped,
            now,
            || {},
            || {
                sent.send(()).unwrap();
                Err(StoreError::ReadIndexReplaced)
            },
        )
    });
    attempts.recv().unwrap();
    assert_eq!(
        attempts.recv_timeout(Duration::from_millis(150)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    events.committed(Committed::default());
    attempts.recv().unwrap();
    stop.store(true, Ordering::Release);
    let _ = wake.try_send(());
    assert!(thread.join().unwrap());
}

#[test]
fn unrecoverable_worker_error_halts_and_shows_halted() {
    for cause in [
        Cause::Environment,
        Cause::StoredData,
        Cause::Bug,
        Cause::Content,
        Cause::Request,
    ] {
        let events = CommitEvents::default();
        let subscription = events.subscribe();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let wake = subscription.wake_handle();
        let (changed, changes) = mpsc::channel();
        let (sent, attempts) = mpsc::channel();
        let error = Failure::stated("", cause, polylogue_model::Code::Refused, "cannot read");
        let expected = error.clone();
        let thread = std::thread::Builder::new()
            .name("halted test".into())
            .spawn(move || {
                follow(
                    "halted test",
                    Some(&subscription),
                    &stopped,
                    now,
                    || changed.send(()).unwrap(),
                    || {
                        sent.send(()).unwrap();
                        Err(StoreError::Failed(error.clone()))
                    },
                )
            })
            .unwrap();
        changes.recv().unwrap();
        attempts.recv().unwrap();
        let state = crate::activity::threads(Duration::ZERO)
            .into_iter()
            .find(|thread| thread.name == "halted test")
            .unwrap()
            .worker
            .unwrap();
        assert_eq!(state, Worker::Halted { error: expected });
        assert_eq!(
            state.problem("halted test").as_deref(),
            Some("halted test: halted: cannot read")
        );
        events.committed(Committed::default());
        assert_eq!(
            attempts.recv_timeout(Duration::from_millis(150)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        stop.store(true, Ordering::Release);
        let _ = wake.try_send(());
        assert!(thread.join().unwrap());
    }
}

#[test]
fn failure_lines_carry_error_priority() {
    const CHILD: &str = "POLYLOGUE_FOLLOW_JOURNAL_TEST";
    if std::env::var_os(CHILD).is_some() {
        follow(
            "priority test",
            None,
            &AtomicBool::new(false),
            now,
            || {},
            || Err(StoreError::internal("synthetic failure")),
        );
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "follow::tests::failure_lines_carry_error_priority",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "<3>polylogue: priority test: synthetic failure; this is a Polylogue bug; report it with the request ID\n"
    );
}

#[test]
fn a_changed_error_is_logged_once_more() {
    let events = CommitEvents::default();
    let subscription = events.subscribe();
    let stop = AtomicBool::new(false);
    let mut messages = [
        "database busy",
        "database busy",
        "reader busy",
        "reader busy",
    ]
    .into_iter();
    assert!(follow(
        "changed failure test",
        Some(&subscription),
        &stop,
        now,
        || {},
        || {
            events.committed(Committed::default());
            if let Some(message) = messages.next() {
                Err(StoreError::Failed(Failure::stated(
                    "",
                    Cause::Transient,
                    polylogue_model::Code::Busy,
                    message,
                )))
            } else {
                stop.store(true, Ordering::Release);
                Ok(())
            }
        },
    ));
    let lines: Vec<_> = crate::log::recent(crate::log::KEPT)
        .into_iter()
        .filter(|line| line.text.starts_with("changed failure test:"))
        .map(|line| (line.priority, line.text))
        .collect();
    assert_eq!(
        lines,
        [
            (
                crate::log::Priority::Error,
                "changed failure test: database busy".into()
            ),
            (
                crate::log::Priority::Error,
                "changed failure test: reader busy".into()
            ),
            (
                crate::log::Priority::Info,
                "changed failure test: recovered".into()
            ),
        ]
    );
}
