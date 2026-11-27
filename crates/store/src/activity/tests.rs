use super::*;
use std::sync::mpsc::{Receiver, Sender, channel};

const THREAD: &str = "activity-test-statement";

/// The statement a connection opened by the store is running is listed
/// with the thread running it and that thread's work while it runs, and is
/// gone once it ends; the work goes when its guard does.
#[test]
fn a_running_statement_is_listed_with_its_thread_until_it_ends() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("activity.db");
    let (entered_tx, entered) = channel::<()>();
    let (release, release_rx) = channel::<()>();
    let (ended_tx, ended) = channel::<()>();
    let (exit, exit_rx) = channel::<()>();
    let worker = std::thread::Builder::new()
        .name(THREAD.into())
        .spawn(move || run_blocked_statement(&path, entered_tx, release_rx, &ended_tx, &exit_rx))
        .unwrap();

    entered.recv().unwrap();
    let running = find(THREAD).expect("the thread is listed while its statement runs");
    assert_eq!(
        (
            running
                .doing
                .iter()
                .map(|work| (work.what.as_str(), work.subject.as_deref()))
                .collect::<Vec<_>>(),
            running.seconds_since_progress.is_some(),
            running
                .statements
                .iter()
                .map(|statement| (statement.connection.as_str(), statement.sql.as_str()))
                .collect::<Vec<_>>(),
        ),
        (
            vec![
                ("testing", None),
                ("waiting in a statement", Some("the test's release")),
            ],
            true,
            vec![("writer", "SELECT count(*) FROM held WHERE test_wait(n) = 1")],
        )
    );

    release.send(()).unwrap();
    ended.recv().unwrap();
    let after = find(THREAD).expect("the thread still has work");
    assert_eq!(
        (
            after
                .doing
                .iter()
                .map(|work| work.what.as_str())
                .collect::<Vec<_>>(),
            after.statements.len()
        ),
        (vec!["testing"], 0)
    );

    exit.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(find(THREAD), None);
}

/// The calling thread reads the list and is left out of it.
#[test]
fn the_reading_thread_is_left_out() {
    let _doing = doing("reading the activity");
    let me = std::thread::current().name().unwrap().to_owned();
    assert_eq!(
        threads(Duration::ZERO)
            .into_iter()
            .find(|thread| thread.name == me),
        None
    );
}

fn find(name: &str) -> Option<Thread> {
    threads(Duration::ZERO)
        .into_iter()
        .find(|thread| thread.name == name)
}

fn run_blocked_statement(
    path: &std::path::Path,
    entered: Sender<()>,
    release: Receiver<()>,
    ended: &Sender<()>,
    exit: &Receiver<()>,
) {
    let conn = crate::archive::open_write(path, crate::archive::cache::SETUP).unwrap();
    conn.execute_batch("CREATE TABLE held(n INTEGER); INSERT INTO held VALUES (1)")
        .unwrap();
    let release = std::sync::Mutex::new(release);
    conn.create_scalar_function(
        "test_wait",
        1,
        rusqlite::functions::FunctionFlags::SQLITE_UTF8,
        move |_| {
            entered.send(()).unwrap();
            release.lock().unwrap().recv().unwrap();
            Ok(1)
        },
    )
    .unwrap();
    let _outer = doing("testing");
    {
        let _inner = doing_on("waiting in a statement", "the test's release");
        let counted: i64 = conn
            .query_row(
                "SELECT count(*) FROM held
                   WHERE test_wait(n) = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(counted, 1);
    }
    ended.send(()).unwrap();
    exit.recv().unwrap();
}

#[test]
fn slow_statements_keep_counts_and_leave_warnings_in_the_event_log() {
    crate::log!(warning, "synthetic warning before slow SQL");
    let _operation = doing("synthetic read");
    for n in 0..300 {
        finished_slow_statement("reader", "SELECT synthetic", Duration::from_secs(1 + n % 3));
    }
    let statements = slow_statements();
    let statement = statements
        .iter()
        .find(|statement| statement.sql == "SELECT synthetic")
        .unwrap();
    assert_eq!(
        (
            statement.count,
            statement.seconds,
            statement.longest_seconds
        ),
        (300, 3.0, 3.0)
    );
    assert_eq!(statement.operation.as_deref(), Some("synthetic read"));
    let warnings: Vec<_> = crate::log::recent(crate::log::KEPT)
        .into_iter()
        .filter(|record| record.text == "synthetic warning before slow SQL")
        .collect();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].priority, crate::log::Priority::Warning);
}
