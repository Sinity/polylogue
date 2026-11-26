use super::*;
use polylogue_store::log::{Priority, recent};

fn unreadable(message: &str) -> Failure {
    Failure::stated(
        "/home/op/a.jsonl",
        Cause::Environment,
        Code::Io,
        format!("read: {message}"),
    )
}

fn logged() -> Vec<(Priority, String)> {
    recent(usize::MAX)
        .into_iter()
        .map(|record| (record.priority, record.text))
        .collect()
}

#[test]
fn a_failure_is_logged_when_it_starts_and_when_it_clears_not_while_it_lasts() {
    let mut log = FailureLog::default();
    let a: HashSet<PathBuf> = [PathBuf::from("/home/op/a.jsonl")].into();
    let b: HashSet<PathBuf> = [PathBuf::from("/home/op/b.jsonl")].into();
    let looked = |files, walked| Looked {
        files,
        walked,
        completed: true,
    };
    // Its sentence differs every pass; it is the same condition.
    for pass in 0..10 {
        log.pass_ended(
            &[unreadable(&format!("pass {pass}: permission denied"))],
            looked(&a, false),
        );
    }
    // Passes over other files neither repeat nor clear it.
    log.pass_ended(&[], looked(&b, false));
    log.pass_ended(&[], looked(&b, false));
    assert_eq!(
        logged(),
        [(
            Priority::Warning,
            "/home/op/a.jsonl: read: pass 0: permission denied (environment, io)".to_owned()
        )]
    );
    // A walk that leaves it alone does not clear it; a pass that reads it
    // without failing does, once.
    log.pass_ended(&[], looked(&b, true));
    log.pass_ended(&[], looked(&a, false));
    log.pass_ended(&[], looked(&a, true));
    assert_eq!(
        logged()[1..],
        [(
            Priority::Info,
            "/home/op/a.jsonl: no longer failing (io)".to_owned()
        )]
    );
}

#[test]
fn a_failed_turn_is_an_error_until_a_turn_completes() {
    let mut log = FailureLog::default();
    let none = HashSet::new();
    let turn = Failure::stated("", Cause::Transient, Code::Busy, "database is locked");
    for _ in 0..3 {
        log.pass_ended(
            std::slice::from_ref(&turn),
            Looked {
                files: &none,
                walked: false,
                completed: false,
            },
        );
    }
    log.pass_ended(
        &[],
        Looked {
            files: &none,
            walked: false,
            completed: true,
        },
    );
    assert_eq!(
        logged(),
        [
            (
                Priority::Error,
                "a pass failed: database is locked (transient, busy)".to_owned()
            ),
            (Priority::Info, "passes complete again (busy)".to_owned()),
        ]
    );
}

#[test]
fn a_file_error_stays_until_the_file_is_read_without_the_error() {
    let mut log = FailureLog::default();
    let none = HashSet::new();
    let file = [PathBuf::from("/home/op/a.jsonl")].into();
    let failure = unreadable("permission denied");
    log.pass_ended(
        std::slice::from_ref(&failure),
        Looked {
            files: &file,
            walked: false,
            completed: true,
        },
    );
    for _ in 0..3 {
        log.pass_ended(
            &[],
            Looked {
                files: &none,
                walked: true,
                completed: true,
            },
        );
        assert_eq!(log.errors(), std::slice::from_ref(&failure));
    }
    log.pass_ended(
        &[],
        Looked {
            files: &file,
            walked: false,
            completed: true,
        },
    );
    assert_eq!(log.errors(), []);
}
