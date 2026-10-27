use super::*;
use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

const T0: i64 = 1_790_000_000_000;

/// Sessions in `/home/op/proj` that ran `cargo test` and `cargo fmt`:
/// each run is (session, command, outcome, minute).
fn ran(runs: &[(&str, &str, &str, i64)]) -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut sessions = std::collections::BTreeMap::new();
    for (session, command, outcome, minute) in runs {
        let (s, m) = *sessions.entry(*session).or_insert_with(|| {
            let s = f
                .session(SessionFacts {
                    id: session,
                    cwd: Some("/home/op/proj"),
                    started: Some(T0),
                    last: Some(T0 + 3_600_000),
                    ..SessionFacts::default()
                })
                .unwrap();
            let m = f
                .message(
                    s,
                    MessageFacts {
                        id: session,
                        role: "assistant",
                        written_by: "model",
                        at: Some(T0),
                        ..MessageFacts::default()
                    },
                )
                .unwrap();
            (s, m)
        });
        f.tool_call(
            s,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some(command),
                outcome,
                exit_code: (*outcome == "error").then_some(101),
                at: Some(T0 + minute * 60_000),
                ..CallFacts::default()
            },
        )
        .unwrap();
    }
    f.derive().unwrap();
    (dir, f)
}

fn advise(f: &Fixture, command: &str) -> Prepared {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let now = Timestamp::from_unix_ms(T0 + 3_600_000).unwrap();
    advice(&reader, Path::new("/home/op/proj"), command, "d1", now).unwrap()
}

#[test]
fn a_check_that_failed_repeatedly_since_its_last_pass_gets_a_hint() {
    let (_dir, f) = ran(&[
        ("claude-code:s1", "cargo test", "ok", 1),
        ("claude-code:s1", "cargo test", "error", 2),
        ("claude-code:s2", "cargo test", "error", 3),
    ]);
    let p = advise(&f, "cargo test");
    assert!(!p.empty);
    assert_eq!(p.event, "advice");
    for expected in [
        "`cargo test` failed 2 times here",
        "across 2 sessions, with no pass since",
        "The last failure is in claude-code:s2, exit 101",
        "`polylogue read claude-code:s2`",
    ] {
        assert!(
            p.text.contains(expected),
            "missing {expected:?} in {}",
            p.text
        );
    }
    assert_eq!(p.refs, ["session:claude-code:s2"]);
}

#[test]
fn anything_less_certain_says_nothing() {
    let (_dir, f) = ran(&[
        ("claude-code:s1", "cargo test", "error", 1),
        ("claude-code:s1", "cargo fmt", "error", 2),
        ("claude-code:s1", "cargo fmt", "error", 3),
        ("claude-code:s1", "cargo fmt", "ok", 4),
    ]);
    // One failure is not a pattern.
    assert!(advise(&f, "cargo test").empty);
    // A later pass clears the failures before it.
    assert!(advise(&f, "cargo fmt").empty);
    // A command that runs no check, and a check never run here.
    assert!(advise(&f, "ls -la").empty);
    assert!(advise(&f, "cargo clippy").empty);
}
