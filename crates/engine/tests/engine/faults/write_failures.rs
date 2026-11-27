use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};
use std::collections::BTreeSet;
use std::io::Read;
use std::sync::Arc;

const POINTS: &[(&str, bool)] = &[
    ("write.archive_commit", false),
    ("capture.after_commit", true),
    ("write.index_commit", true),
];

#[test]
#[ignore = "child process of a_failed_write_step_loses_nothing"]
fn failed_write_child() {
    let base = std::path::PathBuf::from(std::env::var_os("POLYLOGUE_WRITE_BASE").unwrap());
    let point = std::env::var("POLYLOGUE_WRITE_POINT").unwrap();
    let mut w = ClaudeHome::new();
    w.home = base.join("home");
    w.root = base.join("archive");
    let (mut engine, _lock) = w.engine(64 << 20);
    if std::env::var("POLYLOGUE_WRITE_ACTION").unwrap() == "stop" {
        fault::arm(&point, "", Action::StopProcess, 0, 1);
        engine.run_pass().unwrap();
        panic!("the configured crash point was not reached: {point}");
    }
    // Every attempt fails, including the writer's attempts to split a group.
    fault::arm(&point, "", Action::Fail, 0, u32::MAX);
    let failed = engine.run_pass().unwrap();
    assert_eq!(
        fault::recorded(&point)
            .into_iter()
            .map(|(subject, _)| subject)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([String::new()]),
        "{point}: the armed fault did not fire"
    );
    assert_eq!(
        failed
            .errors
            .iter()
            .map(|error| (error.cause, error.code))
            .collect::<Vec<_>>(),
        [(polylogue_model::Cause::Transient, polylogue_model::Code::Io)],
        "{point}: a failed write must be reported"
    );
    // Exit without dropping the engine: no shutdown work may change the archive.
    std::process::exit(86);
}

#[allow(unsafe_code)]
fn interrupt_write(w: &ClaudeHome, point: &str, fail: bool) {
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "faults::write_failures::failed_write_child",
            "--ignored",
            "--nocapture",
        ])
        .env("POLYLOGUE_WRITE_BASE", w.dir.path())
        .env("POLYLOGUE_WRITE_POINT", point)
        .env("POLYLOGUE_WRITE_ACTION", if fail { "fail" } else { "stop" })
        .spawn()
        .unwrap();
    if fail {
        assert_eq!(
            child.wait().unwrap().code(),
            Some(86),
            "{point}: failed-write child"
        );
        return;
    }
    let mut status = 0;
    // The child stops the whole process at the write boundary; no destructor runs.
    assert_eq!(
        unsafe { libc::waitpid(child.id() as libc::pid_t, &mut status, libc::WUNTRACED) },
        child.id() as libc::pid_t
    );
    assert_eq!(
        (libc::WIFSTOPPED(status), libc::WSTOPSIG(status)),
        (true, libc::SIGSTOP),
        "child exited before {point}: {status}"
    );
    child.kill().unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
}

fn stored_inputs(w: &ClaudeHome) -> Vec<(String, Vec<u8>)> {
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let blobs = Arc::new(polylogue_store::Archive::at(&w.root).blobs().unwrap());
    let mut inputs = reader
        .known_watched_inputs("test-machine")
        .unwrap()
        .into_iter()
        .map(|input| {
            let chain = reader
                .stored_chain(&input.input_id, input.chain.unwrap(), input.stored)
                .unwrap();
            let mut bytes = Vec::new();
            chain.reader(blobs.clone()).read_to_end(&mut bytes).unwrap();
            let name = std::path::Path::new(&input.path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            (name, bytes)
        })
        .collect::<Vec<_>>();
    inputs.sort();
    inputs
}

#[test]
fn a_failed_write_step_loses_nothing() {
    check_reopened_archive(true);
}

#[test]
fn a_crash_at_a_write_boundary_loses_only_uncommitted_inputs() {
    check_reopened_archive(false);
}

fn check_reopened_archive(fail: bool) {
    for &(point, committed) in POINTS {
        let w = ClaudeHome::new();
        let first = claude_user("one", "first", "first committed body");
        let second = claude_user("two", "second", "second committed body");
        let pending = claude_user("three", "pending", "pending body");
        w.append("first.jsonl", &first);
        {
            let (mut engine, _lock) = w.engine(64 << 20);
            assert_eq!(engine.run_pass().unwrap().errors, []);
            w.append("second.jsonl", &second);
            assert_eq!(engine.run_pass().unwrap().errors, []);
        }
        w.append("pending.jsonl", &pending);
        let all_inputs = vec![
            ("first.jsonl".to_owned(), first.into_bytes()),
            ("pending.jsonl".to_owned(), pending.into_bytes()),
            ("second.jsonl".to_owned(), second.into_bytes()),
        ];
        let at_reopen = all_inputs
            .iter()
            .filter(|(name, _)| committed || name != "pending.jsonl")
            .cloned()
            .collect::<Vec<_>>();
        let mut uninterrupted = ClaudeHome::new();
        uninterrupted.home = w.home.clone();
        {
            let (mut engine, _lock) = uninterrupted.engine(64 << 20);
            assert_eq!(engine.run_pass().unwrap().errors, []);
        }
        let sessions = [
            "claude-code:first",
            "claude-code:second",
            "claude-code:pending",
        ];
        let expected = sessions.map(|id| uninterrupted.export(id));
        interrupt_write(&w, point, fail);
        let (mut engine, _lock) = w.engine(64 << 20);
        // Opening starts no update. Read the archive before the source is read again.
        assert_eq!(stored_inputs(&w), at_reopen, "{point}: archive at reopen");
        assert_eq!(engine.run_pass().unwrap().errors, []);
        assert_eq!(
            stored_inputs(&w),
            all_inputs,
            "{point}: archive after update"
        );
        assert_eq!(w.texts("first"), ["first committed body"]);
        assert_eq!(w.texts("second"), ["second committed body"]);
        assert_eq!(w.texts("pending"), ["pending body"]);
        assert_eq!(
            sessions.map(|id| w.export(id)),
            expected,
            "{point}: uninterrupted view"
        );
    }
}

/// On reopen after an interrupted index write, status still reflects the
/// published rows. The newly stored input is visible as work left to parse.
#[test]
fn status_is_correct_after_a_writer_killed_mid_transaction() {
    let w = ClaudeHome::new();
    w.append("first.jsonl", &claude_user("one", "first", "first body"));
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        assert_eq!(engine.run_pass().unwrap().errors, []);
    }

    let status = || w.read(|reader| reader.status(&polylogue_engine::reading_contracts()));
    let figures = |status: &polylogue_store::read::Status| {
        (
            status.sessions,
            status.sessions_physical,
            status.messages,
            status.documents,
            status.tool_calls,
        )
    };
    let before = status();
    assert_eq!(figures(&before), (1, 1, 1, 0, 0));

    w.append("second.jsonl", &claude_user("two", "second", "second body"));
    interrupt_write(&w, "write.index_commit", false);

    let (mut engine, _lock) = w.engine(64 << 20);
    let mut reopened = status();
    assert_eq!(figures(&reopened), figures(&before));
    assert_eq!(reopened.parse_states.remove("parsing"), Some(1));
    assert_eq!(reopened.parse_states, before.parse_states);

    assert_eq!(engine.run_pass().unwrap().errors, []);
    let after = status();
    assert_eq!(figures(&after), (2, 2, 2, 0, 0));
    assert_eq!(after.parse_states.get("done"), Some(&2));
    assert_eq!(w.texts("second"), ["second body"]);
}
