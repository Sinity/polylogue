//! Sequences follow session order, not provider time.

use super::*;

/// A session whose provider times run backwards: two failing and
/// succeeding `cargo test` pairs written in that order, each call stamped
/// earlier than the one before it.
fn backwards() -> TestArchive {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = fx
        .session(SessionFacts {
            id: "claude-code:backwards",
            started: at(1, 9),
            last: at(1, 18),
            ..SessionFacts::default()
        })
        .unwrap();
    let calls = [
        ("cargo test", "error", 101, 18),
        ("src/date.rs", "ok", 0, 17),
        ("cargo test", "ok", 0, 16),
        ("cargo test", "error", 101, 15),
        ("cargo test", "ok", 0, 14),
    ];
    for (n, (what, outcome, exit_code, hour)) in calls.into_iter().enumerate() {
        let m = fx
            .message(
                s,
                msg(&format!("b{n}"), "assistant", "model", "", at(1, hour)),
            )
            .unwrap();
        let edit = what.starts_with("src/");
        fx.tool_call(
            s,
            m,
            CallFacts {
                tool: if edit { "Edit" } else { "Bash" },
                action: if edit { "file_edit" } else { "shell" },
                command: (!edit).then_some(what),
                path: edit.then_some(what),
                outcome,
                exit_code: Some(exit_code),
                at: at(1, hour),
                output: Some("done"),
            },
        )
        .unwrap();
    }
    TestArchive { _dir: dir, fx }
}

/// `seq(...)` and "a later success of the same command" match by order in
/// the session, and their match rows are listed in that order, however
/// the provider's times run.
#[test]
fn sequences_follow_session_order_when_times_run_backwards() {
    let w = backwards();
    let fail_then_pass = "sessions where seq(command:\"cargo test\" AND outcome:error -> command:\"cargo test\" AND outcome:ok)";
    assert_eq!(ids(&q(&w, fail_then_pass)), ["claude-code:backwards"]);
    let v = q(&w, &format!("{fail_then_pass} | matches"));
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "{v}");
    let order: Vec<(u64, &str, &str)> = rows
        .iter()
        .map(|r| {
            (
                r["match"].as_u64().unwrap(),
                r["step1_message"].as_str().unwrap(),
                r["step2_message"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(order, [(1, "b0", "b2"), (2, "b3", "b4")]);
    // The first pair's success came one call (the edit) after its failure,
    // though its time is two hours earlier.
    assert_eq!(rows[0]["gap_steps"], 1);
    assert_eq!(rows[0]["gap_time"], -2 * 3_600_000);
    // In session order a success never comes before the first failure.
    let pass_then_fail = "sessions where seq(command:\"cargo test\" AND outcome:ok -> [next] command:\"cargo test\" AND outcome:error)";
    let v = q(&w, &format!("{pass_then_fail} | matches"));
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{v}");
    assert_eq!(rows[0]["step1_message"], "b2");
    assert_eq!(rows[0]["step2_message"], "b3");
    // A time window needs the later call to be later in time too.
    let within = "sessions where seq(command:\"cargo test\" AND outcome:error -> [within:5h] command:\"cargo test\" AND outcome:ok)";
    assert_eq!(q(&w, within)["outcome"], "empty");
}
