use crate::fixture::writer::*;
use polylogue_model::{Json, RepoLookup, Session, SessionKind};
use serde_json::json;

fn one(w: &WriterWorld, sql: &str) -> Json {
    w.rows(sql).into_iter().next().unwrap_or(Json::Null)
}

#[test]
fn every_derivation_runs_after_what_it_reads() {
    for (i, d) in super::ALL.iter().enumerate() {
        for a in d.after {
            assert!(
                super::ALL[..i].iter().any(|x| x.name == *a),
                "{} reads {a}, which does not run before it",
                d.name
            );
        }
    }
}

#[test]
fn a_shell_call_s_command_is_normalized_by_the_derive_step() {
    let mut w = WriterWorld::new();
    w.write("a", bash("a", 1, 1_000, "cargo test -p x", true, "ok"));
    assert_eq!(
        one(&w, "SELECT action, command_norm FROM tool_call"),
        json!({"action": "shell", "command_norm": "cargo test"})
    );
    // Nothing changed: nothing is derived again.
    assert_eq!(w.writer.derive().unwrap(), 0);
}

#[test]
fn a_call_left_behind_has_passed_without_result() {
    let mut w = WriterWorld::new();
    let mut msgs = call(
        "a",
        1,
        1_000,
        "Bash",
        Some("sleep 1"),
        Json::Null,
        Ok("unused"),
    );
    // Drop the result: the call never gets one.
    msgs.truncate(1);
    w.write("a", msgs);
    assert_eq!(
        one(&w, "SELECT outcome_reason FROM tool_call"),
        json!({"outcome_reason": "no_result"})
    );
    // A later model response: the harness moved on without a result.
    w.write("a", bash("a", 2, 2_000, "true", true, ""));
    assert_eq!(
        w.rows("SELECT outcome_reason FROM tool_call ORDER BY at"),
        vec![
            json!({"outcome_reason": "passed_without_result"}),
            json!({"outcome_reason": null})
        ]
    );
}

#[test]
fn a_session_without_a_title_is_named_by_its_first_prompt() {
    let mut w = WriterWorld::new();
    w.write(
        "a",
        prose("a", 1, 1_000, "\n  Fix the flaky crash test\nplease"),
    );
    assert_eq!(
        one(&w, "SELECT title, derived_title FROM session"),
        json!({"title": null, "derived_title": "Fix the flaky crash test"})
    );
}

#[test]
fn a_session_takes_its_repository_from_the_nearest_lookup() {
    let mut w = WriterWorld::new();
    let mut s = Session::new(sid("a"), SessionKind::Main);
    s.offer_cwd("/r/project/x/sub");
    w.lookups.push(RepoLookup {
        path: "/r/project/x".into(),
        remote: Some("https://example.com/x".into()),
        root_commit: Some("abc".into()),
    });
    w.write_session(s.clone(), prose("a", 1, 1_000, "hi"));
    assert_eq!(
        one(&w, "SELECT repo, repo_root FROM session"),
        json!({"repo": "https://example.com/x", "repo_root": "abc"})
    );
    // The lookup changes: the session is attributed again.
    w.lookups.push(RepoLookup {
        path: "/r/project/x".into(),
        remote: Some("https://example.com/y".into()),
        root_commit: Some("abc".into()),
    });
    w.write_session(s, vec![]);
    assert_eq!(
        one(&w, "SELECT repo FROM session"),
        json!({"repo": "https://example.com/y"})
    );
}

#[test]
fn a_repository_the_records_name_wins_over_the_lookup() {
    let mut w = WriterWorld::new();
    let mut s = Session::new(sid("a"), SessionKind::Main);
    s.offer_cwd("/r/project/x");
    s.repo = Some("git@own:repo".into());
    w.lookups.push(RepoLookup {
        path: "/r/project/x".into(),
        remote: Some("https://example.com/x".into()),
        root_commit: Some("abc".into()),
    });
    w.write_session(s, prose("a", 1, 1_000, "hi"));
    assert_eq!(
        one(&w, "SELECT repo, repo_root FROM session"),
        json!({"repo": "git@own:repo", "repo_root": "abc"})
    );
}

/// Two inputs about one session, written in the given order.
fn two_inputs(early_first: bool) -> Json {
    let mut w = WriterWorld::new();
    let mut early = Session::new(sid("s"), SessionKind::Subagent);
    early.offer_cwd("/a");
    early.offer_title("early", polylogue_model::TitleSource::Generated);
    early.git_branch = Some("main".into());
    early.tool_version = Some("1.0".into());
    let mut late = Session::new(sid("s"), SessionKind::Main);
    late.offer_cwd("/b");
    late.offer_cwd("/a");
    late.offer_title("late", polylogue_model::TitleSource::Generated);
    late.git_branch = Some("dev".into());
    let writes = [
        ("in-early", early, prose("s", 1, 1_000, "first")),
        ("in-late", late, prose("s", 2, 5_000, "second")),
    ];
    let order: Vec<usize> = if early_first { vec![0, 1] } else { vec![1, 0] };
    for i in order {
        let (input, s, m) = writes[i].clone();
        w.write_from(input, s, m);
    }
    one(
        &w,
        "SELECT kind, title, title_source, cwd_json, cwd, branch, harness_version FROM session",
    )
}

#[test]
fn session_facts_do_not_depend_on_arrival_order() {
    // Law 4 for a session's own facts: kind from the earliest input, an
    // equal-precedence title and the branch from the latest, working
    // directories by first use.
    let expected = json!({
        "kind": "subagent", "title": "late", "title_source": "generated",
        "cwd_json": "[\"/a\",\"/b\"]", "cwd": "/b", "branch": "dev", "harness_version": "1.0"
    });
    assert_eq!(two_inputs(true), expected);
    assert_eq!(two_inputs(false), expected);
}
