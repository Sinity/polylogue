use super::*;

#[test]
fn a_checkout_root_joins_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("proj/repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("crates/x")).unwrap();
    let dirs = scope_dirs(&repo.join("crates/x"));
    assert_eq!(dirs.len(), 2);
    assert_eq!(dirs[1], repo.to_string_lossy());
}

#[test]
fn start_sources_parse_from_harness_names() {
    assert_eq!(StartSource::parse("compact"), StartSource::Compact);
    assert_eq!(StartSource::parse("anything"), StartSource::Startup);
    assert_eq!(StartSource::parse("resume"), StartSource::Resume);
}

#[test]
fn recall_finds_prose_then_tool_output_as_a_query_sees_them() {
    use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:s1",
            title: Some("quokka work"),
            ..SessionFacts::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "m1",
            role: "user",
            written_by: "human",
            text: Some("the quokka migration plan"),
            ..MessageFacts::default()
        },
    )
    .unwrap();
    let m2 = f
        .message(
            s,
            MessageFacts {
                id: "m2",
                role: "assistant",
                written_by: "model",
                ..MessageFacts::default()
            },
        )
        .unwrap();
    f.tool_call(
        s,
        m2,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("ls"),
            outcome: "ok",
            output: Some("quokka.txt"),
            ..CallFacts::default()
        },
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let ask = Ask::Recall {
        text: "Quokka!".into(),
        cwd: None,
        for_session: None,
        cursor: None,
    };
    let p = prepare(
        &reader,
        &ask,
        "d1",
        Timestamp::from_unix_ms(0).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        p.refs,
        ["session:test:s1", "message:m1", "message:m2"],
        "{}",
        p.text
    );
    assert!(
        p.text.contains("(prose) the quokka migration plan"),
        "{}",
        p.text
    );
    assert!(p.text.contains("(output) quokka.txt"), "{}", p.text);
    let search_summary = p.search_summary.unwrap();
    assert_eq!(
        (
            search_summary.hits,
            search_summary.sessions,
            search_summary.more
        ),
        (2, 1, false)
    );
}

const T0: i64 = 1_790_856_000_000;

/// A directory with one session that edited a file, failed `cargo test`,
/// claimed a Bead, committed, and answered; and a second that failed and
/// then passed `cargo fmt`.
fn worked_directory() -> (tempfile::TempDir, polylogue_store::fixture::Fixture) {
    use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = f
        .session(SessionFacts {
            id: "claude-code:s1",
            title: Some("parser work"),
            cwd: Some("/home/op/proj"),
            started: Some(T0),
            last: Some(T0 + 60_000),
            ..SessionFacts::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "p1",
            role: "user",
            written_by: "human",
            prompt_kind: Some("prose"),
            text: Some("fix the parser"),
            at: Some(T0),
            ..MessageFacts::default()
        },
    )
    .unwrap();
    let a = f
        .message(
            s,
            MessageFacts {
                id: "a1",
                role: "assistant",
                written_by: "model",
                at: Some(T0 + 1_000),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    for (command, path, outcome, exit, at, output) in [
        (
            Some("bd update pl-1 --claim"),
            None,
            "ok",
            None,
            2_000,
            Some("claimed"),
        ),
        (
            None,
            Some("/home/op/proj/src/a.rs"),
            "ok",
            None,
            3_000,
            None,
        ),
        (
            Some("cargo test"),
            None,
            "error",
            Some(101),
            4_000,
            Some("failed"),
        ),
        (
            Some("git commit -m fix"),
            None,
            "ok",
            None,
            5_000,
            Some("[main abc1234f] fix\n 1 file changed"),
        ),
    ] {
        f.tool_call(
            s,
            a,
            CallFacts {
                tool: if path.is_some() { "Edit" } else { "Bash" },
                action: if path.is_some() { "file_edit" } else { "shell" },
                command,
                path,
                outcome,
                exit_code: exit,
                at: Some(T0 + at),
                output,
            },
        )
        .unwrap();
    }
    f.message(
        s,
        MessageFacts {
            id: "a2",
            role: "assistant",
            written_by: "model",
            text: Some("Finished the parser fix; one test still fails."),
            at: Some(T0 + 6_000),
            ..MessageFacts::default()
        },
    )
    .unwrap();
    let s2 = f
        .session(SessionFacts {
            id: "claude-code:s2",
            title: Some("formatting"),
            cwd: Some("/home/op/proj/sub"),
            started: Some(T0 - 60_000),
            last: Some(T0 - 30_000),
            ..SessionFacts::default()
        })
        .unwrap();
    let m = f
        .message(
            s2,
            MessageFacts {
                id: "b1",
                role: "assistant",
                written_by: "model",
                at: Some(T0 - 60_000),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    for (outcome, at) in [("error", -50_000), ("ok", -40_000)] {
        f.tool_call(
            s2,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some("cargo fmt"),
                outcome,
                at: Some(T0 + at),
                ..CallFacts::default()
            },
        )
        .unwrap();
    }
    f.derive().unwrap();
    // The development tooling's run history: a failed check in the
    // checkout, and a test run there that failed and then passed.
    let history = dir.path().join("history.db");
    let runs = rusqlite::Connection::open(&history).unwrap();
    runs.execute_batch(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY, project TEXT NOT NULL, checkout TEXT NOT NULL,
            command TEXT NOT NULL, args TEXT NOT NULL, agentctl_job TEXT, started_ms INTEGER NOT NULL,
            status TEXT NOT NULL, failure TEXT);",
    )
    .unwrap();
    for (command, status, at) in [
        ("check", "failed", 7_000),
        ("test", "failed", 8_000),
        ("test", "passed", 9_000),
    ] {
        runs.execute(
            "INSERT INTO runs(project, checkout, command, args, agentctl_job, started_ms, status, failure)
             VALUES ('proj', '/home/op/proj', ?1, '[]', 'proj-' || ?1, ?3, ?2, 'clippy failed')",
            rusqlite::params![command, status, T0 + at],
        )
        .unwrap();
    }
    f.setting("briefing.check_history", &history.to_string_lossy())
        .unwrap();
    (dir, f)
}

#[test]
fn a_brief_reports_what_recent_sessions_did_and_what_is_open() {
    let (_dir, f) = worked_directory();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let ask = Ask::SessionStart {
        cwd: "/home/op/proj".into(),
        session: None,
        source: StartSource::Startup,
    };
    let now = Timestamp::from_unix_ms(T0 + 3_600_000).unwrap();
    let p = prepare(&reader, &ask, "d1", now, None).unwrap();
    let t = &p.text;
    assert!(!p.empty);
    for expected in [
        "parser work",
        "1 commit (abc1234)",
        "changed 1 file: a.rs",
        "checks 0 passed, 1 failed",
        "claimed pl-1, still open",
        "last answer: \"Finished the parser fix; one test still fails.\"",
        "Beads claimed here this week and still open:\n- pl-1",
        "Checks that failed here with no later pass:\n- `proj check` failed",
        "in /home/op/proj, job proj-check: clippy failed; `xtask history failures 1`\n- `cargo test` exit 101",
        "formatting",
    ] {
        assert!(t.contains(expected), "missing {expected:?} in:\n{t}");
    }
    // `cargo fmt` passed after it failed.
    assert!(!t.contains("`cargo fmt`"), "{t}");
    assert!(t.len() <= DEFAULT_BUDGET as usize * 4);
    assert!(p.refs.contains(&"session:claude-code:s1".to_owned()));
}

#[test]
fn a_brief_fits_its_budget_and_names_what_it_left_out() {
    for budget in [200u64, 500, 1_000, 2_000, 8_000] {
        let mut fit = Fitter::new(budget, "Polylogue brief for /x (delivery d)\n".into());
        let items = |n: usize, width: usize| -> Vec<Item> {
            (0..n)
                .map(|i| Item {
                    text: format!("- item {i} {}", "x".repeat(width)),
                    refs: vec![format!("session:test:{i}")],
                })
                .collect()
        };
        fit.section("Notes:", items(40, 90), "notes", "polylogue note list");
        fit.section(
            "Sessions:",
            items(40, 300),
            "recent sessions",
            "polylogue list",
        );
        let (text, refs, left_out) = fit.finish("Earlier work: `polylogue recall WORDS`.");
        assert!(
            text.len() <= budget as usize * 4,
            "budget {budget}: {} bytes",
            text.len()
        );
        // Every item is either in the text with its reference or counted
        // as left out.
        let shown = refs.len();
        let cut: usize = left_out
            .iter()
            .map(|l| l.what.split(' ').next().unwrap().parse::<usize>().unwrap())
            .sum();
        assert_eq!(shown + cut, 80, "budget {budget}");
        if cut > 0 {
            assert!(text.contains("Left out for the budget:"), "{text}");
        }
    }
}

#[test]
fn recall_pages_continue_by_cursor() {
    use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:s1",
            ..SessionFacts::default()
        })
        .unwrap();
    for i in 0..45 {
        f.message(
            s,
            MessageFacts {
                id: &format!("m{i:02}"),
                role: "user",
                written_by: "human",
                text: Some("the wombat burrow"),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let now = Timestamp::from_unix_ms(0).unwrap();
    let mut seen = Vec::new();
    let mut cursor = None;
    loop {
        let ask = Ask::Recall {
            text: "wombat".into(),
            cwd: None,
            for_session: None,
            cursor: cursor.clone(),
        };
        let p = prepare(&reader, &ask, "d", now, None).unwrap();
        seen.extend(p.refs.iter().filter(|r| r.starts_with("message:")).cloned());
        match p.next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    assert_eq!(seen.len(), 45);
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 45, "no hit repeats across pages");
    // A cursor belongs to its text.
    let ask = Ask::Recall {
        text: "other".into(),
        cwd: None,
        for_session: None,
        cursor: Some(recall_cursor(20, "wombat", None)),
    };
    assert!(prepare(&reader, &ask, "d", now, None).is_err());
}
