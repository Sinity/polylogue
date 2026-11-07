use super::*;
use crate::daemon::scope::SharedScopeIndex;

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
fn recall_fuses_prose_and_tool_output_in_the_querys_order() {
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
        &SharedScopeIndex::default(),
        None,
        &ask,
        "d1",
        &fixture_context(Timestamp::from_unix_ms(0).unwrap()),
        None,
        None,
        None,
    )
    .unwrap();
    // Each list has one first-place hit. Equal ranks and unknown times
    // are ordered by semantic message identity, lowest first.
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
            id: "claude-code:p1",
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
                id: "claude-code:a1",
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
            id: "claude-code:a2",
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
                id: "claude-code:b1",
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
    let mut facts = polylogue_model::live::LiveFacts::default();
    polylogue_model::live::fold(
        &mut facts,
        "hook_task_claimed",
        T0 + 2_000,
        &serde_json::json!({"task":"pl-1"}),
    );
    f.live_facts(s, &facts).unwrap();
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
    let p = prepare(
        &reader,
        &scope_of(&reader),
        None,
        &ask,
        "d1",
        &fixture_context(now),
        None,
        None,
        None,
    )
    .unwrap();
    let t = &p.text;
    assert!(!p.empty);
    for expected in [
        "parser work",
        "1 commit (abc1234)",
        "changed 1 file: a.rs",
        "claimed pl-1, still open",
        "last answer: \"Finished the parser fix; one test still fails.\"",
        "Beads claimed here this week and still open:\n- pl-1",
        "formatting",
    ] {
        assert!(t.contains(expected), "missing {expected:?} in:\n{t}");
    }
    // A session's own failing commands are not reported.
    assert!(!t.contains("`cargo test`"), "{t}");
    assert!(t.len() <= DEFAULT_BUDGET as usize * 4);
    assert!(p.refs.contains(&"session:claude-code:s1".to_owned()));
}

#[test]
fn a_brief_fits_its_budget_and_names_what_it_left_out() {
    for budget in [200u64, 500, 1_000, 2_000, 8_000] {
        let mut fit = Fitter::new(budget, "Polylogue brief for /x (delivery d)\n".into()).unwrap();
        let items = |n: usize, width: usize| -> Vec<Item> {
            (0..n)
                .map(|i| Item {
                    text: format!("- item {i} {}", "x".repeat(width)),
                    refs: vec![format!("session:test:{i}")],
                })
                .collect()
        };
        fit.section(
            "Notes:",
            items(40, 90),
            "notes",
            "polylogue query assertions",
        );
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
    let mut first_next = None;
    loop {
        let ask = Ask::Recall {
            text: "wombat".into(),
            cwd: None,
            for_session: None,
            cursor: cursor.clone(),
        };
        let p = prepare(
            &reader,
            &SharedScopeIndex::default(),
            None,
            &ask,
            "d",
            &fixture_context(now),
            None,
            None,
            None,
        )
        .unwrap();
        seen.extend(p.refs.iter().filter(|r| r.starts_with("message:")).cloned());
        match p.next {
            Some(next) => {
                first_next.get_or_insert_with(|| next.clone());
                cursor = Some(next);
            }
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
        cursor: first_next,
    };
    assert!(
        prepare(
            &reader,
            &SharedScopeIndex::default(),
            None,
            &ask,
            "d",
            &fixture_context(now),
            None,
            None,
            None,
        )
        .is_err()
    );
}

/// The scope index over everything `reader` sees.
fn scope_of(reader: &Reader<'_>) -> SharedScopeIndex {
    let scope = SharedScopeIndex::default();
    scope.refresh(reader).unwrap();
    scope
}

/// law: brief-reads-bounded-by-result
/// A brief's archive reads depend on what it reports, not on how much work
/// happened under its directory: at `/`, under a prefix holding 30,000
/// sessions and over a million messages, in one of its subdirectories and in
/// a directory nobody worked in, a brief reads at most 100 pages through a
/// reader whose page cache starts empty.
#[test]
fn a_brief_reads_a_bounded_number_of_pages_at_any_directory() {
    use polylogue_store::fixture::{Fixture, ManySessions};
    const SESSIONS: usize = 30_000;
    const MESSAGES: usize = 34;
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    let pks = f
        .many_sessions(ManySessions {
            prefix: "/work/dense",
            sessions: SESSIONS,
            messages: MESSAGES,
            answered: 20,
            last: T0 - 60_000,
        })
        .unwrap();
    const { assert!(SESSIONS * MESSAGES >= 1_000_000) };
    // Three of the newest are working there now.
    for &pk in &pks[SESSIONS - 3..] {
        f.live(pk, "hook_pre_tool_use", T0 - 30_000, None, None)
            .unwrap();
    }
    let scope = scope_of(&f.reads().unwrap().reader().unwrap());
    let now = Timestamp::from_unix_ms(T0).unwrap();
    for cwd in ["/", "/work/dense", "/work/dense/d5", "/unvisited/place"] {
        // A new pool: its connections' page caches start empty.
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        reader.take_page_reads().unwrap();
        let p = prepare(
            &reader,
            &scope,
            None,
            &Ask::SessionStart {
                cwd: cwd.into(),
                session: None,
                source: StartSource::Startup,
            },
            "d1",
            &fixture_context(now),
            None,
            None,
            None,
        )
        .unwrap();
        let pages = reader.take_page_reads().unwrap();
        assert!(pages <= 100, "{cwd}: {pages} pages read\n{}", p.text);
        if cwd == "/unvisited/place" {
            assert!(p.empty, "{}", p.text);
            continue;
        }
        let newest = if cwd == "/work/dense/d5" {
            // The newest session in d5 is the highest n with n % 97 == 5.
            (0..SESSIONS).rev().find(|n| n % 97 == 5).unwrap()
        } else {
            SESSIONS - 1
        };
        for expected in [
            "Recent sessions here, and what they did:".to_owned(),
            format!("test:many-{newest}"),
            format!("({MESSAGES} messages)"),
            "older sessions here".to_owned(),
        ] {
            assert!(
                p.text.contains(&expected),
                "{cwd}: missing {expected:?} in\n{}",
                p.text
            );
        }
        if cwd != "/work/dense/d5" {
            for expected in [
                "Other agents working here now:".to_owned(),
                format!("test:many-{}  untitled: working", SESSIONS - 1),
                format!(
                    "last answer: \"Answer {}: the synthetic work is finished.\"",
                    SESSIONS - 1
                ),
            ] {
                assert!(
                    p.text.contains(&expected),
                    "{cwd}: missing {expected:?} in\n{}",
                    p.text
                );
            }
        }
    }
}

/// law: brief-partial-at-deadline
/// A brief that meets its deadline answers with the sections it has, names
/// each section it left out with reason `deadline`, and says how to get the
/// whole brief; it is not empty while a section produced items.
#[test]
fn a_brief_past_its_deadline_answers_with_the_sections_it_has() {
    use polylogue_store::read::limit::ReadLimit;
    let (_dir, f) = worked_directory();
    let pool = f.reads().unwrap();
    let scope = scope_of(&pool.reader().unwrap());
    let limit = ReadLimit::new(std::time::Duration::from_millis(200), Default::default());
    let p = limit.scope(|| {
        let reader = pool.reader().unwrap();
        while limit.check().is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        prepare(
            &reader,
            &scope,
            None,
            &Ask::SessionStart {
                cwd: "/home/op/proj".into(),
                session: None,
                source: StartSource::Startup,
            },
            "d1",
            &fixture_context(Timestamp::from_unix_ms(T0 + 3_600_000).unwrap()),
            None,
            None,
            None,
        )
        .unwrap()
    });
    assert!(!p.empty, "{}", p.text);
    // Open Beads come from the scope index and need no read.
    assert!(
        p.text
            .contains("Beads claimed here this week and still open:\n- pl-1"),
        "{}",
        p.text
    );
    let late: Vec<&str> = p
        .left_out
        .iter()
        .filter(|l| l.reason == "deadline")
        .map(|l| l.what.as_str())
        .collect();
    assert_eq!(
        late,
        [
            "operator notes",
            "recent sessions",
            "notices",
            "assertions by other authors"
        ]
    );
    assert!(
        p.text.contains("Not available before the deadline: operator notes, recent sessions, notices, assertions by other authors; the whole brief: `polylogue brief --cwd /home/op/proj --deadline none`"),
        "{}",
        p.text
    );
    assert!(!p.text.contains("Recent sessions here"), "{}", p.text);
}

fn fixture_context(now: Timestamp) -> std::sync::Arc<crate::operation::Context> {
    let day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    std::sync::Arc::new(crate::operation::Context::standalone(
        crate::operation::Invocation::new(
            now,
            "synthetic-briefing".into(),
            polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
        ),
        day,
    ))
}

#[test]
fn named_context_defaults_and_explicit_tight_budgets_share_one_fitter() {
    let (_dir, f) = worked_directory();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let ask = Ask::SessionStart {
        cwd: "/home/op/proj".into(),
        session: None,
        source: StartSource::Resume,
    };
    assert_eq!(ask.default_budget_tokens(), 4000);
    assert_eq!(
        Ask::Takeover {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            for_session: None
        }
        .default_budget_tokens(),
        10000
    );
    let prepared = prepare(
        &reader,
        &scope_of(&reader),
        None,
        &ask,
        "tight",
        &fixture_context(Timestamp::from_unix_ms(T0 + 3_600_000).unwrap()),
        None,
        None,
        Some(200),
    )
    .unwrap();
    assert!(prepared.text.len() <= 800, "{}", prepared.text);
    assert!(!prepared.left_out.is_empty());
    assert!(Fitter::new(0, "required header".into()).is_err());
}
