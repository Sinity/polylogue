use super::{Selection, drift, quota, survival, tool_calls, usage};
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use polylogue_model::{QuotaLimit, QuotaReading, QuotaWindow, Usage, money::Amount};
use serde_json::json;
use std::borrow::Cow;

const DAY: i64 = 86_400_000;

/// The tools with a subscription quota, as the origin table declares them.
const KNOWN: &[(&str, &[QuotaLimit])] = &[
    (
        "claude-code",
        &[
            QuotaLimit {
                limit: Cow::Borrowed("five_hour"),
                minutes: 300,
                from_first_response: true,
            },
            QuotaLimit {
                limit: Cow::Borrowed("seven_day"),
                minutes: 10_080,
                from_first_response: false,
            },
        ],
    ),
    ("codex", &[]),
];

/// A quota reading of one window, as a parser stores it.
fn reading(
    limit: &str,
    minutes: i64,
    from_first_response: bool,
    used: Option<f64>,
    resets_at: i64,
    status: Option<&str>,
) -> serde_json::Value {
    serde_json::to_value(QuotaReading {
        windows: vec![QuotaWindow {
            limit: QuotaLimit {
                limit: limit.to_owned().into(),
                minutes,
                from_first_response,
            },
            used_percent: used,
            resets_at: polylogue_model::Timestamp::from_unix_ms(resets_at),
            status: status.map(str::to_owned),
        }],
        ..QuotaReading::default()
    })
    .unwrap()
}
const T0: i64 = 1_790_000_000_000;

fn tokens(input: u64, cached: u64, write: Option<u64>, output: u64) -> Usage {
    Usage {
        input: Some(input),
        cached_input: Some(cached),
        cache_write: write,
        output: Some(output),
        ..Usage::default()
    }
}

fn model_message(f: &mut Fixture, s: i64, id: &str, model: &str, at: i64, u: &Usage) -> i64 {
    let pk = f
        .message(
            s,
            MessageFacts {
                id,
                role: "assistant",
                written_by: "model",
                at: Some(at),
                model: Some(model),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    f.usage(pk, u, None).unwrap();
    f.derive().unwrap();
    pk
}

fn session(f: &mut Fixture, id: &str) -> i64 {
    f.session(SessionFacts {
        id,
        kind: "main",
        ..SessionFacts::default()
    })
    .unwrap()
}

/// Each response's usage counts once; archive totals equal the sum
/// over distinct messages, whatever the sessions show.
#[test]
fn usage_counts_each_response_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let parent = session(&mut f, "claude-code:p");
    let child = session(&mut f, "claude-code:c");
    model_message(
        &mut f,
        parent,
        "claude-code:a",
        "claude-opus-4-1",
        T0,
        &tokens(10, 100, Some(5), 20),
    );
    model_message(
        &mut f,
        child,
        "claude-code:b",
        "claude-opus-4-1",
        T0 + 1,
        &tokens(1, 2, Some(3), 4),
    );
    f.link("claude-code:p", "claude-code:c", "continuation", None)
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let u = usage(&r, &Selection::default()).unwrap();
    assert_eq!(u.messages, Some(2));
    assert_eq!(u.tokens.input.tokens, 11);
    assert_eq!(u.tokens.cached_input.tokens, 102);
    assert_eq!(u.tokens.cache_write.tokens, 8);
    assert_eq!(u.tokens.output.tokens, 24);
    // Reasoning was not reported apart: unknown for both, never zero.
    assert_eq!(u.tokens.reasoning.tokens, 0);
    assert_eq!(u.tokens.reasoning.unknown_messages, 2);
    let by_session: u64 = [parent, child]
        .into_iter()
        .map(|pk| {
            usage(
                &r,
                &Selection {
                    sessions: Some(vec![pk]),
                    ..Selection::default()
                },
            )
            .unwrap()
            .tokens
            .output
            .tokens
        })
        .sum();
    assert_eq!(by_session, u.tokens.output.tokens);
}

/// An unpriced model's cost is unknown, never 0, and totals say how
/// many messages are unpriced; a partly priced selection is partial; credits are
/// never added to USD.
#[test]
fn unpriced_models_stay_unknown_and_credits_stay_apart() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let priced = session(&mut f, "claude-code:priced");
    let unpriced = session(&mut f, "codex:unpriced");
    let mixed = session(&mut f, "claude-code:mixed");
    model_message(
        &mut f,
        priced,
        "claude-code:p1",
        "claude-opus-4-1",
        T0,
        &tokens(1_000_000, 0, Some(0), 0),
    );
    let mut credits = reading("codex", 10_080, false, Some(10.0), T0 + DAY, None);
    credits["plan"] = json!("pro");
    credits["credits"] = json!("38816.6156015000");
    let pk = model_message(
        &mut f,
        unpriced,
        "codex:u1",
        "gpt-9-imaginary",
        T0,
        &tokens(5, 0, None, 5),
    );
    f.usage(pk, &tokens(5, 0, None, 5), Some(credits)).unwrap();
    f.derive().unwrap();
    model_message(
        &mut f,
        mixed,
        "claude-code:m1",
        "claude-opus-4-1",
        T0,
        &tokens(1_000_000, 0, Some(0), 0),
    );
    model_message(
        &mut f,
        mixed,
        "claude-code:m2",
        "gpt-9-imaginary",
        T0,
        &tokens(1, 0, None, 1),
    );
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();

    let only = |pk| Selection {
        sessions: Some(vec![pk]),
        ..Selection::default()
    };
    let u = usage(&r, &only(unpriced)).unwrap();
    assert_eq!(u.api_cost.known, None, "unpriced must be unknown, not 0");
    assert_eq!(u.api_cost.unpriced_messages, 1);
    assert_eq!(u.unpriced_models, vec![("gpt-9-imaginary".to_owned(), 1)]);

    let m = usage(&r, &only(mixed)).unwrap();
    assert!(m.api_cost.partial());
    assert_eq!(m.api_cost.known, Some(Amount(15_000_000_000)));
    assert_eq!(m.api_cost.unpriced_messages, 1);

    let all = usage(&r, &Selection::default()).unwrap();
    assert_eq!(all.api_cost.known, Some(Amount(30_000_000_000)));
    assert_eq!(all.api_cost.unpriced_messages, 2);

    // Credits are reported in their own place, never as dollars.
    let o = quota::quota(&r, T0 + 1, KNOWN).unwrap();
    let codex = o.tools.iter().find(|t| t.tool == "codex").unwrap();
    assert_eq!(codex.credits.as_ref().unwrap().balance, "38816.6156015000");
    assert_eq!(codex.plan.as_deref(), Some("pro"));
    let window = &codex.windows[0];
    assert_eq!(window.usage.as_ref().unwrap().api_cost.known, None);
}

/// Hiding a session never changes a usage figure; the figures say
/// a hidden session is included.
#[test]
fn hiding_never_changes_a_usage_figure() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let a = session(&mut f, "claude-code:a");
    let b = session(&mut f, "claude-code:b");
    model_message(
        &mut f,
        a,
        "claude-code:a1",
        "claude-opus-4-1",
        T0,
        &tokens(100, 0, Some(0), 10),
    );
    model_message(
        &mut f,
        b,
        "claude-code:b1",
        "claude-opus-4-1",
        T0 + 60_000,
        &tokens(200, 0, Some(0), 20),
    );
    let now = T0 + 120_000;
    let read = |f: &Fixture| {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        (
            usage(&r, &Selection::default()).unwrap(),
            quota::quota(&r, now, KNOWN).unwrap(),
        )
    };
    let (u0, o0) = read(&f);
    assert_eq!(u0.hidden_sessions, 0);
    f.note("hidden", None, None, "session:claude-code:b")
        .unwrap();
    let (u1, o1) = read(&f);
    assert_eq!(u1.hidden_sessions, 1);
    assert_eq!(
        (u0.tokens, u0.api_cost, u0.messages),
        (u1.tokens.clone(), u1.api_cost.clone(), u1.messages)
    );
    let five = |o: &quota::Quota| {
        o.tools[0]
            .windows
            .iter()
            .find(|w| w.window_minutes == quota::FIVE_HOURS)
            .unwrap()
            .usage
            .clone()
            .unwrap()
    };
    assert_eq!(five(&o0).tokens, five(&o1).tokens);
    assert_eq!(five(&o1).hidden_sessions, 1);
}

#[test]
fn quota_reads_reported_windows_and_projects_exhaustion() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let codex = session(&mut f, "codex:s");
    let reset = T0 + 3 * 3_600_000;
    for (i, pct) in [10.0, 30.0].into_iter().enumerate() {
        let at = T0 + i as i64 * 3_600_000;
        let pk = model_message(
            &mut f,
            codex,
            &format!("codex:c{i}"),
            "gpt-5",
            at,
            &tokens(10, 0, None, 10),
        );
        let q = reading("codex", 300, false, Some(pct), reset, None);
        f.usage(pk, &tokens(10, 0, None, 10), Some(q)).unwrap();
        f.derive().unwrap();
    }
    let claude = session(&mut f, "claude-code:s");
    let rejected = reading(
        "seven_day",
        10_080,
        false,
        Some(100.0),
        T0 + 2 * DAY,
        Some("rejected"),
    );
    let pk = model_message(
        &mut f,
        claude,
        "claude-code:x",
        "claude-opus-4-1",
        T0,
        &tokens(1, 0, Some(0), 1),
    );
    f.usage(pk, &tokens(1, 0, Some(0), 1), Some(rejected))
        .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let o = quota::quota(&r, T0 + 3_600_000 + 1, KNOWN).unwrap();

    let cw = &o.tools.iter().find(|t| t.tool == "codex").unwrap().windows[0];
    assert_eq!(cw.source, quota::WindowSource::Reported);
    assert_eq!(
        (cw.used_percent, cw.remaining_percent),
        (Some(30.0), Some(70.0))
    );
    assert_eq!(cw.burn_percent_per_hour, Some(20.0));
    // 70% left at 20%/h runs out 3.5 h after the last report, after the
    // reset 2 h later: the window lasts.
    assert!(cw.lasts_to_reset);
    assert_eq!(cw.exhausted_at, None);

    let cc = o.tools.iter().find(|t| t.tool == "claude-code").unwrap();
    let week = cc
        .windows
        .iter()
        .find(|w| w.window_minutes == quota::WEEK)
        .unwrap();
    assert_eq!(week.source, quota::WindowSource::Reported);
    assert_eq!(week.status.as_deref(), Some("rejected"));
    assert_eq!(week.exhausted_at, Some(T0));
    let five = cc
        .windows
        .iter()
        .find(|w| w.window_minutes == quota::FIVE_HOURS)
        .unwrap();
    assert_eq!(five.source, quota::WindowSource::Inferred);
    assert_eq!(five.used_percent, None, "nothing reported: unknown, not 0");
}

/// Responses every two hours from half past the hour of `T0`, 21 of them,
/// each block starting at the hour of the first response after the previous
/// block ended: blocks begin at hours 0, 6, 12, ... of the chain. At 40.75 h
/// the window is 36 h to 41 h, though only responses of the last two
/// windows (from 30.75 h) would put it at 38 h to 43 h.
#[test]
fn the_five_hour_block_phase_follows_the_chain_start() {
    let hour = 3_600_000;
    let h0 = T0 - T0.rem_euclid(hour);
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let claude = session(&mut f, "claude-code:s");
    for k in 0..21 {
        let at = h0 + hour / 2 + 2 * hour * k;
        model_message(
            &mut f,
            claude,
            &format!("claude-code:r{k}"),
            "claude-opus-4-1",
            at,
            &tokens(1, 0, Some(0), 1),
        );
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let o = quota::quota(&r, h0 + 40 * hour + 3 * hour / 4, KNOWN).unwrap();
    let five = o.tools[0]
        .windows
        .iter()
        .find(|w| w.window_minutes == quota::FIVE_HOURS)
        .unwrap();
    assert_eq!(five.source, quota::WindowSource::Inferred);
    assert_eq!(
        (five.starts_at, five.resets_at),
        (h0 + 36 * hour, h0 + 41 * hour)
    );
}

/// A reading whose reset is earlier than the reading is ignored and counted;
/// the window shows the last good reading.
#[test]
fn a_stale_reading_is_counted_and_never_shown() {
    let hour = 3_600_000;
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let codex = session(&mut f, "codex:s");
    for (i, (at, pct, reset)) in [
        (T0 + 2 * hour, 20.0, T0 + 4 * hour),
        (T0 + 3 * hour, 90.0, T0 + hour),
    ]
    .into_iter()
    .enumerate()
    {
        let pk = model_message(
            &mut f,
            codex,
            &format!("codex:c{i}"),
            "gpt-5",
            at,
            &tokens(10, 0, None, 10),
        );
        let q = reading("codex", 300, false, Some(pct), reset, None);
        f.usage(pk, &tokens(10, 0, None, 10), Some(q)).unwrap();
        f.derive().unwrap();
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let o = quota::quota(&r, T0 + 3 * hour + 1, KNOWN).unwrap();
    let tool = o.tools.iter().find(|t| t.tool == "codex").unwrap();
    assert_eq!(tool.stale_readings, 1);
    assert_eq!(tool.windows[0].used_percent, Some(20.0));
}

#[test]
fn quota_projects_exhaustion_before_the_reset() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let codex = session(&mut f, "codex:s");
    let reset = T0 + 3 * 3_600_000;
    let half_hour = 30 * 60_000;
    for (i, pct) in [80.0, 90.0].into_iter().enumerate() {
        let at = T0 + i as i64 * half_hour;
        let pk = model_message(
            &mut f,
            codex,
            &format!("codex:c{i}"),
            "gpt-5",
            at,
            &tokens(10, 0, None, 10),
        );
        let q = reading("codex", 300, false, Some(pct), reset, None);
        f.usage(pk, &tokens(10, 0, None, 10), Some(q)).unwrap();
        f.derive().unwrap();
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let o = quota::quota(&r, T0 + half_hour + 1, KNOWN).unwrap();
    let cw = &o.tools.iter().find(|t| t.tool == "codex").unwrap().windows[0];
    // 10 points in half an hour is 20%/h; the last 10% lasts another half hour.
    assert_eq!(cw.burn_percent_per_hour, Some(20.0));
    assert_eq!(cw.exhausted_at, Some(T0 + 2 * half_hour));
    assert!(!cw.lasts_to_reset);
}

#[test]
fn process_keeps_transcript_order_and_outcomes_apart() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = session(&mut f, "claude-code:p");
    let m = f
        .message(
            s,
            MessageFacts {
                id: "claude-code:m",
                role: "assistant",
                written_by: "model",
                at: Some(T0),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    for (cmd, outcome) in [
        ("cargo test", "error"),
        ("cargo build", "ok"),
        ("cargo test", "ok"),
    ] {
        f.tool_call(
            s,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some(cmd),
                outcome,
                at: Some(T0),
                ..CallFacts::default()
            },
        )
        .unwrap();
    }
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let p = tool_calls::process(&r, &Selection::default(), 10).unwrap();
    assert_eq!((p.sessions, p.calls), (1, 3));
    assert_eq!(p.outcomes.error, 1);
    assert_eq!(p.outcomes.ok, 2);
    assert_eq!(p.failure_examples.len(), 1);
    assert_eq!(p.common_sequences.len(), 1);
    let s = survival::first_failure(&r, &Selection::default(), false).unwrap();
    assert_eq!((s.sessions, s.events), (1, 1));
}

/// Tokens at two grains: a fork's figures with parents include what it shows
/// from its parent; its own figures hold only its own responses, so
/// own figures over sessions still add up to the archive's.
#[test]
fn a_fork_shows_its_parents_usage_at_the_logical_grain_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let parent = session(&mut f, "claude-code:p");
    let child = session(&mut f, "claude-code:c");
    model_message(
        &mut f,
        parent,
        "claude-code:a",
        "claude-opus-5-5",
        T0,
        &tokens(10, 0, None, 1),
    );
    model_message(
        &mut f,
        parent,
        "claude-code:b",
        "claude-opus-5-5",
        T0 + 1,
        &tokens(20, 0, None, 2),
    );
    model_message(
        &mut f,
        child,
        "claude-code:d",
        "claude-opus-5-5",
        T0 + 2,
        &tokens(40, 0, None, 4),
    );
    f.link("claude-code:p", "claude-code:c", "fork", None)
        .unwrap();
    crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute(
            "UPDATE session_link SET branch_point='claude-code:b' WHERE parent_id='claude-code:p' AND child_id='claude-code:c'",
            [],
        )
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let all = super::sessions(&r, &Selection::default()).unwrap();
    let get = |id: &str| all.iter().find(|s| s.session == id).unwrap();
    let (p, c) = (get("claude-code:p"), get("claude-code:c"));
    assert_eq!(
        (p.own.messages, p.with_parents.messages),
        (Some(2), Some(2)),
        "a session with no parent shows only its own"
    );
    assert_eq!(
        (c.own.messages, c.with_parents.messages),
        (Some(1), Some(3))
    );
    assert_eq!(c.own.tokens.input.tokens, 40);
    assert_eq!(c.with_parents.tokens.input.tokens, 70);
    let own: u64 = all.iter().map(|s| s.own.tokens.input.tokens).sum();
    assert_eq!(
        own,
        usage(&r, &Selection::default())
            .unwrap()
            .tokens
            .input
            .tokens
    );
    // Priced under the current catalog, at both grains.
    assert!(c.with_parents.api_cost.known.unwrap() > c.own.api_cost.known.unwrap());
}

fn shell(f: &mut Fixture, s: i64, m: i64, cmd: &str, outcome: &str, at: i64) {
    f.tool_call(
        s,
        m,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some(cmd),
            path: None,
            outcome,
            exit_code: (outcome == "error").then_some(1),
            at: Some(at),
            output: Some("…"),
        },
    )
    .unwrap();
}

fn in_dir(f: &mut Fixture, id: &str, cwd: &str) -> i64 {
    f.session(SessionFacts {
        id,
        kind: "main",
        cwd: Some(cwd),
        ..SessionFacts::default()
    })
    .unwrap()
}

/// Failing commands: per repository, commands whose latest decided run
/// failed; a later success resolves a failure, an unknown outcome decides
/// nothing, and two repositories never mix.
#[test]
fn failing_holds_commands_whose_latest_run_failed() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let a1 = in_dir(&mut f, "claude-code:a1", "/work/a");
    let a2 = in_dir(&mut f, "claude-code:a2", "/work/a");
    let b1 = in_dir(&mut f, "claude-code:b1", "/work/b");
    let m = |f: &mut Fixture, s, id, at| {
        model_message(f, s, id, "claude-opus-5-5", at, &tokens(1, 0, None, 1))
    };
    let (ma1, ma2, mb1) = (
        m(&mut f, a1, "claude-code:m1", T0),
        m(&mut f, a2, "claude-code:m2", T0 + DAY),
        m(&mut f, b1, "claude-code:m3", T0),
    );
    shell(&mut f, a1, ma1, "cargo test", "error", T0);
    shell(&mut f, a2, ma2, "cargo test", "ok", T0 + DAY);
    shell(&mut f, a1, ma1, "cargo clippy", "error", T0 + 1);
    shell(&mut f, a2, ma2, "cargo clippy", "error", T0 + DAY + 1);
    shell(&mut f, a2, ma2, "cargo clippy", "unknown", T0 + DAY + 2);
    shell(&mut f, b1, mb1, "cargo test", "error", T0 + 5);
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let got: Vec<_> = tool_calls::failing_commands(&r, &Selection::default())
        .unwrap()
        .into_iter()
        .map(|u| (u.repository, u.command, u.failures_since_ok, u.session))
        .collect();
    assert_eq!(
        got,
        [
            (
                "cwd:/work/a".into(),
                "cargo clippy".into(),
                2,
                "claude-code:a2".into()
            ),
            (
                "cwd:/work/b".into(),
                "cargo test".into(),
                1,
                "claude-code:b1".into()
            ),
        ]
    );
}

fn failing_runs(runs: &[(&str, i64)]) -> Vec<tool_calls::FailingCommand> {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = in_dir(&mut fixture, "test:commands", "/work/morrow");
    let message = fixture
        .message(
            session,
            MessageFacts {
                id: "test:calls",
                role: "assistant",
                written_by: "model",
                at: Some(T0),
                ..Default::default()
            },
        )
        .unwrap();
    for (index, &(command, exit_code)) in runs.iter().enumerate() {
        fixture
            .tool_call(
                session,
                message,
                CallFacts {
                    tool: "Bash",
                    action: "shell",
                    command: Some(command),
                    path: None,
                    outcome: if exit_code == 0 { "ok" } else { "error" },
                    exit_code: Some(exit_code),
                    at: Some(T0 + index as i64),
                    output: Some("synthetic output"),
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    tool_calls::failing_commands(&reader, &Selection::default()).unwrap()
}

#[test]
fn failing_ignores_inline_function_names() {
    let failures = failing_runs(&[("R=x; g(){ cd a; git -C b status; }; g", 1)]);
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].command, "git status");
    assert_eq!(failures[0].exit_code, Some(1));
    assert_eq!(failures[0].failures_since_ok, 1);
}

#[test]
fn failing_ignores_grep_no_match() {
    assert_eq!(
        failing_runs(&[
            ("grep missing file", 1),
            ("egrep missing file", 1),
            ("fgrep missing file", 1),
            ("rg missing", 1),
            ("ag missing", 1),
            ("which missing", 1),
            ("test -e missing", 1),
            ("[ -e missing ]", 1),
            ("git -C repo grep missing", 1),
        ]),
        []
    );
    let failures = failing_runs(&[
        ("/usr/bin/grep pattern unreadable", 2),
        ("grep missing file", 1),
    ]);
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].command, "grep");
    assert_eq!(failures[0].exit_code, Some(2));
    assert_eq!(failures[0].failures_since_ok, 1);
    assert_eq!(failures[0].last_ok_at, None);
}

#[test]
fn failing_reports_cargo_test_101() {
    let failures = failing_runs(&[("cargo test -p parser", 101)]);
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].command, "cargo test");
    assert_eq!(failures[0].exit_code, Some(101));
    assert_eq!(failures[0].failures_since_ok, 1);
}

/// Ending on an unanswered prompt: an event only once the session has been
/// quiet for the quiet period; answered and recent sessions are censored;
/// sessions no human prompted are not in the cohort.
#[test]
fn an_unanswered_last_prompt_ends_a_session_only_after_it_goes_quiet() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let human = |f: &mut Fixture, s, id, at| {
        f.message(
            s,
            MessageFacts {
                id,
                role: "user",
                written_by: "human",
                prompt_kind: Some("prose"),
                at: Some(at),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    };
    let answered = session(&mut f, "claude-code:answered");
    human(&mut f, answered, "claude-code:h1", T0);
    model_message(
        &mut f,
        answered,
        "claude-code:a1",
        "claude-opus-5-5",
        T0 + 60_000,
        &tokens(1, 0, None, 1),
    );
    let left = session(&mut f, "claude-code:left");
    human(&mut f, left, "claude-code:h2", T0);
    model_message(
        &mut f,
        left,
        "claude-code:a2",
        "claude-opus-5-5",
        T0 + 60_000,
        &tokens(1, 0, None, 1),
    );
    human(&mut f, left, "claude-code:h3", T0 + 120_000);
    let recent = session(&mut f, "claude-code:recent");
    let now = T0 + 10 * DAY;
    human(&mut f, recent, "claude-code:h4", now - 60_000);
    let sub = session(&mut f, "claude-code:sub");
    model_message(
        &mut f,
        sub,
        "claude-code:a4",
        "claude-opus-5-5",
        T0,
        &tokens(1, 0, None, 1),
    );
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let s = survival::unanswered_end(&r, &Selection::default(), now).unwrap();
    assert_eq!(
        (s.sessions, s.events, s.censored, s.unmeasured),
        (3, 1, 2, 1)
    );
}

/// Format drift lists what parsers did not know, most frequent first, with
/// items kept as counts by design after the rest.
#[test]
fn drift_lists_frequent_unknowns_first_and_design_counts_last() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.unrecognized("claude-code", "field:message.effort", 3, false)
        .unwrap();
    f.unrecognized("claude-code", "skipped:launched", 9, false)
        .unwrap();
    f.unrecognized("codex", "skipped:event_msg/item_completed", 50, true)
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let got: Vec<_> = drift::drift(&r)
        .unwrap()
        .into_iter()
        .map(|d| (d.what, d.count, d.by_design))
        .collect();
    assert_eq!(
        got,
        [
            ("skipped:launched".into(), 9, false),
            ("field:message.effort".into(), 3, false),
            ("skipped:event_msg/item_completed".into(), 50, true),
        ]
    );
}

/// Survival reads per-session facts, which carry no message times, so a
/// time-bounded selection is refused rather than silently ignored.
#[test]
fn survival_refuses_a_time_bounded_scope() {
    let dir = tempfile::tempdir().unwrap();
    let f = Fixture::new(dir.path()).unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let selection = Selection {
        since: Some(T0),
        ..Selection::default()
    };
    assert!(survival::first_failure(&r, &selection, false).is_err());
}

/// Status's quota: each limit's newest reading within its window, a
/// declared limit nothing reported as `no_reading`, and a measurement whose
/// window has ended as `window_reset` (docs/operations.md#status).
#[test]
fn current_windows_hold_the_newest_reading_of_each_limit() {
    use quota::{CurrentUse, CurrentWindow, NotMeasured};
    let ts = |ms| polylogue_model::Timestamp::from_unix_ms(ms).unwrap();
    let minute = 60_000;
    let hour = 60 * minute;
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let codex = session(&mut f, "codex:s");
    for (id, at, used) in [("codex:a", T0, 10.0), ("codex:b", T0 + 30 * minute, 25.0)] {
        let pk = model_message(&mut f, codex, id, "gpt-5", at, &tokens(1, 0, None, 1));
        let q = reading("codex", 300, false, Some(used), T0 + 2 * hour, None);
        f.usage(pk, &tokens(1, 0, None, 1), Some(q)).unwrap();
    }
    let claude = session(&mut f, "claude-code:s");
    let pk = model_message(
        &mut f,
        claude,
        "claude-code:x",
        "claude-opus-4-1",
        T0 + 10 * minute,
        &tokens(1, 0, Some(0), 1),
    );
    let rejected = reading(
        "seven_day",
        10_080,
        false,
        Some(100.0),
        T0 + 2 * DAY,
        Some("rejected"),
    );
    f.usage(pk, &tokens(1, 0, Some(0), 1), Some(rejected))
        .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let window = |tool: &str, limit: &str, minutes, state| CurrentWindow {
        tool: tool.into(),
        limit: limit.into(),
        window_minutes: minutes,
        state,
    };
    let five_hour = window(
        "claude-code",
        "five_hour",
        300,
        CurrentUse::NotMeasured {
            reason: NotMeasured::NoReading,
        },
    );
    let seven_day = window(
        "claude-code",
        "seven_day",
        10_080,
        CurrentUse::Measured {
            used_percent: 100.0,
            measured_at: ts(T0 + 10 * minute),
            resets_at: Some(ts(T0 + 2 * DAY)),
            status: Some("rejected".into()),
        },
    );
    assert_eq!(
        quota::current(&r, T0 + hour, KNOWN).unwrap(),
        vec![
            five_hour.clone(),
            seven_day.clone(),
            window(
                "codex",
                "codex",
                300,
                CurrentUse::Measured {
                    used_percent: 25.0,
                    measured_at: ts(T0 + 30 * minute),
                    resets_at: Some(ts(T0 + 2 * hour)),
                    status: None,
                },
            ),
        ]
    );
    assert_eq!(
        quota::current(&r, T0 + 3 * hour, KNOWN).unwrap(),
        vec![
            five_hour,
            seven_day,
            window(
                "codex",
                "codex",
                300,
                CurrentUse::NotMeasured {
                    reason: NotMeasured::WindowReset,
                },
            ),
        ]
    );
    // Eight days on, nothing is within the longest window.
    assert_eq!(
        quota::current(&r, T0 + 8 * DAY, KNOWN)
            .unwrap()
            .into_iter()
            .map(|window| (window.tool, window.limit, window.state))
            .collect::<Vec<_>>(),
        vec![
            (
                "claude-code".to_owned(),
                "five_hour".to_owned(),
                CurrentUse::NotMeasured {
                    reason: NotMeasured::NoReading
                }
            ),
            (
                "claude-code".to_owned(),
                "seven_day".to_owned(),
                CurrentUse::NotMeasured {
                    reason: NotMeasured::NoReading
                }
            ),
        ]
    );
}

mod reports;
