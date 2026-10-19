//! Laws for measurement (plnext-r1i.6): L14, L70, L138 here; L109 with the
//! window rules in `outlook`.

use super::{Scope, analysis, outlook, usage};
use crate::fixture::{Fixture, M, S, T};
use polylogue_model::Usage;
use serde_json::json;

const DAY: i64 = 86_400_000;
const T0: i64 = 1_790_000_000_000;

fn lanes(input: u64, cached: u64, write: Option<u64>, output: u64) -> Usage {
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
            M {
                id,
                role: "assistant",
                written_by: "model",
                at: Some(at),
                model: Some(model),
                ..M::default()
            },
        )
        .unwrap();
    f.usage(pk, u, None).unwrap();
    pk
}

fn session(f: &mut Fixture, id: &str) -> i64 {
    f.session(S {
        id,
        kind: "main",
        ..S::default()
    })
    .unwrap()
}

/// Law 14: each response's usage counts once; archive totals equal the sum
/// over distinct messages, whatever the sessions show.
#[test]
fn law_14_usage_counts_each_response_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let parent = session(&mut f, "claude-code:p");
    let child = session(&mut f, "claude-code:c");
    model_message(
        &mut f,
        parent,
        "a",
        "claude-opus-4-1",
        T0,
        &lanes(10, 100, Some(5), 20),
    );
    model_message(
        &mut f,
        child,
        "b",
        "claude-opus-4-1",
        T0 + 1,
        &lanes(1, 2, Some(3), 4),
    );
    f.link("claude-code:p", "claude-code:c", "continuation", "test")
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let u = usage(&r, &Scope::default()).unwrap();
    assert_eq!(u.messages, 2);
    assert_eq!(u.lanes.input.tokens, 11);
    assert_eq!(u.lanes.cached_input.tokens, 102);
    assert_eq!(u.lanes.cache_write.tokens, 8);
    assert_eq!(u.lanes.output.tokens, 24);
    // Reasoning was not reported apart: unknown for both, never zero.
    assert_eq!(u.lanes.reasoning.tokens, 0);
    assert_eq!(u.lanes.reasoning.unknown_messages, 2);
    let by_session: u64 = [parent, child]
        .into_iter()
        .map(|pk| {
            usage(
                &r,
                &Scope {
                    sessions: Some(vec![pk]),
                    ..Scope::default()
                },
            )
            .unwrap()
            .lanes
            .output
            .tokens
        })
        .sum();
    assert_eq!(by_session, u.lanes.output.tokens);
}

/// Law 70: an unpriced model's cost is unknown, never 0, and totals say how
/// many messages are unpriced; a partly priced scope is partial; credits are
/// never added to USD.
#[test]
fn law_70_unpriced_models_and_credits() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let priced = session(&mut f, "claude-code:priced");
    let unpriced = session(&mut f, "codex:unpriced");
    let mixed = session(&mut f, "claude-code:mixed");
    model_message(
        &mut f,
        priced,
        "p1",
        "claude-opus-4-1",
        T0,
        &lanes(1_000_000, 0, Some(0), 0),
    );
    let credits = json!({"limit_id": "codex", "plan_type": "pro",
        "credits": {"balance": "38816.6156015000"},
        "primary": {"used_percent": 10.0, "window_minutes": 10080, "resets_at": (T0 + DAY) / 1000}});
    let pk = model_message(
        &mut f,
        unpriced,
        "u1",
        "gpt-9-imaginary",
        T0,
        &lanes(5, 0, None, 5),
    );
    f.usage(pk, &lanes(5, 0, None, 5), Some(credits)).unwrap();
    model_message(
        &mut f,
        mixed,
        "m1",
        "claude-opus-4-1",
        T0,
        &lanes(1_000_000, 0, Some(0), 0),
    );
    model_message(
        &mut f,
        mixed,
        "m2",
        "gpt-9-imaginary",
        T0,
        &lanes(1, 0, None, 1),
    );
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();

    let only = |pk| Scope {
        sessions: Some(vec![pk]),
        ..Scope::default()
    };
    let u = usage(&r, &only(unpriced)).unwrap();
    assert_eq!(
        u.api_cost.known_usd, None,
        "unpriced must be unknown, not 0"
    );
    assert_eq!(u.api_cost.unpriced_messages, 1);
    assert_eq!(u.unpriced_models, vec![("gpt-9-imaginary".to_owned(), 1)]);

    let m = usage(&r, &only(mixed)).unwrap();
    assert!(m.api_cost.partial());
    assert_eq!(m.api_cost.known_usd, Some(15.0));
    assert_eq!(m.api_cost.unpriced_messages, 1);

    let all = usage(&r, &Scope::default()).unwrap();
    assert_eq!(all.api_cost.known_usd, Some(30.0));
    assert_eq!(all.api_cost.unpriced_messages, 2);

    // Credits are reported in their own place, never as dollars.
    let o = outlook::outlook(&r, T0 + 1).unwrap();
    let codex = o.tools.iter().find(|t| t.tool == "codex").unwrap();
    assert_eq!(codex.credits.as_ref().unwrap().balance, "38816.6156015000");
    assert_eq!(codex.plan.as_deref(), Some("pro"));
    let window = &codex.windows[0];
    assert_eq!(window.usage.api_cost.known_usd, None);
}

/// Law 138: hiding a session never changes a usage figure; the figures say
/// a hidden session is included.
#[test]
fn law_138_hiding_never_changes_a_usage_figure() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let a = session(&mut f, "claude-code:a");
    let b = session(&mut f, "claude-code:b");
    model_message(
        &mut f,
        a,
        "a1",
        "claude-opus-4-1",
        T0,
        &lanes(100, 0, Some(0), 10),
    );
    model_message(
        &mut f,
        b,
        "b1",
        "claude-opus-4-1",
        T0 + 60_000,
        &lanes(200, 0, Some(0), 20),
    );
    let now = T0 + 120_000;
    let read = |f: &Fixture| {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        (
            usage(&r, &Scope::default()).unwrap(),
            outlook::outlook(&r, now).unwrap(),
        )
    };
    let (u0, o0) = read(&f);
    assert_eq!(u0.hidden_sessions, 0);
    f.note("hidden", None, None, "session:claude-code:b")
        .unwrap();
    let (u1, o1) = read(&f);
    assert_eq!(u1.hidden_sessions, 1);
    assert_eq!(
        (u0.lanes, u0.api_cost, u0.messages),
        (u1.lanes.clone(), u1.api_cost.clone(), u1.messages)
    );
    let five = |o: &outlook::Outlook| {
        o.tools[0]
            .windows
            .iter()
            .find(|w| w.window_minutes == outlook::FIVE_HOURS)
            .unwrap()
            .usage
            .clone()
    };
    assert_eq!(five(&o0).lanes, five(&o1).lanes);
    assert_eq!(five(&o1).hidden_sessions, 1);
}

#[test]
fn outlook_reads_reported_windows_and_projects_exhaustion() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let codex = session(&mut f, "codex:s");
    let reset = T0 + 3 * 3_600_000;
    for (i, pct) in [10.0, 30.0].into_iter().enumerate() {
        let at = T0 + i as i64 * 3_600_000;
        let pk = model_message(
            &mut f,
            codex,
            &format!("c{i}"),
            "gpt-5",
            at,
            &lanes(10, 0, None, 10),
        );
        let q = json!({"primary": {"used_percent": pct, "window_minutes": 300,
                                   "resets_at": reset / 1000}});
        f.usage(pk, &lanes(10, 0, None, 10), Some(q)).unwrap();
    }
    let claude = session(&mut f, "claude-code:s");
    let rejected = json!({"status": "rejected", "resetsAt": (T0 + 2 * DAY) / 1000,
                          "rateLimitType": "seven_day"});
    let pk = model_message(
        &mut f,
        claude,
        "x",
        "claude-opus-4-1",
        T0,
        &lanes(1, 0, Some(0), 1),
    );
    f.usage(pk, &lanes(1, 0, Some(0), 1), Some(rejected))
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let o = outlook::outlook(&r, T0 + 3_600_000 + 1).unwrap();

    let cw = &o.tools.iter().find(|t| t.tool == "codex").unwrap().windows[0];
    assert_eq!(cw.basis, outlook::Basis::Reported);
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
        .find(|w| w.window_minutes == outlook::WEEK)
        .unwrap();
    assert_eq!(week.basis, outlook::Basis::Reported);
    assert_eq!(week.status.as_deref(), Some("rejected"));
    assert_eq!(week.exhausted_at, Some(T0));
    let five = cc
        .windows
        .iter()
        .find(|w| w.window_minutes == outlook::FIVE_HOURS)
        .unwrap();
    assert_eq!(five.basis, outlook::Basis::Inferred);
    assert_eq!(five.used_percent, None, "nothing reported: unknown, not 0");
}

#[test]
fn process_keeps_transcript_order_and_outcomes_apart() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = session(&mut f, "claude-code:p");
    let m = f
        .message(
            s,
            M {
                id: "m",
                role: "assistant",
                written_by: "model",
                at: Some(T0),
                ..M::default()
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
            T {
                tool: "Bash",
                action: "shell",
                command: Some(cmd),
                outcome,
                at: Some(T0),
                ..T::default()
            },
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let p = analysis::process(&r, &Scope::default(), 10).unwrap();
    assert_eq!((p.sessions, p.calls), (1, 3));
    assert_eq!(p.outcomes.error, 1);
    assert_eq!(p.outcomes.ok, 2);
    assert_eq!(p.failure_witnesses.len(), 1);
    assert_eq!(p.motifs.len(), 1);
    let s = analysis::first_failure(&r, &Scope::default(), false).unwrap();
    assert_eq!((s.sessions, s.events), (1, 1));
}

/// Tokens at two grains: a fork's logical figures include what it shows
/// from its parent; its physical figures hold only its own responses, so
/// physical figures over sessions still add up to the archive's.
#[test]
fn a_fork_shows_its_parents_usage_at_the_logical_grain_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let parent = session(&mut f, "claude-code:p");
    let child = session(&mut f, "claude-code:c");
    model_message(
        &mut f,
        parent,
        "a",
        "claude-opus-5-5",
        T0,
        &lanes(10, 0, None, 1),
    );
    model_message(
        &mut f,
        parent,
        "b",
        "claude-opus-5-5",
        T0 + 1,
        &lanes(20, 0, None, 2),
    );
    model_message(
        &mut f,
        child,
        "d",
        "claude-opus-5-5",
        T0 + 2,
        &lanes(40, 0, None, 4),
    );
    f.link("claude-code:p", "claude-code:c", "continuation", "fork")
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let all = super::sessions(&r, &Scope::default()).unwrap();
    let get = |id: &str| all.iter().find(|s| s.session == id).unwrap();
    let (p, c) = (get("claude-code:p"), get("claude-code:c"));
    assert_eq!(
        (p.physical.messages, p.logical.messages),
        (2, 2),
        "a session with no parent shows only its own"
    );
    assert_eq!((c.physical.messages, c.logical.messages), (1, 3));
    assert_eq!(c.physical.lanes.input.tokens, 40);
    assert_eq!(c.logical.lanes.input.tokens, 70);
    let physical: u64 = all.iter().map(|s| s.physical.lanes.input.tokens).sum();
    assert_eq!(
        physical,
        usage(&r, &Scope::default()).unwrap().lanes.input.tokens
    );
    // Priced under the current catalog, at both grains.
    assert!(c.logical.api_cost.known_usd.unwrap() > c.physical.api_cost.known_usd.unwrap());
}
