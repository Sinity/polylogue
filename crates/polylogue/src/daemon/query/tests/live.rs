//! The live view, the change feed and the signal notices as queries:
//! `sessions` with live fields, the `changes` unit and `deliveries`.

use super::fixture_context;
use super::run;
use crate::daemon::query::*;
use polylogue_model::{Event, Json, Output, Session, SessionId, SessionKind, Timestamp};
use polylogue_store::ReadPool;
use polylogue_store::fixture::writer::WriterArchive;
use polylogue_store::query::fields::{live_state, live_until};
use serde_json::json;

fn ask(text: &str, now: i64) -> Ask<'_> {
    Ask {
        text,
        cursor: None,
        page_size: Some(100),
        why: false,
        session: None,
        context: fixture_context(now, "UTC", 0),
        embeddings: false,
        ranking_identity: None,
        near: None,
        fields: &[],
    }
}

const MIN: i64 = 60_000;
/// 2026-10-18 14:00 UTC.
const T0: i64 = 1_792_332_000_000;

fn sid(n: &str) -> SessionId {
    SessionId::parse(&format!("claude-code:{n}")).unwrap()
}

/// Stores hook lines `(session, kind, minutes after T0, payload)` as one
/// step of the hook spool.
fn hooks(w: &mut WriterArchive, lines: &[(&str, &str, i64, Json)]) {
    let lines: Vec<(&str, String, i64, Json)> = lines
        .iter()
        .map(|(s, kind, min, data)| (*s, format!("hook_{kind}"), *min, data.clone()))
        .collect();
    evidence(w, &lines);
}

/// Stores live evidence lines `(session, full kind, minutes after T0,
/// data)` as one step of the hook spool.
fn evidence<K: AsRef<str>>(w: &mut WriterArchive, lines: &[(&str, K, i64, Json)]) {
    let start = w.offset();
    let events: Vec<Event> = lines
        .iter()
        .enumerate()
        .map(|(i, (s, kind, min, data))| Event {
            session: Some(sid(s)),
            kind: kind.as_ref().to_owned(),
            at: Timestamp::from_unix_ms(T0 + min * MIN),
            at_quality: polylogue_model::TimeQuality::Intrinsic,
            at_precision: Some(polylogue_model::Precision::Instant),
            at_until: None,
            after: None,
            range: Some((start + i as u64, start + i as u64 + 1)),
            data: data.clone(),
        })
        .collect();
    let mut sessions: Vec<Session> = vec![];
    for (s, ..) in lines {
        if !sessions.iter().any(|x| x.id == sid(s)) {
            sessions.push(Session::new(sid(s), SessionKind::Main));
        }
    }
    let output = Output {
        sessions,
        events,
        ..Output::default()
    };
    w.write_output("spool", "hook-spool", "/spool", output, lines.len() as u64);
}

/// `(session, state, until in minutes after T0)` at `min` minutes.
fn states(w: &WriterArchive, min: i64) -> Vec<(String, String, Option<i64>)> {
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let now = T0 + min * MIN;
    let rows = r
        .query(
            &format!(
                "SELECT s.id, {} AS live, {} AS until FROM session s ORDER BY s.id",
                live_state("s", now),
                live_until("s", now)
            ),
            &[],
        )
        .unwrap();
    rows.iter()
        .map(|r| {
            (
                r["id"]
                    .as_str()
                    .unwrap()
                    .trim_start_matches("claude-code:")
                    .to_owned(),
                r["live"].as_str().unwrap().to_owned(),
                r["until"].as_i64().map(|u| (u - T0) / MIN),
            )
        })
        .collect()
}

fn st(s: &str, state: &str, until: Option<i64>) -> (String, String, Option<i64>) {
    (s.to_owned(), state.to_owned(), until)
}

#[test]
fn live_state_is_a_function_of_stored_facts_and_the_time() {
    // Live state on hook evidence alone (no process lines). Without
    // process evidence, hook evidence holds for the live window and the
    // session is stale afterwards, until a day after its latest evidence
    // (live-sessions design §3.3: the live table holds "sessions with a
    // live root or activity in the last 24 h").
    let mut w = WriterArchive::new();
    hooks(
        &mut w,
        &[
            ("v1", "session_start", 0, json!({})),
            ("v1", "user_prompt_submit", 1, json!({})),
            (
                "v1",
                "notification",
                41,
                json!({"notification_type": "permission_prompt"}),
            ),
            ("v2", "user_prompt_submit", 50, json!({})),
            (
                "v2",
                "pre_tool_use",
                55,
                json!({"tool_name": "Bash", "tool_use_id": "t1"}),
            ),
            ("v3", "user_prompt_submit", 10, json!({})),
            ("v3", "stop", 20, json!({})),
        ],
    );
    assert_eq!(
        states(&w, 56),
        vec![
            st("v1", "waiting", Some(56)),
            st("v2", "working", Some(70)),
            st("v3", "stale", Some(20 + 24 * 60)),
        ]
    );
    // Asking twice at different times with no write between gives the
    // states for each time.
    assert_eq!(
        states(&w, 60),
        vec![
            st("v1", "stale", Some(41 + 24 * 60)),
            st("v2", "working", Some(70)),
            st("v3", "stale", Some(20 + 24 * 60)),
        ]
    );
    // A Stop makes it idle; a SessionEnd ends it for good.
    hooks(&mut w, &[("v2", "stop", 61, json!({}))]);
    assert_eq!(states(&w, 62)[1], st("v2", "idle", Some(76)));
    hooks(&mut w, &[("v2", "session_end", 63, json!({}))]);
    assert_eq!(states(&w, 64)[1], st("v2", "ended", None));
    // A line applied late (older than the stored one) changes nothing.
    hooks(&mut w, &[("v2", "pre_tool_use", 58, json!({}))]);
    assert_eq!(states(&w, 64)[1], st("v2", "ended", None));
    // A day after their latest evidence, stale sessions have ended.
    assert_eq!(
        states(&w, 26 * 60),
        vec![
            st("v1", "ended", None),
            st("v2", "ended", None),
            st("v3", "ended", None),
        ]
    );
}

#[test]
fn a_wait_ends_with_the_next_event_and_an_idle_notice_is_not_a_wait() {
    let mut w = WriterArchive::new();
    hooks(
        &mut w,
        &[
            (
                "s",
                "notification",
                0,
                json!({"notification_type": "permission_prompt"}),
            ),
            ("s", "post_tool_use", 1, json!({})),
        ],
    );
    assert_eq!(states(&w, 2), vec![st("s", "working", Some(16))]);
    hooks(
        &mut w,
        &[(
            "s",
            "notification",
            3,
            json!({"notification_type": "idle_prompt"}),
        )],
    );
    assert_eq!(states(&w, 4), vec![st("s", "idle", Some(18))]);
}

#[test]
fn live_state_is_total_for_sessions_without_hooks() {
    // Historical sessions with no hook or process evidence have
    // ended; one whose transcript grew within the window is `unknown`, and
    // ended once the window has passed (live-sessions design §8: "Time-only
    // inference survives only as the `unknown` row").
    use polylogue_store::fixture::{Fixture, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    f.session(SessionFacts {
        id: "claude-code:old",
        last: Some(T0 - 24 * 60 * MIN),
        ..SessionFacts::default()
    })
    .unwrap();
    f.session(SessionFacts {
        id: "claude-code:new",
        last: Some(T0),
        ..SessionFacts::default()
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let at = |min: i64| -> Vec<Value> {
        let now = T0 + min * MIN;
        r.query(
            &format!(
                "SELECT s.native_id AS id, {} AS live FROM session s ORDER BY s.id",
                live_state("s", now)
            ),
            &[],
        )
        .unwrap()
    };
    assert_eq!(
        at(60),
        vec![
            json!({"id": "new", "live": "ended"}),
            json!({"id": "old", "live": "ended"})
        ]
    );
    assert_eq!(
        at(10),
        vec![
            json!({"id": "new", "live": "unknown"}),
            json!({"id": "old", "live": "ended"})
        ]
    );
}

#[test]
fn peers_are_symmetric() {
    // Peers are symmetric, through the live query.
    use polylogue_store::fixture::{Fixture, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    let now = T0 + MIN;
    // (id, latest event, waiting on, repo_root, cwd)
    let rows = [
        ("a", "hook_pre_tool_use", None, None, "/src/tide"),
        (
            "b",
            "hook_notification",
            Some("permission"),
            None,
            "/src/tide",
        ),
        ("c", "hook_stop", None, None, "/src/tide/web"),
        ("d", "hook_pre_tool_use", None, Some("r1"), "/x"),
        ("e", "hook_stop", None, Some("r1"), "/y"),
        ("f", "hook_stop", None, Some("r1"), "/y"),
    ];
    for (id, event, waiting, repo, cwd) in rows {
        let pk = f
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                cwd: Some(cwd),
                last: Some(if id == "f" { T0 - 48 * 60 * MIN } else { T0 }),
                ..SessionFacts::default()
            })
            .unwrap();
        // f's evidence is two days old: it has ended.
        let at = if id == "f" { T0 - 48 * 60 * MIN } else { T0 };
        f.live(pk, event, at, waiting, repo).unwrap();
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let fields = ["id".to_owned(), "live".to_owned(), "peers".to_owned()];
    let out = run(
        &r,
        Ask {
            fields: &fields,
            ..ask("sessions | sort by id", now)
        },
    )
    .unwrap();
    let peers: std::collections::BTreeMap<String, Vec<String>> = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let id = row["id"]
                .as_str()
                .unwrap()
                .trim_start_matches("claude-code:")
                .to_owned();
            let p = row["peers"]
                .as_str()
                .unwrap_or("")
                .split_whitespace()
                .map(|p| p.trim_start_matches("claude-code:").to_owned())
                .collect();
            (id, p)
        })
        .collect();
    assert_eq!(peers["a"], ["b"]);
    assert_eq!(peers["b"], ["a"]);
    assert!(
        peers["c"].is_empty(),
        "a directory below another with no repository identity is not the same worktree"
    );
    assert_eq!(peers["d"], ["e"]);
    assert!(peers["f"].is_empty(), "an ended session has no peers");
    for (a, pa) in &peers {
        for (b, pb) in &peers {
            assert_eq!(pa.contains(b), pb.contains(a), "{a} {b}");
        }
    }
    // The live view lists waiting before working before idle, and leaves
    // ended sessions out.
    let live = run(&r, ask(crate::commands::LIVE, now)).unwrap();
    let order: Vec<&str> = live["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row["id"]
                .as_str()
                .unwrap()
                .trim_start_matches("claude-code:")
        })
        .collect();
    assert_eq!(order[0], "b");
    assert!(!order.contains(&"f"));
    assert_eq!(order.len(), 5);
}

#[test]
fn the_change_feed_continues_from_its_cursor() {
    let mut w = WriterArchive::new();
    hooks(
        &mut w,
        &[
            ("a", "user_prompt_submit", 0, json!({})),
            ("b", "user_prompt_submit", 0, json!({})),
        ],
    );
    let refs = |v: &Value| -> Vec<String> {
        v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == "session")
            .map(|r| r["ref"].as_str().unwrap().to_owned())
            .collect()
    };
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let all = run(&r, ask("changes", T0)).unwrap();
    assert_eq!(refs(&all), ["claude-code:a", "claude-code:b"]);
    let cursor = all["read_at"]["cursor"].as_str().unwrap().to_owned();
    let after = format!("changes where after:{cursor}");
    assert!(refs(&run(&r, ask(&after, T0)).unwrap()).is_empty());
    drop(r);
    drop(pool);
    hooks(&mut w, &[("b", "stop", 1, json!({}))]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let later = run(&r, ask(&after, T0)).unwrap();
    assert_eq!(refs(&later), ["claude-code:b"]);
    assert!(later["rows"][0]["live_changed"].is_i64());
    // A malformed cursor is refused, not read as "from the start".
    assert!(run(&r, ask("changes where after:12", T0)).is_err());
}

mod change_generation;

#[test]
fn stored_facts_equal_the_fold_of_stored_lines_with_ties_and_late_lines() {
    let lines = [
        ("f", "user_prompt_submit", 1, json!({})),
        ("f", "permission_request", 2, json!({})),
        ("f", "post_tool_use", 2, json!({})),
        ("f", "stop", 4, json!({})),
        // Arrives after the stop but happened before it.
        (
            "f",
            "pre_tool_use",
            3,
            json!({"tool_name": "Bash", "tool_use_id": "t"}),
        ),
    ];
    let mut w = WriterArchive::new();
    // One line per step, so equal and late instants arrive separately.
    for line in &lines {
        hooks(&mut w, std::slice::from_ref(line));
    }
    w.writer.derive().unwrap();
    // The expected rows: the fold over the lines in (time, offset) order.
    let mut ordered: Vec<_> = lines.iter().enumerate().collect();
    ordered.sort_by_key(|(i, (_, _, min, _))| (*min, *i));
    let mut facts = polylogue_model::live::LiveFacts::default();
    let mut closed = Vec::new();
    for (_, (_, kind, min, data)) in ordered {
        closed.extend(polylogue_model::live::fold(
            &mut facts,
            &format!("hook_{kind}"),
            T0 + min * MIN,
            data,
        ));
    }
    let stored = w.rows("SELECT live_facts, live_reparse FROM session WHERE id = 'claude-code:f'");
    assert_eq!(stored[0]["live_reparse"], 0);
    let stored_facts: polylogue_model::live::LiveFacts =
        serde_json::from_str(stored[0]["live_facts"].as_str().unwrap()).unwrap();
    assert_eq!(stored_facts, facts);
    let intervals = w.rows(
        "SELECT aspect, value, start_at, end_at FROM live_interval ORDER BY start_at, aspect",
    );
    closed.sort_by(|a, b| (a.start, a.aspect.as_str()).cmp(&(b.start, b.aspect.as_str())));
    let expected: Vec<Value> = closed
        .iter()
        .map(|i| json!({"aspect": i.aspect.as_str(), "value": i.value, "start_at": i.start, "end_at": i.end}))
        .collect();
    assert_eq!(intervals, expected);
}

/// A session's end closes the rows of its processes, children and harness
/// alike: the watcher stops watching an ended session.
#[test]
fn a_session_end_closes_its_process_rows() {
    let identity = |pid: u32, role: &str| json!({"pid": pid, "start": 7, "boot_id": "boot-a", "role": role, "comm": "x"});
    let mut w = WriterArchive::new();
    evidence(
        &mut w,
        &[
            (
                "p",
                "hook_user_prompt_submit",
                0,
                json!({"harness_pid": 10, "harness_start": 7, "boot_id": "boot-a"}),
            ),
            ("p", "process_started", 1, identity(10, "harness")),
            ("p", "process_started", 2, identity(11, "child")),
            ("p", "process_ended", 3, identity(11, "child")),
            ("p", "process_started", 4, identity(12, "child")),
            ("p", "hook_session_end", 5, json!({})),
        ],
    );
    let rows = w.rows("SELECT pid, ended_at FROM process ORDER BY pid");
    let ended: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            (
                r["pid"].as_i64().unwrap(),
                (r["ended_at"].as_i64().unwrap() - T0) / MIN,
            )
        })
        .collect();
    assert_eq!(ended, vec![(10, 5), (11, 3), (12, 5)]);
}
