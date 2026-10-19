//! The live view, the change feed and the signal notices as queries:
//! `sessions` with live fields, the `changes` unit and `deliveries`.

use crate::daemon::query::*;
use polylogue_model::{Event, Json, Output, Session, SessionId, SessionKind, Timestamp};
use polylogue_query::fields::{live_state, live_until};
use polylogue_store::ReadPool;
use polylogue_store::fixture::writer::WriterWorld;
use serde_json::json;

fn ask(text: &str, now: i64) -> Ask<'_> {
    Ask {
        text,
        cursor: None,
        page_size: Some(100),
        why: false,
        session: None,
        now,
        day: LogicalDay::named("UTC", 0).unwrap(),
        embeddings: false,
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
fn hooks(w: &mut WriterWorld, lines: &[(&str, &str, i64, Json)]) {
    let start = w.offset();
    let events: Vec<Event> = lines
        .iter()
        .enumerate()
        .map(|(i, (s, kind, min, data))| Event {
            session: sid(s),
            kind: format!("hook_{kind}"),
            at: Timestamp::from_unix_ms(T0 + min * MIN),
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
fn states(w: &WriterWorld, min: i64) -> Vec<(String, String, Option<i64>)> {
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
    // Law 85, on hook evidence alone (no process lines).
    let mut w = WriterWorld::new();
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
    // A day after the last evidence, everything has ended.
    assert!(
        states(&w, 26 * 60)
            .iter()
            .all(|(_, s, u)| s == "ended" && u.is_none())
    );
}

#[test]
fn a_wait_ends_with_the_next_event_and_an_idle_notice_is_not_a_wait() {
    let mut w = WriterWorld::new();
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
    // Law 101: historical sessions with no hook or process evidence have
    // ended; one whose transcript grew within the window is `unknown`.
    use polylogue_store::fixture::{Fixture, S};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    f.session(S {
        id: "claude-code:old",
        last: Some(T0 - 24 * 60 * MIN),
        ..S::default()
    })
    .unwrap();
    f.session(S {
        id: "claude-code:new",
        last: Some(T0),
        ..S::default()
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
            json!({"id": "new", "live": "stale"}),
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
    // Law 89, with the spec's own examples, through the live query.
    use polylogue_store::fixture::{Fixture, S};
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
            "/src/tide/web",
        ),
        ("c", "hook_stop", None, None, "/src/tidepool"),
        ("d", "hook_pre_tool_use", None, Some("r1"), "/x"),
        ("e", "hook_stop", None, Some("r1"), "/y"),
        ("f", "hook_stop", None, Some("r1"), "/y"),
    ];
    for (id, event, waiting, repo, cwd) in rows {
        let pk = f
            .session(S {
                id: &format!("claude-code:{id}"),
                cwd: Some(cwd),
                last: Some(if id == "f" { T0 - 48 * 60 * MIN } else { T0 }),
                ..S::default()
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
        "/src/tide and /src/tidepool are not peers"
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
    let live = run(&r, ask(crate::cli::LIVE, now)).unwrap();
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
    let mut w = WriterWorld::new();
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
    let cursor = all["view"]["cursor"].as_str().unwrap().to_owned();
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
