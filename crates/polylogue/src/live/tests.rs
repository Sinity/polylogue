use super::*;
use polylogue_model::{
    Accounting, Disposition, Event, Json, Output, Session, SessionId, SessionKind, Timestamp,
};
use polylogue_store::{Archive, Batch, Change, InputKey, Observation, ParseStep, ReadPool, Writer};

fn row(id: &str, live: &str, repo: Option<&str>, cwd: &str) -> Value {
    json!({"id": id, "live": live, "repo_root": repo, "checkout": null, "cwd": cwd})
}

#[test]
fn peers_are_symmetric() {
    // Law 89, with the spec's own examples.
    let rows = vec![
        row("a", "working", None, "/src/tide"),
        row("b", "waiting", None, "/src/tide/web"),
        row("c", "idle", None, "/src/tidepool"),
        row("d", "working", Some("r1"), "/x"),
        row("e", "idle", Some("r1"), "/y"),
        row("f", "stale", Some("r1"), "/y"),
    ];
    let p = peers(&rows);
    assert_eq!(p[0], ["b"]);
    assert_eq!(p[1], ["a"]);
    assert!(p[2].is_empty(), "/src/tide and /src/tidepool are not peers");
    assert_eq!(p[3], ["e"]);
    assert!(p[5].is_empty(), "a stale session has no peers");
    for (i, a) in rows.iter().enumerate() {
        for (j, b) in rows.iter().enumerate() {
            let a_has_b = p[i].contains(&b["id"].as_str().unwrap().to_owned());
            let b_has_a = p[j].contains(&a["id"].as_str().unwrap().to_owned());
            assert_eq!(a_has_b, b_has_a, "{i} {j}");
        }
    }
}

#[test]
fn cursors_round_trip() {
    let c = Cursor {
        index: 12,
        archive: 3,
    };
    assert_eq!(Cursor::parse(&c.render()), Some(c));
    assert_eq!(Cursor::parse("12"), None);
}

const MIN: i64 = 60_000;
/// 2026-10-18 14:00 UTC.
const T0: i64 = 1_792_332_000_000;

struct World {
    _dir: tempfile::TempDir,
    archive: Archive,
    _lock: polylogue_store::ArchiveLock,
    writer: Writer,
    offset: u64,
}

fn world() -> World {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    World {
        _dir: dir,
        archive,
        _lock: lock,
        writer,
        offset: 0,
    }
}

fn sid(n: &str) -> SessionId {
    SessionId::parse(&format!("claude-code:{n}")).unwrap()
}

impl World {
    /// Stores hook lines `(session, kind, minutes after T0, payload)` as one
    /// step of the hook spool.
    fn hooks(&mut self, lines: &[(&str, &str, i64, Json)]) {
        let start = self.offset;
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
        let n = events.len() as u64;
        self.offset += n + 1;
        let mut sessions: Vec<Session> = vec![];
        for (s, ..) in lines {
            if !sessions.iter().any(|x| x.id == sid(s)) {
                sessions.push(Session::new(sid(s), SessionKind::Main));
            }
        }
        let mut accounting = Accounting {
            framed: n,
            ..Accounting::default()
        };
        for _ in 0..n {
            accounting.add("hook", &Disposition::Materialized);
        }
        let batch = Batch {
            input: InputKey {
                id: "spool".into(),
                format: "hook-spool".into(),
                route_rank: 0,
            },
            change: if start == 0 {
                Change::First
            } else {
                Change::Appended { from: start }
            },
            observation: Some(Observation {
                machine: "m".into(),
                host: "h".into(),
                path: "/spool".into(),
                facts: None,
                seen_at_ms: 1,
            }),
            continues: false,
            piece: None,
            blobs: vec![],
            length: self.offset,
            release: Default::default(),
            parse: Some(ParseStep {
                output: Output {
                    sessions,
                    events,
                    ..Output::default()
                },
                accounting,
                start,
                parsed_upto: self.offset,
                parser_version: 1,
                hashes: vec![],
                packed: Default::default(),
                bodies: vec![],
                events: vec![],
            }),
        };
        self.writer.send(batch).unwrap();
        self.writer.flush().unwrap();
    }

    /// `(session, state, until in minutes after T0)` at `min` minutes.
    fn states(&self, min: i64) -> Vec<(String, String, Option<i64>)> {
        let pool = ReadPool::open(&self.archive).unwrap();
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
}

fn st(s: &str, state: &str, until: Option<i64>) -> (String, String, Option<i64>) {
    (s.to_owned(), state.to_owned(), until)
}

#[test]
fn live_state_is_a_function_of_stored_facts_and_the_time() {
    // Law 85, on hook evidence alone (no process lines).
    let mut w = world();
    w.hooks(&[
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
    ]);
    assert_eq!(
        w.states(56),
        vec![
            st("v1", "waiting", Some(56)),
            st("v2", "working", Some(70)),
            st("v3", "stale", Some(20 + 24 * 60)),
        ]
    );
    // Asking twice at different times with no write between gives the
    // states for each time.
    assert_eq!(
        w.states(60),
        vec![
            st("v1", "stale", Some(41 + 24 * 60)),
            st("v2", "working", Some(70)),
            st("v3", "stale", Some(20 + 24 * 60)),
        ]
    );
    // A Stop makes it idle; a SessionEnd ends it for good.
    w.hooks(&[("v2", "stop", 61, json!({}))]);
    assert_eq!(w.states(62)[1], st("v2", "idle", Some(76)));
    w.hooks(&[("v2", "session_end", 63, json!({}))]);
    assert_eq!(w.states(64)[1], st("v2", "ended", None));
    // A line applied late (older than the stored one) changes nothing.
    w.hooks(&[("v2", "pre_tool_use", 58, json!({}))]);
    assert_eq!(w.states(64)[1], st("v2", "ended", None));
    // A day after the last evidence, everything has ended.
    assert!(
        w.states(26 * 60)
            .iter()
            .all(|(_, s, u)| s == "ended" && u.is_none())
    );
}

#[test]
fn a_wait_ends_with_the_next_event_and_an_idle_notice_is_not_a_wait() {
    let mut w = world();
    w.hooks(&[
        (
            "s",
            "notification",
            0,
            json!({"notification_type": "permission_prompt"}),
        ),
        ("s", "post_tool_use", 1, json!({})),
    ]);
    assert_eq!(w.states(2), vec![st("s", "working", Some(16))]);
    w.hooks(&[(
        "s",
        "notification",
        3,
        json!({"notification_type": "idle_prompt"}),
    )]);
    assert_eq!(w.states(4), vec![st("s", "idle", Some(18))]);
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
fn the_change_feed_continues_from_its_cursor() {
    let mut w = world();
    w.hooks(&[
        ("a", "user_prompt_submit", 0, json!({})),
        ("b", "user_prompt_submit", 0, json!({})),
    ]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let all = changes(&r, Cursor::default()).unwrap();
    let ids = |v: &Value| -> Vec<String> {
        v["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["session"].as_str().unwrap().to_owned())
            .collect()
    };
    assert_eq!(ids(&all), ["claude-code:a", "claude-code:b"]);
    let next = Cursor::parse(all["next"].as_str().unwrap()).unwrap();
    assert!(ids(&changes(&r, next).unwrap()).is_empty());
    drop(r);
    drop(pool);
    w.hooks(&[("b", "stop", 1, json!({}))]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let later = changes(&r, next).unwrap();
    assert_eq!(ids(&later), ["claude-code:b"]);
    assert!(later["sessions"][0]["live_changed"].is_i64());
}
