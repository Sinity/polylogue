use super::*;
use crate::fixture::{Fixture, SessionFacts};

const NOW: i64 = 1_790_856_000_000;

fn session(f: &mut Fixture, id: &str, cwd: &str, repo: Option<&str>, last: i64) -> i64 {
    f.session(SessionFacts {
        id,
        cwd: Some(cwd),
        repo,
        last: Some(last),
        ..SessionFacts::default()
    })
    .unwrap()
}

fn ids(sessions: Vec<&ScopeSession>) -> Vec<String> {
    sessions
        .into_iter()
        .map(|s| s.session.id.to_string())
        .collect()
}

#[test]
fn sessions_under_a_directory_match_whole_path_components_newest_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    session(&mut f, "test:root", "/work/proj", Some("proj"), NOW - 3_000);
    session(
        &mut f,
        "test:sub",
        "/work/proj/crates/a",
        Some("proj"),
        NOW - 1_000,
    );
    session(
        &mut f,
        "test:sibling",
        "/work/proj-old",
        Some("old"),
        NOW - 2_000,
    );
    session(&mut f, "test:other", "/elsewhere", Some("else"), NOW);
    f.session(SessionFacts {
        id: "test:helper",
        kind: "subagent",
        cwd: Some("/work/proj"),
        last: Some(NOW),
        ..SessionFacts::default()
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let index = ScopeIndex::new(pool.reader().unwrap().scope_changes(None).unwrap());
    let dirs = |d: &[&str]| d.iter().map(|d| (*d).to_owned()).collect::<Vec<_>>();
    assert_eq!(
        ids(index.sessions_under(&dirs(&["/work/proj"]), None)),
        ["test:sub", "test:root"]
    );
    assert_eq!(
        ids(index.sessions_under(&dirs(&["/work/proj/"]), Some("test:sub"))),
        ["test:root"]
    );
    assert_eq!(
        ids(index.sessions_under(&dirs(&["/"]), None)),
        ["test:other", "test:sub", "test:sibling", "test:root"]
    );
    assert!(index.sessions_under(&dirs(&["/work/pro"]), None).is_empty());
    assert_eq!(index.repos_under(&dirs(&["/work"])), ["proj", "old"]);
    assert_eq!(index.len(), 4, "only main sessions are held");
}

#[test]
fn live_sessions_and_open_beads_come_from_the_held_facts() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let waiting = session(&mut f, "test:waiting", "/work/proj", None, NOW - 60_000);
    let working = session(&mut f, "test:working", "/work/proj/sub", None, NOW - 60_000);
    let idle = session(&mut f, "test:idle", "/work/proj", None, NOW - 60_000);
    let ended = session(&mut f, "test:ended", "/work/proj", None, NOW - 60_000);
    let stale = session(&mut f, "test:stale", "/work/proj", None, NOW - 60_000);
    f.live(
        waiting,
        "hook_notification",
        NOW - 1_000,
        Some("permission"),
        None,
    )
    .unwrap();
    f.live(working, "hook_pre_tool_use", NOW - 2_000, None, None)
        .unwrap();
    f.live(idle, "hook_stop", NOW - 3_000, None, None).unwrap();
    f.live(ended, "hook_session_end", NOW - 1_000, None, None)
        .unwrap();
    f.live(stale, "hook_pre_tool_use", NOW - 16 * 60 * 1000, None, None)
        .unwrap();
    let pool = f.reads().unwrap();
    let index = ScopeIndex::new(pool.reader().unwrap().scope_changes(None).unwrap());
    let dirs = vec!["/work/proj".to_owned()];
    let live: Vec<(String, &str)> = index
        .live_under(&dirs, Some("test:idle"), NOW)
        .into_iter()
        .map(|l| (l.session.session.id.to_string(), l.state))
        .collect();
    assert_eq!(
        live,
        [
            ("test:waiting".to_owned(), "waiting"),
            ("test:working".to_owned(), "working")
        ]
    );
    assert_eq!(index.live_under(&dirs, None, NOW).len(), 3);
    assert!(index.open_beads(&dirs).is_empty());
}

#[test]
fn the_index_follows_sessions_changed_after_its_position() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    session(&mut f, "test:first", "/work/proj", None, NOW - 2_000);
    let pool = f.reads().unwrap();
    let mut index = ScopeIndex::new(pool.reader().unwrap().scope_changes(None).unwrap());
    let held = index.position().cloned();
    session(
        &mut f,
        "test:second",
        "/work/proj/sub",
        Some("proj"),
        NOW - 1_000,
    );
    let changes = pool.reader().unwrap().scope_changes(held.as_ref()).unwrap();
    assert!(
        !changes.full,
        "a delta after a position of the same generation"
    );
    assert!(!index.is_newer_than(&changes));
    index.apply(changes);
    let dirs = vec!["/work/proj".to_owned()];
    assert_eq!(
        ids(index.sessions_under(&dirs, None)),
        ["test:second", "test:first"]
    );
    assert_eq!(index.repos_under(&dirs), ["proj"]);
}

/// The scope index reads the same folded facts as the query's live view.
#[test]
fn live_facts_are_read_from_the_session_live_columns() {
    assert!(COLUMNS.contains("live_facts"));
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let pk = session(&mut f, "test:live", "/work/proj", None, NOW);
    f.live(pk, "hook_notification", NOW - 1_000, Some("question"), None)
        .unwrap();
    let pool = f.reads().unwrap();
    let index = ScopeIndex::new(pool.reader().unwrap().scope_changes(None).unwrap());
    let held = &index.sessions[&pk];
    assert_eq!(held.last_event_at(), Some(NOW - 1_000));
    assert_eq!(held.waiting_on(), Some("question"));
    assert!(!held.open_tool_call());
    assert_eq!((held.todo(), held.task()), (None, None));
}

#[test]
fn equal_activity_uses_stable_session_identity_instead_of_row_admission() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    session(&mut f, "test:a", "/work/proj", None, NOW);
    session(&mut f, "test:b", "/work/proj", None, NOW);
    let pool = f.reads().unwrap();
    let index = ScopeIndex::new(pool.reader().unwrap().scope_changes(None).unwrap());
    assert_eq!(
        ids(index.sessions_under(&["/work/proj".into()], None)),
        ["test:a", "test:b"]
    );
}
