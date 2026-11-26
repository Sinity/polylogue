use crate::fixture::writer::*;
use serde_json::json;

#[test]
fn no_task_from_prose_or_claim_commands() {
    // docs/agents.md: "A command's name or arguments, including a tracker's
    // exact claim command, establish no task state." A Bead ID in a prompt,
    // read with `bd show`, or claimed with `bd update --claim` gives no task.
    let mut w = WriterArchive::new();
    let mut a = prose("a", 1, 1_000, "please work on demo-ony.7 next");
    a.extend(bash(
        "a",
        2,
        2_000,
        "bd show demo-ony.7",
        true,
        "demo-ony.7 · Work graph",
    ));
    w.write("a", a);
    w.write(
        "b",
        bash(
            "b",
            1,
            1_000,
            "bd --actor me update demo-ony.7 --claim",
            true,
            "✓ claimed",
        ),
    );
    assert_eq!(w.live_task("a"), json!({"task": null}));
    assert_eq!(w.live_task("b"), json!({"task": null}));

    w.write(
        "b",
        bash(
            "b",
            2,
            5_000,
            "bd close demo-ony.7 --reason done",
            true,
            "✓ closed",
        ),
    );
    assert_eq!(w.live_task("b"), json!({"task": null}));
}

#[test]
fn commits_and_pull_requests_come_from_the_output_of_the_commands_that_made_them() {
    let mut w = WriterArchive::new();
    let mut m = bash("s", 1, 1_000, "bd update demo-9 --claim", true, "");
    m.extend(bash(
        "s",
        2,
        4_000,
        "git add -A && git commit -m 'store: fix'",
        true,
        "[lane-x 1a2b3c4] store: fix\n 1 file changed",
    ));
    m.extend(bash(
        "s",
        3,
        5_000,
        "gh pr create --fill",
        true,
        "https://github.com/Sinity/polylogue/pull/6300\n",
    ));
    w.write("s", m);
    assert_eq!(
        w.rows("SELECT commit_hash, evidence FROM commit_link"),
        vec![json!({"commit_hash": "1a2b3c4", "evidence": "commit-output"})]
    );
    assert_eq!(
        w.rows("SELECT repo, number FROM pr_link"),
        vec![json!({"repo": "Sinity/polylogue", "number": 6300})]
    );
    assert_eq!(w.live_task("s")["task"], json!(null));
}

#[test]
fn deriving_again_changes_nothing() {
    let mut w = WriterArchive::new();
    let mut m = bash("s", 1, 1_000, "bd update demo-1 --claim", true, "");
    m.extend(bash(
        "s",
        2,
        2_000,
        "git commit -m fixture",
        true,
        "[main 1a2b3c4] fixture",
    ));
    m.extend(bash(
        "s",
        3,
        3_000,
        "gh pr create --fill",
        true,
        "https://github.com/Sinity/polylogue/pull/6300\n",
    ));
    w.write("s", m);
    let links = |w: &WriterArchive| {
        (
            w.rows("SELECT commit_hash, evidence FROM commit_link"),
            w.rows("SELECT repo, number, url FROM pr_link"),
        )
    };
    let before = links(&w);
    assert_eq!(
        before,
        (
            vec![json!({"commit_hash": "1a2b3c4", "evidence": "commit-output"})],
            vec![json!({
                "repo": "Sinity/polylogue",
                "number": 6300,
                "url": "https://github.com/Sinity/polylogue/pull/6300"
            })]
        )
    );
    assert_eq!(w.writer.derive().unwrap(), 0);
    assert_eq!(links(&w), before);
}

#[test]
fn command_only_links_do_not_read_tool_result_bodies() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

    let mut w = WriterArchive::new();
    let mut messages = bash(
        "s",
        1,
        1_000,
        "cargo test",
        true,
        "large test output".repeat(4096).as_str(),
    );
    messages.extend(bash(
        "s",
        2,
        2_000,
        "git commit -m failed",
        false,
        "failed commit output",
    ));
    messages.extend(bash(
        "s",
        3,
        3_000,
        "gh pr view",
        true,
        "unrelated pull request output",
    ));
    messages.extend(bash(
        "s",
        4,
        4_000,
        "bd update demo-9 --claim",
        true,
        "claimed",
    ));
    w.write("s", messages);
    let mut index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let session_pk: i64 = index
        .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
            r.get(0)
        })
        .unwrap();
    index
        .authorizer(Some(|context: AuthContext<'_>| match context.action {
            AuthAction::Read {
                table_name: "text" | "block",
                column_name: "body",
            } => Authorization::Deny,
            _ => Authorization::Allow,
        }))
        .unwrap();
    let tx = index.transaction().unwrap();
    from_start(&tx, session_pk);
    tx.commit().unwrap();
    index
        .authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(w.live_task("s")["task"], json!(null));
    assert_eq!(
        w.rows("SELECT count(*) AS n FROM commit_link"),
        vec![json!({"n":0})]
    );
}

#[test]
fn compound_commands_keep_text_and_result_source() {
    for output_in_body in [false, true] {
        let mut w = WriterArchive::new();
        let output = "[work 1a2b3c4] store: improve work reads\nhttps://github.com/Sinity/polylogue/pull/6300\n";
        w.write(
            "s",
            bash(
                "s",
                1,
                1_000,
                "git commit -m 'store: improve work reads' && gh pr create --fill",
                true,
                output,
            ),
        );
        let mut index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
        if output_in_body {
            let body = serde_json::to_vec(
                &json!({"type":"tool_result", "outcome":{"kind":"ok"}, "output":output}),
            )
            .unwrap();
            let result: i64 = index.query_row("SELECT b.pk FROM tool_call t JOIN message m ON m.id=t.result_message_id JOIN block b ON b.message_pk=m.pk AND b.idx=t.result_idx", [], |r| r.get(0)).unwrap();
            index
                .execute("UPDATE block SET text_id=NULL WHERE pk=?1", [result])
                .unwrap();
            crate::content::rewrite_piece(&index, "block", result, Some(&body), None).unwrap();
        }
        let session_pk = index
            .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let tx = index.transaction().unwrap();
        from_start(&tx, session_pk);
        tx.commit().unwrap();
        assert_eq!(w.rows("SELECT c.commit_hash, c.evidence, c.evidence_block_pk=t.call_block_pk AS original_call FROM commit_link c JOIN tool_call t ON t.session_pk=c.session_pk"),
            vec![json!({"commit_hash":"1a2b3c4", "evidence":"commit-output", "original_call":1})]);
        assert_eq!(
            w.rows("SELECT repo, number, evidence FROM pr_link"),
            vec![json!({"repo":"Sinity/polylogue", "number":6300, "evidence":"gh-output"})]
        );
    }
}

/// The session links as stored, and as deriving every session again gives.
fn stored_and_derived(w: &WriterArchive) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let links = |conn: &rusqlite::Connection| -> Vec<serde_json::Value> {
        let mut rows = Vec::new();
        for sql in [
            "SELECT json_array(session_pk, commit_hash, evidence_block_pk, evidence) FROM commit_link ORDER BY 1",
            "SELECT json_array(session_pk, repo, number, url, evidence) FROM pr_link ORDER BY 1",
        ] {
            let mut statement = conn.prepare(sql).unwrap();
            for row in statement.query_map([], |r| r.get::<_, String>(0)).unwrap() {
                rows.push(serde_json::from_str(&row.unwrap()).unwrap());
            }
        }
        rows
    };
    let mut conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let stored = links(&conn);
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    let sessions: Vec<i64> = tx
        .prepare("SELECT pk FROM session")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    for pk in sessions {
        from_start(&tx, pk);
    }
    let derived = links(&tx);
    tx.rollback().unwrap();
    (stored, derived)
}

#[test]
fn links_after_append_equal_full_recompute() {
    let mut w = WriterArchive::new();
    let mut seen = Vec::new();
    let mut check = |w: &WriterArchive, stage: &str| {
        let (stored, derived) = stored_and_derived(w);
        assert_eq!(stored, derived, "{stage}");
        seen.push(stored);
    };
    w.write("s", prose("s", 1, 1_000, "commit the fix"));
    check(&w, "new session");
    w.write(
        "s",
        bash(
            "s",
            2,
            2_000,
            "git commit -m 'store: fix'",
            true,
            "[lane-x 1a2b3c4] store: fix\n 1 file changed",
        ),
    );
    check(&w, "call and result appended together");
    // A late result: the call is written first, its output later.
    let mut pr = bash(
        "s",
        3,
        3_000,
        "gh pr create --fill",
        true,
        "https://github.com/Sinity/polylogue/pull/6300\n",
    );
    let result = pr.pop().unwrap();
    w.write("s", pr);
    check(&w, "call without result");
    w.write("s", vec![result]);
    check(&w, "late result");
    assert_eq!(
        w.rows("SELECT repo, number, evidence FROM pr_link"),
        vec![json!({"repo": "Sinity/polylogue", "number": 6300, "evidence": "gh-output"})]
    );
    // A rewrite: a snapshot states the commit's output again, differently.
    w.write_snapshot(
        "s",
        bash(
            "s",
            2,
            2_000,
            "git commit -m 'store: fix'",
            true,
            "[lane-x 5e6f7a8] store: fix\n 1 file changed",
        ),
    );
    check(&w, "rewritten output");
    assert_eq!(
        w.rows("SELECT commit_hash, evidence FROM commit_link"),
        vec![json!({"commit_hash": "5e6f7a8", "evidence": "commit-output"})]
    );
    // Another session's commit leaves this session's links alone.
    w.write(
        "t",
        bash("t", 1, 4_000, "git commit -m x", true, "[main 9b8c7d6] x"),
    );
    check(&w, "another session");
    // The same commit and pull request reported again by calls timed before
    // the ones already stored but later in session order (docs/measures.md,
    // commit and pull-request links): the calls first in session order stay
    // the recorded ones, so nothing changes.
    w.write(
        "t",
        bash(
            "t",
            2,
            6_000,
            "gh pr create --fill",
            true,
            "https://github.com/Sinity/polylogue/pull/6400/files\n",
        ),
    );
    check(&w, "pull request");
    let mut earlier = bash("t", 3, 3_000, "git commit -m x", true, "[main 9b8c7d6] x");
    earlier.extend(bash(
        "t",
        4,
        2_000,
        "gh pr create --fill",
        true,
        "https://github.com/Sinity/polylogue/pull/6400.\n",
    ));
    w.write("t", earlier);
    check(
        &w,
        "same commit and pull request arriving out of time order",
    );
    seen.dedup();
    assert_eq!(seen.len(), 6, "{seen:?}");
}

#[test]
fn links_derived_by_the_earlier_rules_are_derived_again() {
    let mut w = WriterArchive::new();
    w.write(
        "s",
        bash(
            "s",
            1,
            1_000,
            "gh pr create --fill",
            true,
            "https://github.com/Sinity/polylogue/pull/6300/files\n",
        ),
    );
    // An index the earlier rules derived: the raw URL token, at their version.
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let version: String = conn
        .query_row(
            "SELECT version FROM summary_state WHERE name='links'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    // The rules #6808 shipped.
    assert_ne!(version, "2");
    let earlier = "2".to_owned();
    conn.execute(
        "UPDATE pr_link SET url='https://github.com/Sinity/polylogue/pull/6300/files'",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE summary_state SET version=?1 WHERE name='links'",
        [&earlier],
    )
    .unwrap();
    conn.execute(
        "UPDATE meta SET value=json_set(value,'$.links',?1) WHERE key='summary_versions'",
        [&earlier],
    )
    .unwrap();
    drop(conn);
    w.writer.derive().unwrap();
    assert_eq!(
        w.rows("SELECT url FROM pr_link"),
        vec![json!({"url": "https://github.com/Sinity/polylogue/pull/6300"})]
    );
}

/// The session's links folded from its first row, written in `tx`.
fn from_start(tx: &rusqlite::Transaction<'_>, pk: i64) {
    let folded = crate::summary::fold::Run { conn: tx }
        .session(pk, &crate::summary::Key::start())
        .unwrap();
    super::apply(tx, pk, folded.output.links.as_ref().unwrap()).unwrap();
}
