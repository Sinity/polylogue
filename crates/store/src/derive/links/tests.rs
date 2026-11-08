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
fn a_failed_claim_claims_nothing() {
    let mut w = WriterArchive::new();
    w.write(
        "s",
        bash(
            "s",
            1,
            1_000,
            "bd update demo-1 --status in_progress",
            false,
            "error: not found",
        ),
    );
    assert_eq!(w.live_task("s"), json!({"task": null}));
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
    w.write(
        "s",
        bash("s", 1, 1_000, "bd update demo-1 --claim", true, ""),
    );
    let before = w.live_task("s");
    assert_eq!(w.writer.derive().unwrap(), 0);
    assert_eq!(before, w.live_task("s"));
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
    super::derive_session(&tx, session_pk).unwrap();
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
fn compound_artifact_commands_keep_text_and_packed_result_provenance() {
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
            let body = crate::encoding::pack_json(
                &json!({"type":"tool_result", "outcome":{"kind":"ok"}, "output":output}),
            )
            .unwrap();
            index.execute("UPDATE block SET text_id=NULL, body=?1 WHERE pk IN (SELECT result_block_pk FROM tool_call)", [body]).unwrap();
        }
        let session_pk = index
            .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let tx = index.transaction().unwrap();
        super::derive_session(&tx, session_pk).unwrap();
        tx.commit().unwrap();
        assert_eq!(w.rows("SELECT c.commit_hash, c.evidence, c.evidence_block_pk=t.call_block_pk AS original_call FROM commit_link c JOIN tool_call t ON t.session_pk=c.session_pk"),
            vec![json!({"commit_hash":"1a2b3c4", "evidence":"commit-output", "original_call":1})]);
        assert_eq!(
            w.rows("SELECT repo, number, evidence FROM pr_link"),
            vec![json!({"repo":"Sinity/polylogue", "number":6300, "evidence":"gh-output"})]
        );
    }
}
