//! Outcomes over the Morrow archive with generated files (docs/interfaces.md,
//! Exit codes): what waiting ends makes a reply degraded and says how many
//! and of which origin; what waiting does not end (an origin with no
//! parser, a file of no bytes, a failed parse) leaves a read ok or empty,
//! and status names a failed parse.
use super::*;
use crate::test_answers;
use crate::test_daemon::{Call, TestDaemon};
use polylogue_model::chain::ChainStatusName;
use polylogue_store::fault::{self, Action};
use serde_json::json;
use std::path::Path;

fn listing(d: &TestDaemon, selection: &str) -> command_surface::Answer<serde_json::Value> {
    Call::new("rows", json!({ "selection": selection })).answer(&d.daemon)
}

/// Morrow's #5 is its only session in the repository `morrow` (FACTS.md);
/// Morrow is older than a day.
const ROWS: &str = "repo:morrow";
const NO_ROWS: &str = "repo:morrow since:1d";

/// A Codex rollout naming a local image, whose list of references is kept
/// as bytes of the `tool-context` origin, which has no parser; and a Claude
/// Code session file of no bytes.
fn bytes_only_and_empty(home: &Path) {
    let codex = home.join(".codex/sessions");
    std::fs::create_dir_all(&codex).unwrap();
    let header = json!({"type": "session_meta", "payload": {"id": "generated-images"}});
    let row = json!({"type": "event_msg", "payload": {"type": "user_message",
        "message": "attachment", "local_images": ["missing.png"]}});
    std::fs::write(
        codex.join("rollout-generated-images.jsonl"),
        format!("{header}\n{row}\n"),
    )
    .unwrap();
    let claude = home.join(".claude/projects/-generated");
    std::fs::create_dir_all(&claude).unwrap();
    std::fs::write(claude.join("empty.jsonl"), b"").unwrap();
}

#[test]
fn bytes_only_origins_and_empty_files_leave_replies_ok_with_rows_and_empty_without() {
    let d = test_daemon::start_over_morrow(&bytes_only_and_empty);
    let reader = d.daemon.reads.reader().unwrap();
    let census = reader
        .chain_census(&polylogue_engine::reading_contracts())
        .unwrap();
    assert_eq!(
        census
            .chains
            .get(&("tool-context".to_owned(), ChainStatusName::NoParser)),
        Some(&1),
        "{census:?}"
    );
    let empty = reader
        .query("SELECT count(*) AS n FROM a.revision WHERE length = 0", &[])
        .unwrap();
    assert_eq!(empty[0]["n"], 1);
    drop(reader);

    let rows = listing(&d, ROWS);
    assert_eq!(rows.exit_code(), 0, "{rows:?}");
    let rows = test_answers::ok(rows);
    assert_eq!(rows["reply"]["rows"].as_array().unwrap().len(), 1, "{rows}");
    let none = listing(&d, NO_ROWS);
    assert_eq!(none.exit_code(), 1, "{none:?}");
    test_answers::empty(none);
    let status = test_daemon::call(&d.daemon, "status", json!({}));
    assert_eq!(status.exit_code(), 0, "{status:?}");
}

#[test]
fn a_failed_parse_leaves_reads_ok_and_status_names_it_with_its_origin() {
    // A parser that panics fails its chain (docs/archive.md, Reading
    // versions); the failure stands until the bytes or the parser change.
    fault::arm("parse", "generated-broken", Action::Panic, 0, u32::MAX);
    let d = test_daemon::start_over_morrow(&|home| {
        let dir = home.join(".claude/projects/-generated");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("generated-broken.jsonl"),
            polylogue_testkit::claude_user("b1", "generated-broken", "a body the parser fails"),
        )
        .unwrap();
    });
    let rows = listing(&d, ROWS);
    assert_eq!(rows.exit_code(), 0, "{rows:?}");
    test_answers::empty(listing(&d, NO_ROWS));

    let status = test_daemon::call(&d.daemon, "status", json!({}));
    assert_eq!(status.exit_code(), 5, "{status:?}");
    let (value, notes) = test_answers::degraded(status);
    let [count, detail] = notes.as_slice() else {
        panic!("the census and source failure must both be reported: {notes:?}");
    };
    assert_eq!(count, "1 claude-code input failed to parse");
    assert!(
        detail.contains("generated-broken.jsonl: panicked: injected panic at parse"),
        "{detail}"
    );
    assert_eq!(
        value["freshness"]["local"]["failed_parses"],
        json!([{"reason": "parse_failed", "origin": "claude-code", "count": 1, "waiting_ends": false}])
    );
    assert_eq!(
        value["freshness"]["local"]["parsing"],
        json!({"state": "complete"})
    );
    drop(d);
    fault::clear();
}

#[test]
fn a_reply_while_the_index_catches_up_is_degraded_with_the_count_and_origin() {
    let mut d = test_daemon::start_over_morrow(&|_| {});
    let archive = d.daemon.archive.clone();
    let mut older = 0;
    // Every Claude Code chain read under an earlier parser version, and
    // every parse held, so the daemon stays catching up while it is asked.
    d.restart_after_stop(|| {
        older = polylogue_store::fixture::older_reading(&archive, "claude-code")
            .unwrap()
            .0;
        fault::arm("parse", "", Action::Hold, 0, u32::MAX);
    });
    assert!(older > 0);
    fault::wait_held();
    let rows = listing(&d, ROWS);
    fault::release();
    assert_eq!(rows.exit_code(), 5, "{rows:?}");
    let (value, notes) = test_answers::degraded(rows);
    assert_eq!(
        notes,
        [format!(
            "{older} claude-code inputs await the current parser; status counts them, and no query lists stored inputs yet"
        )]
    );
    assert_eq!(
        value["reply"]["read_position"]["freshness"]["local"]["parsing"],
        json!({"state": "incomplete", "reasons": [
            {"reason": "parse_version_changed", "origin": "claude-code", "count": older, "waiting_ends": true}
        ]})
    );
    d.wait_for(|status| status["caught_up"] == true);
    test_answers::ok(listing(&d, ROWS));
    fault::clear();
}
