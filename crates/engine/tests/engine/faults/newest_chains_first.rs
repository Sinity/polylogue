//! A rebuild reads every input's newest revision chain before any older
//! chain, and its first build ends once those are read (docs/operations.md,
//! a full rebuild).

use super::*;
use polylogue_engine::Engine;
use polylogue_store::{Archive, ArchiveLock};
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};

/// The parse states of `file`'s chains, oldest chain first.
fn chains(w: &ClaudeHome, file: &str) -> Vec<serde_json::Value> {
    w.reads()
        .reader()
        .unwrap()
        .query(
            &format!(
                "SELECT p.chain, p.status, p.parsed_upto, p.commit_seq FROM parse_state p
                 JOIN input_ref i ON i.pk = p.input_pk
                 WHERE i.id = (SELECT input FROM a.observation WHERE path LIKE '%/{file}' LIMIT 1)
                 ORDER BY p.chain"
            ),
            &[],
        )
        .unwrap()
}

/// What the index holds, named by natural keys rather than by row keys,
/// which follow the order of writes: every session's export, every
/// message's sources, and every chain's parse state.
fn readings(
    w: &ClaudeHome,
) -> (
    Vec<polylogue_model::export::SessionExport>,
    serde_json::Value,
) {
    let reads = w.reads();
    let reader = reads.reader().unwrap();
    let sessions: Vec<String> = reader
        .query("SELECT id FROM session ORDER BY id", &[])
        .unwrap()
        .into_iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    let exports = sessions.iter().map(|id| w.export(id)).collect();
    let rows = serde_json::json!({
        "sources": reader.query(
            "SELECT m.id AS message, i.id AS input, s.chain, s.source_start, s.source_end,
                    s.main, s.completion, hex((SELECT version_hash FROM message_version WHERE pk=s.version_pk)) AS version
             FROM message_source s JOIN message m ON m.pk = s.message_pk
             JOIN input_ref i ON i.pk = s.input_pk
             ORDER BY m.id, i.id, s.chain, s.source_start",
            &[],
        ).unwrap(),
        "parse_state": reader.query(
            "SELECT i.id AS input, p.chain, p.status, p.parsed_upto, p.at_length, p.records,
                    p.used, p.skipped, p.unreadable, p.reading_version
             FROM parse_state p JOIN input_ref i ON i.pk = p.input_pk
             ORDER BY i.id, p.chain",
            &[],
        ).unwrap(),
    });
    (exports, rows)
}

fn building(w: &ClaudeHome) -> i64 {
    w.count("SELECT CAST(value AS INTEGER) AS n FROM meta WHERE key = 'building'")
}

/// One worker, so the order of parses is the order the pass plans them.
fn one_worker(w: &ClaudeHome, chunk: u64) -> (Engine, ArchiveLock) {
    let (archive, lock) = Archive::open_or_create(&w.root).unwrap();
    let mut config = w.config(chunk);
    config.workers = 1;
    (
        Engine::new(config, archive, polylogue_testkit::stopped_attempt()).unwrap(),
        lock,
    )
}

/// Runs a pass on its own thread and waits until it holds at the armed
/// `parse` point.
fn held_pass(
    mut engine: Engine,
    lock: ArchiveLock,
) -> std::thread::JoinHandle<Result<polylogue_engine::Pass, polylogue_engine::EngineError>> {
    let pass = std::thread::spawn(move || {
        let pass = engine.run_pass();
        drop(engine);
        drop(lock);
        pass
    });
    while !fault::held() {
        assert!(!pass.is_finished(), "the pass ended before the held parse");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    pass
}

/// A file rewritten once holds an older chain and a newer one. A rebuild
/// holds at the second parse of that file: the newest chains of both files
/// are read, the first build has ended, and the older chain is not read yet.
/// Once released, the pass reads the older chain, and the index holds what
/// capture wrote, which read the older chain first.
#[test]
fn a_first_build_ends_once_every_newest_chain_is_read() {
    fault::clear();
    let w = ClaudeHome::new();
    let draft = user("u1", "edited", "first draft")
        + &assistant("a1", "edited", "first answer")
        + &user("u2", "edited", "a prompt only the draft holds");
    let edited = user("u1", "edited", "final wording") + &assistant("a1", "edited", "final answer");
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        w.append("edited.jsonl", &draft);
        w.append("steady.jsonl", &user("s1", "steady", "unchanged"));
        assert!(engine.run_pass().unwrap().errors.is_empty());
        fs::write(w.file("edited.jsonl"), &edited).unwrap();
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let captured = readings(&w);
    assert_eq!(chains(&w, "edited.jsonl").len(), 2);

    delete_index(&w.root);
    let (engine, lock) = one_worker(&w, 64 << 20);
    fault::arm("parse", "edited.jsonl", Action::Hold, 1, 1);
    let pass = held_pass(engine, lock);
    assert_eq!(
        building(&w),
        0,
        "the first build ended before the older chain"
    );
    let read = chains(&w, "edited.jsonl");
    assert_eq!(read.len(), 1, "only the newest chain is read: {read:?}");
    assert_eq!(read[0]["status"], "done");
    assert_eq!(read[0]["parsed_upto"], edited.len());
    assert_eq!(chains(&w, "steady.jsonl")[0]["status"], "done");
    assert_eq!(w.texts("edited"), ["final wording", "final answer"]);
    fault::release();
    let pass = pass.join().unwrap().unwrap();
    fault::clear();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let read = chains(&w, "edited.jsonl");
    assert_eq!(read.len(), 2);
    assert!(
        read.iter().all(|chain| chain["status"] == "done"),
        "{read:?}"
    );
    assert_eq!(readings(&w), captured);
}

/// A large session read in small ranges, with a record across every range
/// boundary, is rewritten once, so it holds an older chain too. A rebuild is
/// stopped while it reads the newest chain's fourth range, as a daemon
/// restart stops it. The next start resumes the newest chain after its last
/// committed record instead of reading it from the start, finishes it
/// before the older chain, and ends holding what capture wrote.
#[test]
fn a_rebuild_stopped_mid_chain_finishes_that_chain_first_on_restart() {
    fault::clear();
    const CHUNK: u64 = 4 << 10;
    let w = ClaudeHome::new();
    let record = |n: usize, version: &str| {
        let text = format!(
            "{version} turn {n} {}",
            "words of a long turn ".repeat(n % 7 + 6)
        );
        if n.is_multiple_of(2) {
            user(&format!("u{n}"), "big", &text)
        } else {
            assistant(&format!("a{n}"), "big", &text)
        }
    };
    let older: String = (0..40).map(|n| record(n, "older")).collect();
    let newest: String = (0..120).map(|n| record(n, "newest")).collect();
    let (older_end, newest_end) = (older.len() as u64, newest.len() as u64);
    assert!(newest_end > 8 * CHUNK);
    {
        let (mut engine, _lock) = w.engine(CHUNK);
        w.append("big.jsonl", &older);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        fs::write(w.file("big.jsonl"), &newest).unwrap();
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let captured = readings(&w);
    assert_eq!(w.texts("big").len(), 120);

    delete_index(&w.root);
    let (engine, lock) = one_worker(&w, CHUNK);
    let stop = engine.stop_request();
    fault::arm("parse", "big.jsonl", Action::Hold, 3, 1);
    let pass = held_pass(engine, lock);
    stop.request();
    fault::release();
    let _ = pass.join().unwrap();
    fault::clear();
    let stopped = chains(&w, "big.jsonl");
    assert_eq!(
        stopped.len(),
        1,
        "only the newest chain was started: {stopped:?}"
    );
    assert_eq!(stopped[0]["status"], "parsing");
    let resumed_at = stopped[0]["parsed_upto"].as_u64().unwrap();
    assert!(
        resumed_at > 2 * CHUNK && resumed_at < newest_end,
        "{resumed_at}"
    );
    // It stopped after a whole record; the next one crosses the range's end.
    assert_eq!(newest.as_bytes()[resumed_at as usize - 1], b'\n');

    let (mut engine, _lock) = one_worker(&w, CHUNK);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let read = chains(&w, "big.jsonl");
    assert_eq!(read.len(), 2);
    assert!(
        read.iter().all(|chain| chain["status"] == "done"),
        "{read:?}"
    );
    assert_eq!(read[0]["parsed_upto"], older_end);
    assert_eq!(read[1]["parsed_upto"], newest_end);
    assert!(
        read[1]["commit_seq"].as_i64() < read[0]["commit_seq"].as_i64(),
        "the newest chain finished before the older chain: {read:?}"
    );
    // Ranges overlap only by the record crossing each boundary, far less
    // than the newest chain's committed prefix.
    assert!(
        pass.bytes_read < older_end + newest_end,
        "the restart read {} bytes, the newest chain again from its start",
        pass.bytes_read
    );
    assert_eq!(readings(&w), captured);
}

/// A rebuild whose newest chain has a range the writer refuses for its
/// content (here a wire declaration the index already holds differently, as
/// two declarations sharing a kind once produced): the chain stops at the
/// refused range's start, failed with the writer's reason, which the pass
/// and the parse-state counts report; the first build still ends, and the
/// next pass does not read the refused range again.
#[test]
fn a_range_the_writer_refuses_fails_its_chain_and_the_first_build_ends() {
    fault::clear();
    const CHUNK: u64 = 4 << 10;
    let w = ClaudeHome::new();
    let session: String = (0..120)
        .map(|n| {
            let text = format!("turn {n} {}", "words of a long turn ".repeat(n % 7 + 6));
            if n % 2 == 0 {
                user(&format!("u{n}"), "big", &text)
            } else {
                assistant(&format!("a{n}"), "big", &text)
            }
        })
        .collect();
    assert!(session.len() as u64 > 6 * CHUNK);
    {
        let (mut engine, _lock) = w.engine(CHUNK);
        w.append("big.jsonl", &session);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }

    delete_index(&w.root);
    let (engine, lock) = one_worker(&w, CHUNK);
    fault::arm("parse", "big.jsonl", Action::Hold, 2, 1);
    let pass = held_pass(engine, lock);
    let cursor = chains(&w, "big.jsonl")[0]["parsed_upto"].as_u64().unwrap();
    assert!(cursor > 0, "the first ranges are committed");
    let index = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
    index
        .busy_timeout(std::time::Duration::from_secs(30))
        .unwrap();
    let changed = index
        .execute(
            "UPDATE wire_declaration SET expected = '[\"$.declared_otherwise\"]'",
            [],
        )
        .unwrap();
    assert!(
        changed > 0,
        "the committed ranges declared their wire shapes"
    );
    drop(index);
    fault::release();
    let pass = pass.join().unwrap().unwrap();
    fault::clear();
    let refusal = "wire declarations changed without a reading version change";
    assert!(
        pass.errors.iter().any(|error| error.contains(refusal)),
        "{:?}",
        pass.errors
    );
    assert_eq!(building(&w), 0, "the first build ended");
    let reads = w.reads();
    let reader = reads.reader().unwrap();
    let state = reader
        .query(
            "SELECT status, parsed_upto, at_length, instr(reason, ?1) > 0 AS has_reason
             FROM parse_state",
            &[polylogue_store::SqlValue::Text(refusal.into())],
        )
        .unwrap();
    assert_eq!(
        state,
        [
            serde_json::json!({"status": "failed", "parsed_upto": cursor,
                            "at_length": session.len(), "has_reason": 1})
        ]
    );
    assert_eq!(
        reader.status().unwrap().parse_states.get("failed"),
        Some(&1)
    );
    drop(reader);
    drop(reads);

    let (mut engine, _lock) = one_worker(&w, CHUNK);
    let again = engine.run_pass().unwrap();
    assert!(again.errors.is_empty(), "{:?}", again.errors);
    assert_eq!(again.bytes_read, 0, "the refused range is not read again");
}
