use super::*;
use polylogue_model::freshness::{CompletenessDimension, CompletenessReason};
use polylogue_model::{Session, SessionId, SessionKind};
use polylogue_sources::Output;
use polylogue_testkit::{ClaudeHome, claude_user};
use std::sync::mpsc::{Sender, channel};
use std::time::Duration;

fn import(engine: &mut Engine, path: &Path) -> ImportReport {
    engine
        .import_with_options(
            &[path.to_path_buf()],
            &ImportOptions {
                origin: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap()
}

/// Add a real reading without requesting its derived values. This models work
/// left pending by an earlier writer, independently of the imported selection.
fn unrelated_reading(engine: &Engine, home: &ClaudeHome) {
    let output = Output {
        sessions: vec![Session::new(
            SessionId::parse("claude-code:unrelated").unwrap(),
            SessionKind::Main,
        )],
        ..Output::default()
    };
    let mut batch = polylogue_store::fixture::writer::step(
        "unrelated-input",
        "claude-code",
        "/project/unrelated.jsonl",
        output,
        0,
        1,
    );
    let mut raw = batch.clone();
    raw.parse.clear();
    let prepared = engine.archive.prepare_bytes(&b"r\n"[..]).unwrap();
    let stored = engine.writer.store_bytes(prepared, raw).unwrap();
    batch.change = polylogue_store::Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    engine.writer.send(batch).unwrap();
    let report = engine.writer.flush_rows().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(home.count("SELECT COUNT(*) AS n FROM summary_due"), 1);
}

/// Observe whether the caller returned while the derive wave was held. Always
/// release it, so the old blocking behaviour fails an annotation instead of
/// leaving the test's engine waiting forever.
fn held_derivation(engine: &Engine) -> (Sender<()>, std::thread::JoinHandle<bool>) {
    let pause = engine.writer.pause_derive_for_test();
    let (returned, receive) = channel();
    let held = std::thread::spawn(move || {
        pause.wait().expect("derivation reached its computed wave");
        let returned_while_held = receive.recv_timeout(Duration::from_secs(1)).is_ok();
        pause.release();
        returned_while_held
    });
    (returned, held)
}

fn derivation(home: &ClaudeHome) -> CompletenessDimension {
    home.read(|reader| reader.completeness(&[])).derivation
}

#[test]
fn unchanged_import_returns_while_unrelated_derivation_is_held_and_wakes_it() {
    let home = ClaudeHome::new();
    let path = home.dir.path().join("selected.jsonl");
    let original = claude_user("one", "selected", "kept original");
    std::fs::write(&path, &original).unwrap();
    let (mut engine, _lock) = engine(&home, 4096);
    let initial = import(&mut engine, &path);
    assert!(initial.pass.errors.is_empty(), "{initial:?}");
    assert_eq!(derivation(&home), CompletenessDimension::Complete);
    unrelated_reading(&engine, &home);
    let revisions = home.count("SELECT COUNT(*) AS n FROM a.revision");
    let blobs = home.count("SELECT COUNT(*) AS n FROM a.blob");
    let commits = engine.writer.completion_commits();
    let (returned, held) = held_derivation(&engine);

    let repeat = import(&mut engine, &path);
    // The wave remains held while the caller sees the durable pending state.
    let pending = derivation(&home);
    let _ = returned.send(());
    assert!(
        held.join().unwrap(),
        "unchanged import waited for derivation"
    );
    assert!(repeat.pass.errors.is_empty(), "{repeat:?}");
    assert_eq!(repeat.already_stored, [path]);
    assert!(repeat.imported.is_empty(), "{repeat:?}");
    assert!(repeat.pass.derivation_pending, "{repeat:?}");
    assert_eq!(repeat.pass.inputs_seen, 0, "{repeat:?}");
    assert_eq!(repeat.pass.bytes_read, 0, "{repeat:?}");
    assert_eq!(repeat.pass.bytes_stored, 0, "{repeat:?}");
    assert_eq!(repeat.pass.sessions_derived, 0, "{repeat:?}");
    assert_eq!(
        pending,
        CompletenessDimension::Incomplete(vec![CompletenessReason::DerivationBehind])
    );
    assert_eq!(
        home.count("SELECT COUNT(*) AS n FROM a.revision"),
        revisions
    );
    assert_eq!(home.count("SELECT COUNT(*) AS n FROM a.blob"), blobs);
    assert_eq!(home.texts("selected"), ["kept original"]);

    // No further import, flush or derive request: the import's ordinary wake
    // must finish the queued work and clear its durable marks.
    let deadline = Instant::now() + Duration::from_secs(5);
    while derivation(&home) != CompletenessDimension::Complete {
        commits
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("background derivation committed");
    }
    assert_eq!(home.count("SELECT COUNT(*) AS n FROM summary_due"), 0);
    let current = import(&mut engine, &repeat.already_stored[0]);
    assert!(!current.pass.derivation_pending, "{current:?}");
}

#[test]
fn changed_import_waits_for_its_derivation() {
    let home = ClaudeHome::new();
    let path = home.dir.path().join("selected.jsonl");
    let first = claude_user("one", "selected", "first");
    std::fs::write(&path, &first).unwrap();
    let (mut engine, _lock) = engine(&home, 4096);
    let initial = import(&mut engine, &path);
    assert!(initial.pass.errors.is_empty(), "{initial:?}");
    unrelated_reading(&engine, &home);
    let changed_bytes = first + &claude_user("two", "selected", "second");
    std::fs::write(&path, &changed_bytes).unwrap();
    let (returned, held) = held_derivation(&engine);
    let changed = import(&mut engine, &path);
    let _ = returned.send(());
    assert!(!held.join().unwrap(), "changed import skipped derivation");
    assert!(changed.pass.errors.is_empty(), "{changed:?}");
    assert_eq!(
        changed.pass.bytes_read,
        changed_bytes.len() as u64,
        "{changed:?}"
    );
    assert_eq!(changed.pass.messages_written, 1, "{changed:?}");
    assert!(changed.pass.sessions_derived > 0, "{changed:?}");
    assert!(!changed.pass.derivation_pending, "{changed:?}");
    assert_eq!(home.texts("selected"), ["first", "second"]);
    assert_eq!(derivation(&home), CompletenessDimension::Complete);
    assert_eq!(home.count("SELECT COUNT(*) AS n FROM summary_due"), 0);
}
