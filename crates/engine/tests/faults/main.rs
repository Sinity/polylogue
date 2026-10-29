//! Properties that need a failure injected at a chosen point (the `faults`
//! feature): run with `cargo nextest run -p polylogue-engine --features
//! faults --test faults`.

#![cfg(feature = "faults")]

use polylogue_store::SqlValue;
use polylogue_store::fault::{self, Action};
use polylogue_testkit::{Morrow, Set, session};
use std::fs;

mod older_chains;
mod reparse;

/// a failure at any step of a write loses nothing. The writer fails
/// at its n-th archive commit, between its archive and index commits, or at
/// its index commit; the engine is dropped as a crashed process would leave
/// it, and a fresh one runs a pass and comes back to the uninterrupted view with every input
/// reading back byte for byte.
#[test]
fn a_failed_write_step_loses_nothing() {
    let source = Morrow::build(Set::Main);
    let reference = source.view();
    for point in [
        "write.archive_commit",
        "write.between_commits",
        "write.index_commit",
    ] {
        for skip in 0..4 {
            fault::clear();
            let mut w = Morrow::lay_out(Set::Main);
            // Source IDs include paths, so both builds read the same home.
            w.home = source.home.clone();
            fault::arm(point, "", Action::Fail, skip, 1);
            {
                let (mut engine, _lock) = w.engine();
                let _ = engine.run_pass();
                let _ = w.import_exports(&mut engine);
            }
            fault::clear();
            w.run_pass();
            assert_eq!(
                w.view(),
                reference,
                "{point} after {skip} writes: the view differs"
            );
            for file in w.provider_files() {
                if let Ok(stored) = w.stored_bytes(&file) {
                    if polylogue_sources::ORIGINS
                        .iter()
                        .any(|origin| origin.is_row_dump(&file))
                    {
                        assert!(!stored.is_empty());
                        for record in stored.split(|b| *b == b'\n').filter(|r| !r.is_empty()) {
                            serde_json::from_slice::<serde_json::Value>(record).unwrap();
                        }
                        continue;
                    }
                    assert_eq!(
                        stored,
                        fs::read(&file).unwrap(),
                        "{point}/{skip}: {}",
                        file.display()
                    );
                }
            }
        }
    }
}

/// a failing parser loses only its input. #2's parser panics: every
/// other session is built, the pass names the failure, an unchanged file is
/// not tried again, and a change to it parses it once the parser works.
#[test]
fn a_panicking_parser_loses_only_its_input() {
    fault::clear();
    let w = Morrow::lay_out(Set::Main);
    let two = "7f3a2c10-0002-4000-8000-000000000002.jsonl";
    fault::arm("parse", two, Action::Panic, 0, u32::MAX);
    let (mut engine, _lock) = w.engine();
    let pass = engine.run_pass().unwrap();
    assert!(
        pass.errors
            .iter()
            .any(|e| e.contains("panicked") && e.contains(two)),
        "{:?}",
        pass.errors
    );
    let pool = w.reads();
    let r = pool.reader().unwrap();
    for n in [1, 3, 4, 5, 6] {
        assert!(r.session(&session(n)).unwrap().is_some(), "#{n} is missing");
    }
    drop(r);
    drop(pool);
    // Unchanged: not tried again.
    let again = engine.run_pass().unwrap();
    assert!(
        again.errors.is_empty(),
        "retried an unchanged failing input: {:?}",
        again.errors
    );
    assert_eq!(
        again.bytes_read, 0,
        "unchanged failing input is not read again"
    );
    // Changed, with the parser working again: parsed.
    fault::clear();
    let file = w.file(&format!(".claude/projects/-home-op-morrow/{two}"));
    let mut text = fs::read_to_string(&file).unwrap();
    text.push_str("{\"type\":\"user\",\"uuid\":\"u2-late\",\"sessionId\":\"7f3a2c10-0002-4000-8000-000000000002\",\"timestamp\":\"2026-10-08T09:00:00Z\",\"promptSource\":\"typed\",\"message\":{\"role\":\"user\",\"content\":\"later\"}}\n");
    fs::write(&file, text).unwrap();
    let after = engine.run_pass().unwrap();
    assert!(after.errors.is_empty(), "{:?}", after.errors);
    drop(engine);
    assert!(
        w.reads()
            .reader()
            .unwrap()
            .session(&session(2))
            .unwrap()
            .is_some()
    );
}

/// a failed read is an error, never an empty answer.
#[test]
fn a_failed_read_is_never_empty() {
    fault::clear();
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    fault::arm("read.query", "", Action::Fail, 0, 1);
    let got = r.query(
        "SELECT id FROM session WHERE origin = ?1",
        &[SqlValue::Text("claude-code".into())],
    );
    assert!(got.is_err(), "a failed read answered {got:?}");
    let ok = r.query("SELECT count(*) AS n FROM session", &[]).unwrap();
    assert_eq!(ok[0]["n"], 8);
}

mod asset_lookup;

mod tool_exports;

mod writer_retry;
