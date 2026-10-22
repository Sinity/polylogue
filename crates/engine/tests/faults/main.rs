//! Laws that need a failure injected at a chosen point (the `faults`
//! feature): run with `cargo nextest run -p polylogue-engine --features
//! faults --test faults`.

#![cfg(feature = "faults")]

use polylogue_store::SqlValue;
use polylogue_store::fault::{self, Action};
use polylogue_testkit::{Morrow, Set, session};
use std::fs;

/// Law 7: a failure at any step of a write loses nothing. The writer fails
/// at its n-th archive commit, between its archive and index commits, or at
/// its index commit; the engine is dropped as a crashed process would leave
/// it, and a fresh one converges to the uninterrupted view with every input
/// reading back byte for byte.
#[test]
fn law_7_a_failed_write_step_loses_nothing() {
    let reference = Morrow::build(Set::Main).view();
    for point in [
        "write.archive_commit",
        "write.between_commits",
        "write.index_commit",
    ] {
        for skip in 0..4 {
            fault::clear();
            let w = Morrow::lay_out(Set::Main);
            fault::arm(point, "", Action::Fail, skip, 1);
            {
                let (mut engine, _lock) = w.engine();
                let _ = engine.converge();
                let _ = w.import_exports(&mut engine);
            }
            fault::clear();
            w.converge();
            assert_eq!(
                w.view(),
                reference,
                "{point} after {skip} writes: the view differs"
            );
            for file in w.provider_files() {
                if let Ok(stored) = w.stored_bytes(&file) {
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

/// Law 63: a failing parser loses only its input. #2's parser panics: every
/// other session is built, the pass names the failure, an unchanged file is
/// not tried again, and a change to it parses it once the parser works.
#[test]
fn law_63_a_panicking_parser_loses_only_its_input() {
    fault::clear();
    let w = Morrow::lay_out(Set::Main);
    let two = "7f3a2c10-0002-4000-8000-000000000002.jsonl";
    fault::arm("parse", two, Action::Panic, 0, u32::MAX);
    let (mut engine, _lock) = w.engine();
    let pass = engine.converge().unwrap();
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
    let again = engine.converge().unwrap();
    assert!(
        again.errors.is_empty(),
        "retried an unchanged failing input: {:?}",
        again.errors
    );
    // Changed, with the parser working again: parsed.
    fault::clear();
    let file = w.file(&format!(".claude/projects/-home-op-morrow/{two}"));
    let mut text = fs::read_to_string(&file).unwrap();
    text.push_str("{\"type\":\"user\",\"uuid\":\"u2-late\",\"sessionId\":\"7f3a2c10-0002-4000-8000-000000000002\",\"timestamp\":\"2026-10-08T09:00:00Z\",\"promptSource\":\"typed\",\"message\":{\"role\":\"user\",\"content\":\"later\"}}\n");
    fs::write(&file, text).unwrap();
    let after = engine.converge().unwrap();
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

/// Law 45 (store part): a failed read is an error, never an empty answer.
#[test]
fn law_45_a_failed_read_is_never_empty() {
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

mod excision;

mod blob_cleanup;

mod index_cleanup;
