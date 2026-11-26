//! Properties that need a failure injected at a chosen point (the `faults`
//! feature): run with `cargo xtask test -p polylogue-engine -E 'test(/^faults::/)'`.

#![cfg(feature = "faults")]

use polylogue_store::SqlValue;
use polylogue_store::fault::{self, Action};
use polylogue_testkit::{Home, Set, session};
use std::fs;

mod identical_chains;
mod older_chains;
mod prefix_copies;
mod reparse;

#[test]
fn arming_an_undeclared_point_names_it() {
    let error = std::panic::catch_unwind(|| {
        fault::arm("write.between_commits", "", Action::Fail, 0, 1);
    })
    .expect_err("an undeclared point must fail when armed");
    assert_eq!(
        error.downcast_ref::<String>().unwrap(),
        "undeclared fault point: write.between_commits"
    );
}

#[test]
fn checking_an_undeclared_point_names_it() {
    let error = std::panic::catch_unwind(|| {
        fault::check("write.misspelled_commit", "").unwrap();
    })
    .expect_err("an undeclared point must fail when checked");
    assert_eq!(
        error.downcast_ref::<String>().unwrap(),
        "undeclared fault point: write.misspelled_commit"
    );
}

mod write_failures;

/// a failing parser loses only its input. #2's parser panics: every
/// other session is built, the pass names the failure, an unchanged file is
/// not tried again, and a change to it parses it once the parser works.
#[test]
fn a_panicking_parser_loses_only_its_input() {
    fault::clear();
    let w = Home::morrow(Set::Main);
    let two = "7f3a2c10-0002-4000-8000-000000000002.jsonl";
    fault::arm("parse", two, Action::Panic, 0, u32::MAX);
    let (mut engine, _lock) = w.engine(polylogue_testkit::WHOLE_FILES);
    let pass = engine.run_pass().unwrap();
    assert!(
        pass.errors
            .iter()
            .any(|e| e.to_string().contains("panicked") && e.to_string().contains(two)),
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
    let file = w.path(&format!(".claude/projects/-home-op-morrow/{two}"));
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
    let w = Home::morrow(Set::Main).stored();
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
