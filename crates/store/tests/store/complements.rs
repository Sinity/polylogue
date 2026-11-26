//! `cwd:<dir>` and its negation (docs/query.md, field rules: `cwd` matches
//! a subtree by whole path components, case-sensitively; an unknown value
//! matches neither a condition nor its `NOT`, and `cwd:unknown` names it).
//! Over physical sessions the three together list every session once.

use crate::session_list;
use polylogue_testkit::{ClaudeHome, Morrow, Set};
use serde_json::json;

/// Morrow's agent sessions all work in `/home/op/morrow`; the two web
/// chats record no directory (FACTS.md). Physical sessions, newest
/// activity first: #6, #5, #4, #2 (until 11:15), #3 (until 10:26), #1.
#[test]
fn morrow_splits_into_its_directory_and_the_chats_without_one() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    assert_eq!(
        session_list(&reader, "physical:true cwd:/home/op/morrow"),
        ["#6", "#5", "#4", "#2", "#3", "#1"]
    );
    assert_eq!(
        session_list(&reader, "physical:true cwd:/home/op"),
        ["#6", "#5", "#4", "#2", "#3", "#1"]
    );
    assert_eq!(
        session_list(&reader, "physical:true cwd:/home/op/mor"),
        Vec::<String>::new()
    );
    assert_eq!(
        session_list(&reader, "physical:true NOT cwd:/home/op/morrow"),
        Vec::<String>::new()
    );
    assert_eq!(
        session_list(&reader, "physical:true cwd:unknown"),
        ["#8", "#7"]
    );
}

/// Five sessions in neighbouring directories, one with none.
#[test]
fn a_directory_and_its_negation_list_every_session_with_a_directory_once() {
    let home = ClaudeHome::new();
    for (n, cwd) in [
        (1, Some("/home/op/morrow")),
        (2, Some("/home/op/morrow/crates/parse")),
        (3, Some("/home/op/morrow-old")),
        (4, Some("/home/op/Morrow")),
        (5, None),
    ] {
        let id = format!("c0000000-0000-4000-8000-00000000000{n}");
        let mut line = json!({
            "type": "user", "uuid": format!("u{n}"), "sessionId": id,
            "timestamp": format!("2026-10-01T0{n}:00:00Z"),
            "message": {"role": "user", "content": format!("Work in place {n}.")},
        });
        if let Some(cwd) = cwd {
            line["cwd"] = json!(cwd);
        }
        home.append(&format!("{id}.jsonl"), &format!("{line}\n"));
    }
    let (mut engine, _lock) = home.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let id = |n: u8| format!("claude-code:c0000000-0000-4000-8000-00000000000{n}");
    assert_eq!(session_list(&reader, "cwd:/home/op/morrow"), [id(2), id(1)]);
    assert_eq!(
        session_list(&reader, "NOT cwd:/home/op/morrow"),
        [id(4), id(3)]
    );
    assert_eq!(session_list(&reader, "cwd:unknown"), [id(5)]);
}
