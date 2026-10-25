use super::*;
use polylogue_engine::{ExciseRequest, ExciseScope};
use polylogue_testkit::{ClaudeHome, claude_user, delete_index};

#[test]
fn an_older_chain_parse_failure_waits_without_deleting_bytes_and_retries_after_restart_or_excision() {
    for excision in [false, true] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append(
            "history.jsonl",
            &claude_user("old", "history", "historical unique secret"),
        );
        let before;
        {
            let (mut engine, _lock) = w.engine(64 << 20);
            assert!(engine.converge().unwrap().errors.is_empty());
            fs::write(
                w.file("history.jsonl"),
                claude_user("new", "history", "current sibling"),
            )
            .unwrap();
            assert!(engine.converge().unwrap().errors.is_empty());
            before = w.texts("history");
            assert_eq!(before.len(), 2);
        }
        delete_index(&w.root);
        fault::arm("parse", "history.jsonl", Action::Panic, 0, 1);
        let (mut engine, lock) = w.engine(64 << 20);
        let first = engine.converge().unwrap();
        assert_eq!(first.errors.len(), 1, "{:?}", first.errors);
        assert!(first.errors[0].contains("panicked"));
        assert_eq!(w.texts("history"), ["current sibling"]);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 2);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM parse_state"),
            1,
            "failed history is not marked parsed"
        );
        // The fault was one-shot: retry would now succeed, but unchanged
        // failure still waits rather than silently retrying on every pass.
        let unchanged = engine.converge().unwrap();
        assert!(unchanged.errors.is_empty(), "{:?}", unchanged.errors);
        assert_eq!(unchanged.bytes_read, 0);
        assert_eq!(w.texts("history"), ["current sibling"]);
        fault::clear();
        if excision {
            let done = engine
                .excise(ExciseRequest {
                    id: "clear-historical-secret".into(),
                    scope: ExciseScope::parse("text:historical unique secret").unwrap(),
                    reason: None,
                    dry_run: false,
                })
                .unwrap();
            assert!(!done.inputs_reparsed.is_empty());
            let after = engine.converge().unwrap();
            assert!(after.errors.is_empty(), "{:?}", after.errors);
            assert!(after.bytes_read > 0, "changed stored history is retried");
            let messages = w.texts("history");
            assert_eq!(messages.len(), 2);
            assert!(messages.iter().any(|text| text == "current sibling"));
            assert!(
                messages
                    .iter()
                    .all(|text| !text.contains("historical unique secret"))
            );
        } else {
            drop(engine);
            drop(lock);
            let (mut restarted, _lock) = w.engine(64 << 20);
            let after = restarted.converge().unwrap();
            assert!(after.errors.is_empty(), "{:?}", after.errors);
            assert_eq!(w.texts("history"), before);
        }
    }
}
