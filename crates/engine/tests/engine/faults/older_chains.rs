use super::*;
use polylogue_testkit::{ClaudeHome, claude_user, delete_index};

#[test]
fn an_older_chain_parse_failure_waits_across_restart_until_the_parser_version_changes() {
    {
        fault::clear();
        let w = ClaudeHome::new();
        w.append(
            "history.jsonl",
            &claude_user("old", "history", "historical unique secret"),
        );
        let before;
        {
            let (mut engine, _lock) = w.engine(64 << 20);
            assert!(engine.run_pass().unwrap().errors.is_empty());
            fs::write(
                w.file("history.jsonl"),
                claude_user("new", "history", "current sibling"),
            )
            .unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
            before = w.texts("history");
            let mut sorted = before.clone();
            sorted.sort();
            assert_eq!(sorted, ["current sibling", "historical unique secret"]);
        }
        delete_index(&w.root);
        fault::arm("parse", "history.jsonl", Action::Panic, 1, 1);
        let (mut engine, lock) = w.engine(64 << 20);
        // One worker reads the newest chain first, so the second parse,
        // the one that fails, is the older chain's.
        engine.config.workers = 1;
        let first = engine.run_pass().unwrap();
        assert_eq!(first.errors.len(), 1, "{:?}", first.errors);
        assert!(first.errors[0].to_string().contains("panicked"));
        assert_eq!(w.texts("history"), ["current sibling"]);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 2);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM parse_state WHERE state<>'failed'"),
            1,
            "failed history is not marked parsed"
        );
        // The failure is the older chain's parse state (docs/archive.md).
        assert_eq!(
            w.count("SELECT count(*) AS n FROM parse_state WHERE state='failed'"),
            1
        );
        // The fault was one-shot: retry would now succeed, but unchanged
        // failure still waits rather than silently retrying on every pass.
        let unchanged = engine.run_pass().unwrap();
        assert!(unchanged.errors.is_empty(), "{:?}", unchanged.errors);
        assert_eq!(unchanged.bytes_read, 0);
        assert_eq!(w.texts("history"), ["current sibling"]);
        fault::clear();
        drop(engine);
        drop(lock);
        let (mut restarted, _lock) = w.engine(64 << 20);
        let unchanged = restarted.run_pass().unwrap();
        assert!(unchanged.errors.is_empty(), "{unchanged:?}");
        assert_eq!(unchanged.bytes_read, 0, "{unchanged:?}");
        assert_eq!(w.texts("history"), ["current sibling"]);
        polylogue_store::fixture::older_reading(&restarted.archive, "claude-code").unwrap();
        let after = restarted.run_pass().unwrap();
        assert!(after.errors.is_empty(), "{:?}", after.errors);
        assert_eq!(after.bytes_stored, 0, "{after:?}");
        assert_eq!(w.texts("history"), before);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 2);
    }
}
