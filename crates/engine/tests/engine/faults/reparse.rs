use super::*;
use polylogue_engine::Engine;
use polylogue_sources::origin::by_name;
use polylogue_store::Archive;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn a_reparse_write_failure_retries_without_changed_source_bytes() {
    for point in ["reparse.prepare", "reparse.commit"] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append("s.jsonl", &claude_user("message", "s", "stored text"));
        let (mut engine, lock) = w.engine(64 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        drop(engine);
        let archive = Archive::at(&w.root);
        assert_eq!(
            polylogue_store::fixture::older_reading(&archive, "claude-code")
                .unwrap()
                .0,
            1
        );
        let mut engine = Engine::new(w.config(64 << 20), archive).unwrap();
        fault::arm(point, "", Action::Fail, 0, 1);
        let failed = engine.run_pass().unwrap();
        assert!(!failed.errors.is_empty());
        assert_eq!(w.texts("s"), ["stored text"]);
        let retried = engine.run_pass().unwrap();
        assert!(retried.errors.is_empty(), "{retried:?}");
        assert!(retried.bytes_read > 0);
        let rows=w.reads().reader().unwrap().query("SELECT reading_version FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.format='claude-code'",&[]).unwrap();
        assert_eq!(
            rows[0]["reading_version"],
            by_name("claude-code").unwrap().reading_version()
        );
        fault::clear();
        drop(engine);
        drop(lock);
    }
}
