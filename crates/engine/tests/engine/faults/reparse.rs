use super::*;
use polylogue_engine::Engine;
use polylogue_sources::origin::by_name;
use polylogue_store::Archive;
use polylogue_testkit::{ClaudeHome, claude_user};

fn admitted_continuation(
    w: &ClaudeHome,
) -> (Vec<polylogue_model::Json>, Vec<polylogue_model::Json>) {
    let reads = w.reads();
    let reader = reads.reader().unwrap();
    (
        reader.query("SELECT p.chain,p.status,p.parsed_upto,p.reading_version,hex(p.checkpoint) AS checkpoint FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.format='claude-code' ORDER BY p.chain", &[]).unwrap(),
        reader.query("SELECT c.chain,c.reading_version,c.cursor,c.key,hex(c.value) AS value,hex((SELECT version_hash FROM message_version WHERE pk=c.version_pk)) AS version FROM parser_context c JOIN input_ref i ON i.pk=c.input_pk WHERE i.format='claude-code' ORDER BY c.chain,c.reading_version,c.key,c.cursor", &[]).unwrap(),
    )
}

#[test]
fn a_rejected_replacement_preserves_the_admitted_reading_and_retries_without_source_changes() {
    for point in ["reparse.prepare", "reparse.commit", "parse"] {
        fault::clear();
        let w = ClaudeHome::new();
        let first = claude_user("message", "s", "stored text");
        let second = claude_user("later", "s", "second body");
        w.append("s.jsonl", &(first.clone() + &second));
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
        let before = admitted_continuation(&w);
        let messages = w.export("claude-code:s").messages;
        let mut engine = Engine::new(
            w.config(first.len() as u64),
            archive,
            polylogue_testkit::stopped_attempt(),
        )
        .unwrap();
        fault::arm(point, "", Action::Fail, u32::from(point == "parse"), 1);
        let failed = engine.run_pass().unwrap();
        assert_eq!(failed.errors.len(), 1, "{failed:?}");
        assert!(
            failed.errors[0].contains(&format!("injected fault at {point}")),
            "{failed:?}"
        );
        if point == "parse" {
            assert!(
                failed.bytes_read > first.len() as u64,
                "the parser failed after the first staged chunk: {failed:?}"
            );
        }
        assert_eq!(
            admitted_continuation(&w),
            before,
            "a rejected {point} changed admitted progress or keyed context"
        );
        assert_eq!(w.export("claude-code:s").messages, messages);
        let retried = engine.run_pass().unwrap();
        assert!(retried.errors.is_empty(), "{retried:?}");
        assert!(retried.bytes_read > 0);
        assert_eq!(retried.bytes_stored, 0, "{retried:?}");
        assert_eq!(w.export("claude-code:s").messages, messages);
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
