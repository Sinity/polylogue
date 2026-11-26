use super::*;
use polylogue_engine::Engine;
use polylogue_sources::origin::by_name;
use polylogue_store::Archive;
use polylogue_testkit::{ClaudeHome, claude_user};

fn committed_continuation(
    w: &ClaudeHome,
) -> (
    Vec<polylogue_model::Json>,
    Vec<polylogue_model::Json>,
    Vec<polylogue_model::Json>,
) {
    let reads = w.reads();
    let reader = reads.reader().unwrap();
    (
        reader.query("SELECT p.chain,CASE WHEN p.state='failed' AND p.failure_kind='refused' THEN 'refused' WHEN p.state='failed' THEN 'failed' WHEN p.waiting>0 THEN 'waiting' ELSE 'done' END AS status,p.parsed_to AS parsed_upto,p.parser_version AS parse_version,hex(p.resume) AS checkpoint FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.format='claude-code' ORDER BY p.chain", &[]).unwrap(),
        reader.query("SELECT s.chain,s.response_key,s.source_start,s.source_end,hex(s.hash) AS version FROM message_parse s JOIN input_ref i ON i.pk=s.input_pk WHERE i.format='claude-code' AND s.response_key IS NOT NULL ORDER BY s.chain,s.response_key", &[]).unwrap(),
        reader.query("SELECT a.chain,a.name,a.target FROM message_alias a JOIN input_ref i ON i.pk=a.input_pk WHERE i.format='claude-code' ORDER BY a.chain,a.name", &[]).unwrap(),
    )
}

#[test]
fn rejected_replacement_keeps_the_committed_reading() {
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
        let before = committed_continuation(&w);
        let messages = w
            .reads()
            .reader()
            .unwrap()
            .messages(
                &polylogue_model::SessionId::parse("claude-code:s").unwrap(),
                false,
            )
            .unwrap();
        assert_eq!(
            messages.iter().map(|m| m.text()).collect::<Vec<_>>(),
            ["stored text", "second body"]
        );
        let mut engine = Engine::new(w.config(first.len() as u64), archive).unwrap();
        fault::arm(point, "", Action::Fail, u32::from(point == "parse"), 1);
        let failed = engine.run_pass().unwrap();
        assert_eq!(failed.errors.len(), 1, "{failed:?}");
        assert!(
            failed.errors[0]
                .to_string()
                .contains(&format!("injected fault at {point}")),
            "{failed:?}"
        );
        if point == "parse" {
            assert!(
                failed.bytes_read > first.len() as u64,
                "the parser failed after the first staged chunk: {failed:?}"
            );
        }
        assert_eq!(
            committed_continuation(&w),
            before,
            "a rejected {point} changed committed progress or keyed context"
        );
        assert_eq!(
            w.reads()
                .reader()
                .unwrap()
                .messages(
                    &polylogue_model::SessionId::parse("claude-code:s").unwrap(),
                    false
                )
                .unwrap(),
            messages
        );
        let retried = engine.run_pass().unwrap();
        assert!(retried.errors.is_empty(), "{retried:?}");
        assert!(retried.bytes_read > 0);
        assert_eq!(retried.bytes_stored, 0, "{retried:?}");
        assert_eq!(
            w.reads()
                .reader()
                .unwrap()
                .messages(
                    &polylogue_model::SessionId::parse("claude-code:s").unwrap(),
                    false
                )
                .unwrap(),
            messages
        );
        let rows=w.reads().reader().unwrap().query("SELECT parser_version AS parse_version FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.format='claude-code'",&[]).unwrap();
        assert_eq!(
            rows[0]["parse_version"],
            by_name("claude-code").unwrap().parse_version()
        );
        fault::clear();
        drop(engine);
        drop(lock);
    }
}
