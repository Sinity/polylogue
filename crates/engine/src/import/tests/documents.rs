use super::*;
use polylogue_testkit::ClaudeHome;
use serde_json::json;

#[test]
fn whole_documents_cite_the_retained_bytes_and_preserve_reading_order_on_rebuild() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("OpenRouter Chat example.json");
    let bytes = br#"{
        "version":"example", "title":"Example", "characters":{}, "messages":{
            "a-last":{"id":"a-last","type":"assistant","createdAt":"2026-01-01T00:01:00Z",
                "parentMessageId":"z-first","content":"Second"},
            "z-first":{"id":"z-first","type":"user","createdAt":"2026-01-01T00:00:00Z","content":"First"}
        }}"#.to_vec();
    std::fs::write(&path, &bytes).unwrap();
    let sid = polylogue_model::SessionId::parse("openrouter:z-first").unwrap();
    for rebuild in [false, true] {
        if rebuild {
            std::fs::remove_file(&path).unwrap();
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut e, _lock) = engine(&w, 16);
        let errors = if rebuild {
            e.converge().unwrap().errors
        } else {
            e.import(std::slice::from_ref(&path)).unwrap().pass.errors
        };
        assert!(errors.is_empty(), "{errors:?}");
        let r = e.reads.reader().unwrap();
        let messages = r.messages(&sid, false).unwrap();
        assert_eq!(
            messages.iter().map(|m| m.text()).collect::<Vec<_>>(),
            ["First", "Second"]
        );
        assert_eq!(messages[1].parent.as_ref(), Some(&messages[0].id));
        let rows = r.query("SELECT source_start,source_end,source_order FROM message_source ORDER BY source_order",&[]).unwrap();
        assert_eq!(
            rows,
            vec![
                json!({"source_start":0,"source_end":bytes.len(),"source_order":0}),
                json!({"source_start":0,"source_end":bytes.len(),"source_order":1})
            ]
        );
        let events = r.events(&sid).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].range, Some((0, bytes.len() as u64)));
        assert_eq!(events[0].data, json!({}));
    }
}
