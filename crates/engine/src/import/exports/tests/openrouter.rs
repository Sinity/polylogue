use super::*;

const ROOM: &str = r#"{"version":"example","messages":{
    "repeated":{"id":"old-first","type":"user","content":"superseded","createdAt":"2020-01-01T00:00:00Z"},
    "middle":{"id":"m-kept","type":"user","content":"First","createdAt":"2026-01-01T00:00:00Z"},
    "repeated":{"id":"z-after","type":"assistant","characterId":"char","parentMessageId":"m-kept",
        "content":"Second","createdAt":"2026-01-02T00:00:00Z"}},
    "title":"Late header","characters":{"char":{"model":"example-model"}}}"#;

#[test]
fn keyed_room_replacements_choose_the_first_retained_message_and_rebuild_exactly() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("OpenRouter Chat example.json");
    fs::write(&path, ROOM).unwrap();
    let sid = polylogue_model::SessionId::parse("openrouter:m-kept").unwrap();
    let mut previous = None;
    for rebuild in [false, true] {
        if rebuild {
            fs::remove_file(&path).unwrap();
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
        assert_eq!(messages[1].model.as_deref(), Some("example-model"));
        let sources=r.query("SELECT source_start,source_end,source_order FROM message_source ORDER BY source_order",&[]).unwrap();
        assert_eq!(sources.len(), 2);
        for (index, source) in sources.iter().enumerate() {
            let range = source["source_start"].as_u64().unwrap() as usize
                ..source["source_end"].as_u64().unwrap() as usize;
            let raw: serde_json::Value = serde_json::from_slice(&ROOM.as_bytes()[range]).unwrap();
            assert_eq!(raw["id"], messages[index].id.native().as_str());
            assert_eq!(source["source_order"], index + 1);
        }
        let events = r.events(&sid).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, json!({"char":{"model":"example-model"}}));
        assert_eq!(events[0].range, Some((0, ROOM.len() as u64)));
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 3);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
        let snapshot = serde_json::to_value((&messages, &events)).unwrap();
        if let Some(old) = previous {
            assert_eq!(snapshot, old);
        }
        previous = Some(snapshot);
    }
}

#[test]
fn a_bad_room_suffix_refuses_before_any_header_or_message_is_published() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("OpenRouter Chat example.json");
    fs::write(&path, format!("{ROOM} false")).unwrap();
    let (mut e, _lock) = engine(&w, 16);
    let report = e.import(&[path]).unwrap();
    assert!(report.pass.errors.is_empty());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 0);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM session"), 0);
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 1);
}
