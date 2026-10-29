use super::*;

#[test]
fn openai_message_arrays_and_wrappers_rebuild_the_same_tool_dialogue_from_stored_bytes() {
    let messages = r#"[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"reasoning_content":"need a tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}}]},{"role":"tool","tool_call_id":"call-1","content":"rain"},{"role":"assistant","content":"Rain."}]"#;
    for wrapped in [false, true] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("chat.json");
        let bytes = if wrapped {
            format!(
                "{{\"messages\":{messages},\"id\":\"chosen\",\"model\":\"example\",\"title\":\"Weather\"}}"
            )
        } else {
            messages.to_owned()
        };
        fs::write(&path, &bytes).unwrap();
        let sid = polylogue_model::SessionId::parse(if wrapped {
            "openai-chat:chosen"
        } else {
            "openai-chat:chat"
        })
        .unwrap();
        let mut previous = None;
        for rebuild in [false, true] {
            if rebuild {
                fs::remove_file(&path).unwrap();
                polylogue_testkit::delete_index(&w.root);
            }
            let (mut e, _lock) = engine(&w, 8);
            if rebuild {
                assert!(e.run_pass().unwrap().errors.is_empty());
            } else {
                let report = e
                    .import_with_options(
                        std::slice::from_ref(&path),
                        &ImportOptions {
                            format: Some("openai-chat".into()),
                            explain: false,
                        },
                    )
                    .unwrap();
                assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
                assert_eq!(report.imported.as_slice(), std::slice::from_ref(&path));
            }
            let r = e.reads.reader().unwrap();
            let shown = r.messages(&sid, false).unwrap();
            assert_eq!(shown.len(), 4);
            assert_eq!(shown[3].parent.as_ref(), Some(&shown[2].id));
            assert_eq!(shown[3].text(), "Rain.");
            assert_eq!(
                shown[3].id.native().as_str(),
                if wrapped { "chosen#3" } else { "chat#3" }
            );
            assert_eq!(shown[3].model.as_deref(), wrapped.then_some("example"));
            assert_eq!(
                shown[2].prompt_kind,
                Some(polylogue_model::PromptKind::Harness)
            );
            assert!(
                matches!(&shown[1].blocks[1],polylogue_model::Block::ToolCall{input,..} if input["city"]=="Oslo")
            );
            assert!(
                matches!(&shown[2].blocks[0],polylogue_model::Block::ToolResult{output:Some(s),..} if s=="rain")
            );
            let ranges = r
                .query("SELECT source_start,source_end FROM message_source", &[])
                .unwrap();
            assert_eq!(ranges.len(), 4);
            for row in ranges {
                let start = row["source_start"].as_u64().unwrap() as usize;
                let end = row["source_end"].as_u64().unwrap() as usize;
                let raw: serde_json::Value =
                    serde_json::from_slice(&bytes.as_bytes()[start..end]).unwrap();
                assert!(raw.get("role").is_some());
            }
            assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 5);
            assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
            let view = serde_json::to_value(&shown).unwrap();
            if let Some(old) = previous {
                assert_eq!(view, old);
            }
            previous = Some(view);
        }
    }
}

#[test]
fn malformed_openai_wrapper_suffix_commits_no_prefix_messages() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("chat.json");
    fs::write(
        &path,
        r#"{"messages":[{"role":"user","content":"never committed"}],"model":"example"} false"#,
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let report = e
        .import_with_options(
            &[path],
            &ImportOptions {
                format: Some("openai-chat".into()),
                explain: false,
            },
        )
        .unwrap();
    assert!(report.pass.errors.is_empty());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 0);
    assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 1);
}
