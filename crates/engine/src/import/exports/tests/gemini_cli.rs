use super::*;

#[test]
fn watched_gemini_turns_keep_tools_notices_and_header_facts_through_rewrites_and_rebuild() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".gemini/tmp/project/chats");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("session-stream.json");
    let dialogue = |edited: bool| {
        let mut messages = vec![
            json!({"id":"u1","type":"user","content":if edited{"revised question"}else{"question"},"timestamp":"2026-05-17T12:00:00Z"}),
            json!({"id":"g1","type":"gemini","content":"Listing.","model":"example","timestamp":"2026-05-17T12:00:05Z",
                "tokens":{"input":100,"output":7,"cached":60,"thoughts":20},
                "thoughts":[{"subject":"Plan","description":"inspect files"}],
                "toolCalls":[{"id":"call-1","name":"run_shell_command","args":{"command":"ls"},"status":"success",
                "result":[{"functionResponse":{"response":{"output":"a\nb"}}}]}]}),
            json!({"id":"notice","type":"info","content":"Request cancelled."}),
            json!({"id":"error","type":"error","content":"Synthetic API error."}),
        ];
        if edited {
            messages.push(json!({"id":"u2","type":"user","content":"continue","timestamp":"2026-05-17T12:00:10Z"}));
        }
        // Recognition's sessionId precedes messages; all other facts follow them.
        format!(
            "{{\"sessionId\":\"native-session\",\"messages\":{},\"summary\":\"{}\",\"directories\":[\"/synthetic/project\"],\"kind\":\"subagent\",\"startTime\":\"2026-05-17T12:00:00Z\",\"projectHash\":\"project-hash\"}}",
            serde_json::to_string(&messages).unwrap(),
            if edited {
                "Revised summary"
            } else {
                "Initial summary"
            }
        )
    };
    let mut final_bytes = String::new();
    {
        let (mut e, _lock) = engine(&w, 8);
        for edited in [false, true] {
            final_bytes = dialogue(edited);
            fs::write(&path, &final_bytes).unwrap();
            let pass = e.converge().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
            let r = e.reads.reader().unwrap();
            let sid = polylogue_model::SessionId::parse("gemini-cli:native-session").unwrap();
            let messages = r.messages(&sid, false).unwrap();
            assert_eq!(messages.len(), if edited { 3 } else { 2 });
            assert_eq!(messages[0].parent, None);
            assert_eq!(messages[1].parent.as_ref(), Some(&messages[0].id));
            assert_eq!(messages[1].usage.input, Some(40));
            assert_eq!(messages[1].usage.cached_input, Some(60));
            assert!(
                matches!(&messages[1].blocks[3],polylogue_model::Block::ToolResult{outcome:polylogue_model::ToolOutcome::Ok,output:Some(s),..} if s=="a\nb")
            );
            let session = r.session(&sid).unwrap().unwrap();
            assert_eq!(session.kind, polylogue_model::SessionKind::Subagent);
            assert_eq!(session.cwd, ["/synthetic/project"]);
            assert_eq!(
                session.title.unwrap().text,
                if edited {
                    "Revised summary"
                } else {
                    "Initial summary"
                }
            );
            let events = r.events(&sid).unwrap();
            for kind in ["notice", "api_error"] {
                let event = events.iter().find(|event| event.kind == kind).unwrap();
                assert_eq!(event.after.as_ref(), Some(&messages[1].id));
                assert_eq!(event.at, None);
                let (start, end) = event.range.unwrap();
                let raw: serde_json::Value =
                    serde_json::from_slice(&final_bytes.as_bytes()[start as usize..end as usize])
                        .unwrap();
                assert_eq!(raw["type"], if kind == "notice" { "info" } else { "error" });
            }
        }
    }
    fs::remove_file(&path).unwrap();
    let sid = polylogue_model::SessionId::parse("gemini-cli:native-session").unwrap();
    let before = w
        .reads()
        .reader()
        .unwrap()
        .session_export(&sid)
        .unwrap()
        .unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 8);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let after = e
        .reads
        .reader()
        .unwrap()
        .session_export(&sid)
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_value(after).unwrap(),
        serde_json::to_value(before).unwrap()
    );
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 6);
    assert!(!final_bytes.is_empty());
}
