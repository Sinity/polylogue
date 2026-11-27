use super::*;
use polylogue_model::ToolOutcome;
use serde_json::json;

/// The rule as a file found at a location meets it: its first bytes alone.
fn recognize(path: &std::path::Path, head: &[u8]) -> bool {
    super::recognize(path, &crate::document::Head::new(head))
}

fn session(messages: Vec<Json>) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "sessionId": "s1", "projectHash": "abc", "startTime": "2026-03-14T18:01:43.756Z",
        "lastUpdated": "2026-03-14T18:46:59.577Z", "kind": "main", "summary": "Review the code",
        "messages": messages
    }))
    .unwrap()
}

#[test]
fn a_turn_with_a_tool_call_and_its_result() {
    let bytes = session(vec![
        json!({"id": "u1", "timestamp": "2026-03-14T18:01:43.756Z", "type": "user",
               "content": [{"text": "list files"}]}),
        json!({"id": "g1", "timestamp": "2026-03-14T18:01:50Z", "type": "gemini",
               "content": "Listing.", "model": "gemini-3-pro",
               "thoughts": [{"subject": "Plan", "description": "run ls", "timestamp": "x"}],
               "tokens": {"input": 100, "output": 7, "cached": 60, "thoughts": 20, "tool": 0, "total": 127},
               "toolCalls": [{"id": "c1", "name": "run_shell_command", "args": {"command": "ls"},
                              "result": [{"functionResponse": {"id": "c1", "name": "run_shell_command",
                                                               "response": {"output": "a\nb"}}}],
                              "status": "success", "timestamp": "x", "resultDisplay": "a\nb",
                              "displayName": "Shell", "description": "d", "renderOutputAsMarkdown": false}]}),
        json!({"id": "i1", "timestamp": "2026-03-14T18:02:00Z", "type": "info",
               "content": "Request cancelled."}),
    ]);
    let step =
        crate::export::fixture::parse(&mut Parse::Export(Box::new(GeminiCli::default())), &bytes)
            .unwrap();
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs[0].text(), "list files");
    assert_eq!(msgs[1].parent.as_ref().unwrap().native().as_str(), "u1");
    assert_eq!(msgs[1].usage.input, Some(40));
    assert_eq!(msgs[1].usage.cached_input, Some(60));
    assert_eq!(msgs[1].usage.output, Some(7));
    assert_eq!(msgs[1].usage.reasoning, Some(20));
    match &msgs[1].blocks[2] {
        Block::ToolCall {
            run_by_script: false,
            call_id,
            tool,
            command,
            action,
            ..
        } => {
            assert_eq!(call_id.as_deref(), Some("c1"));
            assert_eq!(tool.as_deref(), Some("run_shell_command"));
            assert_eq!(command.as_deref(), Some("ls"));
            assert_eq!(*action, polylogue_model::ToolAction::Shell);
        }
        other => panic!("{other:?}"),
    }
    match &msgs[1].blocks[3] {
        Block::ToolResult {
            call_id,
            outcome,
            output,
            ..
        } => {
            assert_eq!(call_id.as_deref(), Some("c1"));
            assert_eq!(*outcome, ToolOutcome::Ok);
            assert_eq!(output.as_deref(), Some("a\nb"));
        }
        other => panic!("{other:?}"),
    }
    let notice = step
        .output
        .events
        .iter()
        .find(|e| e.kind == "notice")
        .unwrap();
    assert_eq!(notice.after.as_ref().unwrap().native().as_str(), "g1");
    assert_eq!(notice.data, json!("Request cancelled."));
    assert_eq!(
        step.counts.unread_fields,
        std::collections::BTreeMap::from([
            ("gemini.thoughts[].timestamp".into(), 1),
            ("gemini.tokens.tool".into(), 1),
            ("gemini.tokens.total".into(), 1),
            ("gemini.toolCalls[].description".into(), 1),
            ("gemini.toolCalls[].displayName".into(), 1),
            ("gemini.toolCalls[].renderOutputAsMarkdown".into(), 1),
            ("gemini.toolCalls[].timestamp".into(), 1),
        ])
    );
    assert_eq!(step.counts.records, 4);
}

#[test]
fn cached_above_input_leaves_input_unknown() {
    let bytes = session(vec![
        json!({"id": "g1", "timestamp": "2026-03-14T18:01:50Z",
        "type": "gemini", "content": "ok", "model": "gemini-3-pro",
        "tokens": {"input": 50, "output": 7, "cached": 60}}),
    ]);
    let step =
        crate::export::fixture::parse(&mut Parse::Export(Box::new(GeminiCli::default())), &bytes)
            .unwrap();
    let usage = &step.output.messages[0].message.usage;
    assert_eq!(usage.input, None);
    assert_eq!(usage.cached_input, Some(60));
}

#[test]
fn recognizes_chat_files_only() {
    let p = Path::new("/h/.gemini/tmp/proj/chats/session-1.json");
    assert!(recognize(p, b"{\n  \"sessionId\": \"4ab5\""));
    assert!(!recognize(
        Path::new("/h/.gemini/state.json"),
        b"{\"sessionId\": \"x\""
    ));
    let chats = Path::new("/h/.gemini/tmp/proj/chats/session-1.json");
    assert!(!recognize(chats, b"{\"messages\": []}"));
    assert!(!recognize(chats, b"[{\"sessionId\": \"x\"}]"));
    assert!(!recognize(
        Path::new("/h/.gemini/tmp/proj/chats/session-1.txt"),
        b"{\"sessionId\": \"x\""
    ));
}

#[test]
fn an_unknown_session_kind_reads_as_main_and_is_reported() {
    let parse = |kind: &str| {
        let bytes =
            serde_json::to_vec(&json!({"sessionId": "s1", "kind": kind, "messages": []})).unwrap();
        crate::export::fixture::parse(&mut Parse::Export(Box::new(GeminiCli::default())), &bytes)
            .unwrap()
            .output
    };
    let out = parse("delegate");
    assert_eq!(out.sessions[0].kind, SessionKind::Main);
    let reported: Vec<_> = out
        .events
        .iter()
        .filter(|event| event.kind == "session_kind_unknown")
        .map(|event| event.data.clone())
        .collect();
    assert_eq!(reported, [json!({"kind": "delegate"})]);
    for known in ["main", "subagent"] {
        assert!(
            parse(known)
                .events
                .iter()
                .all(|event| event.kind != "session_kind_unknown")
        );
    }
    assert_eq!(parse("subagent").sessions[0].kind, SessionKind::Subagent);
}
