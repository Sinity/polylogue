use super::*;
use polylogue_model::ToolOutcome;
use serde_json::json;

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
    assert_eq!(msgs[1].usage.reasoning, Some(20));
    match &msgs[1].blocks[2] {
        Block::ToolCall {
            run_by_script: false,
            command,
            action,
            ..
        } => {
            assert_eq!(command.as_deref(), Some("ls"));
            assert_eq!(*action, polylogue_model::ToolAction::Shell);
        }
        other => panic!("{other:?}"),
    }
    match &msgs[1].blocks[3] {
        Block::ToolResult {
            outcome, output, ..
        } => {
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
    assert!(
        step.accounting.unread_fields.is_empty(),
        "{:?}",
        step.accounting.unread_fields
    );
    assert_eq!(step.accounting.framed, 4);
}

#[test]
fn recognizes_chat_files_only() {
    let p = Path::new("/h/.gemini/tmp/proj/chats/session-1.json");
    assert!(recognize(p, b"{\n  \"sessionId\": \"4ab5\""));
    assert!(!recognize(
        Path::new("/h/.gemini/state.json"),
        b"{\"sessionId\": \"x\""
    ));
}
