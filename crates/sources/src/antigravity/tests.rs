use super::*;
use crate::parse::run_document;
use serde_json::json;

fn parse(markdown: &str) -> crate::Step {
    let value = json!({"source":"antigravity_language_server", "cascadeId":"cascade-1", "title":"Checks", "markdown":markdown,
        "workspaceName":"example", "lastModifiedTime":"2026-03-05T04:21:34Z"});
    run_document(&Antigravity, &serde_json::to_vec(&value).unwrap()).unwrap()
}

#[test]
fn headings_preserve_authorship_and_unknown_times() {
    let result = parse(
        "# Chat Session\nNote: exported\n\n### User Input\n\nRun checks.\n\n### Planner Response\n\nDone.\n",
    );
    assert_eq!(
        result.output.sessions[0].id.to_string(),
        "antigravity:cascade-1"
    );
    assert_eq!(
        result.output.sessions[0].title.as_ref().unwrap().text,
        "Checks"
    );
    assert_eq!(result.output.messages.len(), 2);
    assert_eq!(result.output.messages[0].message.role, Role::User);
    assert_eq!(
        result.output.messages[0].message.prompt_kind,
        Some(PromptKind::Prose)
    );
    assert_eq!(result.output.messages[1].message.role, Role::Assistant);
    assert!(
        result
            .output
            .messages
            .iter()
            .all(|m| m.message.at.is_none())
    );
    assert_eq!(result.counts.records, 3);
}

#[test]
fn activity_inside_user_sections_is_an_assistant_action() {
    let result = parse(
        "### User Input\nPrompt\n\n*Viewed [file](src/main.rs)*\n\n*Edited relevant file*\n\nMore prose\n*I think this is useful*\n",
    );
    let messages: Vec<_> = result.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, Role::Assistant);
    assert!(
        matches!(&messages[1].blocks[0], Block::ToolCall {tool:Some(t), path:Some(p), ..} if t=="viewed_file" && p=="src/main.rs")
    );
    assert_eq!(messages[1].blocks.len(), 2);
    assert!(matches!(&messages[2].blocks[0], Block::Text {text} if text.contains("*I think")));
}

#[test]
fn command_markers_accept_heredocs_and_incomplete_markers_stay_prose() {
    let result = parse(
        "### Planner Response\n*User accepted the command `cat <<EOF\nhello\nEOF`*\n\n*User accepted the command `unfinished\nrest",
    );
    assert!(
        matches!(&result.output.messages[0].message.blocks[0], Block::ToolCall {command:Some(c), ..} if c=="cat <<EOF\nhello\nEOF")
    );
    assert!(
        matches!(&result.output.messages[1].message.blocks[0], Block::Text{text} if text.contains("unfinished\nrest"))
    );
}

#[test]
fn identical_runs_have_distinct_ids_stable_after_unrelated_insertions() {
    let first = parse("### Planner Response\nA\n*Edited relevant file*\nB\n*Edited relevant file*");
    let second = parse(
        "### Planner Response\nNew\n*Checked command status*\nA\n*Edited relevant file*\nB\n*Edited relevant file*",
    );
    let edits = |s: &crate::Step| {
        s.output.messages.iter().filter(|r| matches!(&r.message.blocks[0], Block::ToolCall {tool:Some(t), ..} if t=="edited_file")).map(|r| r.message.id.clone()).collect::<Vec<_>>()
    };
    let ids = edits(&first);
    assert_eq!(ids, edits(&second));
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn unstructured_exports_and_empty_exports_keep_their_meaning() {
    let result = parse("# Chat Session\n\nNote: generated export\n\nUnstructured transcript body.");
    assert_eq!(result.output.messages.len(), 1);
    assert!(
        matches!(&result.output.messages[0].message.blocks[0], Block::Text{text} if text=="Unstructured transcript body.")
    );
    assert!(parse("# Chat Session\n\n").output.messages.is_empty());
    let invalid = run_document(&Antigravity, b"{\"markdown\":\"text\"}").unwrap();
    assert!(invalid.output.sessions.is_empty());
}

#[test]
fn an_unclosed_command_does_not_swallow_later_activity() {
    let result = parse(
        "### Planner Response\n*User accepted the command `unfinished\n*Edited relevant file*\n",
    );
    assert_eq!(result.output.messages.len(), 2);
    assert!(
        matches!(&result.output.messages[1].message.blocks[0],Block::ToolCall {tool:Some(t),..} if t=="edited_file")
    );
}

#[test]
fn source_update_time_belongs_to_metadata_without_inventing_message_times() {
    let parsed = parse("### User Input\nQuestion\n");
    let event = parsed
        .output
        .events
        .iter()
        .find(|e| e.kind == "antigravity_session")
        .unwrap();
    assert_eq!(
        event.at,
        crate::time::rfc3339(&json!("2026-03-05T04:21:34Z"))
    );
    assert_eq!(event.data["workspaceName"], "example");
    assert!(event.data.get("markdown").is_none());
    assert!(
        parsed
            .output
            .messages
            .iter()
            .all(|row| row.message.at.is_none())
    );
    let value = json!({"source":"antigravity_language_server","cascadeId":"one","markdown":"### User Input\nQuestion","lastModifiedTime":"unreadable"});
    let parsed = run_document(&Antigravity, &serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(parsed.output.events[0].at.is_none());
    assert_eq!(
        parsed.output.events[0].data["lastModifiedTime"],
        "unreadable"
    );
}
