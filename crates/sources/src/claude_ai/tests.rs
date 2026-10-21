use super::*;
use crate::parse::run_document;
use serde_json::json;

fn conv(messages: Vec<Json>) -> Json {
    json!({"uuid": "c1", "name": "Named", "summary": "", "created_at": "2026-01-01T00:00:00Z",
           "updated_at": "2026-01-01T00:10:00Z", "account": {"uuid": "acc"},
           "chat_messages": messages})
}

fn m(uuid: &str, sender: &str, parent: &str, at: &str, content: Json) -> Json {
    json!({"uuid": uuid, "text": "", "content": content, "sender": sender,
           "created_at": at, "updated_at": at, "attachments": [], "files": [],
           "parent_message_uuid": parent})
}

fn run(convs: Vec<Json>) -> crate::parse::Step {
    crate::array::fixture::parse(
        &mut Parse::Array(Box::new(Export)),
        &serde_json::to_vec(&convs).unwrap(),
    )
    .unwrap()
}

#[test]
fn a_turn_with_a_server_tool_keeps_call_and_result_together() {
    let step = run(vec![conv(vec![
        m(
            "u1",
            "human",
            ROOT_PARENT,
            "2026-01-01T00:00:00Z",
            json!([{"type": "text", "text": "search it"}]),
        ),
        m(
            "a1",
            "assistant",
            "u1",
            "2026-01-01T00:00:05Z",
            json!([
                {"type": "thinking", "thinking": "", "summaries": [{"summary": "planning"}],
                 "thinking_hidden": true, "signature": "sig"},
                {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "q"}},
                {"type": "tool_result", "tool_use_id": "t1", "name": "web_search", "is_error": false,
                 "content": [{"type": "text", "text": "found"}]},
                {"type": "token_budget", "remaining": 1000},
                {"type": "text", "text": "Here it is."}
            ]),
        ),
    ])]);
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].parent, None, "the root sentinel is no parent");
    assert_eq!(msgs[1].parent.as_ref().unwrap().native().as_str(), "u1");
    let kinds: Vec<&str> = msgs[1]
        .blocks
        .iter()
        .map(|b| match b {
            Block::Thinking { .. } => "thinking",
            Block::ToolCall { .. } => "call",
            Block::ToolResult { .. } => "result",
            Block::Text { .. } => "text",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["thinking", "call", "result", "text"]);
    match &msgs[1].blocks[0] {
        Block::Thinking { text, redacted, .. } => {
            assert_eq!(text.as_deref(), Some("planning"));
            assert!(*redacted);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(step.accounting.framed, 3);
}

#[test]
fn branches_are_kept_and_old_flat_text_is_read() {
    let mut old = m(
        "u0",
        "human",
        ROOT_PARENT,
        "2026-01-01T00:00:00Z",
        json!([]),
    );
    old["text"] = json!("plain old text");
    let step = run(vec![conv(vec![
        old,
        m(
            "a1",
            "assistant",
            "u0",
            "2026-01-01T00:00:01Z",
            json!([{"type": "text", "text": "first answer"}]),
        ),
        m(
            "a2",
            "assistant",
            "u0",
            "2026-01-01T00:00:02Z",
            json!([{"type": "text", "text": "retried answer"}]),
        ),
    ])]);
    let texts: Vec<String> = step
        .output
        .messages
        .iter()
        .map(|r| r.message.text())
        .collect();
    assert_eq!(texts, ["plain old text", "first answer", "retried answer"]);
    assert!(
        step.output.messages[1..].iter().all(|r| r
            .message
            .parent
            .as_ref()
            .unwrap()
            .native()
            .as_str()
            == "u0")
    );
}

#[test]
fn uploads_keep_name_size_and_extracted_text() {
    let mut u = m(
        "u1",
        "human",
        ROOT_PARENT,
        "2026-01-01T00:00:00Z",
        json!([{"type": "text", "text": "read this"}]),
    );
    u["attachments"] = json!([{"file_name": "a.txt", "file_size": 5, "file_type": "text/plain",
                               "extracted_content": "hello"}]);
    u["files"] = json!([{"file_uuid": "f-1", "file_name": "pic.png", "file_kind": "image"}]);
    let step = run(vec![conv(vec![u])]);
    let blocks = &step.output.messages[0].message.blocks;
    assert!(
        matches!(&blocks[1], Block::File(a) if a.name.as_deref() == Some("a.txt") && a.size == Some(5))
    );
    assert!(matches!(&blocks[2], Block::Other { type_name, .. } if type_name == "attachment_text"));
    assert!(matches!(&blocks[3], Block::Image(a) if a.source_id.as_deref() == Some("f-1")));
}

#[test]
fn a_message_without_a_uuid_keeps_its_identity_by_position() {
    let at = |text: &str| {
        let mut x = m("", "human", ROOT_PARENT, "2026-01-01T00:00:00Z", json!([]));
        x.as_object_mut().unwrap().remove("uuid");
        x["text"] = json!(text);
        x
    };
    let before = run(vec![conv(vec![at("draft"), at("second")])]);
    let after = run(vec![conv(vec![at("edited draft"), at("second")])]);
    let ids = |s: &crate::parse::Step| -> Vec<String> {
        s.output
            .messages
            .iter()
            .map(|r| r.message.id.to_string())
            .collect()
    };
    assert_eq!(ids(&before), ["claude-ai:c1#0", "claude-ai:c1#1"]);
    assert_eq!(
        ids(&after),
        ids(&before),
        "an edit is a new version, not a new message"
    );
    assert_eq!(after.output.messages[0].message.text(), "edited draft");
}

#[test]
fn recognizers_tell_the_exports_apart() {
    let claude = b"[{\"uuid\": \"3b33\", \"name\": \"x\"";
    let chatgpt = b"[{\"title\": \"Draft\", \"create_time\": 1";
    let p = Path::new("/x.zip!/conversations.json");
    assert!(recognize(p, claude));
    assert!(!recognize(p, chatgpt));
    assert!(!crate::chatgpt::recognize(p, claude));
    assert!(crate::chatgpt::recognize(p, chatgpt));
    let d = Path::new("/x.zip!/design_chats/05c0.json");
    assert!(recognize_design(d, b"{\"uuid\": \"05c0\""));
    assert!(!recognize_design(
        Path::new("/x/projects/1.json"),
        b"{\"uuid\": \"1\""
    ));
}

#[test]
fn a_design_chat_is_a_linear_session() {
    let chat = json!({"uuid": "d1", "title": "Landing page", "project": {"uuid": "p", "name": "P"},
    "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:01:00Z",
    "messages": [
        {"uuid": "m1", "role": "user", "created_at": "2026-01-01T00:00:00Z",
         "content": {"role": "user", "content": "make it blue", "attachments": [],
                     "authorName": "O", "id": "x", "timestamp": "2026-01-01T00:00:00Z"}},
        {"uuid": "m2", "role": "assistant", "created_at": "2026-01-01T00:00:30Z",
         "content": {"role": "assistant", "content": "Done.", "attachments": []}}
    ]});
    let step = run_document(&Design, &serde_json::to_vec(&chat).unwrap()).unwrap();
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs[0].session.to_string(), "claude-design:d1");
    assert_eq!(msgs[1].parent.as_ref().unwrap().native().as_str(), "m1");
    assert_eq!(msgs[1].text(), "Done.");
    assert_eq!(step.accounting.framed, 3);
}

#[test]
fn a_compaction_summary_marks_the_discontinuity_before_its_message() {
    let mut later = m(
        "u2",
        "human",
        "a1",
        "2026-01-01T00:05:00Z",
        json!([{"type": "text", "text": "go on"}]),
    );
    later["compaction_summary"] = json!([{"type": "text", "text": "Summary of earlier turns.",
                                          "citations": []}]);
    let step = run(vec![conv(vec![
        m(
            "u1",
            "human",
            ROOT_PARENT,
            "2026-01-01T00:00:00Z",
            json!([{"type": "text", "text": "hi"}]),
        ),
        m(
            "a1",
            "assistant",
            "u1",
            "2026-01-01T00:00:05Z",
            json!([{"type": "text", "text": "hello"}]),
        ),
        later,
    ])]);
    let event = step
        .output
        .events
        .iter()
        .find(|e| e.kind == "compaction")
        .unwrap();
    assert_eq!(event.after.as_ref().unwrap().native().as_str(), "a1");
    assert_eq!(
        event.data.pointer("/summary/0/text").and_then(Json::as_str),
        Some("Summary of earlier turns.")
    );
    assert!(
        !step
            .accounting
            .unread_fields
            .keys()
            .any(|k| k.contains("compaction_summary")),
        "{:?}",
        step.accounting.unread_fields
    );
}
