//! An attachment's direction comes from the role of the message holding
//! it, in every origin and every attachment shape (the Python version's
//! `test_attachment_direction.py`).

use crate::support::{message, parse};
use serde_json::{Value, json};

fn attachments(m: &Value) -> Vec<&Value> {
    m["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["type"] == "file" || b["type"] == "image")
        .collect()
}

fn directions(m: &Value) -> Vec<String> {
    attachments(m)
        .iter()
        .map(|a| a["direction"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn chatgpt_uses_the_message_role() {
    let export = json!([{"id": "c1", "title": "t", "create_time": 1785578400.0, "current_node": "a", "mapping": {
        "root": {"id": "root", "parent": null, "children": ["u"], "message": null},
        "u": {"id": "u", "parent": "root", "children": ["a"], "message": {"id": "user-message",
            "author": {"role": "user"}, "create_time": 1785578400.0,
            "content": {"content_type": "text", "parts": ["Read this"]},
            "metadata": {"attachments": [{"id": "user-file", "name": "input.txt"}]}}},
        "a": {"id": "a", "parent": "u", "children": [], "message": {"id": "assistant-message",
            "author": {"role": "assistant"}, "create_time": 1785578401.0,
            "content": {"content_type": "text", "parts": ["Here is the result"]},
            "metadata": {"attachments": [{"id": "output-file", "name": "output.txt"}]}}}}}]);
    let p = parse(
        "chatgpt",
        "Downloads/chatgpt-export/conversations.json",
        export.to_string().as_bytes(),
    );
    assert_eq!(directions(message(&p, ":user-message")), ["input"]);
    assert_eq!(
        attachments(message(&p, ":user-message"))[0]["source_id"],
        "user-file"
    );
    assert_eq!(directions(message(&p, ":assistant-message")), ["output"]);
}

#[test]
fn claude_ai_uses_the_sender() {
    let export = json!([{"uuid": "c", "name": "n", "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z", "chat_messages": [
        {"uuid": "user-message", "sender": "human", "text": "Read this", "created_at": "2026-01-01T00:00:00Z",
         "attachments": [{"file_name": "input.txt", "extracted_content": "x"}]},
        {"uuid": "assistant-message", "sender": "assistant", "text": "Generated this",
         "created_at": "2026-01-01T00:00:01Z",
         "attachments": [{"file_name": "output.txt", "extracted_content": "y"}],
         "files": [{"file_uuid": "f", "file_name": "o.png", "file_type": "image/png"}]}]}]);
    let p = parse(
        "claude-ai",
        "Downloads/claude-export/conversations.json",
        export.to_string().as_bytes(),
    );
    assert_eq!(directions(message(&p, ":user-message")), ["input"]);
    assert_eq!(
        directions(message(&p, ":assistant-message")),
        ["output", "output"]
    );
}

#[test]
fn ai_studio_uses_the_chunk_role_for_every_attachment_shape() {
    let prompt = json!({"chunkedPrompt": {"chunks": [
        {"id": "user-turn", "role": "user", "text": "Use these files", "driveDocument": {"id": "user-drive-file"}},
        {"id": "model-turn", "role": "model", "parts": [
            {"inlineData": {"mimeType": "text/plain", "data": "b3V0"}},
            {"fileData": {"mimeType": "text/plain", "fileUri": "drive://output-file"}}]}]}});
    let p = parse(
        "ai-studio",
        ".local/state/polylogue/drive/ai-studio/1abc.json",
        prompt.to_string().as_bytes(),
    );
    let user = message(&p, "#user-turn");
    assert_eq!(directions(user), ["input"]);
    assert_eq!(attachments(user)[0]["source_id"], "user-drive-file");
    let model = message(&p, "#model-turn");
    assert_eq!(directions(model), ["output", "output"]);
    assert!(
        attachments(model)[0]["blob"].is_string(),
        "inline bytes are stored"
    );
    assert_eq!(
        attachments(model)[1]["url"],
        "drive://output-file",
        "a file part keeps its URI"
    );
    assert!(p["counts"]["unread_fields"].as_object().unwrap().is_empty());
}

fn capture(turns: Value, attachments: Value) -> Value {
    json!({"polylogue_capture_kind": "browser_llm_session", "schema_version": 1, "capture_id": "gemini:direction",
        "provenance": {"source_url": "https://example.test/chat", "captured_at": "2026-08-31T00:00:00Z", "adapter_name": "test"},
        "session": {"provider": "gemini", "provider_session_id": "direction", "turns": turns, "attachments": attachments}})
}

const CAPTURE: &str = ".local/state/polylogue/browser-capture/gemini/capture.json";

#[test]
fn browser_capture_turn_attachments_use_the_turn_role() {
    let c = capture(
        json!([
        {"provider_turn_id": "user-turn", "role": "user", "text": "Read this",
         "attachments": [{"provider_attachment_id": "input", "name": "in.txt", "size_bytes": 3}]},
        {"provider_turn_id": "assistant-turn", "role": "assistant", "text": "Generated this",
         "attachments": [{"provider_attachment_id": "output", "mime_type": "image/png"}]}]),
        json!([]),
    );
    let p = parse("browser-capture", CAPTURE, c.to_string().as_bytes());
    let user = message(&p, ":user-turn");
    assert_eq!(directions(user), ["input"]);
    assert_eq!(attachments(user)[0]["source_id"], "input");
    assert_eq!(attachments(user)[0]["size"], 3);
    let assistant = message(&p, ":assistant-turn");
    assert_eq!(directions(assistant), ["output"]);
    assert_eq!(attachments(assistant)[0]["type"], "image");
    assert!(p["counts"]["unread_fields"].as_object().unwrap().is_empty());
}

#[test]
fn a_session_level_capture_attachment_takes_the_role_of_the_message_it_names() {
    let c = capture(
        json!([{"provider_turn_id": "assistant-1", "role": "assistant", "text": "generated"}]),
        json!([{"provider_attachment_id": "session-file", "message_provider_id": "assistant-1",
                "name": "out.bin", "size_bytes": 3}]),
    );
    let p = parse("browser-capture", CAPTURE, c.to_string().as_bytes());
    let m = message(&p, ":assistant-1");
    assert_eq!(
        directions(m),
        ["output"],
        "a reference without bytes is kept"
    );
    assert_eq!(attachments(m)[0]["name"], "out.bin");
}

#[test]
fn a_capture_attachment_already_on_its_message_is_not_added_twice() {
    let c = capture(
        json!([{"provider_turn_id": "u1", "role": "user", "text": "see",
                "attachments": [{"provider_attachment_id": "f1", "name": "a.txt"}]}]),
        json!([{"provider_attachment_id": "f1", "message_provider_id": "u1", "name": "a.txt", "content_base64": "YWJj"}]),
    );
    let p = parse("browser-capture", CAPTURE, c.to_string().as_bytes());
    let files = attachments(message(&p, ":u1"));
    assert_eq!(files.len(), 1);
    assert_eq!(files[0]["size"], 3);
    assert!(files[0]["blob"].is_string());
}
