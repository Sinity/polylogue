use super::*;
use crate::parse::run_document;
use serde_json::json;

fn capture(raw: Json, session: Json) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "polylogue_capture_kind": "conversation",
        "schema_version": 1,
        "source": "browser-extension",
        "provenance": {"source_url": "https://example.invalid/c/1",
                       "captured_at": "2026-02-01T00:00:00Z", "capture_mode": "auto"},
        "provider_meta": {},
        "raw_provider_payload": raw,
        "session": session,
    }))
    .unwrap()
}

fn chatgpt_payload() -> Json {
    json!({"id": "c1", "conversation_id": "c1", "title": "T", "create_time": 1.0,
           "current_node": "n1",
           "mapping": {"n1": {"id": "n1", "parent": null, "children": [], "message": {
               "id": "m1", "author": {"role": "user"}, "create_time": 1.0,
               "content": {"content_type": "multimodal_text", "parts": [
                   {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-Pic"},
                   "see"]},
               "status": "finished_successfully", "recipient": "all", "metadata": {}}}}})
}

#[test]
fn a_chatgpt_capture_reads_the_provider_json_under_the_export_ids() {
    let bytes = capture(
        chatgpt_payload(),
        json!({"provider": "chatgpt", "provider_session_id": "c1", "turns": [],
               "attachments": [{"provider_attachment_id": "file-Pic", "message_provider_id": "m1",
                                "name": "pic.png", "content_base64": "aGVsbG8="}]}),
    );
    let step = run_document(&Capture, &bytes).unwrap();
    let msg = &step.output.messages[0].message;
    assert_eq!(msg.id.to_string(), "chatgpt:m1");
    assert_eq!(msg.session.to_string(), "chatgpt:c1");
    match &msg.blocks[0] {
        Block::Image(a) => {
            assert_eq!(a.blob, Some(BlobHash::of(b"hello")));
            assert_eq!(a.size, Some(5));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(step.output.blobs.len(), 1);
    assert!(
        step.output
            .events
            .iter()
            .any(|e| e.kind == "browser_capture")
    );
    assert_eq!(step.accounting.materialized["capture"], 1);
}

#[test]
fn without_provider_json_the_extension_turns_are_read() {
    let bytes = capture(
        json!({}),
        json!({"provider": "gemini", "provider_session_id": "g1", "title": "Chat",
        "turns": [
            {"role": "model", "text": "answer", "ordinal": 1, "blocks": []},
            {"role": "user", "text": "question", "ordinal": 0, "blocks": []}
        ]}),
    );
    let step = run_document(&Capture, &bytes).unwrap();
    let got: Vec<(String, String)> = step
        .output
        .messages
        .iter()
        .map(|r| (r.message.id.to_string(), r.message.text()))
        .collect();
    assert_eq!(
        got,
        [
            ("gemini:g1#0".to_string(), "question".to_string()),
            ("gemini:g1#1".to_string(), "answer".to_string())
        ]
    );
}

#[test]
fn recognizes_captures_by_their_first_key() {
    let p = Path::new("/s/polylogue/browser-capture/chatgpt/x.json");
    assert!(recognize(
        p,
        b"{\"polylogue_capture_kind\": \"conversation\""
    ));
    assert!(!recognize(p, b"{\"title\": \"x\""));
    assert!(recognize(p, b"{\n  \"capture_id\": \"chatgpt:6a67\""));
}
