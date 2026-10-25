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

#[test]
fn grok_endpoint_bundle_preserves_native_identity_parents_and_files() {
    let bytes = capture(
        json!({
            "conversation": {"conversationId":"g1", "title":"Grok fixture", "createTime":"2026-06-27T12:00:00Z"},
            "responses":{"responses":[
                {"responseId":"r1", "sender":"human", "message":"question", "createTime":"2026-06-27T12:00:01Z",
                 "fileAttachmentsMetadata":[{"fileMetadataId":"f1","fileName":"notes.txt","fileMimeType":"text/plain","fileUri":"users/u/f1/content"}]},
                {"responseId":"r2", "parentResponseId":"r1", "sender":"ASSISTANT", "message":"answer", "model":"grok-3", "steps":[{"text":["step"]}]}
            ]},"response_nodes":{"nodes":[]}
        }),
        json!({"provider":"grok","provider_session_id":"g1", "turns":[],
        "attachments":[{"provider_attachment_id":"f1","message_provider_id":"r1","content_base64":"aGVsbG8="}]}),
    );
    let step = run_document(&Capture, &bytes).unwrap();
    assert_eq!(step.output.sessions[0].id.to_string(), "grok:g1");
    assert_eq!(step.output.messages.len(), 2);
    let assistant = &step.output.messages[1].message;
    assert_eq!(assistant.parent.as_ref().unwrap().to_string(), "grok:r1");
    assert_eq!(assistant.model.as_deref(), Some("grok-3"));
    assert!(
        assistant
            .blocks
            .iter()
            .any(|b| matches!(b, Block::Other { type_name, .. } if type_name == "grok_step"))
    );
    assert_eq!(step.output.blobs[0].bytes, b"hello");
    let inspected = inspect::inspect(&bytes).unwrap();
    assert_eq!(inspected["summary"]["turn_count"], 2);
    assert_eq!(inspected["attachments"][0]["url"], "users/u/f1/content");
}

#[test]
fn inspection_keeps_mapping_keys_separate_from_native_message_ids() {
    let bytes = capture(
        chatgpt_payload(),
        json!({"provider":"chatgpt", "provider_session_id":"c1"}),
    );
    let inspected = inspect::inspect(&bytes).unwrap();
    assert_eq!(inspected["attachments"][0]["original_record_key"], "n1");
    assert_eq!(inspected["attachments"][0]["message_provider_id"], "m1");
    assert_eq!(
        inspected["attachments"][0]["provider_meta"]["provider_file_id"],
        "file-Pic"
    );
}

#[test]
fn inspection_refuses_a_raw_conversation_with_another_identity() {
    let bytes = capture(
        chatgpt_payload(),
        json!({"provider":"chatgpt", "provider_session_id":"another"}),
    );
    assert!(inspect::inspect(&bytes).unwrap_err().contains("identity"));
}

#[test]
fn claude_inline_media_is_linked_to_stored_bytes_and_reported_as_inline() {
    let bytes = capture(
        json!({"uuid":"claude-inline", "chat_messages":[
            {"uuid":"inline-m1", "sender":"human", "content":[
                {"type":"image", "source":{"type":"base64", "media_type":"image/png", "data":"aGVsbG8="}},
                {"type":"document", "title":"notes.txt", "source":{"type":"base64", "media_type":"text/plain", "data":"bm90ZXM="}}
            ]}
        ]}),
        json!({"provider":"claude-ai", "provider_session_id":"claude-inline"}),
    );
    let parsed = run_document(&Capture, &bytes).unwrap();
    assert_eq!(parsed.output.blobs.len(), 2);
    assert!(
        matches!(&parsed.output.messages[0].message.blocks[0], Block::Image(file) if file.blob == Some(BlobHash::of(b"hello")))
    );
    let inspected = inspect::inspect(&bytes).unwrap();
    assert_eq!(
        inspected["attachments"][0]["provider_meta"]["native_inline_size_bytes"],
        5
    );
    assert_eq!(
        inspected["attachments"][0]["provider_meta"]["native_inline_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(inspected["attachments"][1]["name"], "notes.txt");
}

#[test]
fn inspection_refuses_local_raw_refs_but_accepts_empty_provider_conversations() {
    let session = json!({"provider":"grok", "provider_session_id":"g1"});
    let refs = capture(
        json!({"conversation":{"staged_raw_json":"local-conversation"}, "responses":{"staged_raw_json":"local-responses"}}),
        session.clone(),
    );
    assert_eq!(
        inspect::inspect(&refs).unwrap_err(),
        "unresolved_staged_raw_json"
    );
    let empty = capture(
        json!({"conversation":{"conversationId":"g1"}, "responses":{"responses":[]}}),
        session,
    );
    assert_eq!(
        inspect::inspect(&empty).unwrap()["summary"]["turn_count"],
        0
    );
}

#[test]
fn streamed_native_envelopes_are_recognized_with_schema_version_first() {
    let bytes = br#"{"schema_version":1,"polylogue_capture_kind":"browser_llm_session","session":{"provider":"grok","provider_session_id":"g1","turns":[]}}"#;
    assert!(recognize(Path::new("capture.json"), &bytes[..64]));
    assert!(!recognize(
        Path::new("other.json"),
        br#"{"schema_version":1,"unrelated":true}"#
    ));
}

#[test]
fn inspection_preserves_literal_staging_names_inside_provider_data() {
    let bytes = capture(
        json!({"conversation":{"conversationId":"g1", "metadata":{"staged_raw_json":"provider-literal"}}, "responses":{"responses":[]}}),
        json!({"provider":"grok","provider_session_id":"g1"}),
    );
    assert_eq!(
        inspect::inspect(&bytes).unwrap()["summary"]["turn_count"],
        0
    );
}
