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
    assert_eq!(step.counts.used["capture"], 1);
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
            ("gemini-app:g1#0".to_string(), "question".to_string()),
            ("gemini-app:g1#1".to_string(), "answer".to_string())
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
fn a_grok_capture_keeps_the_providers_ids_parents_and_files() {
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
    let read = files(&bytes).unwrap();
    assert_eq!(read["conversation"]["messages"], 2);
    assert_eq!(read["files"][0]["url"], "users/u/f1/content");
}

#[test]
fn the_files_of_a_capture_name_their_message_by_its_provider_id() {
    let bytes = capture(
        chatgpt_payload(),
        json!({"provider":"chatgpt", "provider_session_id":"c1"}),
    );
    let read = files(&bytes).unwrap();
    assert_eq!(read["files"][0]["message"], "m1");
    assert_eq!(read["files"][0]["file_id"], "file-Pic");
    assert_eq!(read["files"][0]["inline"], false);
    assert_eq!(read["conversation"]["temporary"], false);
}

#[test]
fn a_capture_whose_provider_conversation_has_another_identity_is_refused() {
    let bytes = capture(
        chatgpt_payload(),
        json!({"provider":"chatgpt", "provider_session_id":"another"}),
    );
    assert!(files(&bytes).unwrap_err().contains("identity"));
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
    let read = files(&bytes).unwrap();
    assert_eq!(read["files"][0]["inline"], true);
    assert_eq!(read["files"][1]["name"], "notes.txt");
}

#[test]
fn an_empty_provider_conversation_is_a_conversation_with_no_messages() {
    let empty = capture(
        json!({"conversation":{"conversationId":"g1"}, "responses":{"responses":[]}}),
        json!({"provider":"grok", "provider_session_id":"g1"}),
    );
    assert_eq!(files(&empty).unwrap()["conversation"]["messages"], 0);
}

#[test]
fn captures_are_recognized_with_schema_version_first() {
    let bytes = br#"{"schema_version":1,"polylogue_capture_kind":"browser_llm_session","session":{"provider":"grok","provider_session_id":"g1","turns":[]}}"#;
    assert!(recognize(Path::new("capture.json"), &bytes[..64]));
    assert!(!recognize(
        Path::new("other.json"),
        br#"{"schema_version":1,"unrelated":true}"#
    ));
}

#[test]
fn a_gemini_capture_and_a_gemini_app_export_of_one_conversation_are_one_session() {
    let link = "https://gemini.google.com/app/abc123";
    let html = format!(
        "<html><body><div class=\"outer-cell mdl-cell\"><div class=\"mdl-grid\">\
         <div class=\"header-cell\"><p>Gemini Apps<br></p></div>\
         <div class=\"content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1\">\
         Prompted\u{a0}question<br>Jul 27, 2026, 10:59:30\u{202f}PM CEST<br><p>answer</p></div>\
         <div class=\"content-cell mdl-cell mdl-cell--12-col mdl-typography--caption\">\
         Products:<br>&emsp;Gemini Apps<br>Details:<br>&emsp;{link}: <a href=\"{link}\">{link}</a><br>\
         </div></div></div></body></html>"
    );
    let export = crate::export::fixture::parse(
        &mut crate::gemini_app::open(std::path::Path::new("MyActivity.html")).unwrap(),
        html.as_bytes(),
    )
    .unwrap();
    let captured = run_document(
        &Capture,
        &capture(
            json!({}),
            json!({"provider": "gemini", "provider_session_id": "abc123",
                   "turns": [{"role": "user", "text": "question", "ordinal": 0, "blocks": []}]}),
        ),
    )
    .unwrap();
    let exported = &export.output.sessions[0].id;
    assert_eq!(exported.to_string(), "gemini-app:abc123");
    assert_eq!(&captured.output.sessions[0].id, exported);
    assert_eq!(&captured.output.messages[0].message.session, exported);
    assert_eq!(session_for("gemini", "abc123").as_ref(), Some(exported));
}
