use super::*;
use crate::parse::run_document;
use serde_json::json;

fn room(messages: Json) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "version": "orpg.2.0", "title": "Nix help",
        "characters": {"char-1": {"id": "char-1", "model": "openrouter/horizon-beta",
                                  "description": "d", "samplingParameters": {}}},
        "messages": messages,
        "artifacts": {}, "artifactFiles": {}, "artifactVersions": {}, "artifactFileContents": {}
    }))
    .unwrap()
}

#[test]
fn a_room_becomes_one_session_named_by_its_first_message() {
    let bytes = room(json!({
        "msg-2-bbb": {"id": "msg-2-bbb", "type": "assistant", "characterId": "char-1",
                      "contentType": "text", "context": "main-chat", "parentMessageId": "msg-1-aaa",
                      "citations": [], "files": [], "createdAt": "2025-08-04T20:01:00.000Z",
                      "updatedAt": "2025-08-04T20:01:00.000Z", "isGenerating": false,
                      "isRetrying": false, "isEdited": false, "isCollapsed": false,
                      "content": "Use flakes.",
                      "metadata": {"plugins": [], "generateDuration": 41006, "latencyDuration": 5889,
                                   "tokensCount": 2394, "tokensPerSecond": 58.3,
                                   "variantSlug": "openrouter/horizon-beta", "provider": "Stealth",
                                   "cost": "0.0012"}},
        "msg-1-aaa": {"id": "msg-1-aaa", "type": "user", "characterId": "USER",
                      "contentType": "text", "context": "main-chat",
                      "attachments": [{"type": "text", "content": "my config"}],
                      "createdAt": "2025-08-04T20:00:00.000Z", "updatedAt": "2025-08-04T20:00:00.000Z",
                      "isGenerating": false, "isRetrying": false, "isEdited": false,
                      "isCollapsed": false, "content": "How do I pin nixpkgs?"}
    }));
    let step = run_document(&OpenRouter, &bytes).unwrap();
    let s = &step.output.sessions[0];
    assert_eq!(s.id.native().as_str(), "msg-1-aaa");
    assert_eq!(s.title.as_ref().unwrap().text, "Nix help");
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs[0].text(), "How do I pin nixpkgs?");
    assert!(matches!(&msgs[0].blocks[1], Block::File(a) if a.blob.is_some()));
    assert_eq!(step.output.blobs[0].bytes, b"my config");
    let a = msgs[1];
    assert_eq!(a.parent.as_ref().unwrap().native().as_str(), "msg-1-aaa");
    assert_eq!(a.model.as_deref(), Some("openrouter/horizon-beta"));
    assert_eq!(a.usage.output, Some(2394));
    assert_eq!(a.duration_ms, Some(41006));
    assert_eq!(a.production.first_token_ms, Some(5889));
    assert_eq!(a.reported_cost_usd.as_ref().unwrap().to_string(), "0.0012");
    assert!(step.output.events.iter().any(|e| e.kind == "room_models"));
    assert!(
        step.accounting.unread_fields.is_empty(),
        "{:?}",
        step.accounting.unread_fields
    );
    assert_eq!(step.accounting.framed, 3);
}

#[test]
fn recognizes_room_exports_by_name_and_first_key() {
    assert!(recognize(
        Path::new("/a/OpenRouter Chat Mon Aug 04 2025 (1).json"),
        br#"{"version": "orpg"#
    ));
    assert!(!recognize(
        Path::new("/a/chat.json"),
        br#"{"version": "orpg"#
    ));
}
