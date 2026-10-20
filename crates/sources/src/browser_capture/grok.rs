use super::*;

/// Grok exposes metadata and responses through separate endpoints. Keep
/// provider IDs and parent links when adapting that bundle to shared turns.
pub(super) fn conversation(raw: &Json, fallback: &Json, sink: &mut Sink<'_>) {
    let metadata = raw
        .get("conversation")
        .and_then(|v| v.get("conversation").or(Some(v)))
        .unwrap_or(&Json::Null);
    let responses = raw
        .get("responses")
        .and_then(|v| v.get("responses").or(Some(v)))
        .and_then(Json::as_array);
    let mut session = fallback.clone();
    session["provider_session_id"] = metadata
        .get("conversationId")
        .or_else(|| fallback.get("provider_session_id"))
        .cloned()
        .unwrap_or(Json::Null);
    session["title"] = metadata.get("title").cloned().unwrap_or(Json::Null);
    session["created_at"] = metadata.get("createTime").cloned().unwrap_or(Json::Null);
    session["turns"] = Json::Array(responses.into_iter().flatten().enumerate().map(|(ordinal, r)| {
        let role = str_of(r, "sender").unwrap_or("").to_ascii_lowercase();
        let mut files = Vec::new();
        for f in r.get("fileAttachmentsMetadata").and_then(Json::as_array).into_iter().flatten() {
            files.push(serde_json::json!({"provider_attachment_id": f.get("fileMetadataId"), "name": f.get("fileName"), "mime_type": f.get("fileMimeType"), "url": f.get("fileUri")}));
        }
        for image in r.get("imageAttachments").and_then(Json::as_array).into_iter().flatten() {
            let url = image.as_str().or_else(|| str_of(image, "url"));
            files.push(serde_json::json!({"provider_attachment_id": url, "url": url, "mime_type": "image/*"}));
        }
        let mut blocks = vec![serde_json::json!({"type": "text", "text": r.get("message").and_then(Json::as_str).unwrap_or("")})];
        for step in r.get("steps").and_then(Json::as_array).into_iter().flatten() {
            blocks.push(serde_json::json!({"type":"grok_step", "step":step}));
        }
        serde_json::json!({"provider_turn_id":r.get("responseId"), "parent_turn_id":r.get("parentResponseId"), "role":role, "timestamp":r.get("createTime"), "model":r.get("model"), "ordinal":ordinal, "blocks":blocks, "attachments":files})
    }).collect());
    turns("grok", &session, sink);
}
