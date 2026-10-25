//! What a capture holds, read with the same parser the loop uses: its
//! conversation and the files its messages name. The extension asks before
//! storing a capture, then fetches the bytes of the files it can.
use super::*;
use crate::parse::run_document;
use serde_json::json;

pub fn files(bytes: &[u8]) -> Result<Json, String> {
    let capture: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let step = run_document(&Capture, bytes).map_err(|e| e.to_string())?;
    if step.output.sessions.len() != 1 {
        return Err("the capture holds no single conversation".into());
    }
    let session = &step.output.sessions[0];
    let provider = capture.pointer("/session/provider").and_then(Json::as_str);
    let conversation = capture
        .pointer("/session/provider_session_id")
        .and_then(Json::as_str);
    let expected = provider
        .zip(conversation)
        .and_then(|(provider, id)| session_for(provider, id));
    if expected.as_ref() != Some(&session.id) {
        return Err(
            "the provider's conversation has another identity than the capture names".into(),
        );
    }
    let mut files = Vec::new();
    for row in &step.output.messages {
        let id = row.message.id.to_string();
        let message = id.split_once(':').map_or(id.as_str(), |(_, native)| native);
        for block in &row.message.blocks {
            let found: Vec<&Attachment> = match block {
                Block::File(a) | Block::Image(a) => vec![a],
                Block::ToolResult { attachments, .. } => attachments.iter().collect(),
                _ => Vec::new(),
            };
            for file in found {
                files.push(json!({
                    "id": file
                        .source_id
                        .clone()
                        .or_else(|| file.name.clone())
                        .or_else(|| file.url.clone())
                        .unwrap_or_else(|| format!("{message}:{}", files.len())),
                    "message": message,
                    "name": file.name,
                    "media_type": file.media_type,
                    "size": file.size,
                    "url": file.url,
                    "file_id": file.source_id,
                    // The provider's JSON already carries these bytes.
                    "inline": file.blob.is_some(),
                }));
            }
        }
    }
    let temporary = capture
        .pointer("/raw_provider_payload/is_temporary")
        .and_then(Json::as_bool)
        .unwrap_or(false)
        || capture
            .pointer("/raw_provider_payload/conversation/temporary")
            .and_then(Json::as_bool)
            .unwrap_or(false);
    Ok(json!({
        "conversation": {
            "provider": provider,
            "id": conversation,
            "title": session.title.as_ref().map(|t| &t.text),
            "messages": step.output.messages.len(),
            "temporary": temporary,
        },
        "files": files,
    }))
}
