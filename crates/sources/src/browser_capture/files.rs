//! What a capture holds, read with the same parser the loop uses: its
//! conversation and the files its messages name. The extension asks before
//! storing a capture, then fetches the bytes of the files it can.
/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use super::*;
use crate::parse::run_document;
use serde_json::json;

pub fn files(bytes: &[u8]) -> Result<Json, String> {
    // The capture tree is dropped before the parser builds its own, so a
    // capture with inlined files is never held twice.
    let (provider, conversation) = {
        let capture: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let text = |pointer: &str| {
            capture
                .pointer(pointer)
                .and_then(Json::as_str)
                .map(str::to_owned)
        };
        (
            text("/session/provider"),
            text("/session/provider_session_id"),
        )
    };
    let step = run_document(&Capture, bytes).map_err(|e| e.to_string())?;
    if step.output.sessions.len() != 1 {
        return Err("the capture holds no single conversation".into());
    }
    let session = &step.output.sessions[0];
    let expected = provider
        .as_deref()
        .zip(conversation.as_deref())
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
            for file in block.attachments() {
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
    Ok(json!({
        "conversation": {
            "provider": provider,
            "id": conversation,
            "title": session.title.as_ref().map(|t| &t.text),
            "messages": step.output.messages.len(),
            "temporary": session.mode.as_deref() == Some(TEMPORARY_MODE),
        },
        "files": files,
    }))
}
