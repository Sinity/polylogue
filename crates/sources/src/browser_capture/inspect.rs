//! Stateless inspection uses the same parser as durable archive ingestion.
use super::*;
use crate::parse::run_document;
use serde_json::json;
use sha2::{Digest, Sha256};

pub fn inspect(bytes: &[u8]) -> Result<Json, String> {
    let envelope: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let step = run_document(&Capture, bytes).map_err(|e| e.to_string())?;
    if step.output.sessions.len() != 1 {
        return Err("capture must contain one usable session".into());
    }
    let session = &step.output.sessions[0];
    let expected = envelope
        .pointer("/session/provider")
        .and_then(Json::as_str)
        .zip(
            envelope
                .pointer("/session/provider_session_id")
                .and_then(Json::as_str),
        )
        .and_then(|(provider, id)| session_for(provider, id));
    if expected.as_ref() != Some(&session.id) {
        return Err("capture identity differs from provider payload".into());
    }
    let mut attachments = Vec::new();
    for (ordinal, row) in step.output.messages.iter().enumerate() {
        let native = row.message.id.to_string();
        let native = native.split_once(':').map_or(native.as_str(), |(_, id)| id);
        let record_key = envelope
            .pointer("/raw_provider_payload/mapping")
            .and_then(Json::as_object)
            .and_then(|mapping| {
                mapping
                    .iter()
                    .find(|(_, node)| {
                        node.pointer("/message/id").and_then(Json::as_str) == Some(native)
                    })
                    .map(|(key, _)| key.as_str())
            })
            .unwrap_or(native);
        for block in &row.message.blocks {
            let files: Vec<&Attachment> = match block {
                Block::File(a) | Block::Image(a) => vec![a],
                Block::ToolResult { attachments, .. } => attachments.iter().collect(),
                _ => Vec::new(),
            };
            for file in files {
                let inline = file
                    .blob
                    .and_then(|hash| step.output.blobs.iter().find(|blob| blob.hash == hash));
                let inline_hash = inline.map(|blob| format!("{:x}", Sha256::digest(&blob.bytes)));
                let attachment_id = file
                    .source_id
                    .clone()
                    .or_else(|| file.name.clone())
                    .or_else(|| file.url.clone())
                    .unwrap_or_else(|| format!("{native}:inline:{}", attachments.len()));
                attachments.push(json!({
                    "provider_attachment_id": attachment_id,
                    "message_provider_id": native,
                    "original_record_key": record_key,
                    "original_record_ordinal": ordinal,
                    "name": file.name, "mime_type": file.media_type,
                    "attachment_kind": if matches!(block, Block::Image(_)) { "image" } else { "file" },
                    "provider_meta": {"provider_file_id": file.source_id, "native_inline_sha256": inline_hash, "native_inline_size_bytes": inline.map(|blob| blob.size)},
                    "size_bytes": file.size, "url": file.url,
                }));
            }
        }
    }
    Ok(json!({"summary": {
        "provider": envelope.pointer("/session/provider"),
        "provider_session_id": envelope.pointer("/session/provider_session_id"),
        "title":session.title.as_ref().map(|t| &t.text),
        "message_count":step.output.messages.len(), "turn_count":step.output.messages.len(),
        "attachment_count":attachments.len(),
        "session_kind": if envelope.pointer("/raw_provider_payload/is_temporary").and_then(Json::as_bool).unwrap_or(false) || envelope.pointer("/raw_provider_payload/conversation/temporary").and_then(Json::as_bool).unwrap_or(false) {"temporary"} else {"main"},
        "needs_follow_up":false,
    }, "attachments":attachments}))
}
