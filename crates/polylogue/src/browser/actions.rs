//! Chat actions as the operator queues them and the extension sees them.
//! The daemon's writer stores them ([`polylogue_store::chat_actions`]);
//! this module checks requests and shapes the answers.
//!
//! A new action drafts unless it asks to submit; a submit into an existing
//! conversation waits for the operator's approval. One action is handed out
//! at a time, as a claim that expires and returns the action to the queue
//! if the extension goes quiet. Replies come back only through ordinary
//! capture.

use polylogue_model::BlobHash;
use polylogue_model::user::{CHAT_CLAIM_MS, ChatAction, ChatStatus};
use polylogue_model::user::{ChatAttachment, NewChatAction};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

/// The providers whose pages the extension can draft and submit on, by
/// the extension's names for them.
pub fn providers() -> impl Iterator<Item = &'static str> {
    polylogue_sources::origin::ORIGINS
        .iter()
        .filter_map(|o| o.web.as_ref())
        .filter(|w| w.actions)
        .filter_map(|w| w.extension_names.first().copied())
}

/// A refusal: an HTTP status for the route and a stable code.
#[derive(Debug, PartialEq, Eq)]
pub struct Refused(pub u16, pub &'static str);

/// A request to draft or submit one turn in a chat web page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub provider: String,
    /// The conversation to reply in; none starts a new one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    /// The provider's project to start a new conversation in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub text: String,
    /// The model as the provider's model menu names it.
    pub model: String,
    /// The reasoning effort as the provider's menu names it, where it
    /// offers one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Submit the turn; without it the extension fills the composer and
    /// stops, leaving a draft.
    #[serde(default)]
    pub submit: bool,
}

/// A file the operator attaches, with its bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatFile {
    pub name: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

/// What the extension instance holding an action reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Update {
    /// The extension instance holding the claim.
    pub by: String,
    /// `progress` renews the claim; anything else is an outcome.
    pub outcome: String,
    pub phase: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub receipt: Option<Value>,
}

/// Checks a request and makes the row the writer stores. `files` are the
/// attachments already stored as blobs, with their names and media types.
pub fn new_action(
    request_key: &str,
    req: &ChatRequest,
    files: Vec<ChatAttachment>,
) -> Result<NewChatAction, Refused> {
    if !providers().any(|p| p == req.provider) {
        return Err(Refused(400, "unsupported_provider"));
    }
    if req.text.trim().is_empty() {
        return Err(Refused(400, "empty_text"));
    }
    if req.model.trim().is_empty() {
        return Err(Refused(400, "model_required"));
    }
    for file in &files {
        let basename = Path::new(&file.name)
            .file_name()
            .is_some_and(|n| n == file.name.as_str());
        if !basename || file.name.chars().any(char::is_control) {
            return Err(Refused(400, "attachment_name_not_a_basename"));
        }
    }
    let request = serde_json::to_value(req).map_err(|_| Refused(400, "invalid_action"))?;
    let mut hashed = request.clone();
    hashed["files"] = json!(files.iter().map(|f| f.blob.to_string()).collect::<Vec<_>>());
    let request_hash = BlobHash::of(&serde_json::to_vec(&hashed).unwrap_or_default());
    Ok(NewChatAction {
        id: uuid::Uuid::now_v7().to_string(),
        request_key: request_key.to_owned(),
        request_hash,
        request,
        stage_only: !req.submit,
        needs_approval: req.submit && req.conversation.is_some(),
        attachments: files,
    })
}

/// A file's attachment row, once its bytes are stored under `blob`.
pub fn attachment(file: &ChatFile, blob: BlobHash) -> ChatAttachment {
    ChatAttachment {
        id: uuid::Uuid::now_v7().to_string(),
        blob,
        name: file.name.clone(),
        media_type: file.media_type.clone(),
        size: file.bytes.len() as u64,
    }
}

fn iso(ms: i64) -> Value {
    jiff::Timestamp::from_millisecond(ms).map_or(Value::Null, |t| json!(t.to_string()))
}

/// One action as the CLI and the extension read it, at `now`.
pub fn view(a: &ChatAction, now: i64) -> Value {
    let (status, holder) = a.status(now);
    let last = a.events.last();
    let claim = a
        .events
        .iter()
        .rev()
        .find(|e| matches!(e.kind.as_str(), "claimed" | "progress"))
        .filter(|_| holder.is_some());
    let failure = a
        .events
        .iter()
        .rev()
        .find(|e| {
            matches!(
                polylogue_store::chat_actions::outcome(&e.kind),
                Some(Some(ChatStatus::Failed | ChatStatus::Blocked))
            )
        })
        .map(|e| e.detail.clone().unwrap_or_else(|| e.kind.clone()));
    let mut out = a.request.clone();
    let fields = json!({
        "id": a.id,
        "attachments": a.attachments.iter().map(|x| json!({
            "id": x.id,
            "name": x.name,
            "media_type": x.media_type,
            "size": x.size,
            "hash": x.blob.to_string(),
        })).collect::<Vec<_>>(),
        "status": status,
        "phase": last.map_or("queued", |e| e.phase.as_str()),
        "created_at": iso(a.created_at),
        "updated_at": iso(last.map_or(a.created_at, |e| e.at)),
        "claimed_by": holder,
        "claim_expires_at": claim.map(|e| iso(e.at + CHAT_CLAIM_MS)),
        "needs_approval": a.needs_approval,
        "last_error": failure,
        "receipt": a.events.iter().rev().find_map(|e| e.receipt.clone()),
        "events": a.events,
    });
    if let (Some(out), Value::Object(fields)) = (out.as_object_mut(), fields) {
        out.extend(fields);
    }
    out
}

/// The HTTP status and code a store refusal maps to.
pub fn refusal(code: &'static str) -> Refused {
    let status = match code {
        "unknown_chat_action"
        | "unknown_chat_action_attachment"
        | "attachment_missing"
        | "unknown_backfill_job" => 404,
        "unknown_outcome" | "unknown_attachment" | "unknown_report" => 400,
        _ => 409,
    };
    Refused(status, code)
}

#[cfg(test)]
mod tests;
