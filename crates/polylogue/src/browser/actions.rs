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
use polylogue_store::chat_actions::{
    CLAIM_MS, ChatAction, ChatAttachment, ChatStatus, NewChatAction,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

pub const OPERATIONS: &[&str] = &["conversation.create", "conversation.reply"];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmitPolicy {
    /// Fill the composer and stop: the operator sees a draft.
    StageOnly,
    /// Submit once.
    SubmitOnce,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    #[serde(default = "new_conversation")]
    pub conversation_id: String,
    #[serde(default)]
    pub conversation_url: Option<String>,
    #[serde(default)]
    pub project_ref: Option<String>,
}

fn new_conversation() -> String {
    "new".into()
}

impl Default for Target {
    fn default() -> Self {
        Self {
            conversation_id: new_conversation(),
            conversation_url: None,
            project_ref: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Presentation {
    #[serde(default = "chat")]
    pub surface: String,
    pub model_slug: String,
    pub model_label: String,
    pub effort_label: String,
}

fn chat() -> String {
    "chat".into()
}

/// A file to attach: bytes stored earlier by an upload, named by their
/// BLAKE3 hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentInput {
    pub name: String,
    #[serde(default = "octet")]
    pub mime_type: String,
    pub attachment_ref: String,
}

fn octet() -> String {
    "application/octet-stream".into()
}

/// A request to draft or submit one turn in a chat web page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    #[serde(default)]
    pub action_id: Option<String>,
    #[serde(default)]
    pub idempotency_key: Option<String>,
    pub provider: String,
    /// `conversation.create` or `conversation.reply`.
    pub operation: String,
    #[serde(default)]
    pub target: Target,
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<AttachmentInput>,
    pub presentation: Presentation,
    #[serde(default = "stage_only")]
    pub submit_policy: SubmitPolicy,
}

fn stage_only() -> SubmitPolicy {
    SubmitPolicy::StageOnly
}

/// What the extension instance holding an action reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Update {
    pub owner_instance_id: String,
    #[serde(default = "progress")]
    pub outcome: String,
    pub phase: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub receipt: Option<Value>,
}

fn progress() -> String {
    "progress".into()
}

/// Checks a request and makes the row the writer stores. `size` gives the
/// size of a stored upload, `None` when no upload has that hash.
pub fn new_action(
    req: &ActionRequest,
    size: impl Fn(&BlobHash) -> Option<u64>,
) -> Result<NewChatAction, Refused> {
    if !providers().any(|p| p == req.provider) || !OPERATIONS.contains(&req.operation.as_str()) {
        return Err(Refused(400, "unsupported_action"));
    }
    if req.text.trim().is_empty() {
        return Err(Refused(400, "empty_text"));
    }
    let is_new = req.target.conversation_id == "new";
    if (req.operation == "conversation.create") != is_new {
        return Err(Refused(400, "target_does_not_match_operation"));
    }
    let mut attachments = Vec::new();
    for a in &req.attachments {
        let blob =
            BlobHash::parse(&a.attachment_ref).map_err(|_| Refused(400, "unknown_attachment"))?;
        let size = size(&blob).ok_or(Refused(400, "unknown_attachment"))?;
        let basename = Path::new(&a.name)
            .file_name()
            .is_some_and(|n| n == a.name.as_str());
        if !basename || a.name.chars().any(char::is_control) {
            return Err(Refused(400, "attachment_name_not_a_basename"));
        }
        attachments.push(ChatAttachment {
            id: uuid::Uuid::now_v7().to_string(),
            blob,
            name: a.name.clone(),
            media_type: a.mime_type.clone(),
            size,
        });
    }
    let request = serde_json::to_value(req).map_err(|_| Refused(400, "invalid_action"))?;
    let request_hash = BlobHash::of(&serde_json::to_vec(&request).unwrap_or_default());
    Ok(NewChatAction {
        id: req
            .action_id
            .clone()
            .filter(|id| uuid::Uuid::parse_str(id).is_ok())
            .unwrap_or_else(|| uuid::Uuid::now_v7().to_string()),
        request_key: req
            .idempotency_key
            .clone()
            .unwrap_or_else(|| request_hash.to_string()),
        request_hash,
        request,
        stage_only: req.submit_policy == SubmitPolicy::StageOnly,
        needs_approval: req.submit_policy == SubmitPolicy::SubmitOnce && !is_new,
        attachments,
    })
}

fn iso(ms: i64) -> Value {
    jiff::Timestamp::from_millisecond(ms).map_or(Value::Null, |t| json!(t.to_string()))
}

/// One action as the extension reads it, at `now`.
pub fn view(a: &ChatAction, now: i64) -> Value {
    let (status, holder) = a.status(now);
    let last = a.events.last();
    let claim = a
        .events
        .iter()
        .rev()
        .find(|e| matches!(e.kind.as_str(), "claimed" | "progress"))
        .filter(|_| holder.is_some());
    let at = |kind: &str| {
        a.events
            .iter()
            .rev()
            .find(|e| e.kind == kind)
            .map(|e| iso(e.at))
    };
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
        "action_id": a.id,
        "idempotency_key": a.request_key,
        "request_hash": a.request_hash.to_string(),
        "attachments": a.attachments.iter().map(|x| json!({
            "attachment_id": x.id,
            "name": x.name,
            "mime_type": x.media_type,
            "size_bytes": x.size,
            "hash": x.blob.to_string(),
        })).collect::<Vec<_>>(),
        "status": status,
        "phase": last.map_or("queued", |e| e.phase.as_str()),
        "created_at": iso(a.created_at),
        "updated_at": iso(last.map_or(a.created_at, |e| e.at)),
        "claimed_by": holder,
        "claim_expires_ms": claim.map(|e| e.at + CLAIM_MS),
        "requires_operator_approval": a.needs_approval,
        "approved_at": at("approved"),
        "declined_at": at("declined"),
        "last_error": failure,
        "receipt": a.events.iter().rev().find_map(|e| e.receipt.clone()),
        "events": a.events,
    });
    if let (Some(out), Value::Object(fields)) = (out.as_object_mut(), fields) {
        out.remove("action_id");
        out.remove("idempotency_key");
        out.extend(fields);
    }
    out
}

/// The HTTP status and code a store refusal maps to.
pub fn refusal(code: &'static str) -> Refused {
    let status = match code {
        "unknown_chat_action" | "unknown_chat_action_attachment" | "attachment_missing" => 404,
        "unknown_outcome" | "unknown_attachment" => 400,
        _ => 409,
    };
    Refused(status, code)
}

#[cfg(test)]
mod tests;
