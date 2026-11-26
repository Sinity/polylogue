//! Chat actions as `send`, `start --in` and `answer` make them and the
//! extension sees them. The daemon's writer stores them
//! ([`polylogue_store::chat_actions`]); this module checks requests and
//! shapes the answers.
//!
//! A new conversation drafts unless it asks to submit; a turn into an
//! existing conversation, and a submit by anyone but the operator, waits
//! for the operator's answer. One action is handed out
//! at a time, as a claim that expires and returns the action to the queue
//! if the extension goes quiet before sending. Once a send may have begun,
//! an expired claim has an unknown outcome and is not queued again. Replies
//! come back only through the usual capture.

use polylogue_model::BlobHash;
use polylogue_model::user::{CHAT_CLAIM_MS, ChatAction, ChatEvent, ChatStatus};
use polylogue_model::user::{ChatAttachment, NewChatAction};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

/// The providers whose pages the extension can draft and submit on, by
/// the extension's names for them.
pub fn providers() -> impl Iterator<Item = &'static str> {
    providers_where(|web| web.actions)
}

/// The extension's names for the providers whose web declaration passes `keep`.
pub(super) fn providers_where(
    keep: impl Fn(&polylogue_sources::origin::WebChat) -> bool,
) -> impl Iterator<Item = &'static str> {
    polylogue_sources::origin::ORIGINS
        .iter()
        .filter_map(|o| o.web.as_ref())
        .filter(move |w| keep(w))
        .filter_map(|w| w.extension_names.first().copied())
}

/// A refusal with its stable code; the daemon's operation boundary maps
/// the code to an HTTP status.
#[derive(Debug, PartialEq, Eq)]
pub struct Refused(pub polylogue_model::Code);

/// A request to draft or submit one turn in a chat web page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChatRequest {
    pub provider: String,
    /// The conversation to reply in; none starts a new one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    /// The provider's project to start a new conversation in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub text: String,
    /// The model as the provider's model menu names it; none keeps what
    /// the page has selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The reasoning effort as the provider's menu names it, where it
    /// offers one; it is chosen in the model's menu, so it needs a model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Submit the turn; without it the extension fills the composer and
    /// stops, leaving a draft.
    #[serde(default)]
    pub submit: bool,
}

/// A file attached to a turn, with its bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChatFile {
    pub name: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

/// What the extension instance holding an action reports.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
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

/// An attachment's name must be a plain file name, so it cannot name a path.
pub fn check_file_name(name: &str) -> Result<(), Refused> {
    let basename = Path::new(name).file_name().is_some_and(|n| n == name);
    if !basename || name.chars().any(char::is_control) {
        return Err(Refused(polylogue_model::Code::AttachmentNameNotABasename));
    }
    Ok(())
}

/// Checks a request and makes the row the writer stores. `files` are the
/// attachments already stored as blobs, with their names and media types;
/// a `held` action waits for the operator's answer before the extension
/// takes it.
pub fn new_action(
    request_key: &str,
    req: &ChatRequest,
    files: Vec<ChatAttachment>,
    held: bool,
) -> Result<NewChatAction, Refused> {
    if !providers().any(|p| p == req.provider) {
        return Err(Refused(polylogue_model::Code::UnsupportedProvider));
    }
    if req.text.trim().is_empty() {
        return Err(Refused(polylogue_model::Code::EmptyText));
    }
    if req.effort.is_some() && req.model.as_deref().is_none_or(|m| m.trim().is_empty()) {
        return Err(Refused(polylogue_model::Code::ModelRequired));
    }
    for file in &files {
        check_file_name(&file.name)?;
    }
    let request =
        serde_json::to_value(req).map_err(|_| Refused(polylogue_model::Code::InvalidAction))?;
    let mut hashed = request.clone();
    hashed["files"] = json!(files.iter().map(|f| f.blob.to_string()).collect::<Vec<_>>());
    let request_hash = BlobHash(
        *canonical_json::canonical_hash("polylogue-browser-action", &hashed)
            .map_err(|_| Refused(polylogue_model::Code::InvalidAction))?
            .as_bytes(),
    );
    Ok(NewChatAction {
        id: uuid::Uuid::now_v7().to_string(),
        request_key: request_key.to_owned(),
        request_hash,
        request,
        stage_only: !req.submit,
        needs_approval: held,
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

pub(super) fn iso(ms: i64) -> Option<String> {
    jiff::Timestamp::from_millisecond(ms)
        .ok()
        .map(|t| t.to_string())
}

/// One chat action as the CLI and the extension read it: the request as
/// queued, then where it stands.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ActionSummary {
    #[serde(flatten)]
    pub request: ChatRequest,
    pub id: String,
    pub attachments: Vec<AttachmentSummary>,
    #[schemars(with = "String")]
    pub status: ChatStatus,
    pub phase: String,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    /// The extension instance whose claim has not expired.
    pub claimed_by: Option<String>,
    pub claim_expires_at: Option<String>,
    pub needs_approval: bool,
    /// The detail of the latest failed or blocked outcome.
    pub last_error: Option<String>,
    /// What the extension reported last about the sent turn.
    pub receipt: Option<Value>,
    pub events: Vec<ChatEvent>,
}

/// A file a chat action attaches, without its bytes.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentSummary {
    pub id: String,
    pub name: String,
    pub media_type: String,
    pub size: u64,
    /// The hash of its stored bytes.
    pub hash: String,
}

/// One action as the CLI and the extension read it, at `now`. Its stored
/// request is the [`ChatRequest`] it was queued with.
pub fn view(a: &ChatAction, now: i64) -> Result<ActionSummary, serde_json::Error> {
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
                polylogue_model::user::chat_outcome(&e.kind),
                Some(Some(ChatStatus::Failed | ChatStatus::Blocked))
            )
        })
        .map(|e| e.detail.clone().unwrap_or_else(|| e.kind.clone()));
    Ok(ActionSummary {
        request: serde_json::from_value(a.request.clone())?,
        id: a.id.clone(),
        attachments: a
            .attachments
            .iter()
            .map(|x| AttachmentSummary {
                id: x.id.clone(),
                name: x.name.clone(),
                media_type: x.media_type.clone(),
                size: x.size,
                hash: x.blob.to_string(),
            })
            .collect(),
        status,
        phase: last.map_or("queued", |e| e.phase.as_str()).to_owned(),
        created_at: iso(a.created_at),
        updated_at: iso(last.map_or(a.created_at, |e| e.at)),
        claimed_by: holder.map(str::to_owned),
        claim_expires_at: claim.and_then(|e| iso(e.at + CHAT_CLAIM_MS)),
        needs_approval: a.needs_approval,
        last_error: failure,
        receipt: a.events.iter().rev().find_map(|e| e.receipt.clone()),
        events: a.events.clone(),
    })
}

#[cfg(test)]
mod tests;
