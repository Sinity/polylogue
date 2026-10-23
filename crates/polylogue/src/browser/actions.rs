//! The chat-action queue: one JSON file per action under
//! `browser-actions/`. Chrome starts a host process per request, so every
//! change holds an exclusive lock on `browser-actions/.lock`.
//!
//! A new action drafts unless it asks to submit; a submit into an existing
//! conversation waits for the operator's approval. One action is handed out
//! at a time, as a claim that expires and returns the action to the queue
//! if the extension goes quiet. Replies come back only through ordinary capture.

use super::{Ctx, new_id, now_iso, now_ms, safe_token, sha256_hex, write_durable};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

/// HTTP and daemon attachment input bound.
pub const BODY_LIMIT: usize = 2 << 20;

/// How long a claimed action stays with the extension instance that took
/// it without a report.
const CLAIM_MS: i64 = 180_000;
const PROVIDERS: &[&str] = &["chatgpt", "claude"];
const OPERATIONS: &[&str] = &["conversation.create", "conversation.reply"];

/// A refusal: an HTTP status for the route and a stable code.
#[derive(Debug, PartialEq, Eq)]
pub struct Refused(pub u16, pub &'static str);

type Result<T> = std::result::Result<T, Refused>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmitPolicy {
    /// Fill the composer and stop: the operator sees a draft.
    StageOnly,
    /// Submit once.
    SubmitOnce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    AwaitingApproval,
    Queued,
    Claimed,
    Drafted,
    Submitted,
    OutcomeUnknown,
    Blocked,
    Failed,
    Cancelled,
}

impl Status {
    fn finished(self) -> bool {
        matches!(
            self,
            Status::Drafted | Status::Submitted | Status::Failed | Status::Cancelled
        )
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub attachment_id: String,
    pub name: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// A request to draft or submit one turn in a chat web UI.
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub event_id: String,
    pub at: String,
    pub kind: String,
    pub phase: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub owner_instance_id: Option<String>,
}

/// One action as stored: the request, where it stands, and what the
/// extension observed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub contract: String,
    pub action_id: String,
    pub idempotency_key: String,
    pub request_sha256: String,
    pub receiver_id: String,
    pub provider: String,
    pub operation: String,
    pub target: Target,
    pub text: String,
    pub attachments: Vec<Attachment>,
    pub presentation: Presentation,
    pub submit_policy: SubmitPolicy,
    pub status: Status,
    pub phase: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub claimed_by: Option<String>,
    #[serde(default)]
    pub claim_expires_ms: Option<i64>,
    #[serde(default)]
    pub requires_operator_approval: bool,
    #[serde(default)]
    pub approved_at: Option<String>,
    #[serde(default)]
    pub declined_at: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    /// The provider's own record of the result, as the extension saw it.
    #[serde(default)]
    pub receipt: Option<Value>,
    #[serde(default)]
    pub events: Vec<Event>,
}

/// Holds the queue's lock until dropped.
struct Locked {
    _file: fs::File,
}

fn lock(ctx: &Ctx) -> Result<Locked> {
    let f = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(ctx.actions.join(".lock"))
        .map_err(|_| Refused(500, "action_storage_unavailable"))?;
    f.lock()
        .map_err(|_| Refused(500, "action_storage_unavailable"))?;
    Ok(Locked { _file: f })
}

fn path(ctx: &Ctx, id: &str) -> Option<PathBuf> {
    (safe_token(id) == id).then(|| ctx.actions.join(format!("{id}.json")))
}

fn load(ctx: &Ctx, id: &str) -> Option<Action> {
    serde_json::from_slice(&fs::read(path(ctx, id)?).ok()?).ok()
}

fn save(ctx: &Ctx, a: &Action) -> Result<()> {
    let p = path(ctx, &a.action_id).ok_or(Refused(400, "invalid_action_id"))?;
    let bytes = serde_json::to_vec_pretty(a).map_err(|_| Refused(500, "write_failed"))?;
    write_durable(&p, &bytes).map_err(|_| Refused(500, "write_failed"))
}

fn all(ctx: &Ctx) -> Vec<Action> {
    let mut out: Vec<Action> = fs::read_dir(&ctx.actions)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_slice(&fs::read(e.path()).ok()?).ok())
        .collect();
    out.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    out
}

fn event(kind: &str, phase: &str, detail: Option<String>, owner: Option<String>) -> Event {
    Event {
        event_id: new_id(),
        at: now_iso(),
        kind: kind.into(),
        phase: phase.into(),
        detail,
        owner_instance_id: owner,
    }
}

fn input_path(ctx: &Ctx, digest: &str) -> Option<PathBuf> {
    (digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| ctx.actions.join("inputs").join(digest))
}

/// Queues an action. The caller has already established it is the
/// operator; `enabled` is the chat-actions switch.
pub fn enqueue(ctx: &Ctx, req: &ActionRequest, enabled: bool) -> Result<(bool, Action)> {
    if !enabled {
        return Err(Refused(409, "browser_actions_off"));
    }
    if !PROVIDERS.contains(&req.provider.as_str()) || !OPERATIONS.contains(&req.operation.as_str())
    {
        return Err(Refused(400, "unsupported_action"));
    }
    if req.text.trim().is_empty() {
        return Err(Refused(400, "empty_text"));
    }
    let is_new = req.target.conversation_id == "new";
    if (req.operation == "conversation.create") != is_new {
        return Err(Refused(400, "target_does_not_match_operation"));
    }
    let request_sha = sha256_hex(&serde_json::to_vec(req).unwrap_or_default());
    let key = req
        .idempotency_key
        .clone()
        .unwrap_or_else(|| request_sha.clone());
    let _lock = lock(ctx)?;
    // A resent request with the same key is the same action.
    if let Some(existing) = all(ctx).into_iter().find(|a| a.idempotency_key == key) {
        return if existing.request_sha256 == request_sha {
            Ok((false, existing))
        } else {
            Err(Refused(409, "idempotency_key_reused"))
        };
    }
    let mut attachments = Vec::new();
    for a in &req.attachments {
        let size = input_path(ctx, &a.attachment_ref)
            .and_then(|p| fs::metadata(p).ok())
            .map(|m| m.len())
            .ok_or(Refused(400, "unknown_attachment"))?;
        let basename = Path::new(&a.name)
            .file_name()
            .is_some_and(|n| n == a.name.as_str());
        if !basename || a.name.chars().any(char::is_control) {
            return Err(Refused(400, "attachment_name_not_a_basename"));
        }
        attachments.push(Attachment {
            attachment_id: new_id(),
            name: a.name.clone(),
            mime_type: a.mime_type.clone(),
            size_bytes: size,
            sha256: a.attachment_ref.clone(),
        });
    }
    let needs_approval = req.submit_policy == SubmitPolicy::SubmitOnce && !is_new;
    let now = now_iso();
    let action = Action {
        contract: "polylogue.browser-actions/v1".into(),
        action_id: req
            .action_id
            .clone()
            .filter(|id| safe_token(id) == *id)
            .unwrap_or_else(new_id),
        idempotency_key: key,
        request_sha256: request_sha,
        receiver_id: ctx.receiver_id.clone(),
        provider: req.provider.clone(),
        operation: req.operation.clone(),
        target: req.target.clone(),
        text: req.text.clone(),
        attachments,
        presentation: req.presentation.clone(),
        submit_policy: req.submit_policy,
        status: if needs_approval {
            Status::AwaitingApproval
        } else {
            Status::Queued
        },
        phase: if needs_approval {
            "awaiting_approval"
        } else {
            "queued"
        }
        .into(),
        created_at: now.clone(),
        updated_at: now,
        claimed_by: None,
        claim_expires_ms: None,
        requires_operator_approval: needs_approval,
        approved_at: None,
        declined_at: None,
        last_error: None,
        receipt: None,
        events: vec![event("enqueued", "queued", None, None)],
    };
    save(ctx, &action)?;
    Ok((true, action))
}

pub fn list(ctx: &Ctx) -> Vec<Action> {
    all(ctx)
}

pub fn read(ctx: &Ctx, id: &str) -> Result<Action> {
    load(ctx, id).ok_or(Refused(404, "unknown_browser_action"))
}

/// Hands the oldest queued action to `owner`, if chat actions are on and no
/// other action is out.
pub fn claim(ctx: &Ctx, owner: &str, enabled: bool) -> Result<Option<Action>> {
    if !enabled {
        return Ok(None);
    }
    let _lock = lock(ctx)?;
    let now = now_ms();
    let mut actions = all(ctx);
    // An expired claim returns its action to the queue.
    for a in &mut actions {
        if a.status == Status::Claimed && a.claim_expires_ms.is_some_and(|t| t < now) {
            a.status = Status::Queued;
            a.phase = "queued".into();
            a.claimed_by = None;
            a.claim_expires_ms = None;
            a.events.push(event("claim_expired", "queued", None, None));
            save(ctx, a)?;
        }
    }
    if actions.iter().any(|a| a.status == Status::Claimed) {
        return Ok(None);
    }
    let Some(a) = actions.iter_mut().find(|a| a.status == Status::Queued) else {
        return Ok(None);
    };
    a.status = Status::Claimed;
    a.phase = "claimed".into();
    a.claimed_by = Some(owner.to_owned());
    a.claim_expires_ms = Some(now + CLAIM_MS);
    a.updated_at = now_iso();
    a.events
        .push(event("claimed", "claimed", None, Some(owner.to_owned())));
    save(ctx, a)?;
    Ok(Some(a.clone()))
}

#[derive(Debug, Deserialize)]
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

/// A report from the extension instance holding the claim. A final
/// outcome ends the claim; any other report renews it.
pub fn report(ctx: &Ctx, id: &str, u: Update) -> Result<Action> {
    let _lock = lock(ctx)?;
    let mut a = load(ctx, id).ok_or(Refused(404, "unknown_browser_action"))?;
    if a.claimed_by.as_deref() != Some(u.owner_instance_id.as_str()) {
        return Err(Refused(409, "not_the_claimant"));
    }
    if a.status.finished() {
        return Err(Refused(409, "action_finished"));
    }
    let next = match u.outcome.as_str() {
        "progress" => None,
        "drafted" => Some(Status::Drafted),
        "submitted" => Some(Status::Submitted),
        "outcome_unknown" => Some(Status::OutcomeUnknown),
        "rate_limited" | "auth_challenge" | "safety_locked" | "provider_warning" => {
            Some(Status::Blocked)
        }
        "failed" | "network_error" | "capability_mismatch" | "provider_drift" => {
            Some(Status::Failed)
        }
        _ => return Err(Refused(400, "unknown_outcome")),
    };
    if a.submit_policy == SubmitPolicy::StageOnly && next == Some(Status::Submitted) {
        return Err(Refused(409, "stage_only_action_cannot_submit"));
    }
    if let Some(s) = next {
        a.status = s;
        a.claimed_by = None;
        a.claim_expires_ms = None;
    } else {
        a.claim_expires_ms = Some(now_ms() + CLAIM_MS);
    }
    if matches!(next, Some(Status::Failed | Status::Blocked)) {
        a.last_error = u.detail.clone().or_else(|| Some(u.outcome.clone()));
    }
    if u.receipt.is_some() {
        a.receipt = u.receipt;
    }
    a.phase.clone_from(&u.phase);
    a.updated_at = now_iso();
    a.events.push(event(
        &u.outcome,
        &u.phase,
        u.detail,
        Some(u.owner_instance_id),
    ));
    save(ctx, &a)?;
    Ok(a)
}

/// The operator's approve or decline of a held submit.
pub fn decide(
    ctx: &Ctx,
    id: &str,
    approve: bool,
    by: &str,
    detail: Option<String>,
) -> Result<Action> {
    let _lock = lock(ctx)?;
    let mut a = load(ctx, id).ok_or(Refused(404, "unknown_browser_action"))?;
    if a.status != Status::AwaitingApproval {
        return Err(Refused(409, "not_awaiting_approval"));
    }
    let now = now_iso();
    if approve {
        a.status = Status::Queued;
        a.phase = "queued".into();
        a.approved_at = Some(now.clone());
    } else {
        a.status = Status::Cancelled;
        a.phase = "declined".into();
        a.declined_at = Some(now.clone());
    }
    a.updated_at = now;
    let kind = if approve { "approve" } else { "decline" };
    a.events
        .push(event(kind, &a.phase.clone(), detail, Some(by.to_owned())));
    save(ctx, &a)?;
    Ok(a)
}

/// Stores an attachment's bytes for a later action, named by their
/// SHA-256.
pub fn upload(ctx: &Ctx, bytes: &[u8]) -> Result<Value> {
    if bytes.len() > BODY_LIMIT {
        return Err(Refused(413, "attachment_too_large"));
    }
    let digest = sha256_hex(bytes);
    let p = input_path(ctx, &digest).ok_or(Refused(500, "write_failed"))?;
    if !p.exists() {
        write_durable(&p, bytes).map_err(|_| Refused(500, "write_failed"))?;
    }
    Ok(json!({ "ok": true, "attachment_ref": digest, "size_bytes": bytes.len() }))
}

/// An attachment's bytes and type, checked against their hash.
pub fn attachment(ctx: &Ctx, id: &str, attachment: &str) -> Result<(String, Vec<u8>)> {
    let a = load(ctx, id).ok_or(Refused(404, "unknown_browser_action"))?;
    let att = a
        .attachments
        .iter()
        .find(|x| x.attachment_id == attachment)
        .ok_or(Refused(404, "unknown_browser_action_attachment"))?;
    let p = input_path(ctx, &att.sha256).ok_or(Refused(404, "attachment_missing"))?;
    let bytes = fs::read(p).map_err(|_| Refused(404, "attachment_missing"))?;
    if sha256_hex(&bytes) != att.sha256 {
        return Err(Refused(409, "attachment_integrity_mismatch"));
    }
    Ok((att.mime_type.clone(), bytes))
}
