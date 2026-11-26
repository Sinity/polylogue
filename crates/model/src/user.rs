//! Authored request payloads and their portable values. Storage and the
//! clock that creates a request belong to the caller.

use crate::{BlobHash, DeliveryOutcome, Timestamp, authorship::Authorship};
use serde::{Deserialize, Serialize};

/// Trusted host and wall-clock inputs captured by the binary. Authored writes
/// receive one timestamp from this handle; storage never samples another clock.
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    machine: String,
    clock: crate::WallClock,
}
impl RequestMetadata {
    pub fn new(machine: String, clock: crate::WallClock) -> Self {
        Self { machine, clock }
    }
    pub fn request(&self, id: String, write: UserWrite, authorship: Authorship) -> UserRequest {
        UserRequest {
            id,
            machine: self.machine.clone(),
            authorship,
            at: self.clock.now().into(),
            write,
        }
    }
    pub fn now(&self) -> crate::Timestamp {
        self.clock.now().into()
    }
    pub fn machine(&self) -> &str {
        &self.machine
    }
}

/// Accepted inputs for Save. Native keyed rows are hydrated at the step;
/// derived relations and match rows retain their resolved values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LineSelection {
    pub binding: crate::query_objects::ExecutionBinding,
    pub relation: Option<serde_json::Value>,
    pub matches: Option<serde_json::Value>,
}

/// One Save step's accepted rows and optional named question and watch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Save {
    pub name: Option<String>,
    pub query: String,
    pub denominator: Option<String>,
    pub rows_json: String,
    pub read_position: String,
    pub watch: Option<SaveWatch>,
    pub outcomes: Option<SaveReport>,
}
/// The exact objects the accepting Save wrote or reused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveReport {
    pub request: String,
    pub saved: Option<SavedPart>,
    pub saved_rows: KeptPart,
    pub watch: Option<WatchPart>,
    pub repeat: bool,
    pub pending: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SavedPart {
    /// The actual accepted annotation revision, including a reused revision.
    pub id: String,
    pub name: String,
    pub revision: u32,
    pub new_revision: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeptPart {
    pub result: String,
    pub rows: u64,
    pub at: Timestamp,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WatchPart {
    /// The actual accepted annotation revision, including a reused rule.
    pub id: String,
    pub name: String,
    pub new_rule: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveWatch {
    pub hold: Option<String>,
    pub notify: bool,
}

/// One reader's displayed position in a target, using the captured change position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeenMark {
    pub target: String,
    pub item: Option<String>,
    pub at: crate::ChangePosition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "write", rename_all = "snake_case", deny_unknown_fields)]
pub enum UserWrite {
    /// Advances only this computed author's displayed positions.
    MarkSeen {
        marks: Vec<SeenMark>,
    },
    /// Names and keeps one accepted selection in one transaction.
    Save(Box<Save>),
    /// A line's accepted intent and fixed population, shared by every step and retry.
    ResolveLine {
        request: serde_json::Value,
        resolved: Option<Vec<crate::Reference>>,
        retained: Option<Box<LineSelection>>,
    },
    /// A trusted daemon checkpoint with its immutable signal pending_delivery and receipts.
    SignalCommit(Box<crate::signal::Commit>),
    Alternate {
        alternate: crate::NewAlternate,
    },
    Assert {
        annotations: Vec<crate::annotation::NewAnnotation>,
    },
    AssertQuery {
        query: String,
        annotation: crate::annotation::NewAnnotation,
        resolved: Vec<crate::Reference>,
    },
    ReviseAnnotations {
        #[serde(with = "crate::reference::annotation_revisions")]
        #[schemars(with = "Vec<crate::Reference>")]
        bases: Vec<crate::RevisionReference>,
        patch: crate::annotation::AnnotationPatch,
    },
    WithdrawAnnotations {
        targets: Vec<crate::Reference>,
        reason: Option<String>,
    },
    /// Context prepared for a recipient, with its exact text.
    Deliver(Delivery),
    /// Advice already printed by a hook, recorded with its delivered outcome.
    DeliveredAdvice(Delivery),
    /// What happened to a prepared delivery.
    DeliveryEvent {
        delivery: String,
        outcome: DeliveryOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    SetSetting {
        key: String,
        value: String,
        /// `None` for an archive-wide key, otherwise this machine.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        machine: Option<String>,
    },
    /// A chat action the operator queued; a resent request finds the
    /// action it made.
    ChatEnqueue(NewChatAction),
    /// The extension instance `by` takes the oldest queued chat action;
    /// `NotFound` when none is queued or another is out.
    ChatClaim {
        by: String,
    },
    /// What the instance holding a chat action's claim reports.
    ChatReport {
        action: String,
        by: String,
        outcome: String,
        phase: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        receipt: Option<serde_json::Value>,
    },
    /// The operator approves or declines a held submit.
    ChatDecide {
        action: String,
        approve: bool,
        by: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// The operator starts an import job; the provider's job that has not
    /// ended is found instead of a second one.
    ImportJobStart(NewImportJob),
    /// The extension instance `by` takes the job it holds or the oldest
    /// waiting one; `NotFound` when there is none.
    ImportJobClaim {
        by: String,
    },
    /// What the instance holding an import job's claim reports.
    ImportJobReport {
        job: String,
        by: String,
        report: ImportJobReport,
    },
    /// The operator stops an import job.
    ImportJobCancel {
        job: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Delivery {
    pub recipient_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipient_id: Option<String>,
    pub sender: crate::DeliverySender,
    pub event: crate::DeliveryEvent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    pub text: String,
    pub refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub left_out: Vec<LeftOut>,
}

/// A reference or section a delivery did not include, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct LeftOut {
    pub item: String,
    pub reason: String,
}

impl UserWrite {
    /// The caller's content, without data resolved when the write runs.
    pub fn intent(&self) -> Self {
        let mut intent = match self {
            Self::Save(save) => Self::Save(Box::new(Save {
                name: save.name.clone(),
                query: save.query.clone(),
                denominator: save.denominator.clone(),
                rows_json: String::new(),
                read_position: String::new(),
                watch: save.watch.clone(),
                outcomes: None,
            })),
            _ => self.clone(),
        };
        match &mut intent {
            Self::AssertQuery { resolved, .. } => resolved.clear(),
            Self::ResolveLine {
                resolved, retained, ..
            } => {
                *resolved = None;
                *retained = None;
            }
            _ => {}
        }
        intent
    }
}

/// What a user write created, or found already created by the same
/// request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UserOutcome {
    /// The note, delivery or setting row the request wrote.
    pub id: String,
    /// The request had already been written; nothing changed.
    pub repeat: bool,
    #[serde(default)]
    pub pending: bool,
}

/// A bulk annotation retains the exact resolved target population count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AssertQueryOutcome {
    /// Absent when every row was already tagged and no write was submitted.
    pub id: Option<String>,
    pub repeat: bool,
    pub pending: bool,
    pub targets: usize,
    /// Revisions in the accepting request's recorded effects; confirmation
    /// is still reported separately by `pending`.
    pub written: usize,
    /// Human reference spellings for targets skipped by the original request.
    pub already: Vec<String>,
}
impl AssertQueryOutcome {
    pub fn new(
        outcome: UserOutcome,
        targets: usize,
        written: usize,
        already: Vec<crate::Reference>,
    ) -> Self {
        Self {
            id: Some(outcome.id),
            repeat: outcome.repeat,
            pending: outcome.pending,
            targets,
            written,
            already: already.iter().map(ToString::to_string).collect(),
        }
    }
    pub fn all_tagged(already: Vec<crate::Reference>) -> Self {
        Self {
            id: None,
            repeat: false,
            pending: false,
            targets: already.len(),
            written: 0,
            already: already.iter().map(ToString::to_string).collect(),
        }
    }
}

/// One user write with the request that carries it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UserRequest {
    pub id: String,
    pub machine: String,
    pub authorship: Authorship,
    pub at: Timestamp,
    pub write: UserWrite,
}

/// A new chat action, as the writer stores it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NewChatAction {
    pub id: String,
    /// The request's idempotency key: a resent request is the same action.
    pub request_key: String,
    /// BLAKE3 of the request's JSON.
    pub request_hash: BlobHash,
    pub request: serde_json::Value,
    pub stage_only: bool,
    pub needs_approval: bool,
    pub attachments: Vec<ChatAttachment>,
}

/// A file attached to a chat action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChatAttachment {
    pub id: String,
    pub blob: BlobHash,
    pub name: String,
    pub media_type: String,
    pub size: u64,
}

/// A new import job, as the writer stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NewImportJob {
    pub id: String,
    /// The web chat, by the extension's name for it.
    pub provider: String,
    /// Only conversations the provider changed at or after this time (ms).
    pub since: Option<i64>,
}

/// Conversations a job has handled so far.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
pub struct ImportJobCounts {
    /// Listed by the provider.
    pub listed: u64,
    /// Captured and stored.
    pub captured: u64,
    /// Already in the archive as the provider has them.
    pub unchanged: u64,
    /// Could not be captured.
    pub failed: u64,
}

/// What the extension instance holding a job reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImportJobReport {
    /// `progress`, `finished` or `failed`.
    pub kind: String,
    /// Where in the provider's conversation list to continue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    pub counts: ImportJobCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// How long an extension's chat claim lasts, in milliseconds.
pub const CHAT_CLAIM_MS: i64 = 180_000;
/// One thing that happened to a chat action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChatEvent {
    pub request: String,
    pub at: i64,
    pub kind: String,
    pub phase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<serde_json::Value>,
}

crate::stored_names! {
    /// Where a chat action stands.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ChatStatus {
        AwaitingApproval => "awaiting_approval",
        Queued => "queued",
        Claimed => "claimed",
        Drafted => "drafted",
        Submitted => "submitted",
        OutcomeUnknown => "outcome_unknown",
        Blocked => "blocked",
        Failed => "failed",
        Cancelled => "cancelled",
    }
}

impl ChatStatus {
    pub fn finished(self) -> bool {
        matches!(
            self,
            Self::Drafted | Self::Submitted | Self::Failed | Self::Cancelled
        )
    }
}

/// What an extension instance may report about the action it holds, and
/// the state each outcome leaves; `progress` renews the claim.
pub fn chat_outcome(kind: &str) -> Option<Option<ChatStatus>> {
    Some(match kind {
        "progress" => None,
        "drafted" => Some(ChatStatus::Drafted),
        "submitted" => Some(ChatStatus::Submitted),
        "outcome_unknown" => Some(ChatStatus::OutcomeUnknown),
        "rate_limited" | "auth_challenge" | "safety_locked" | "provider_warning" => {
            Some(ChatStatus::Blocked)
        }
        "failed" | "network_error" | "capability_mismatch" | "provider_drift" => {
            Some(ChatStatus::Failed)
        }
        _ => return None,
    })
}

/// A stored chat action with everything that happened to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ChatAction {
    pub id: String,
    pub request_key: String,
    pub request_hash: BlobHash,
    pub request: serde_json::Value,
    pub stage_only: bool,
    pub needs_approval: bool,
    pub created_at: i64,
    pub execution: crate::authored::RunsHere,
    pub attachments: Vec<ChatAttachment>,
    pub events: Vec<ChatEvent>,
}

impl ChatAction {
    /// The state at `now`, and the instance holding an unexpired claim.
    pub fn status(&self, now: i64) -> (ChatStatus, Option<&str>) {
        chat_status(self.needs_approval, &self.events, now)
    }

    /// The instance that started sending may still confirm the outcome, without
    /// renewing its claim.
    pub fn unconfirmed_submitter(&self, now: i64) -> Option<&str> {
        let state = chat_state(self.needs_approval, &self.events);
        (state.status(now).0 == ChatStatus::OutcomeUnknown)
            .then_some(state.sending_by)
            .flatten()
    }
}

/// The state a chat action's events leave, before the clock decides
/// whether a claim has expired.
fn chat_state(needs_approval: bool, events: &[ChatEvent]) -> ChatState<'_> {
    let mut status = if needs_approval {
        ChatStatus::AwaitingApproval
    } else {
        ChatStatus::Queued
    };
    let mut claim: Option<(&str, i64)> = None;
    // A later display phase or claim renewal cannot undo a possible send.
    let mut sending_by = None;
    for e in events {
        match e.kind.as_str() {
            "approved" => status = ChatStatus::Queued,
            "declined" => status = ChatStatus::Cancelled,
            "claimed"
                if sending_by.is_none()
                    && matches!(status, ChatStatus::Queued | ChatStatus::Claimed) =>
            {
                status = ChatStatus::Claimed;
                claim = e.by.as_deref().map(|by| (by, e.at + CHAT_CLAIM_MS));
            }
            "progress" => {
                if status == ChatStatus::Claimed
                    && let Some((by, until)) = claim
                    && e.by.as_deref() == Some(by)
                    && e.at <= until
                {
                    if e.phase == "sending" {
                        sending_by.get_or_insert(by);
                    }
                    claim = Some((by, e.at + CHAT_CLAIM_MS));
                }
            }
            kind => {
                if let Some(Some(next)) = chat_outcome(kind) {
                    status = next;
                    claim = None;
                }
            }
        }
    }
    ChatState {
        status,
        claim,
        sending_by,
    }
}

/// The state at `now` that `events` leave a chat action in, and the
/// instance holding an unexpired claim.
pub fn chat_status(
    needs_approval: bool,
    events: &[ChatEvent],
    now: i64,
) -> (ChatStatus, Option<&str>) {
    chat_state(needs_approval, events).status(now)
}

struct ChatState<'a> {
    status: ChatStatus,
    claim: Option<(&'a str, i64)>,
    sending_by: Option<&'a str>,
}

impl<'a> ChatState<'a> {
    fn status(&self, now: i64) -> (ChatStatus, Option<&'a str>) {
        match (self.status, self.claim) {
            (ChatStatus::Claimed, Some((by, until))) if until >= now => (self.status, Some(by)),
            (ChatStatus::Claimed, _) if self.sending_by.is_some() => {
                (ChatStatus::OutcomeUnknown, None)
            }
            (ChatStatus::Claimed, _) => (ChatStatus::Queued, None),
            _ => (self.status, None),
        }
    }
}

/// How long an extension's import job claim lasts, in milliseconds.
pub const IMPORT_JOB_CLAIM_MS: i64 = 180_000;
/// One thing that happened to an import job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImportJobEvent {
    pub request: String,
    pub at: i64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    pub counts: ImportJobCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

crate::stored_names! {
    /// Where an import job stands.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ImportJobStatus {
        /// Started and not held by an extension instance.
        Waiting => "waiting",
        Running => "running",
        /// Stopped by the provider refusing the browser's sign-in; not claimed
        /// until the operator starts the import again, which resumes it from
        /// its saved position.
        Paused => "paused",
        Finished => "finished",
        Failed => "failed",
        Cancelled => "cancelled",
    }
}

impl ImportJobStatus {
    pub fn ended(self) -> bool {
        matches!(self, Self::Finished | Self::Failed | Self::Cancelled)
    }
}

/// What the extension instance holding a job may report, and the state
/// each report leaves; `progress` renews the claim.
pub fn import_job_report_status(kind: &str) -> Option<ImportJobStatus> {
    Some(match kind {
        "progress" => ImportJobStatus::Running,
        "paused" => ImportJobStatus::Paused,
        "finished" => ImportJobStatus::Finished,
        "failed" => ImportJobStatus::Failed,
        _ => return None,
    })
}

/// A stored import job with everything that happened to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImportJob {
    pub id: String,
    pub provider: String,
    pub since: Option<i64>,
    pub created_at: i64,
    pub execution: crate::authored::RunsHere,
    pub events: Vec<ImportJobEvent>,
}

impl ImportJob {
    /// The state at `now`, and the instance holding an unexpired claim.
    pub fn status(&self, now: i64) -> (ImportJobStatus, Option<&str>) {
        import_job_status(&self.events, now)
    }

    /// The totals of the latest event, and the latest reported list position: a
    /// later event without a position keeps the earlier one, which is where a
    /// later claim resumes.
    pub fn progress(&self) -> (ImportJobCounts, Option<&str>) {
        let counts = self.events.last().map(|e| e.counts).unwrap_or_default();
        let position = self.events.iter().rev().find_map(|e| e.position.as_deref());
        (counts, position)
    }
}

/// The state at `now` that `events` leave an import job in, and the
/// instance holding an unexpired claim.
pub fn import_job_status(events: &[ImportJobEvent], now: i64) -> (ImportJobStatus, Option<&str>) {
    let mut status = ImportJobStatus::Waiting;
    let mut claim: Option<(&str, i64)> = None;
    for e in events {
        match e.kind.as_str() {
            "claimed" | "progress" => {
                status = ImportJobStatus::Running;
                claim = e.by.as_deref().map(|by| (by, e.at + IMPORT_JOB_CLAIM_MS));
            }
            "cancelled" => {
                status = ImportJobStatus::Cancelled;
                claim = None;
            }
            "resumed" => {
                status = ImportJobStatus::Waiting;
                claim = None;
            }
            kind => {
                if let Some(next) = import_job_report_status(kind) {
                    status = next;
                    claim = None;
                }
            }
        }
    }
    match (status, claim) {
        (ImportJobStatus::Running, Some((by, until))) if until >= now => (status, Some(by)),
        (ImportJobStatus::Running, _) => (ImportJobStatus::Waiting, None),
        _ => (status, None),
    }
}

#[cfg(test)]
mod tests;
