//! Authored request payloads and their portable values. Storage and the
//! clock that creates a request belong to the caller.

use crate::{Author, BlobHash, DeliveryOutcome, NewNote, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "write", rename_all = "snake_case")]
pub enum UserWrite {
    Variant {
        variant: crate::NewVariant,
    },
    AddNote {
        note: NewNote,
    },
    /// A saved query whose previous revision is resolved by name.
    SaveQuery {
        note: NewNote,
    },
    /// The query intent and its resolved targets, kept together for retries and replay.
    AnnotateQuery {
        query: String,
        note: NewNote,
    },
    /// Withdraws a note family: the newest revision and every older one.
    Withdraw {
        note: String,
        author: Author,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Context prepared for a recipient, with its exact text.
    Deliver(Delivery),
    /// Advice already printed by a hook, recorded with its delivered outcome.
    DeliveredAdvice(Delivery),
    /// What happened to a prepared delivery.
    DeliveryEvent {
        delivery: String,
        what: DeliveryOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// The exact rows a query returned, kept to be read back as
    /// `result:<id>`.
    KeepResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        query: String,
        /// The stored rows, match rows and their columns, as JSON.
        rows_json: String,
        /// What the query read: its resolved time, as JSON.
        read_at: String,
    },
    SetSetting {
        key: String,
        value: String,
        author: Author,
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
    /// The operator starts a backfill job; the provider's job that has not
    /// ended is found instead of a second one.
    BackfillStart(NewBackfillJob),
    /// The extension instance `by` takes the job it holds or the oldest
    /// waiting one; `NotFound` when there is none.
    BackfillClaim {
        by: String,
    },
    /// What the instance holding a backfill job's claim reports.
    BackfillReport {
        job: String,
        by: String,
        report: BackfillReport,
    },
    /// The operator stops a backfill job.
    BackfillCancel {
        job: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Delivery {
    pub recipient_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipient_id: Option<String>,
    pub sender: String,
    pub event: String,
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
    pub what: String,
    pub reason: String,
}

impl UserWrite {
    /// The caller's content, without data resolved when the write runs.
    pub fn intent(&self) -> Self {
        let mut intent = match self {
            Self::KeepResult { name, query, .. } => Self::KeepResult {
                name: name.clone(),
                query: query.clone(),
                rows_json: String::new(),
                read_at: String::new(),
            },
            _ => self.clone(),
        };
        match &mut intent {
            Self::AnnotateQuery { note, .. } => note.targets.clear(),
            Self::KeepResult {
                rows_json, read_at, ..
            } => {
                rows_json.clear();
                read_at.clear();
            }
            // An explicit revision in AddNote remains part of the request.
            Self::SaveQuery { note } => note.revises = None,
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

/// One user write with the request that carries it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UserRequest {
    pub id: String,
    pub machine: String,
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

/// A new backfill job, as the writer stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NewBackfillJob {
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
pub struct BackfillCounts {
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
pub struct BackfillReport {
    /// `progress`, `finished` or `failed`.
    pub kind: String,
    /// Where in the provider's conversation list to continue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    pub counts: BackfillCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// An extension chat claim lease, in milliseconds.
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

/// Where a chat action stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChatStatus {
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
    pub execution: crate::authored::ExecutionEligibility,
    pub attachments: Vec<ChatAttachment>,
    pub events: Vec<ChatEvent>,
}

impl ChatAction {
    /// The state at `now`, and the instance holding an unexpired claim.
    pub fn status(&self, now: i64) -> (ChatStatus, Option<&str>) {
        let mut status = if self.needs_approval {
            ChatStatus::AwaitingApproval
        } else {
            ChatStatus::Queued
        };
        let mut claim: Option<(&str, i64)> = None;
        for e in &self.events {
            match e.kind.as_str() {
                "approved" => status = ChatStatus::Queued,
                "declined" => status = ChatStatus::Cancelled,
                "claimed" | "progress" => {
                    status = ChatStatus::Claimed;
                    claim = e.by.as_deref().map(|by| (by, e.at + CHAT_CLAIM_MS));
                }
                kind => {
                    if let Some(Some(next)) = chat_outcome(kind) {
                        status = next;
                        claim = None;
                    }
                }
            }
        }
        match (status, claim) {
            (ChatStatus::Claimed, Some((by, until))) if until >= now => (status, Some(by)),
            (ChatStatus::Claimed, _) => (ChatStatus::Queued, None),
            _ => (status, None),
        }
    }
}

/// An extension backfill claim lease, in milliseconds.
pub const BACKFILL_CLAIM_MS: i64 = 180_000;
/// One thing that happened to a backfill job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BackfillEvent {
    pub request: String,
    pub at: i64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    pub counts: BackfillCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Where a backfill job stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BackfillStatus {
    /// Started and not held by an extension instance.
    Waiting,
    Running,
    Finished,
    Failed,
    Cancelled,
}

impl BackfillStatus {
    pub fn ended(self) -> bool {
        matches!(self, Self::Finished | Self::Failed | Self::Cancelled)
    }
}

/// What the extension instance holding a job may report, and the state
/// each report leaves; `progress` renews the claim.
pub fn backfill_report_status(kind: &str) -> Option<BackfillStatus> {
    Some(match kind {
        "progress" => BackfillStatus::Running,
        "finished" => BackfillStatus::Finished,
        "failed" => BackfillStatus::Failed,
        _ => return None,
    })
}

/// A stored backfill job with everything that happened to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BackfillJob {
    pub id: String,
    pub provider: String,
    pub since: Option<i64>,
    pub created_at: i64,
    pub execution: crate::authored::ExecutionEligibility,
    pub events: Vec<BackfillEvent>,
}

impl BackfillJob {
    /// The state at `now`, and the instance holding an unexpired claim.
    pub fn status(&self, now: i64) -> (BackfillStatus, Option<&str>) {
        let mut status = BackfillStatus::Waiting;
        let mut claim: Option<(&str, i64)> = None;
        for e in &self.events {
            match e.kind.as_str() {
                "claimed" | "progress" => {
                    status = BackfillStatus::Running;
                    claim = e.by.as_deref().map(|by| (by, e.at + BACKFILL_CLAIM_MS));
                }
                "cancelled" => {
                    status = BackfillStatus::Cancelled;
                    claim = None;
                }
                kind => {
                    if let Some(next) = backfill_report_status(kind) {
                        status = next;
                        claim = None;
                    }
                }
            }
        }
        match (status, claim) {
            (BackfillStatus::Running, Some((by, until))) if until >= now => (status, Some(by)),
            (BackfillStatus::Running, _) => (BackfillStatus::Waiting, None),
            _ => (status, None),
        }
    }

    /// The totals and list position of the latest report.
    pub fn progress(&self) -> (BackfillCounts, Option<&str>) {
        let counts = self.events.last().map(|e| e.counts).unwrap_or_default();
        let position = self.events.iter().rev().find_map(|e| e.position.as_deref());
        (counts, position)
    }
}
