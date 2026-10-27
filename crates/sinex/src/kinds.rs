//! AI sessions as Polylogue reads them, and what people and agents add to
//! them. Polylogue captures the files AI tools write and sends each as an
//! ordinary material of its real input (a Claude Code session file, a Codex
//! rollout, an export). It parses them itself and sends these events, each
//! anchored at its record in that material and keyed by Polylogue's stable
//! IDs: a message carried by two files, or read again by a changed parser, is
//! one record, and the newer reading replaces the older. Sinex never
//! parses AI sessions.
//!
//! Notes, settings, kept results, deliveries and excisions are requests
//! Polylogue writes: each is a small material holding the request (a request
//! material) and one event below.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinex_model::{MaterialId, Timestamp};

/// One AI session: a conversation or agent run in one tool.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.session", key = ["session"], replace = newer, time = started, quality = intrinsic)]
pub struct Session {
    /// `<origin>:<native id>`, such as `claude-code:7f3a…`.
    pub session: String,
    /// The tool that wrote it: `claude-code`, `codex`, `chatgpt`, ….
    pub origin: String,
    /// `main`, `subagent`, `sidechain` or `compaction`.
    pub kind: String,
    pub title: Option<String>,
    /// Working directories in the order the session used them.
    pub cwd: Vec<String>,
    /// The repository, as the session names it or as its directory resolves.
    pub repo: Option<String>,
    pub branch: Option<String>,
    /// The tool's version.
    pub tool_version: Option<String>,
    /// How the tool ran it, when not interactively (`bg`).
    pub mode: Option<String>,
    /// The machine whose files hold it (`/etc/machine-id`), when known.
    pub machine: Option<String>,
    pub started: Option<Timestamp>,
    pub last_activity: Option<Timestamp>,
    /// How it relates to other sessions, as their records state it.
    pub links: Vec<SessionLink>,
}

/// A relationship between two sessions. Lineage is a reference between
/// sessions, never event ancestry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionLink {
    /// `fork`, `spawn`, `continuation` or `copy`.
    pub kind: String,
    pub parent: String,
    pub child: String,
    /// The parent's message the child branched from or was spawned by.
    pub at: Option<String>,
    /// The parent's tool call that spawned the child.
    pub call_id: Option<String>,
}

/// One message of a session, stored once where it originated. Its text is
/// what people and models wrote, so it is guarded.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.message", key = ["message"], replace = newer, time = at, quality = intrinsic, class = guarded)]
pub struct Message {
    pub message: String,
    pub session: String,
    /// The message it replies to, when the source records it.
    pub parent: Option<String>,
    /// `user`, `assistant` or `system`.
    pub role: String,
    /// Who wrote it: `human`, `model`, `harness` or `unknown`.
    pub written_by: String,
    /// A typed prompt's kind: `prose`, `command`, `harness` or `model`.
    pub prompt_kind: Option<String>,
    pub at: Option<Timestamp>,
    pub model: Option<String>,
    /// Its text blocks, joined by blank lines. Tool inputs and outputs stay
    /// in the material.
    pub text: String,
    pub usage: Usage,
    /// Polylogue's cost of the message in billionths of a dollar; unknown
    /// when absent.
    pub cost_nusd: Option<i64>,
    pub stop_reason: Option<String>,
    pub duration_ms: Option<u64>,
    pub attachments: Vec<Attachment>,
    /// The other records that carry a version of this message (the same
    /// message in an export and in a live file). The event's own location is
    /// the version Polylogue shows.
    pub copies: Vec<RecordRef>,
}

/// Token counts as the source reported them; unknown when absent.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Usage {
    pub input: Option<u64>,
    pub cached_input: Option<u64>,
    pub cache_write: Option<u64>,
    pub output: Option<u64>,
    pub reasoning: Option<u64>,
}

/// An image or file a message carried.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Attachment {
    pub name: Option<String>,
    pub media_type: Option<String>,
    pub size: Option<u64>,
    /// `blake3:<hex>` of the bytes, when they were stored.
    pub blob: Option<String>,
    /// `input` (to the model), `output` (from it) or `unknown`.
    pub direction: String,
}

/// A record in a material: another version of a message, or a tool call's
/// output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecordRef {
    pub material: MaterialId,
    /// Byte position of the record in that material.
    pub anchor: u64,
}

/// One tool call and how it ended. Its arguments and output stay in the
/// material: `result` names the record holding the output.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.tool_call", key = ["call"], replace = newer, time = at, quality = intrinsic, class = guarded)]
pub struct ToolCall {
    /// `tool:<session>#<call id>`, Polylogue's reference to the call.
    pub call: String,
    pub session: String,
    /// The message that made the call.
    pub message: String,
    pub tool: String,
    /// `shell`, `code`, `file_read`, `file_write`, `file_edit`, `search`,
    /// `web`, `subagent`, `mcp` or `other`.
    pub action: String,
    /// The shell command, when the call runs one.
    pub command: Option<String>,
    /// The file it reads or changes, when it names one.
    pub path: Option<String>,
    /// `ok`, `error` or `unknown`.
    pub outcome: String,
    pub outcome_reason: Option<String>,
    pub exit_code: Option<i64>,
    pub at: Option<Timestamp>,
    /// The record holding the call's output.
    pub result: Option<RecordRef>,
}

/// One turn: a person's prompts and what answered them, so other sources can
/// be read during it.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.turn", key = ["turn"], replace = newer, time = start, quality = intrinsic)]
pub struct Turn {
    /// Its first prompt's message ID.
    pub turn: String,
    pub session: String,
    pub start: Option<Timestamp>,
    pub end: Option<Timestamp>,
    /// The prompt messages, in order.
    pub prompts: Vec<String>,
    /// The last model message of the turn.
    pub answer: Option<String>,
    /// Whether the turn opened with a slash command.
    pub slash_command: bool,
    pub tool_calls: u64,
    pub errors: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cost_nusd: Option<i64>,
    /// The session's last turn, with no answer.
    pub abandoned: bool,
}

/// Context Polylogue prepared for a model or the operator: what it held and
/// what it left out. The text itself is in the record's material.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.delivery", key = ["delivery"], replace = repeat, time = at, quality = intrinsic)]
pub struct Delivery {
    pub delivery: String,
    /// `session` or `operator`.
    pub recipient_kind: String,
    pub recipient_id: Option<String>,
    /// `hook`, `mcp`, `cli`, `ping` or `signal`.
    pub sender: String,
    /// `session_start`, `resume`, `compaction`, `recall`, ….
    pub event: String,
    pub cwd: Option<String>,
    pub at: Timestamp,
    /// `blake3:<hex>` of the delivered text.
    pub content_hash: String,
    pub bytes: u64,
    /// References included, in order.
    pub refs: Vec<String>,
    pub left_out: Vec<LeftOut>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LeftOut {
    pub what: String,
    pub reason: String,
}

/// What happened to a prepared delivery.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.delivery_outcome", key = ["delivery", "what", "at"], replace = repeat, time = at, quality = intrinsic)]
pub struct DeliveryOutcome {
    pub delivery: String,
    /// `delivered`, `failed`, `not_sent` or `used`.
    pub what: String,
    pub at: Timestamp,
    pub detail: Option<String>,
}

/// A note, tag, title, judgment, saved query or other assertion a person or
/// agent wrote about something. A revision is a new note with the same
/// `first_note`; a withdrawal is a note of kind `withdrawal`.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.note", key = ["note"], replace = repeat, time = created_at, quality = intrinsic, class = guarded)]
pub struct Note {
    pub note: String,
    /// The first note of its revisions; the note itself for a first note.
    pub first_note: String,
    pub kind: String,
    pub key: Option<String>,
    pub value: Option<String>,
    /// `operator`, an agent session ID, `agent` or `script`.
    pub author: String,
    pub machine: String,
    pub created_at: Timestamp,
    pub valid_until: Option<Timestamp>,
    /// What it is about: `session:…`, `message:…`, `tool:…`, ….
    pub targets: Vec<String>,
    pub relations: Vec<NoteRelation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoteRelation {
    /// `revises`, `withdraws`, `answers`, `challenges`, ….
    pub kind: String,
    pub other: String,
}

/// One value of a Polylogue setting; a change is a new record.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.setting", key = ["setting"], replace = repeat, time = set_at, quality = intrinsic)]
pub struct Setting {
    pub setting: String,
    /// The machine it applies to, or `*` for every machine.
    pub machine: String,
    pub key: String,
    pub value: String,
    pub author: String,
    pub set_at: Timestamp,
}

/// A query's answer kept as it was, cited as `result:<id>`.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.kept_result", key = ["result"], replace = repeat, time = ran_at, quality = intrinsic, class = guarded)]
pub struct KeptResult {
    pub result: String,
    pub name: Option<String>,
    pub query: String,
    pub ran_at: Timestamp,
    /// What its read transaction read (the query's resolved time).
    pub read_at: Value,
    /// The rows it returned.
    pub rows: Value,
}

/// The operator's request to remove something from Polylogue for good.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, sinex_model::EventPayload,
)]
#[serde(deny_unknown_fields)]
#[event(version = 2, type = "polylogue.excision", key = ["excision"], replace = repeat, time = created_at, quality = operator)]
pub struct Excision {
    pub excision: String,
    /// What it removed: `input:<id>`, `session:<id>` or `document:<id>`.
    pub target: String,
    pub reason: Option<String>,
    pub created_at: Timestamp,
}

pub fn definition_hash(producer: &str) -> String {
    let hashes: Vec<_> = declarations(producer)
        .iter()
        .map(|d| d.definition_hash())
        .collect();
    blake3::hash(&serde_json::to_vec(&hashes).expect("declaration hashes serialize"))
        .to_hex()
        .to_string()
}

pub fn declarations(producer: &str) -> Vec<sinex_model::KindDeclaration> {
    use sinex_model::EventPayload;
    let mut declarations = vec![
        Session::declaration(),
        Message::declaration(),
        ToolCall::declaration(),
        Turn::declaration(),
        Delivery::declaration(),
        DeliveryOutcome::declaration(),
        Note::declaration(),
        Setting::declaration(),
        KeptResult::declaration(),
        Excision::declaration(),
    ];
    for d in &mut declarations {
        d.declaring_producer = producer.to_string();
        d.normalization = sinex_model::kinds::Normalization::Json;
    }
    declarations
}

#[cfg(test)]
mod tests;
