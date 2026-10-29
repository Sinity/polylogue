//! Typed product field declarations. SQL expressions are owned by Store.
use crate::{ColKind, ast::Unit};
use polylogue_model::{
    BlockKind, CallOutcome, DELIVERY_STATES, DeliveryEvent, DeliverySender, DocumentAuthorship,
    DocumentPurpose, LINK_HOW_NAMES, LINK_KIND_NAMES, PromptKind, Role, SessionKind, WorkNodeKind,
    WrittenBy,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Whole value, case-insensitive.
    Match,
    /// The value or a unique prefix of it (IDs).
    Prefix,
    /// A substring, case- and accent-blind.
    Substring,
    /// A path and the directories below it, by whole components, exactly.
    Subtree,
    Number,
    /// Dollars, stored as nano-dollars.
    Money,
    /// Milliseconds.
    Duration,
    /// Unix milliseconds.
    Time,
    Bool,
    /// Product-specific matching semantics.
    Special,
}

pub struct Field {
    pub unit: Unit,
    pub name: &'static str,
    pub kind: Kind,
    /// The closed set of values, when there is one.
    pub values: Option<&'static [&'static str]>,
    /// The stored word representing an unknown value, when present.
    pub stored_unknown: Option<&'static str>,
    pub help: &'static str,
    /// The column kind of the value when the matching kind does not say.
    pub value_kind: Option<ColKind>,
}

impl Field {
    const fn stored_unknown(mut self, word: &'static str) -> Self {
        self.stored_unknown = Some(word);
        self
    }

    /// The value is true or false, though the field matches by its own rule.
    const fn boolean(mut self) -> Self {
        self.value_kind = Some(ColKind::Bool);
        self
    }

    const fn ids(mut self) -> Self {
        self.value_kind = Some(ColKind::Ids);
        self
    }

    const fn json(mut self) -> Self {
        self.value_kind = Some(ColKind::Json);
        self
    }

    /// The kind of the field's value as a column.
    pub fn column_kind(&self) -> ColKind {
        if let Some(k) = self.value_kind {
            return k;
        }
        match self.kind {
            Kind::Number => ColKind::Number,
            Kind::Money => ColKind::Money,
            Kind::Duration => ColKind::Duration,
            Kind::Time => ColKind::Time,
            Kind::Bool => ColKind::Bool,
            _ => ColKind::Text,
        }
    }
}

const fn f(unit: Unit, name: &'static str, kind: Kind, help: &'static str) -> Field {
    Field {
        unit,
        name,
        kind,
        values: None,
        stored_unknown: None,
        help,
        value_kind: None,
    }
}

/// A field with product-specific matching semantics.
const fn sp(unit: Unit, name: &'static str, help: &'static str) -> Field {
    Field {
        unit,
        name,
        kind: Kind::Special,
        values: None,
        stored_unknown: None,
        help,
        value_kind: None,
    }
}

const fn v(
    unit: Unit,
    name: &'static str,
    values: &'static [&'static str],
    help: &'static str,
) -> Field {
    Field {
        unit,
        name,
        kind: Kind::Match,
        values: Some(values),
        stored_unknown: None,
        help,
        value_kind: None,
    }
}

/// A field with a closed set of values and its own compile rule.
const fn sv(
    unit: Unit,
    name: &'static str,
    values: &'static [&'static str],
    help: &'static str,
) -> Field {
    Field {
        values: Some(values),
        ..sp(unit, name, help)
    }
}

use Kind::*;
use Unit::*;

const SESSION_HAS_NAMES: [&str; 8] = [
    "thinking",
    "tool_calls",
    "errors",
    "attachments",
    "compaction",
    "branches",
    "note",
    "conflict",
];
const MESSAGE_HAS_NAMES: [&str; 4] = ["thinking", "tool_calls", "attachments", "code"];
pub const LIVE_STATES: &[&str] = &["working", "waiting", "idle", "unknown", "stale", "ended"];
pub const SESSION_OUTCOMES: &[&str] = &["solved", "partial", "abandoned", "none"];
pub static FIELDS: &[Field] = &[
    f(Documents, "id", Prefix, "document ID or a unique prefix"),
    f(Documents, "origin", Match, "the provider export family"),
    v(
        Documents,
        "purpose",
        DocumentPurpose::NAMES,
        "instruction, context, memory or transcript",
    ),
    v(
        Documents,
        "authorship",
        DocumentAuthorship::NAMES,
        "who wrote the text, unknown without source evidence",
    )
    .stored_unknown("unknown"),
    f(Documents, "title", Substring, "the supplied source title"),
    f(
        Documents,
        "source_time",
        Time,
        "the provider's source-record time",
    ),
    f(
        Documents,
        "versions",
        Number,
        "the number of distinct versions its sources hold",
    ),
    f(Sessions, "id", Prefix, "session ID or a unique prefix"),
    f(Sessions, "origin", Match, "claude-code, codex, chatgpt, …"),
    v(
        Sessions,
        "kind",
        SessionKind::NAMES,
        "who started the session: main, or one another session spawned",
    ),
    f(Sessions, "title", Substring, "the displayed title"),
    sp(Sessions, "repo", "the repository: name, owner/name or URL"),
    f(Sessions, "branch", Match, "the git branch, as recorded"),
    sp(
        Sessions,
        "commit",
        "a commit recorded or linked (prefix, 7+ hex digits)",
    ),
    sp(Sessions, "pr", "a pull request linked to the session"),
    v(
        Sessions,
        "outcome",
        SESSION_OUTCOMES,
        "the session's outcome note",
    ),
    f(Sessions, "hidden", Bool, "has a hidden note"),
    sp(
        Sessions,
        "cwd",
        "working directory, and the directories below it",
    ),
    v(
        Sessions,
        "source",
        &["present", "missing"],
        "whether a transcript was stored",
    ),
    sv(
        Sessions,
        "live",
        LIVE_STATES,
        "the live state now, from hook records and the time",
    ),
    sp(Sessions, "waiting", "the live state is waiting").boolean(),
    sp(
        Sessions,
        "running",
        "the live state is working, waiting, idle or unknown",
    )
    .boolean(),
    v(
        Sessions,
        "waiting_on",
        &["permission", "question", "notification"],
        "what a waiting session waits for",
    ),
    f(
        Sessions,
        "task",
        Match,
        "the live task: a job ID or Bead ID",
    ),
    f(
        Sessions,
        "live_until",
        Time,
        "when the live state changes with no new record; empty when only a new record can change it",
    ),
    f(
        Sessions,
        "last_event",
        Match,
        "the latest live record: a hook event",
    ),
    f(
        Sessions,
        "last_event_at",
        Time,
        "when the latest live record happened",
    ),
    f(
        Sessions,
        "waiting_since",
        Time,
        "when a waiting session started waiting",
    ),
    f(
        Sessions,
        "todo",
        Substring,
        "the live todo list, as recorded",
    ),
    f(
        Sessions,
        "live_rank",
        Number,
        "0 waiting, 1 working, 2 idle, 3 unknown, 4 stale, 5 ended: the order the live view lists",
    ),
    f(
        Sessions,
        "peers",
        Substring,
        "other working, waiting or idle sessions sharing its repository or working directory",
    ),
    f(
        Sessions,
        "repo_root",
        Prefix,
        "the repository's identity: its root commit hash, when found",
    ),
    f(
        Sessions,
        "harness",
        Match,
        "claude-code, codex-tui, codex-exec, …",
    ),
    sp(Sessions, "model", "a model that wrote at least one message"),
    sp(
        Sessions,
        "tag",
        "a tag note; tag:key:value for key:value tags",
    ),
    sv(
        Sessions,
        "has",
        &SESSION_HAS_NAMES,
        "the session has thinking, tool calls, errors, …",
    ),
    sp(Sessions, "parent", "sessions linked from the given session"),
    sp(
        Sessions,
        "root",
        "the subagent tree under the given session",
    ),
    sp(Sessions, "depth", "depth in the subagent tree"),
    sp(
        Sessions,
        "link",
        "the kind of an incoming link; link:fork, link:resume",
    ),
    sp(
        Sessions,
        "continues",
        "sessions continuing the given session",
    ),
    sp(
        Sessions,
        "continued_by",
        "sessions the given session continues",
    ),
    sp(
        Sessions,
        "connected",
        "every session connected to the given one through links",
    ),
    sp(
        Sessions,
        "path",
        "a file path the session read, wrote or edited",
    ),
    sp(Sessions, "tool", "the session made such a tool call"),
    sp(Sessions, "action", "the session made such a tool call"),
    f(Sessions, "started", Time, "first own message or event"),
    f(
        Sessions,
        "first_seen",
        Time,
        "when Polylogue first had any of its messages",
    ),
    sp(
        Sessions,
        "evidence_until",
        "the evidence Polylogue had by the time (asked of messages, blocks, tool calls, events and notes)",
    ),
    f(Sessions, "last_active", Time, "last own message or event"),
    sp(Sessions, "since", "active on or after the time"),
    sp(
        Sessions,
        "until",
        "begun before the instant or through the logical day",
    ),
    f(Sessions, "messages", Number, "own messages"),
    f(Sessions, "human_prompts", Number, "own prose prompts"),
    f(Sessions, "human_commands", Number, "own typed commands"),
    f(Sessions, "words", Number, "words in own prose"),
    f(Sessions, "tool_calls", Number, "own tool calls"),
    f(
        Sessions,
        "errors",
        Number,
        "own tool calls with outcome error",
    ),
    f(Sessions, "input_tokens", Number, "summed over own messages"),
    f(
        Sessions,
        "output_tokens",
        Number,
        "summed over own messages",
    ),
    f(
        Sessions,
        "cache_read_tokens",
        Number,
        "summed over own messages",
    ),
    f(
        Sessions,
        "cache_write_tokens",
        Number,
        "summed over own messages",
    ),
    f(
        Sessions,
        "cost",
        Money,
        "own cost; unknown when nothing is priced",
    ),
    f(
        Sessions,
        "tree_cost",
        Money,
        "own cost plus every session below in the subagent tree",
    ),
    f(
        Sessions,
        "wall_span",
        Duration,
        "Elapsed time between first and last own recorded time",
    ),
    f(
        Sessions,
        "clustered_time",
        Duration,
        "Union of consecutive message gaps at most 5 minutes, extended across paired tools",
    ),
    f(
        Sessions,
        "tool_time",
        Duration,
        "Sum of strictly positive paired tool intervals; untimed calls counted separately",
    ),
    f(
        Sessions,
        "tool_time_union",
        Duration,
        "Union of strictly positive paired tool intervals",
    ),
    f(
        Sessions,
        "tool_untimed",
        Number,
        "Own tool calls whose durations are unknown; shown beside tool-time sums",
    ),
    f(
        Sessions,
        "ends_with",
        Match,
        "Kind of the last recorded own item",
    )
    .stored_unknown("unknown"),
    f(
        Sessions,
        "last_stop_reason",
        Match,
        "Last model stop reason as recorded",
    ),
    f(
        Sessions,
        "time_source",
        Match,
        "Messages, or events when there are no messages",
    )
    .stored_unknown("absent"),
    f(
        Sessions,
        "waiting_model",
        Duration,
        "Wall time waiting for the model",
    ),
    f(
        Sessions,
        "waiting_human",
        Duration,
        "Wall time waiting for the human",
    ),
    f(Sessions, "human_words", Number, "Own human words"),
    f(Sessions, "model_words", Number, "Own model words"),
    f(
        Sessions,
        "thinking_messages",
        Number,
        "Own thinking messages",
    ),
    f(Sessions, "compactions", Number, "Own compactions"),
    sp(Sessions, "similar", "semantic similarity (a ranking)"),
    sp(Sessions, "output", "words in tool output"),
    sp(Sessions, "thinking", "words in thinking"),
    f(Messages, "id", Prefix, "message ID"),
    v(Messages, "role", Role::NAMES, "who the source says spoke"),
    v(Messages, "written_by", WrittenBy::NAMES, "who wrote it").stored_unknown("unknown"),
    v(
        Messages,
        "prompt_kind",
        PromptKind::NAMES,
        "for user messages: prose, command, or text the harness or another model wrote",
    ),
    f(Messages, "model", Match, "the model that wrote it"),
    f(Messages, "at", Time, "provider time"),
    f(
        Messages,
        "first_seen",
        Time,
        "when Polylogue first had its bytes",
    ),
    sp(
        Messages,
        "evidence_until",
        "first seen by the time: the evidence Polylogue had then",
    ),
    f(
        Messages,
        "duplicate",
        Bool,
        "a turn another input of the session also holds, shown from that input",
    ),
    f(Messages, "active", Bool, "on the active path"),
    sv(
        Messages,
        "has",
        &MESSAGE_HAS_NAMES,
        "the message has thinking, tool calls, attachments or code",
    ),
    f(Messages, "words", Number, "words in its text"),
    f(Messages, "input_tokens", Number, ""),
    f(Messages, "output_tokens", Number, ""),
    f(Messages, "cache_read_tokens", Number, ""),
    f(Messages, "cache_write_tokens", Number, ""),
    f(
        Messages,
        "cost",
        Money,
        "per message; unknown when unpriced",
    ),
    f(Messages, "stop_reason", Match, ""),
    sp(Messages, "since", "at or after the time"),
    sp(
        Messages,
        "until",
        "before the instant or through the logical day",
    ),
    sp(Messages, "similar", "semantic similarity (a ranking)"),
    sp(Messages, "output", "words in tool output"),
    sp(Messages, "thinking", "words in thinking"),
    v(Blocks, "kind", BlockKind::NAMES, "the block kind"),
    f(Blocks, "media_type", Match, ""),
    f(Blocks, "file_name", Substring, ""),
    f(Blocks, "at", Time, "the message's time"),
    f(
        Blocks,
        "first_seen",
        Time,
        "when Polylogue first had its message's bytes",
    ),
    sp(
        Blocks,
        "evidence_until",
        "its message first seen by the time",
    ),
    sp(Blocks, "since", "at or after the time"),
    sp(
        Blocks,
        "until",
        "before the instant or through the logical day",
    ),
    f(ToolCalls, "tool", Match, "the tool name as written"),
    sp(
        ToolCalls,
        "mcp_server",
        "for MCP tools, the server part of the name",
    ),
    v(
        ToolCalls,
        "action",
        polylogue_model::ToolAction::NAMES,
        "shell, file_read, file_write, file_edit, search, web, …",
    ),
    sp(ToolCalls, "command", "the shell command (substring)"),
    f(
        ToolCalls,
        "command_norm",
        Match,
        "the normalized command: cargo test, git commit",
    ),
    sp(ToolCalls, "input", "text inside the arguments (substring)"),
    sp(ToolCalls, "output", "words in the result"),
    sp(
        ToolCalls,
        "path",
        "a file path in the arguments (substring)",
    ),
    v(
        ToolCalls,
        "outcome",
        CallOutcome::NAMES,
        "ok, error, pending or unknown",
    )
    .stored_unknown("unknown"),
    f(ToolCalls, "exit_code", Number, ""),
    f(ToolCalls, "at", Time, ""),
    f(
        ToolCalls,
        "result_at",
        Time,
        "native time of the message or notification reporting the result",
    ),
    f(
        ToolCalls,
        "reported_duration",
        Duration,
        "provider-reported tool duration in milliseconds; unknown when absent",
    ),
    f(
        ToolCalls,
        "first_seen",
        Time,
        "when Polylogue first had the bytes of the message that made the call",
    ),
    sp(
        ToolCalls,
        "evidence_until",
        "its call first seen by the time",
    ),
    sp(
        ToolCalls,
        "since",
        "ended at or after the time, or still open",
    ),
    sp(
        ToolCalls,
        "until",
        "started before the instant or through the logical day",
    ),
    sp(ToolCalls, "child", "for subagent calls: the child session"),
    sp(ToolCalls, "next", "the session's next tool call"),
    f(Files, "path", Substring, "the file path"),
    f(Files, "reads", Number, ""),
    f(Files, "writes", Number, ""),
    f(Files, "edits", Number, ""),
    f(Files, "first_at", Time, ""),
    f(Files, "last_at", Time, ""),
    f(Events, "id", Prefix, "the observation ID or its prefix"),
    sp(
        Events,
        "text",
        "a substring in one textual observation value",
    ),
    f(
        Events,
        "input",
        Match,
        "the input this observation came from",
    ),
    f(Events, "source", Match, "the input's source parser"),
    f(
        Events,
        "origin",
        Match,
        "conversation origin, or source format when unbound",
    ),
    f(Events, "data", Substring, "the observation payload").json(),
    f(Events, "source_chain", Number, "the source revision chain"),
    f(Events, "source_start", Number, "record byte start"),
    f(Events, "source_end", Number, "record byte end"),
    f(
        Events,
        "kind",
        Match,
        "compaction, model_switch, permission, …",
    ),
    f(Events, "at", Time, ""),
    f(
        Events,
        "first_seen",
        Time,
        "when Polylogue first had its bytes",
    ),
    sp(Events, "evidence_until", "first seen by the time"),
    sp(Events, "since", "at or after the time"),
    sp(
        Events,
        "until",
        "before the instant or through the logical day",
    ),
    v(
        Links,
        "kind",
        LINK_KIND_NAMES,
        "the link's family: subagent or continuation",
    ),
    v(Links, "how", LINK_HOW_NAMES, "how the child relates"),
    sp(Links, "parent", "the parent session"),
    sp(Links, "child", "the child session"),
    f(Links, "pending", Bool, "the parent is not in the archive"),
    f(
        Links,
        "evidence",
        Match,
        "the input field that established it",
    ),
    f(
        Links,
        "branch_point",
        Prefix,
        "the message where the child branched off",
    ),
    f(Notes, "id", Prefix, "note revision ID or a unique prefix"),
    f(Notes, "kind", Match, "tag, note, title, decision, claim, …"),
    f(Notes, "key", Match, ""),
    f(Notes, "value", Substring, ""),
    f(
        Notes,
        "author",
        Match,
        "operator, a session ID, agent or script",
    ),
    f(Notes, "agent", Bool, "an agent wrote it"),
    f(
        Notes,
        "judged",
        Match,
        "the operator's judgment of this revision: accepted or rejected",
    ),
    sp(
        Notes,
        "target",
        "a reference it is about: session:ID, repo:NAME, cwd:PATH, …",
    ),
    f(
        Notes,
        "write_counter",
        Number,
        "the archive's write counter when it was written",
    ),
    f(Notes, "created_at", Time, ""),
    f(
        Notes,
        "first_note",
        Prefix,
        "the first note of its revisions",
    ),
    f(Notes, "current", Bool, "no later revision replaces it"),
    f(Notes, "withdrawn", Bool, "a withdrawal names it"),
    sp(Notes, "since", "created at or after the time"),
    sp(
        Notes,
        "evidence_until",
        "the notes stored by the time, each family at its revision then",
    ),
    sp(
        Notes,
        "until",
        "created before the instant or through the logical day",
    ),
    f(Deliveries, "id", Prefix, ""),
    sp(Deliveries, "recipient", "the receiving session"),
    v(
        Deliveries,
        "event",
        DeliveryEvent::NAMES,
        "what the delivery answered",
    ),
    v(
        Deliveries,
        "sender",
        DeliverySender::NAMES,
        "what prepared it; signal notices have sender signal",
    ),
    v(
        Deliveries,
        "outcome",
        DELIVERY_STATES,
        "what last happened to it; prepared until a delivery event says more",
    ),
    f(
        Deliveries,
        "detail",
        Substring,
        "the latest event's detail: a failure's reason, or what used it",
    ),
    f(Deliveries, "text", Substring, "the delivered text"),
    sp(Deliveries, "refs", "a reference the delivery included"),
    sp(Deliveries, "left_out", "a reference left out for budget"),
    f(Deliveries, "at", Time, ""),
    f(Deliveries, "bytes", Number, ""),
    v(
        Changes,
        "kind",
        &["session", "removed", "note"],
        "what changed",
    ),
    f(Changes, "ref", Prefix, "the session or note ID"),
    f(
        Changes,
        "seq",
        Number,
        "the write counter of the change: index.db's for sessions, archive.db's for notes",
    ),
    f(Changes, "note_kind", Match, "a note's kind"),
    sp(
        Changes,
        "after",
        "changes after a cursor `GEN:COUNTER:ARCHIVE`: the `view` of an earlier answer",
    ),
    sp(
        Changes,
        "sessions",
        "selection array, available as output only",
    )
    .ids(),
    sp(
        Changes,
        "messages",
        "selection array, available as output only",
    )
    .ids(),
    f(Turns, "id", Prefix, "its first prompt's message ID"),
    f(Turns, "start", Time, "the first message's time"),
    f(
        Turns,
        "end",
        Time,
        "the latest time of any message in the turn",
    ),
    f(Turns, "burst", Number, "how many prompts it has"),
    f(
        Turns,
        "prompts",
        Substring,
        "the human prompt messages of the turn, in order (`prompts:ID` holds that message)",
    ),
    f(Turns, "slash_command", Bool, "starts with a typed command"),
    f(
        Turns,
        "answer",
        Prefix,
        "the model's final message in the turn",
    ),
    f(
        Turns,
        "attempts",
        Number,
        "replies under the turn's last prompt, abandoned branches included",
    ),
    v(
        Turns,
        "path",
        &["active", "unknown"],
        "unknown when the session's active path is unknown",
    )
    .stored_unknown("unknown"),
    f(
        Turns,
        "latency",
        Duration,
        "from the last prompt to the first model message",
    ),
    f(Turns, "tool_calls", Number, ""),
    f(Turns, "errors", Number, "tool calls with outcome error"),
    f(Turns, "input_tokens", Number, ""),
    f(Turns, "output_tokens", Number, ""),
    f(Turns, "cost", Money, ""),
    f(
        Turns,
        "abandoned",
        Bool,
        "the turn ended the session with no answer",
    ),
    f(
        Turns,
        "corrected",
        Bool,
        "a current correction note targets the turn's answer",
    ),
    v(
        Work,
        "kind",
        WorkNodeKind::NAMES,
        "goal: a claimed Bead or a declared todo item; attempt: one session pursuing it; \
         verification: a check command; artifact: a commit or pull request",
    ),
    f(
        Work,
        "title",
        Substring,
        "the Bead ID, item text, check command or commit",
    ),
    f(
        Work,
        "key",
        Prefix,
        "bead:<id>, todo:…, attempt:…, check:…, commit:…, pr:…",
    ),
    f(
        Work,
        "state",
        Match,
        "attempt: open, closed, released, completed, pending; verification: ok, error, unknown",
    ),
    f(
        Work,
        "at",
        Time,
        "when the node's first evidence was recorded",
    ),
    f(
        Work,
        "goal",
        Substring,
        "the goal an attempt pursues, or the goal of the attempts a node checks or was made for",
    ),
    f(
        Work,
        "checks_passed",
        Number,
        "attempt: passing checks while it was open",
    ),
    f(
        Work,
        "checks_failed",
        Number,
        "attempt: failing checks while it was open",
    ),
    f(
        Work,
        "checks_unknown",
        Number,
        "attempt: checks without a known outcome while it was open",
    ),
    f(
        Work,
        "artifacts",
        Number,
        "attempt: commits and pull requests made while it was open",
    ),
];
pub fn lookup(unit: Unit, name: &str) -> Option<&'static Field> {
    FIELDS
        .iter()
        .find(|f| f.unit == unit && f.name.eq_ignore_ascii_case(name))
}

pub fn names(unit: Unit) -> impl Iterator<Item = &'static str> {
    FIELDS
        .iter()
        .filter(move |f| f.unit == unit)
        .map(|f| f.name)
}

/// Units with a row that names a session, so `session.<field>` (and a bare
/// session field) applies to them.
pub fn has_session(unit: Unit) -> bool {
    crate::units::spec(unit).in_session
}

/// The prefixes a unit accepts before a field, with the unit they name.
pub fn relation(unit: Unit, prefix: &str) -> Option<Unit> {
    crate::units::spec(unit).relation(prefix).map(|r| r.target)
}

/// Evidence and unknown rules for fields backed by declared session products.
pub fn construct(field: &Field) -> Option<polylogue_model::derived::Construct> {
    use polylogue_model::derived;
    if field.unit != Unit::Sessions {
        return None;
    }
    if polylogue_model::derived::COUNT_FIELDS
        .iter()
        .any(|(name, _)| *name == field.name)
    {
        return Some(derived::SESSION_COUNTS);
    }
    match field.name {
        "wall_span" | "clustered_time" | "tool_time" | "tool_time_union" | "tool_untimed"
        | "ends_with" | "last_stop_reason" | "time_source" => Some(derived::SESSION_SPAN),
        "waiting_model" | "waiting_human" => Some(derived::SESSION_TIME),
        _ => None,
    }
}
