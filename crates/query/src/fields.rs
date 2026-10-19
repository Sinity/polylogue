//! The field table, as data: every field each unit accepts, its type, its
//! closed set of values when it has one, and the SQL it reads. Completion,
//! help and error suggestions read [`FIELDS`]; [`sql`] gives the expression
//! for a field over the unit's table aliases. Nothing keeps a parallel list.
//!
//! Aliases: `s` session, `m` message, `b` block, `t` tool call (`cb` its
//! call block), `e` event, `l` link, `n` note, `f` file row, `r` run, `p`
//! process, `d` delivery. Nested field paths (`session.repo`, `child.cost`,
//! `next.action`) are resolved by [`crate::compile`] with other aliases.

use crate::ast::Unit;

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
    /// Compiled by its own rule in [`crate::compile`].
    Special,
}

pub struct Field {
    pub unit: Unit,
    pub name: &'static str,
    pub kind: Kind,
    /// The closed set of values, when there is one.
    pub values: Option<&'static [&'static str]>,
    pub help: &'static str,
}

const fn f(unit: Unit, name: &'static str, kind: Kind, help: &'static str) -> Field {
    Field {
        unit,
        name,
        kind,
        values: None,
        help,
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
        help,
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
        unit,
        name,
        kind: Kind::Special,
        values: Some(values),
        help,
    }
}

use Kind::*;
use Unit::*;

pub const SESSION_KINDS: &[&str] = &["main", "subagent", "sidechain", "chat"];
pub const SESSION_HAS: &[&str] = &[
    "thinking",
    "tool_calls",
    "errors",
    "attachments",
    "compaction",
    "branches",
    "note",
    "conflict",
    "secret",
];
pub const TOOL_OUTCOMES: &[&str] = &["ok", "error", "pending", "unknown"];
pub const SESSION_OUTCOMES: &[&str] = &["solved", "partial", "abandoned", "none"];
pub const ACTIONS: &[&str] = &[
    "shell",
    "code",
    "file_read",
    "file_write",
    "file_edit",
    "search",
    "web",
    "subagent",
    "mcp",
    "other",
];
pub const LINK_KINDS: &[&str] = &["subagent", "continuation", "reference"];

pub static FIELDS: &[Field] = &[
    // sessions
    f(Sessions, "id", Prefix, "session ID or a unique prefix"),
    f(Sessions, "origin", Match, "claude-code, codex, chatgpt, …"),
    v(
        Sessions,
        "kind",
        SESSION_KINDS,
        "main, subagent, sidechain or chat",
    ),
    f(Sessions, "title", Substring, "the displayed title"),
    f(
        Sessions,
        "repo",
        Special,
        "the repository: name, owner/name or URL",
    ),
    f(Sessions, "checkout", Subtree, "the worktree path"),
    f(
        Sessions,
        "collection",
        Special,
        "the provider's grouping: name or ID",
    ),
    f(Sessions, "branch", Match, "the git branch, as recorded"),
    f(
        Sessions,
        "commit",
        Special,
        "a commit recorded or linked (prefix, 7+ hex digits)",
    ),
    f(
        Sessions,
        "pr",
        Special,
        "a pull request linked to the session",
    ),
    v(
        Sessions,
        "outcome",
        SESSION_OUTCOMES,
        "the session's outcome note",
    ),
    f(Sessions, "hidden", Bool, "has a hidden note"),
    f(
        Sessions,
        "cwd",
        Special,
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
        "the live state now, from hook and process records and the time",
    ),
    f(Sessions, "waiting", Special, "the live state is waiting"),
    f(
        Sessions,
        "running",
        Special,
        "the live state is working, waiting, idle or unknown",
    ),
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
        "process_alive",
        Bool,
        "whether the harness process is alive",
    ),
    f(
        Sessions,
        "harness",
        Match,
        "claude-code, codex-tui, codex-exec, …",
    ),
    f(
        Sessions,
        "model",
        Special,
        "a model that wrote at least one message",
    ),
    f(
        Sessions,
        "tag",
        Special,
        "a tag note; tag:key:value for key:value tags",
    ),
    sv(
        Sessions,
        "has",
        SESSION_HAS,
        "thinking, tool_calls, errors, attachments, compaction, …",
    ),
    f(Sessions, "temporary", Bool, "temporary web chats"),
    f(
        Sessions,
        "parent",
        Special,
        "sessions linked from the given session",
    ),
    f(
        Sessions,
        "root",
        Special,
        "the subagent tree under the given session",
    ),
    f(Sessions, "depth", Special, "depth in the subagent tree"),
    f(
        Sessions,
        "link",
        Special,
        "the kind of an incoming link; link:fork, link:resume",
    ),
    f(
        Sessions,
        "continues",
        Special,
        "sessions continuing the given session",
    ),
    f(
        Sessions,
        "continued_by",
        Special,
        "sessions the given session continues",
    ),
    f(
        Sessions,
        "lineage",
        Special,
        "every session connected through links",
    ),
    f(Sessions, "team", Match, "the agent team"),
    f(
        Sessions,
        "agent_name",
        Match,
        "the member's name in an agent team",
    ),
    f(
        Sessions,
        "path",
        Special,
        "a file path the session read, wrote or edited",
    ),
    f(
        Sessions,
        "tool",
        Special,
        "the session made such a tool call",
    ),
    f(
        Sessions,
        "action",
        Special,
        "the session made such a tool call",
    ),
    f(Sessions, "started", Time, "first own message or event"),
    f(Sessions, "last_active", Time, "last own message or event"),
    f(Sessions, "since", Special, "active on or after the time"),
    f(Sessions, "until", Special, "begun on or before the time"),
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
    f(Sessions, "duration", Duration, "last_active - started"),
    f(Sessions, "near", Special, "semantic similarity (a ranking)"),
    f(Sessions, "output", Special, "words in tool output"),
    f(Sessions, "thinking", Special, "words in thinking"),
    // messages
    f(Messages, "id", Prefix, "message ID"),
    v(
        Messages,
        "role",
        &["user", "assistant", "system", "tool"],
        "user, assistant, system or tool",
    ),
    v(
        Messages,
        "written_by",
        &["human", "model", "harness", "unknown"],
        "who wrote it",
    ),
    v(
        Messages,
        "prompt_kind",
        &["prose", "command", "harness"],
        "for human messages: prose or command",
    ),
    v(
        Messages,
        "authorship",
        &["authored", "pasted", "mixed", "unknown"],
        "for human messages",
    ),
    f(Messages, "model", Match, "the model that wrote it"),
    f(Messages, "at", Time, "provider time"),
    f(
        Messages,
        "inherited",
        Bool,
        "a copy of an earlier session's message",
    ),
    f(Messages, "active", Bool, "on the active path"),
    sv(
        Messages,
        "has",
        &["thinking", "tool_calls", "attachments", "code"],
        "thinking, tool_calls, attachments, code",
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
    f(
        Messages,
        "from_session",
        Prefix,
        "the sending session of an agent-to-agent message",
    ),
    f(Messages, "from_name", Match, "the sender's name as written"),
    f(Messages, "since", Special, "at or after the time"),
    f(Messages, "until", Special, "at or before the time"),
    f(Messages, "near", Special, "semantic similarity (a ranking)"),
    f(Messages, "output", Special, "words in tool output"),
    f(Messages, "thinking", Special, "words in thinking"),
    // blocks
    v(
        Blocks,
        "kind",
        &[
            "text",
            "thinking",
            "tool_call",
            "tool_result",
            "image",
            "file",
            "other",
        ],
        "the block kind",
    ),
    f(Blocks, "media_type", Match, ""),
    f(Blocks, "file_name", Substring, ""),
    f(Blocks, "at", Time, "the message's time"),
    f(Blocks, "since", Special, "at or after the time"),
    f(Blocks, "until", Special, "at or before the time"),
    // tool_calls
    f(ToolCalls, "tool", Match, "the tool name as written"),
    f(
        ToolCalls,
        "mcp_server",
        Special,
        "for MCP tools, the server part of the name",
    ),
    v(
        ToolCalls,
        "action",
        ACTIONS,
        "shell, file_read, file_write, file_edit, search, web, …",
    ),
    f(
        ToolCalls,
        "command",
        Special,
        "the shell command (substring)",
    ),
    f(
        ToolCalls,
        "command_norm",
        Match,
        "the normalized command: cargo test, git commit",
    ),
    f(
        ToolCalls,
        "input",
        Special,
        "text inside the arguments (substring)",
    ),
    f(ToolCalls, "output", Special, "words in the result"),
    f(
        ToolCalls,
        "path",
        Special,
        "a file path in the arguments (substring)",
    ),
    v(
        ToolCalls,
        "outcome",
        TOOL_OUTCOMES,
        "ok, error, pending or unknown",
    ),
    f(ToolCalls, "exit_code", Number, ""),
    f(ToolCalls, "at", Time, ""),
    f(ToolCalls, "duration", Duration, ""),
    f(
        ToolCalls,
        "since",
        Special,
        "ended at or after the time, or still open",
    ),
    f(ToolCalls, "until", Special, "started at or before the time"),
    f(
        ToolCalls,
        "child",
        Special,
        "for subagent calls: the child session",
    ),
    f(ToolCalls, "next", Special, "the session's next tool call"),
    // files
    f(Files, "path", Substring, "the file path"),
    f(Files, "reads", Number, ""),
    f(Files, "writes", Number, ""),
    f(Files, "edits", Number, ""),
    f(Files, "first_at", Time, ""),
    f(Files, "last_at", Time, ""),
    // events
    f(
        Events,
        "kind",
        Match,
        "compaction, model_switch, permission, …",
    ),
    f(Events, "at", Time, ""),
    f(
        Events,
        "model_saw",
        Bool,
        "the content was part of the model's context",
    ),
    f(Events, "since", Special, "at or after the time"),
    f(Events, "until", Special, "at or before the time"),
    // links
    v(
        Links,
        "kind",
        LINK_KINDS,
        "subagent, continuation or reference",
    ),
    v(
        Links,
        "how",
        &[
            "fresh",
            "copied",
            "summarized",
            "resume",
            "fork",
            "after_compaction",
            "briefing",
            "ping",
            "mention",
        ],
        "how the child relates",
    ),
    f(Links, "parent", Special, "the parent session"),
    f(Links, "child", Special, "the child session"),
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
    f(Links, "spawned_by", Prefix, "the spawning tool call"),
    f(
        Links,
        "retry_of",
        Prefix,
        "the earlier spawning call this one retries",
    ),
    v(
        Links,
        "result_use",
        &["returned", "referenced", "unknown"],
        "how the child's result reached the parent",
    ),
    // notes
    f(Notes, "kind", Match, "tag, note, title, decision, claim, …"),
    f(Notes, "key", Match, ""),
    f(Notes, "value", Substring, ""),
    f(
        Notes,
        "author",
        Match,
        "operator, a session ID, agent or script",
    ),
    f(Notes, "target", Special, "a reference"),
    f(
        Notes,
        "source",
        Prefix,
        "the message an inline marker came from",
    ),
    f(Notes, "created_at", Time, ""),
    f(Notes, "root", Prefix, "the revision family"),
    f(Notes, "current", Bool, "no later revision replaces it"),
    f(
        Notes,
        "withdrawn",
        Bool,
        "a current withdrawal withdraws it",
    ),
    f(Notes, "since", Special, "created at or after the time"),
    f(Notes, "until", Special, "created at or before the time"),
    // runs
    f(Runs, "id", Prefix, ""),
    v(Runs, "kind", &["task", "attempt"], ""),
    f(Runs, "task", Prefix, "an attempt's task"),
    v(
        Runs,
        "state",
        &["running", "completed", "failed", "cancelled", "unknown"],
        "",
    ),
    f(Runs, "instruction", Substring, ""),
    f(Runs, "attempts", Number, "for a task: how many attempts"),
    f(Runs, "started", Time, ""),
    f(Runs, "ended", Time, ""),
    f(Runs, "duration", Duration, ""),
    f(Runs, "evidence", Match, ""),
    // deliveries
    f(Deliveries, "id", Prefix, ""),
    f(Deliveries, "recipient", Special, "the receiving session"),
    v(
        Deliveries,
        "event",
        &[
            "session_start",
            "resume",
            "compaction",
            "prompt_reference",
            "mcp_brief",
            "ping",
            "advisory",
        ],
        "",
    ),
    v(
        Deliveries,
        "outcome",
        &["delivered", "failed", "unknown", "not_sent"],
        "",
    ),
    f(
        Deliveries,
        "refs",
        Special,
        "a reference the delivery included",
    ),
    f(
        Deliveries,
        "left_out",
        Special,
        "a reference left out for budget",
    ),
    f(Deliveries, "at", Time, ""),
    f(Deliveries, "bytes", Number, ""),
    // processes
    v(Processes, "role", &["harness", "child"], ""),
    f(Processes, "comm", Match, "the command name"),
    f(
        Processes,
        "argv",
        Substring,
        "the argument list, as observed",
    ),
    f(Processes, "alive", Bool, "no end seen yet"),
    f(Processes, "started_at", Time, ""),
    f(Processes, "ended_at", Time, ""),
    v(
        Work,
        "kind",
        &["goal", "attempt", "verification", "artifact"],
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
    !matches!(
        unit,
        Unit::Sessions | Unit::Links | Unit::Notes | Unit::Runs | Unit::Processes
    )
}

/// The prefixes a unit accepts before a field, with the unit they name.
pub fn relation(unit: Unit, prefix: &str) -> Option<Unit> {
    Some(match (unit, prefix) {
        (u, "session") if has_session(u) || matches!(u, Unit::Runs | Unit::Processes) => {
            Unit::Sessions
        }
        (Unit::Blocks | Unit::ToolCalls, "message") => Unit::Messages,
        (Unit::Blocks, "call") => Unit::ToolCalls,
        (Unit::ToolCalls, "child") => Unit::Sessions,
        (Unit::ToolCalls, "next") => Unit::ToolCalls,
        (Unit::Links, "parent" | "child") => Unit::Sessions,
        (Unit::Deliveries, "recipient") => Unit::Sessions,
        (Unit::Runs, "task") => Unit::Runs,
        (Unit::Processes, "tool_call") => Unit::ToolCalls,
        _ => return None,
    })
}

/// SQL over the aliases a unit's row is read through.
#[derive(Debug, Clone)]
pub struct Aliases {
    pub s: String,
    pub m: String,
    pub b: String,
    pub t: String,
    pub cb: String,
    pub row: String,
}

impl Aliases {
    pub fn plain() -> Self {
        Self {
            s: "s".into(),
            m: "m".into(),
            b: "b".into(),
            t: "t".into(),
            cb: "cb".into(),
            row: "r".into(),
        }
    }

    pub fn suffixed(n: usize) -> Self {
        Self {
            s: format!("s{n}"),
            m: format!("m{n}"),
            b: format!("b{n}"),
            t: format!("t{n}"),
            cb: format!("cb{n}"),
            row: format!("r{n}"),
        }
    }
}

/// Own messages of session `s`: not inherited, on the active path.
pub fn own_messages(s: &str, x: &str) -> String {
    format!(
        "{x}.session_pk = {s}.pk AND {x}.inherited_from IS NULL AND COALESCE({x}.on_active_path, 1) = 1"
    )
}

/// A Codex code-mode wrapper whose children are counted in its place
/// (spec query.md, Defaults; sources/codex.md, code mode). Children are the
/// `exec_command` calls the wrapper ran; a wrapper with none counts itself.
pub fn counted_call(t: &str) -> String {
    format!(
        "NOT ({t}.action = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = {t}.session_pk AND cw.tool = 'exec_command'))"
    )
}

fn own_calls(s: &str, x: &str, extra: &str) -> String {
    format!(
        "(SELECT count(*) FROM tool_call {x} JOIN block {x}b ON {x}b.pk = {x}.call_block_pk \
         JOIN message {x}m ON {x}m.pk = {x}b.message_pk \
         WHERE {x}.session_pk = {s}.pk AND {x}m.inherited_from IS NULL \
         AND COALESCE({x}m.on_active_path, 1) = 1 AND {}{extra})",
        counted_call(x)
    )
}

/// The title of the goal a work node belongs to: an attempt's goal, or the
/// goal of the attempt a verification checks or an artifact was made for.
pub fn work_goal(r: &str) -> String {
    format!(
        "COALESCE(\
         (SELECT g.title FROM work_edge e JOIN work_node g ON g.pk = e.to_pk \
          WHERE e.from_pk = {r}.pk AND e.kind = 'pursues'), \
         (SELECT g.title FROM work_edge e JOIN work_edge p ON p.from_pk = e.to_pk AND p.kind = 'pursues' \
          JOIN work_node g ON g.pk = p.to_pk WHERE e.from_pk = {r}.pk AND e.kind = 'checks' LIMIT 1), \
         (SELECT g.title FROM work_edge e JOIN work_edge p ON p.from_pk = e.from_pk AND p.kind = 'pursues' \
          JOIN work_node g ON g.pk = p.to_pk WHERE e.to_pk = {r}.pk AND e.kind = 'produces' LIMIT 1), \
         CASE WHEN {r}.kind = 'goal' THEN {r}.title END)"
    )
}

/// The live states (spec model.md, Live state).
pub const LIVE_STATES: &[&str] = &["working", "waiting", "idle", "unknown", "stale", "ended"];

/// The live window: hook evidence this recent makes a session live.
pub const LIVE_WINDOW_MS: i64 = 15 * 60 * 1000;
/// Evidence older than this, with nothing newer, means the session ended.
pub const LIVE_DAY_MS: i64 = 24 * 60 * 60 * 1000;

/// The live state of session alias `s` at `now` (spec model.md, Live
/// state): computed when asked from the stored live facts and the time,
/// never written by a clock. Rules apply in order and the result is total.
/// A fact dated after `now` counts as at `now`.
pub fn live_state(s: &str, now: i64) -> String {
    let w = LIVE_WINDOW_MS;
    let d = LIVE_DAY_MS;
    let ev = format!("MIN({s}.live_last_event_at, {now})");
    let grew = format!("MIN({s}.last_activity_at, {now})");
    let live = format!(
        "({s}.live_root_alive = 1 OR ({s}.live_root_alive IS NULL \
         AND {s}.live_last_event_at IS NOT NULL AND {now} - {ev} <= {w}))"
    );
    format!(
        "(CASE \
         WHEN {s}.live_last_event = 'hook_session_end' \
              OR ({s}.live_root_alive = 0 AND {s}.live_last_event = 'hook_stop') THEN 'ended' \
         WHEN {s}.live_root_alive = 0 THEN 'stale' \
         WHEN {live} AND {s}.live_waiting_on IS NOT NULL THEN 'waiting' \
         WHEN {live} AND {s}.live_last_event IN ('hook_stop', 'hook_notification') THEN 'idle' \
         WHEN {live} THEN 'working' \
         WHEN {s}.live_last_event IS NULL AND {s}.live_root_alive IS NULL \
              AND {s}.last_activity_at IS NOT NULL AND {now} - {grew} <= {w} THEN 'unknown' \
         WHEN COALESCE({s}.live_last_event_at, {s}.last_activity_at) IS NOT NULL \
              AND {now} - MAX(COALESCE({ev}, 0), COALESCE({grew}, 0)) <= {d} THEN 'stale' \
         ELSE 'ended' END)"
    )
}

/// When `s`'s live state changes with no new fact: the end of the live
/// window or of the 24 hours after the last evidence; NULL when only a new
/// fact can change it.
pub fn live_until(s: &str, now: i64) -> String {
    let w = LIVE_WINDOW_MS;
    let d = LIVE_DAY_MS;
    let ev = format!("MIN({s}.live_last_event_at, {now})");
    let grew = format!("MIN({s}.last_activity_at, {now})");
    let last = format!("MAX(COALESCE({ev}, 0), COALESCE({grew}, 0))");
    format!(
        "(CASE {state} \
         WHEN 'ended' THEN NULL \
         WHEN 'stale' THEN CASE WHEN {s}.live_root_alive = 0 THEN NULL ELSE {last} + {d} END \
         WHEN 'unknown' THEN {grew} + {w} \
         ELSE CASE WHEN {s}.live_root_alive = 1 THEN NULL ELSE {ev} + {w} END END)",
        state = live_state(s, now)
    )
}

/// The expression for a plain field (every kind but `Special`).
pub fn sql(unit: Unit, name: &str, a: &Aliases) -> Option<String> {
    let s = &a.s;
    let m = &a.m;
    let b = &a.b;
    let t = &a.t;
    let r = &a.row;
    let own_sum = |col: &str| {
        format!(
            "(SELECT sum(x.{col}) FROM message x WHERE {})",
            own_messages(s, "x")
        )
    };
    Some(match (unit, name) {
        (Unit::Sessions, "id") => format!("{s}.id"),
        (Unit::Sessions, "origin") => format!("{s}.origin"),
        (Unit::Sessions, "kind") => format!("{s}.kind"),
        (Unit::Sessions, "title") => format!("{s}.title"),
        (Unit::Sessions, "checkout") => format!("{s}.checkout"),
        (Unit::Sessions, "branch") => format!("{s}.branch"),
        (Unit::Sessions, "hidden") => hidden(s),
        (Unit::Sessions, "source") => {
            format!("CASE WHEN {s}.stub = 1 THEN 'missing' ELSE 'present' END")
        }
        (Unit::Sessions, "process_alive") => format!("{s}.live_root_alive"),
        (Unit::Work, "kind") => format!("{r}.kind"),
        (Unit::Work, "title") => format!("{r}.title"),
        (Unit::Work, "key") => format!("{r}.key"),
        (Unit::Work, "state") => format!("{r}.state"),
        (Unit::Work, "at") => format!("{r}.at"),
        (Unit::Work, "goal") => work_goal(r),
        (Unit::Work, "checks_passed") => format!("json_extract({r}.attrs_json, '$.checks_passed')"),
        (Unit::Work, "checks_failed") => format!("json_extract({r}.attrs_json, '$.checks_failed')"),
        (Unit::Work, "artifacts") => format!("json_extract({r}.attrs_json, '$.artifacts')"),
        (Unit::Sessions, "waiting_on") => format!("{s}.live_waiting_on"),
        (Unit::Sessions, "task") => format!("{s}.live_task"),
        (Unit::Sessions, "harness") => format!("{s}.harness"),
        (Unit::Sessions, "temporary") => format!("{s}.temporary"),
        (Unit::Sessions, "team") => format!("{s}.team"),
        (Unit::Sessions, "agent_name") => format!("{s}.agent_name"),
        (Unit::Sessions, "started") => format!("{s}.started_at"),
        (Unit::Sessions, "last_active") => format!("{s}.last_activity_at"),
        (Unit::Sessions, "duration") => format!("({s}.last_activity_at - {s}.started_at)"),
        (Unit::Sessions, "outcome") => format!(
            "COALESCE((SELECT lower(n.value) FROM a.note n JOIN a.note_target nt ON nt.note_id = n.id \
             WHERE n.kind = 'outcome' AND nt.ref = 'session:' || {s}.id AND {} ORDER BY n.local_seq DESC LIMIT 1), 'none')",
            live_note("n")
        ),
        (Unit::Sessions, "messages") => format!(
            "(SELECT count(*) FROM message x WHERE {})",
            own_messages(s, "x")
        ),
        (Unit::Sessions, "human_prompts") => format!(
            "(SELECT count(*) FROM message x WHERE {} AND x.written_by = 'human' AND x.prompt_kind = 'prose')",
            own_messages(s, "x")
        ),
        (Unit::Sessions, "human_commands") => format!(
            "(SELECT count(*) FROM message x WHERE {} AND x.written_by = 'human' AND x.prompt_kind = 'command')",
            own_messages(s, "x")
        ),
        (Unit::Sessions, "words") => format!(
            "(SELECT COALESCE(sum(x.words), 0) FROM message x WHERE {} AND x.written_by IN ('human', 'model'))",
            own_messages(s, "x")
        ),
        (Unit::Sessions, "tool_calls") => own_calls(s, "tx", ""),
        (Unit::Sessions, "errors") => own_calls(s, "tx", " AND tx.outcome = 'error'"),
        (Unit::Sessions, "input_tokens") => own_sum("input_tokens"),
        (Unit::Sessions, "output_tokens") => own_sum("output_tokens"),
        (Unit::Sessions, "cache_read_tokens") => own_sum("cache_read_tokens"),
        (Unit::Sessions, "cache_write_tokens") => own_sum("cache_write_tokens"),
        (Unit::Sessions, "cost") => own_sum("cost_nusd"),
        (Unit::Sessions, "tree_cost") => format!(
            "(SELECT sum(x.cost_nusd) FROM message x WHERE x.inherited_from IS NULL \
             AND COALESCE(x.on_active_path, 1) = 1 AND x.session_pk IN ({}))",
            subtree(s)
        ),

        (Unit::Messages, "id") => format!("{m}.id"),
        (Unit::Messages, "role") => format!("{m}.role"),
        (Unit::Messages, "written_by") => format!("{m}.written_by"),
        (Unit::Messages, "prompt_kind") => format!("{m}.prompt_kind"),
        (Unit::Messages, "authorship") => format!("COALESCE({m}.authorship, 'unknown')"),
        (Unit::Messages, "model") => format!("{m}.model"),
        (Unit::Messages, "at") => format!("{m}.at"),
        (Unit::Messages, "inherited") => format!("({m}.inherited_from IS NOT NULL)"),
        (Unit::Messages, "active") => format!("COALESCE({m}.on_active_path, 1)"),
        (Unit::Messages, "words") => format!("{m}.words"),
        (Unit::Messages, "input_tokens") => format!("{m}.input_tokens"),
        (Unit::Messages, "output_tokens") => format!("{m}.output_tokens"),
        (Unit::Messages, "cache_read_tokens") => format!("{m}.cache_read_tokens"),
        (Unit::Messages, "cache_write_tokens") => format!("{m}.cache_write_tokens"),
        (Unit::Messages, "cost") => format!("{m}.cost_nusd"),
        (Unit::Messages, "stop_reason") => format!("{m}.stop_reason"),
        (Unit::Messages, "from_session") => format!("{m}.from_session"),
        (Unit::Messages, "from_name") => format!("{m}.from_name"),

        (Unit::Blocks, "kind") => format!("{b}.kind"),
        (Unit::Blocks, "media_type") => format!("{b}.media_type"),
        (Unit::Blocks, "file_name") => format!("{b}.file_name"),
        (Unit::Blocks, "at") => format!("{m}.at"),

        (Unit::ToolCalls, "tool") => format!("{t}.tool"),
        (Unit::ToolCalls, "action") => format!("{t}.action"),
        (Unit::ToolCalls, "command_norm") => format!("{t}.command_norm"),
        (Unit::ToolCalls, "outcome") => format!(
            "CASE WHEN {t}.result_block_pk IS NULL AND {t}.outcome = 'unknown' AND {s}.live_root_alive = 1 \
             THEN 'pending' ELSE {t}.outcome END"
        ),
        (Unit::ToolCalls, "exit_code") => format!("{t}.exit_code"),
        (Unit::ToolCalls, "at") => format!("{t}.at"),
        (Unit::ToolCalls, "duration") => format!("{t}.duration_ms"),

        (Unit::Files, "path") => format!("{r}.path"),
        (Unit::Files, "reads") => format!("{r}.reads"),
        (Unit::Files, "writes") => format!("{r}.writes"),
        (Unit::Files, "edits") => format!("{r}.edits"),
        (Unit::Files, "first_at") => format!("{r}.first_at"),
        (Unit::Files, "last_at") => format!("{r}.last_at"),

        (Unit::Events, "kind") => format!("{r}.kind"),
        (Unit::Events, "at") => format!("{r}.at"),
        (Unit::Events, "model_saw") => format!("{r}.model_saw"),

        (Unit::Links, "kind") => format!("{r}.kind"),
        (Unit::Links, "how") => format!("{r}.how"),
        (Unit::Links, "pending") => {
            format!("(NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = {r}.parent_id))")
        }
        (Unit::Links, "evidence") => format!("{r}.evidence"),
        (Unit::Links, "branch_point") => format!("{r}.branch_point"),
        (Unit::Links, "spawned_by") => format!("{r}.spawned_by"),
        (Unit::Links, "retry_of") => format!("{r}.retry_of"),
        (Unit::Links, "result_use") => format!("COALESCE({r}.result_use, 'unknown')"),

        (Unit::Notes, "kind") => format!("{r}.kind"),
        (Unit::Notes, "key") => format!("{r}.key"),
        (Unit::Notes, "value") => format!("{r}.value"),
        (Unit::Notes, "author") => format!("{r}.author"),
        (Unit::Notes, "source") => format!("{r}.source"),
        (Unit::Notes, "created_at") => format!("{r}.created_at"),
        (Unit::Notes, "root") => format!("{r}.root"),
        (Unit::Notes, "current") => format!("({})", current_note(r)),
        (Unit::Notes, "withdrawn") => format!("({})", withdrawn_note(r)),

        (Unit::Runs, "id") => format!("{r}.id"),
        (Unit::Runs, "kind") => format!("{r}.kind"),
        (Unit::Runs, "task") => format!("{r}.task_id"),
        (Unit::Runs, "state") => format!("{r}.state"),
        (Unit::Runs, "instruction") => format!("{r}.instruction"),
        (Unit::Runs, "attempts") => {
            format!("(SELECT count(*) FROM run ra WHERE ra.task_id = {r}.id)")
        }
        (Unit::Runs, "started") => format!("{r}.started_at"),
        (Unit::Runs, "ended") => format!("{r}.ended_at"),
        (Unit::Runs, "duration") => format!("({r}.ended_at - {r}.started_at)"),
        (Unit::Runs, "evidence") => format!("{r}.evidence"),

        (Unit::Deliveries, "id") => format!("{r}.id"),
        (Unit::Deliveries, "event") => format!("{r}.event"),
        (Unit::Deliveries, "outcome") => format!("{r}.outcome"),
        (Unit::Deliveries, "at") => format!("{r}.at"),
        (Unit::Deliveries, "bytes") => format!("{r}.bytes"),

        (Unit::Processes, "role") => format!("{r}.role"),
        (Unit::Processes, "comm") => format!("{r}.comm"),
        (Unit::Processes, "argv") => format!("{r}.argv_json"),
        (Unit::Processes, "alive") => format!("({r}.ended_at IS NULL)"),
        (Unit::Processes, "started_at") => format!("{r}.started_at"),
        (Unit::Processes, "ended_at") => format!("{r}.ended_at"),
        _ => return None,
    })
}

/// A note that is its family's current revision and not withdrawn.
pub fn live_note(n: &str) -> String {
    format!("{} AND NOT {}", current_note(n), withdrawn_note(n))
}

pub fn current_note(n: &str) -> String {
    format!(
        "NOT EXISTS (SELECT 1 FROM a.note nr WHERE nr.root = {n}.root AND nr.local_seq > {n}.local_seq)"
    )
}

pub fn withdrawn_note(n: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM a.note nw JOIN a.note_target nwt ON nwt.note_id = nw.id \
         WHERE nw.kind = 'withdrawal' AND nwt.ref = 'note:' || {n}.root \
         AND NOT EXISTS (SELECT 1 FROM a.note nw2 WHERE nw2.root = nw.root AND nw2.local_seq > nw.local_seq))"
    )
}

/// A current `hidden` note targets the session.
pub fn hidden(s: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
         WHERE hn.kind = 'hidden' AND ht.ref = 'session:' || {s}.id AND {})",
        live_note("hn")
    )
}

/// An excision row names the session.
pub fn excised(s: &str) -> String {
    format!("EXISTS (SELECT 1 FROM a.excision ex WHERE ex.scope = 'session:' || {s}.id)")
}

/// The session and every session below it in its subagent tree.
pub fn subtree(s: &str) -> String {
    format!(
        "WITH RECURSIVE tr(pk) AS (SELECT {s}.pk UNION SELECT l.child_pk FROM link l \
         JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_non_special_field_has_sql() {
        let a = Aliases::plain();
        for f in FIELDS {
            if f.kind != Kind::Special {
                assert!(sql(f.unit, f.name, &a).is_some(), "{:?}.{}", f.unit, f.name);
            }
        }
    }

    #[test]
    fn names_are_unique_per_unit() {
        let mut seen = std::collections::HashSet::new();
        for f in FIELDS {
            assert!(seen.insert((f.unit, f.name)), "{:?}.{}", f.unit, f.name);
        }
    }
}
