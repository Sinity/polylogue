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
use crate::compile::custom_fields::{self as custom, Custom};
use crate::compile::{ColKind, ColumnReference};
use polylogue_model::{
    BlockKind, DocumentAuthorship, DocumentPurpose, LINK_HOW_NAMES, LINK_KIND_NAMES, PromptKind,
    Role, SessionKind, WorkNodeKind,
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
    /// Compiled by its own rule ([`Field::compile`]).
    Special,
}

pub struct Field {
    pub unit: Unit,
    pub name: &'static str,
    pub kind: Kind,
    /// The closed set of values, when there is one.
    pub values: Option<&'static [&'static str]>,
    pub help: &'static str,
    /// The SQL of the field's value. A `Special` field matches by its own
    /// rule but may still have a value to group, sort and show by.
    pub sql: Option<Expr>,
    /// The column kind of the value when the matching kind does not say.
    pub value_kind: Option<ColKind>,
    /// A special field's compile rule; every special field has one.
    pub(crate) compile: Option<Custom>,
}

/// SQL over a unit's aliases: a template whose `{s}`, `{m}`, `{b}`, `{t}`,
/// `{cb}` and `{r}` name them, or a function for what a template cannot
/// say; `Clock` also takes the query's instant.
#[derive(Clone, Copy)]
pub enum Expr {
    Template(&'static str),
    Reference(&'static str, ColumnReference),
    Fn(fn(&Aliases) -> String),
    Clock(fn(&Aliases, i64) -> String),
}

impl Expr {
    pub fn reference(&self) -> Option<ColumnReference> {
        match self {
            Self::Reference(_, kind) => Some(*kind),
            _ => None,
        }
    }

    pub fn render(&self, a: &Aliases, now: i64) -> String {
        match self {
            Expr::Template(t) | Expr::Reference(t, _) => a.expand(t),
            Expr::Fn(f) => f(a),
            Expr::Clock(f) => f(a, now),
        }
    }
}

impl Field {
    const fn sql(mut self, template: &'static str) -> Self {
        self.sql = Some(Expr::Template(template));
        self
    }

    const fn sql_ref(mut self, template: &'static str, reference: ColumnReference) -> Self {
        self.sql = Some(Expr::Reference(template, reference));
        self
    }

    const fn sql_fn(mut self, f: fn(&Aliases) -> String) -> Self {
        self.sql = Some(Expr::Fn(f));
        self
    }

    const fn sql_clock(mut self, f: fn(&Aliases, i64) -> String) -> Self {
        self.sql = Some(Expr::Clock(f));
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

    /// The field's value over `a` at `now`.
    pub fn expr(&self, a: &Aliases, now: i64) -> Option<String> {
        self.sql.map(|e| e.render(a, now))
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
        help,
        sql: None,
        value_kind: None,
        compile: None,
    }
}

/// A special field: matched by its own compile rule.
const fn sp(unit: Unit, name: &'static str, help: &'static str, compile: Custom) -> Field {
    Field {
        unit,
        name,
        kind: Kind::Special,
        values: None,
        help,
        sql: None,
        value_kind: None,
        compile: Some(compile),
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
        sql: None,
        value_kind: None,
        compile: None,
    }
}

/// A field with a closed set of values and its own compile rule.
const fn sv(
    unit: Unit,
    name: &'static str,
    values: &'static [&'static str],
    help: &'static str,
    compile: Custom,
) -> Field {
    Field {
        values: Some(values),
        ..sp(unit, name, help, compile)
    }
}

use Kind::*;
use Unit::*;

/// One value of a `has:` field and the condition it stands for, over the
/// row's alias (`s` for sessions, `m` for messages).
pub struct Has {
    pub name: &'static str,
    pub sql: fn(&str) -> String,
}

const fn has_names<const N: usize>(table: &[Has; N]) -> [&'static str; N] {
    let mut out = [""; N];
    let mut i = 0;
    while i < N {
        out[i] = table[i].name;
        i += 1;
    }
    out
}

pub const SESSION_HAS: [Has; 9] = [
    Has {
        name: "thinking",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM block xb JOIN current_message xm ON xm.pk = xb.message_pk WHERE {} AND xb.kind = 'thinking')",
                own_messages(s, "xm")
            )
        },
    },
    Has {
        name: "tool_calls",
        sql: |s| format!("EXISTS (SELECT 1 FROM tool_call xt WHERE xt.session_pk = {s}.pk)"),
    },
    Has {
        name: "errors",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM tool_call xt WHERE xt.session_pk = {s}.pk AND xt.outcome = 'error')"
            )
        },
    },
    Has {
        name: "attachments",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.session_pk = {s}.pk AND xb.kind IN ('image', 'file'))"
            )
        },
    },
    Has {
        name: "compaction",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM current_event xe WHERE xe.session_pk = {s}.pk AND xe.kind = 'compaction')"
            )
        },
    },
    Has {
        name: "branches",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM current_message xm WHERE xm.session_pk = {s}.pk AND xm.on_active_path = 0)"
            )
        },
    },
    Has {
        name: "note",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
                 WHERE (ht.ref = 'session:' || {s}.id OR ht.ref IN (SELECT 'message:' || xm.id FROM current_message xm WHERE xm.session_pk = {s}.pk)) \
                 AND {})",
                live_note("hn")
            )
        },
    },
    Has {
        name: "conflict",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
                 WHERE ht.ref = 'session:' || {s}.id AND {} \
                 GROUP BY hn.root HAVING count(*) > 1)",
                current_note("hn")
            )
        },
    },
    Has {
        name: "secret",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.session_pk = {s}.pk AND xb.looks_like_secret = 1)"
            )
        },
    },
];
static SESSION_HAS_NAMES: [&str; 9] = has_names(&SESSION_HAS);

pub const MESSAGE_HAS: [Has; 4] = [
    Has {
        name: "thinking",
        sql: |m| {
            format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind = 'thinking')"
            )
        },
    },
    Has {
        name: "tool_calls",
        sql: |m| {
            format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind = 'tool_call')"
            )
        },
    },
    Has {
        name: "attachments",
        sql: |m| {
            format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind IN ('image', 'file'))"
            )
        },
    },
    Has {
        name: "code",
        sql: |m| {
            format!(
                "EXISTS (SELECT 1 FROM block xb JOIN text tx ON tx.id = xb.text_id WHERE xb.message_pk = {m}.pk \
                 AND xb.kind = 'text' AND instr(pl_text(tx.body), char(96, 96, 96)) > 0)"
            )
        },
    },
];
static MESSAGE_HAS_NAMES: [&str; 4] = has_names(&MESSAGE_HAS);

pub const TOOL_OUTCOMES: &[&str] = &["ok", "error", "pending", "unknown"];
pub const SESSION_OUTCOMES: &[&str] = &["solved", "partial", "abandoned", "none"];

pub static FIELDS: &[Field] = &[
    // documents have no conversation role, session or usage defaults.
    f(Documents,"id",Prefix,"document ID or a unique prefix").sql_ref("{r}.id", ColumnReference::Document),
    f(Documents,"origin",Match,"the provider export family").sql("{r}.origin"),
    v(Documents,"purpose",DocumentPurpose::NAMES,"instruction, context or memory").sql("{r}.purpose"),
    v(Documents,"authorship",DocumentAuthorship::NAMES,"who wrote the text, unknown without source evidence").sql("{r}.authorship"),
    f(Documents,"title",Substring,"the supplied source title").sql("{r}.title"),
    f(Documents,"source_time",Time,"the provider's source-record time").sql("{r}.source_time"),
    f(Documents,"versions",Number,"the number of retained content versions").sql("(SELECT count(DISTINCT ds.version_hash) FROM document_source ds WHERE ds.document_pk={r}.pk)"),
    // sessions
    f(Sessions, "id", Prefix, "session ID or a unique prefix").sql_ref("{s}.id", ColumnReference::Session),
    f(Sessions, "origin", Match, "claude-code, codex, chatgpt, …").sql("{s}.origin"),
    v(
        Sessions,
        "kind",
        SessionKind::NAMES,
        "who started the session: main, or one another session spawned",
    ).sql("{s}.kind"),
    f(Sessions, "title", Substring, "the displayed title").sql("COALESCE({s}.title, {s}.derived_title)"),
    sp(Sessions, "repo", "the repository: name, owner/name or URL", custom::repo).sql("{s}.repo"),
    f(Sessions, "checkout", Subtree, "the worktree path").sql("{s}.checkout"),
    sp(Sessions, "collection", "the provider's grouping: name or ID", custom::collection).sql("COALESCE({s}.collection_name, {s}.collection_id)"),
    f(Sessions, "branch", Match, "the git branch, as recorded").sql("{s}.branch"),
    sp(Sessions, "commit", "a commit recorded or linked (prefix, 7+ hex digits)", custom::commit),
    sp(Sessions, "pr", "a pull request linked to the session", custom::pr),
    v(
        Sessions,
        "outcome",
        SESSION_OUTCOMES,
        "the session's outcome note",
    ).sql_fn(sessions_outcome),
    f(Sessions, "hidden", Bool, "has a hidden note").sql_fn(sessions_hidden),
    sp(Sessions, "cwd", "working directory, and the directories below it", custom::cwd).sql("{s}.cwd"),
    v(
        Sessions,
        "source",
        &["present", "missing"],
        "whether a transcript was stored",
    ).sql("CASE WHEN {s}.stub = 1 THEN 'missing' ELSE 'present' END"),
    sv(Sessions, "live", LIVE_STATES, "the live state now, from hook and process records and the time", custom::live).sql_clock(sessions_live),
    sp(Sessions, "waiting", "the live state is waiting", custom::waiting).sql_clock(sessions_waiting).boolean(),
    sp(Sessions, "running", "the live state is working, waiting, idle or unknown", custom::running).sql_clock(sessions_running).boolean(),
    v(
        Sessions,
        "waiting_on",
        &["permission", "question", "notification"],
        "what a waiting session waits for",
    ).sql("{s}.live_waiting_on"),
    f(
        Sessions,
        "task",
        Match,
        "the live task: a job ID or Bead ID",
    ).sql("{s}.live_task"),
    f(
        Sessions,
        "process_alive",
        Bool,
        "whether the harness process is alive",
    ).sql("{s}.live_root_alive"),
    f(
        Sessions,
        "live_until",
        Time,
        "when the live state changes with no new record; empty when only a new record can change it",
    )
    .sql_clock(sessions_live_until),
    f(Sessions, "last_event", Match, "the latest live record: a hook event").sql("{s}.live_last_event"),
    f(Sessions, "last_event_at", Time, "when the latest live record happened").sql("{s}.live_last_event_at"),
    f(Sessions, "waiting_since", Time, "when a waiting session started waiting").sql("{s}.live_waiting_since"),
    f(Sessions, "todo", Substring, "the live todo list, as recorded").sql("{s}.live_todo"),
    f(
        Sessions,
        "live_rank",
        Number,
        "0 waiting, 1 working, 2 idle, 3 unknown, 4 stale, 5 ended: the order the live view lists",
    )
    .sql_clock(sessions_live_rank),
    f(
        Sessions,
        "peers",
        Substring,
        "other working, waiting or idle sessions sharing its repository, checkout or working directory",
    )
    .sql_clock(sessions_peers),
    f(Sessions, "repo_root", Prefix, "the repository's identity: its root commit hash, when found").sql("{s}.repo_root"),
    f(
        Sessions,
        "harness",
        Match,
        "claude-code, codex-tui, codex-exec, …",
    ).sql("{s}.harness"),
    sp(Sessions, "model", "a model that wrote at least one message", custom::model).sql(
        "(SELECT x.model FROM current_message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL GROUP BY x.model ORDER BY count(*) DESC LIMIT 1)",
    ),
    sp(Sessions, "tag", "a tag note; tag:key:value for key:value tags", custom::tag),
    sv(Sessions, "has", &SESSION_HAS_NAMES, "the session has thinking, tool calls, errors, …", custom::has),
    f(Sessions, "temporary", Bool, "temporary web chats").sql("{s}.temporary"),
    sp(Sessions, "parent", "sessions linked from the given session", custom::parent),
    sp(Sessions, "root", "the subagent tree under the given session", custom::root),
    sp(Sessions, "depth", "depth in the subagent tree", custom::depth),
    sp(Sessions, "link", "the kind of an incoming link; link:fork, link:resume", custom::link),
    sp(Sessions, "continues", "sessions continuing the given session", custom::continues),
    sp(Sessions, "continued_by", "sessions the given session continues", custom::continued_by),
    sp(Sessions, "lineage", "every session connected through links", custom::lineage),
    f(Sessions, "team", Match, "the agent team").sql("{s}.team"),
    f(
        Sessions,
        "agent_name",
        Match,
        "the member's name in an agent team",
    ).sql("{s}.agent_name"),
    sp(Sessions, "path", "a file path the session read, wrote or edited", custom::session_tool_call),
    sp(Sessions, "tool", "the session made such a tool call", custom::session_tool_call),
    sp(Sessions, "action", "the session made such a tool call", custom::session_tool_call),
    f(Sessions, "started", Time, "first own message or event").sql("{s}.started_at"),
    f(Sessions, "last_active", Time, "last own message or event").sql("{s}.last_activity_at"),
    sp(Sessions, "since", "active on or after the time", custom::since),
    sp(Sessions, "until", "begun on or before the time", custom::until),
    f(Sessions, "messages", Number, "own messages").sql_fn(sessions_messages),
    f(Sessions, "human_prompts", Number, "own prose prompts").sql_fn(sessions_human_prompts),
    f(Sessions, "human_commands", Number, "own typed commands").sql_fn(sessions_human_commands),
    f(Sessions, "words", Number, "words in own prose").sql_fn(sessions_words),
    f(Sessions, "tool_calls", Number, "own tool calls").sql_fn(sessions_tool_calls),
    f(
        Sessions,
        "errors",
        Number,
        "own tool calls with outcome error",
    ).sql_fn(sessions_errors),
    f(Sessions, "input_tokens", Number, "summed over own messages").sql_fn(sessions_input_tokens),
    f(
        Sessions,
        "output_tokens",
        Number,
        "summed over own messages",
    ).sql_fn(sessions_output_tokens),
    f(
        Sessions,
        "cache_read_tokens",
        Number,
        "summed over own messages",
    ).sql_fn(sessions_cache_read_tokens),
    f(
        Sessions,
        "cache_write_tokens",
        Number,
        "summed over own messages",
    ).sql_fn(sessions_cache_write_tokens),
    f(
        Sessions,
        "cost",
        Money,
        "own cost; unknown when nothing is priced",
    ).sql_fn(sessions_cost),
    f(
        Sessions,
        "tree_cost",
        Money,
        "own cost plus every session below in the subagent tree",
    ).sql_fn(sessions_tree_cost),
    f(Sessions, "duration", Duration, "last_active - started").sql("({s}.last_activity_at - {s}.started_at)"),
    sp(Sessions, "near", "semantic similarity (a ranking)", custom::near),
    sp(Sessions, "output", "words in tool output", custom::output),
    sp(Sessions, "thinking", "words in thinking", custom::thinking),
    // messages
    f(Messages, "id", Prefix, "message ID").sql_ref("{m}.id", ColumnReference::Message),
    v(
        Messages,
        "role",
        Role::NAMES,
        "who the source says spoke",
    ).sql("{m}.role"),
    v(
        Messages,
        "written_by",
        &["human", "model", "harness", "unknown"],
        "who wrote it",
    ).sql("{m}.written_by"),
    v(
        Messages,
        "prompt_kind",
        PromptKind::NAMES,
        "for user messages: prose, command, or text the harness or another model wrote",
    ).sql("{m}.prompt_kind"),
    v(
        Messages,
        "authorship",
        &["authored", "pasted", "mixed", "unknown"],
        "for human messages",
    ).sql("COALESCE({m}.authorship, 'unknown')"),
    f(Messages, "model", Match, "the model that wrote it").sql("{m}.model"),
    f(Messages, "at", Time, "provider time").sql("{m}.at"),
    f(
        Messages,
        "inherited",
        Bool,
        "a copy of an earlier session's message",
    ).sql("({m}.inherited_from IS NOT NULL)"),
    f(Messages, "active", Bool, "on the active path").sql("COALESCE({m}.on_active_path, 1)"),
    sv(Messages, "has", &MESSAGE_HAS_NAMES, "the message has thinking, tool calls, attachments or code", custom::has),
    f(Messages, "words", Number, "words in its text").sql("{m}.words"),
    f(Messages, "input_tokens", Number, "").sql("{m}.input_tokens"),
    f(Messages, "output_tokens", Number, "").sql("{m}.output_tokens"),
    f(Messages, "cache_read_tokens", Number, "").sql("{m}.cache_read_tokens"),
    f(Messages, "cache_write_tokens", Number, "").sql("{m}.cache_write_tokens"),
    f(
        Messages,
        "cost",
        Money,
        "per message; unknown when unpriced",
    ).sql("{m}.cost_nusd"),
    f(Messages, "stop_reason", Match, "").sql("{m}.stop_reason"),
    f(
        Messages,
        "from_session",
        Prefix,
        "the sending session of an agent-to-agent message",
    ).sql("{m}.from_session"),
    f(Messages, "from_name", Match, "the sender's name as written").sql("{m}.from_name"),
    sp(Messages, "since", "at or after the time", custom::since),
    sp(Messages, "until", "at or before the time", custom::until),
    sp(Messages, "near", "semantic similarity (a ranking)", custom::near),
    sp(Messages, "output", "words in tool output", custom::output),
    sp(Messages, "thinking", "words in thinking", custom::thinking),
    // blocks
    v(
        Blocks,
        "kind",
        BlockKind::NAMES,
        "the block kind",
    ).sql("{b}.kind"),
    f(Blocks, "media_type", Match, "").sql("{b}.media_type"),
    f(Blocks, "file_name", Substring, "").sql("{b}.file_name"),
    f(Blocks, "at", Time, "the message's time").sql("{m}.at"),
    sp(Blocks, "since", "at or after the time", custom::since),
    sp(Blocks, "until", "at or before the time", custom::until),
    // tool_calls
    f(ToolCalls, "tool", Match, "the tool name as written").sql("{t}.tool"),
    sp(ToolCalls, "mcp_server", "for MCP tools, the server part of the name", custom::mcp_server).sql(
        "CASE WHEN {t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' THEN substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1) END",
    ),
    v(
        ToolCalls,
        "action",
        polylogue_model::ToolAction::NAMES,
        "shell, file_read, file_write, file_edit, search, web, …",
    ).sql("{t}.action"),
    sp(ToolCalls, "command", "the shell command (substring)", custom::tool_call_text).sql("{t}.command"),
    f(
        ToolCalls,
        "command_norm",
        Match,
        "the normalized command: cargo test, git commit",
    ).sql("{t}.command_norm"),
    sp(ToolCalls, "input", "text inside the arguments (substring)", custom::input),
    sp(ToolCalls, "output", "words in the result", custom::output),
    sp(ToolCalls, "path", "a file path in the arguments (substring)", custom::tool_call_text).sql("{t}.path"),
    v(
        ToolCalls,
        "outcome",
        TOOL_OUTCOMES,
        "ok, error, pending or unknown",
    ).sql(
        "CASE WHEN {t}.result_block_pk IS NULL AND {t}.outcome = 'unknown' AND {s}.live_root_alive = 1 \
             THEN 'pending' ELSE {t}.outcome END",
    ),
    f(ToolCalls, "exit_code", Number, "").sql("{t}.exit_code"),
    f(ToolCalls, "at", Time, "").sql("{t}.at"),
    f(ToolCalls, "duration", Duration, "").sql("{t}.duration_ms"),
    sp(ToolCalls, "since", "ended at or after the time, or still open", custom::since),
    sp(ToolCalls, "until", "started at or before the time", custom::until),
    sp(ToolCalls, "child", "for subagent calls: the child session", custom::spawned_child),
    sp(ToolCalls, "next", "the session's next tool call", custom::next_call),
    // files
    f(Files, "path", Substring, "the file path").sql("{r}.path"),
    f(Files, "reads", Number, "").sql("{r}.reads"),
    f(Files, "writes", Number, "").sql("{r}.writes"),
    f(Files, "edits", Number, "").sql("{r}.edits"),
    f(Files, "first_at", Time, "").sql("{r}.first_at"),
    f(Files, "last_at", Time, "").sql("{r}.last_at"),
    // events
    sp(Events, "text", "a substring in one textual observation value", custom::event_text),
    f(Events, "input", Match, "the retained input owning this observation").sql("{r}i.id"),
    f(Events, "source", Match, "the input's source parser").sql("{r}i.format"),
    f(Events, "origin", Match, "conversation origin, or source format when unbound").sql("COALESCE({s}.origin,{r}i.format)"),
    f(Events, "data", Substring, "the retained observation payload").sql("pl_block({r}.data)").json(),
    f(Events, "source_chain", Number, "the retained source revision").sql("{r}.source_chain"),
    f(Events, "source_start", Number, "record byte start").sql("{r}.source_start"),
    f(Events, "source_end", Number, "record byte end").sql("{r}.source_end"),
    f(
        Events,
        "kind",
        Match,
        "compaction, model_switch, permission, …",
    ).sql("{r}.kind"),
    f(Events, "at", Time, "").sql("{r}.at"),
    sp(Events, "since", "at or after the time", custom::since),
    sp(Events, "until", "at or before the time", custom::until),
    // links
    v(
        Links,
        "kind",
        LINK_KIND_NAMES,
        "the link's family: subagent or continuation",
    ).sql("{r}.kind"),
    v(
        Links,
        "how",
        LINK_HOW_NAMES,
        "how the child relates",
    ).sql("{r}.how"),
    sp(Links, "parent", "the parent session", custom::link_parent).sql_ref("{r}.parent_id", ColumnReference::Session),
    sp(Links, "child", "the child session", custom::link_child).sql_ref("{r}c.id", ColumnReference::Session),
    f(Links, "pending", Bool, "the parent is not in the archive").sql("(NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = {r}.parent_id))"),
    f(
        Links,
        "evidence",
        Match,
        "the input field that established it",
    ).sql("{r}.evidence"),
    f(
        Links,
        "branch_point",
        Prefix,
        "the message where the child branched off",
    ).sql("{r}.branch_point"),
    f(Links, "spawned_by", Prefix, "the spawning tool call").sql("{r}.spawned_by"),
    f(
        Links,
        "retry_of",
        Prefix,
        "the earlier spawning call this one retries",
    ).sql("{r}.retry_of"),
    v(
        Links,
        "result_use",
        &["returned", "referenced", "unknown"],
        "how the child's result reached the parent",
    ).sql("COALESCE({r}.result_use, 'unknown')"),
    // notes
    f(Notes, "kind", Match, "tag, note, title, decision, claim, …").sql("{r}.kind"),
    f(Notes, "key", Match, "").sql("{r}.key"),
    f(Notes, "value", Substring, "").sql("{r}.value"),
    f(
        Notes,
        "author",
        Match,
        "operator, a session ID, agent or script",
    ).sql("{r}.author"),
    sp(Notes, "target", "a reference", custom::note_target),
    f(
        Notes,
        "source",
        Prefix,
        "the message an inline marker came from",
    ).sql("{r}.source"),
    f(Notes, "created_at", Time, "").sql("{r}.created_at"),
    f(Notes, "root", Prefix, "the revision family").sql("{r}.root"),
    f(Notes, "current", Bool, "no later revision replaces it").sql_fn(notes_current),
    f(
        Notes,
        "withdrawn",
        Bool,
        "a current withdrawal withdraws it",
    ).sql_fn(notes_withdrawn),
    sp(Notes, "since", "created at or after the time", custom::since),
    sp(Notes, "until", "created at or before the time", custom::until),
    // runs
    f(Runs, "id", Prefix, "").sql("{r}.id"),
    v(Runs, "kind", &["task", "attempt"], "").sql("{r}.kind"),
    f(Runs, "task", Prefix, "an attempt's task").sql("{r}.task_id"),
    v(
        Runs,
        "state",
        &["running", "completed", "failed", "cancelled", "unknown"],
        "",
    ).sql("{r}.state"),
    f(Runs, "instruction", Substring, "").sql("{r}.instruction"),
    f(Runs, "attempts", Number, "for a task: how many attempts").sql("(SELECT count(*) FROM run ra WHERE ra.task_id = {r}.id)"),
    f(Runs, "started", Time, "").sql("{r}.started_at"),
    f(Runs, "ended", Time, "").sql("{r}.ended_at"),
    f(Runs, "duration", Duration, "").sql("({r}.ended_at - {r}.started_at)"),
    f(Runs, "evidence", Match, "").sql("{r}.evidence"),
    // deliveries
    f(Deliveries, "id", Prefix, "").sql("{r}.id"),
    sp(Deliveries, "recipient", "the receiving session", custom::recipient).sql("{r}.recipient_id"),
    v(
        Deliveries,
        "event",
        &[
            "session_start",
            "resume",
            "compaction",
            "takeover",
            "recall",
            "advisory",
            "ping",
            "signal",
        ],
        "what the delivery answered",
    ).sql("{r}.event"),
    v(
        Deliveries,
        "sender",
        &["hook", "mcp", "cli", "ping", "signal"],
        "what prepared it; signal notices have sender signal",
    ).sql("{r}.sender"),
    v(
        Deliveries,
        "outcome",
        &["prepared", "delivered", "failed", "not_sent", "used"],
        "what last happened to it; prepared until a delivery event says more",
    ).sql_fn(deliveries_outcome),
    f(
        Deliveries,
        "detail",
        Substring,
        "the latest event's detail: a failure's reason, or what used it",
    ).sql_fn(deliveries_detail),
    f(Deliveries, "text", Substring, "the delivered text").sql("pl_text({r}.body)"),
    sp(Deliveries, "refs", "a reference the delivery included", custom::refs),
    sp(Deliveries, "left_out", "a reference left out for budget", custom::left_out),
    f(Deliveries, "at", Time, "").sql("{r}.at"),
    f(Deliveries, "bytes", Number, "").sql("{r}.bytes"),
    // changes: sessions written/removed, notes and permanent removal rules
    v(
        Changes,
        "kind",
        &["session", "removed", "note", "excision"],
        "what changed",
    )
    .sql("{r}.kind"),
    f(Changes, "ref", Prefix, "the session, note or excision ID").sql("{r}.ref"),
    f(
        Changes,
        "seq",
        Number,
        "the write counter of the change: index.db's for sessions, archive.db's for notes and excisions",
    )
    .sql("{r}.seq"),
    f(Changes, "note_kind", Match, "a note's kind").sql("{r}.note_kind"),
    f(Changes, "scope", Prefix, "an excision's session/input scope or text hash").sql("{r}.scope"),
    sp(Changes, "sessions", "an excision's retained session selection; null when unknown", custom::selection).sql("{r}.sessions").ids(),
    sp(Changes, "messages", "an excision's retained message selection", custom::selection).sql("{r}.messages").ids(),
    sp(Changes, "after", "changes after a cursor `GEN:COUNTER:ARCHIVE`: the `view` of an earlier answer", custom::changes_after),
    // turns (spec query.md, turns): computed along the active path
    f(Turns, "id", Prefix, "its first prompt's message ID").sql("{r}.id"),
    f(Turns, "start", Time, "the first message's time").sql("{r}.start"),
    f(
        Turns,
        "end",
        Time,
        "the latest time of any message in the turn",
    ).sql("{r}.end"),
    f(Turns, "burst", Number, "how many prompts it has").sql("{r}.burst"),
    f(
        Turns,
        "prompts",
        Substring,
        "the human prompt messages of the turn, in order (`prompts:ID` holds that message)",
    ).sql("{r}.prompts"),
    f(Turns, "slash_command", Bool, "starts with a typed command").sql("{r}.slash_command"),
    f(
        Turns,
        "answer",
        Prefix,
        "the model's final message in the turn",
    ).sql("{r}.answer"),
    f(
        Turns,
        "attempts",
        Number,
        "replies under the turn's last prompt, abandoned branches included",
    ).sql("{r}.attempts"),
    v(
        Turns,
        "path",
        &["active", "unknown"],
        "unknown when the session's active path is unknown",
    ).sql("{r}.path"),
    f(
        Turns,
        "latency",
        Duration,
        "from the last prompt to the first model message",
    ).sql("{r}.latency"),
    f(Turns, "tool_calls", Number, "").sql("{r}.tool_calls"),
    f(Turns, "errors", Number, "tool calls with outcome error").sql("{r}.errors"),
    f(Turns, "input_tokens", Number, "").sql("{r}.input_tokens"),
    f(Turns, "output_tokens", Number, "").sql("{r}.output_tokens"),
    f(Turns, "cost", Money, "").sql("{r}.cost_nusd"),
    f(
        Turns,
        "abandoned",
        Bool,
        "the turn ended the session with no answer",
    ).sql("{r}.abandoned"),
    f(
        Turns,
        "corrected",
        Bool,
        "a current correction note targets the turn's answer",
    ).sql_fn(turns_corrected),
    // processes
    v(Processes, "role", &["harness", "child"], "").sql("{r}.role"),
    f(Processes, "comm", Match, "the command name").sql("{r}.comm"),
    f(
        Processes,
        "argv",
        Substring,
        "the argument list, as observed",
    ).sql("{r}.argv_json"),
    f(Processes, "alive", Bool, "no end seen yet").sql("({r}.ended_at IS NULL)"),
    f(Processes, "started_at", Time, "").sql("{r}.started_at"),
    f(Processes, "ended_at", Time, "").sql("{r}.ended_at"),
    v(
        Work,
        "kind",
        WorkNodeKind::NAMES,
        "goal: a claimed Bead or a declared todo item; attempt: one session pursuing it; \
         verification: a check command; artifact: a commit or pull request",
    ).sql("{r}.kind"),
    f(
        Work,
        "title",
        Substring,
        "the Bead ID, item text, check command or commit",
    ).sql("{r}.title"),
    f(
        Work,
        "key",
        Prefix,
        "bead:<id>, todo:…, attempt:…, check:…, commit:…, pr:…",
    ).sql("{r}.key"),
    f(
        Work,
        "state",
        Match,
        "attempt: open, closed, released, completed, pending; verification: ok, error, unknown",
    ).sql("{r}.state"),
    f(
        Work,
        "at",
        Time,
        "when the node's first evidence was recorded",
    ).sql("{r}.at"),
    f(
        Work,
        "goal",
        Substring,
        "the goal an attempt pursues, or the goal of the attempts a node checks or was made for",
    ).sql_fn(work_goal_field),
    f(
        Work,
        "checks_passed",
        Number,
        "attempt: passing checks while it was open",
    ).sql("json_extract({r}.attrs_json, '$.checks_passed')"),
    f(
        Work,
        "checks_failed",
        Number,
        "attempt: failing checks while it was open",
    ).sql("json_extract({r}.attrs_json, '$.checks_failed')"),
    f(
        Work,
        "artifacts",
        Number,
        "attempt: commits and pull requests made while it was open",
    ).sql("json_extract({r}.attrs_json, '$.artifacts')"),
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

    /// A template with its alias placeholders filled in.
    pub fn expand(&self, template: &str) -> String {
        template
            .replace("{s}", &self.s)
            .replace("{m}", &self.m)
            .replace("{cb}", &self.cb)
            .replace("{b}", &self.b)
            .replace("{t}", &self.t)
            .replace("{r}", &self.row)
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
         JOIN current_message {x}m ON {x}m.pk = {x}b.message_pk \
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

/// The value of a field over `a` at `now`.
pub fn sql(unit: Unit, name: &str, a: &Aliases, now: i64) -> Option<String> {
    lookup(unit, name).and_then(|f| f.expr(a, now))
}

/// The sum of a column over session `s`'s own messages.
fn own_sum(s: &str, col: &str) -> String {
    format!(
        "(SELECT sum(x.{col}) FROM current_message x WHERE {})",
        own_messages(s, "x")
    )
}

/// Session `s`'s own messages matching `extra`, counted.
fn own_count(s: &str, extra: &str) -> String {
    format!(
        "(SELECT count(*) FROM current_message x WHERE {}{extra})",
        own_messages(s, "x")
    )
}

fn sessions_hidden(a: &Aliases) -> String {
    hidden(&a.s)
}

fn sessions_outcome(a: &Aliases) -> String {
    format!(
        "COALESCE((SELECT lower(n.value) FROM a.note n JOIN a.note_target nt ON nt.note_id = n.id \
         WHERE n.kind = 'outcome' AND nt.ref = 'session:' || {s}.id AND {} ORDER BY n.commit_seq DESC LIMIT 1), 'none')",
        live_note("n"),
        s = a.s
    )
}

fn sessions_messages(a: &Aliases) -> String {
    own_count(&a.s, "")
}

fn sessions_human_prompts(a: &Aliases) -> String {
    own_count(
        &a.s,
        " AND x.written_by = 'human' AND x.prompt_kind = 'prose'",
    )
}

fn sessions_human_commands(a: &Aliases) -> String {
    own_count(
        &a.s,
        " AND x.written_by = 'human' AND x.prompt_kind = 'command'",
    )
}

fn sessions_words(a: &Aliases) -> String {
    format!(
        "(SELECT COALESCE(sum(x.words), 0) FROM current_message x WHERE {} AND x.written_by IN ('human', 'model'))",
        own_messages(&a.s, "x")
    )
}

/// The latest delivery event's `col` for delivery `r`.
fn latest_delivery_event(r: &str, col: &str) -> String {
    format!(
        "(SELECT e.{col} FROM a.delivery_event e WHERE e.delivery = {r}.id ORDER BY e.at DESC LIMIT 1)"
    )
}

pub(crate) fn deliveries_outcome(a: &Aliases) -> String {
    format!(
        "COALESCE({}, 'prepared')",
        latest_delivery_event(&a.row, "what")
    )
}

fn deliveries_detail(a: &Aliases) -> String {
    latest_delivery_event(&a.row, "detail")
}

fn turns_corrected(a: &Aliases) -> String {
    format!(
        "EXISTS (SELECT 1 FROM a.note nc JOIN a.note_target nct ON nct.note_id = nc.id \
         WHERE nc.kind = 'correction' AND nct.ref = 'message:' || {r}.answer \
         AND {} AND NOT {})",
        current_note("nc"),
        withdrawn_note("nc"),
        r = a.row
    )
}

fn sessions_tool_calls(a: &Aliases) -> String {
    own_calls(&a.s, "tx", "")
}

fn sessions_errors(a: &Aliases) -> String {
    own_calls(&a.s, "tx", " AND tx.outcome = 'error'")
}

fn sessions_input_tokens(a: &Aliases) -> String {
    own_sum(&a.s, "input_tokens")
}

fn sessions_output_tokens(a: &Aliases) -> String {
    own_sum(&a.s, "output_tokens")
}

fn sessions_cache_read_tokens(a: &Aliases) -> String {
    own_sum(&a.s, "cache_read_tokens")
}

fn sessions_cache_write_tokens(a: &Aliases) -> String {
    own_sum(&a.s, "cache_write_tokens")
}

fn sessions_cost(a: &Aliases) -> String {
    own_sum(&a.s, "cost_nusd")
}

fn sessions_tree_cost(a: &Aliases) -> String {
    format!(
        "(SELECT sum(x.cost_nusd) FROM current_message x WHERE x.inherited_from IS NULL \
         AND COALESCE(x.on_active_path, 1) = 1 AND x.session_pk IN ({}))",
        subtree(&a.s)
    )
}

fn sessions_live(a: &Aliases, now: i64) -> String {
    live_state(&a.s, now)
}

fn sessions_waiting(a: &Aliases, now: i64) -> String {
    format!("({} = 'waiting')", live_state(&a.s, now))
}

fn sessions_running(a: &Aliases, now: i64) -> String {
    format!(
        "({} IN ('working', 'waiting', 'idle', 'unknown'))",
        live_state(&a.s, now)
    )
}

fn sessions_live_until(a: &Aliases, now: i64) -> String {
    live_until(&a.s, now)
}

fn sessions_live_rank(a: &Aliases, now: i64) -> String {
    format!(
        "(CASE {} WHEN 'waiting' THEN 0 WHEN 'working' THEN 1 WHEN 'idle' THEN 2 \
         WHEN 'unknown' THEN 3 WHEN 'stale' THEN 4 ELSE 5 END)",
        live_state(&a.s, now)
    )
}

/// Peers (spec model.md): the other sessions working, waiting or idle that
/// share a repository or checkout, or, when either has no git evidence, a
/// working directory where one lies at or below the other, by whole path
/// components. Symmetric by construction (law 89).
fn sessions_peers(a: &Aliases, now: i64) -> String {
    let s = &a.s;
    let active = |x: &str| format!("{} IN ('working', 'waiting', 'idle')", live_state(x, now));
    let below = |outer: &str, inner: &str| {
        format!(
            "({inner}.cwd = rtrim({outer}.cwd, '/') OR substr({inner}.cwd, 1, length(rtrim({outer}.cwd, '/')) + 1) = rtrim({outer}.cwd, '/') || '/')"
        )
    };
    format!(
        "(SELECT group_concat(pe.id, ' ') FROM (SELECT pp.id FROM session pp WHERE pp.pk <> {s}.pk \
         AND {mine} AND {theirs} AND ((pp.repo_root IS NOT NULL AND pp.repo_root = {s}.repo_root) \
         OR (pp.checkout IS NOT NULL AND pp.checkout = {s}.checkout) \
         OR (((pp.repo_root IS NULL AND pp.checkout IS NULL) OR ({s}.repo_root IS NULL AND {s}.checkout IS NULL)) \
             AND pp.cwd IS NOT NULL AND {s}.cwd IS NOT NULL AND ({a_in_b} OR {b_in_a}))) \
         ORDER BY pp.id) pe)",
        mine = active(s),
        theirs = active("pp"),
        a_in_b = below(s, "pp"),
        b_in_a = below("pp", s),
    )
}

fn notes_current(a: &Aliases) -> String {
    format!("({})", current_note(&a.row))
}

fn notes_withdrawn(a: &Aliases) -> String {
    format!("({})", withdrawn_note(&a.row))
}

fn work_goal_field(a: &Aliases) -> String {
    work_goal(&a.row)
}

/// A note that is its family's current revision and not withdrawn.
pub fn live_note(n: &str) -> String {
    format!("{} AND NOT {}", current_note(n), withdrawn_note(n))
}

pub fn current_note(n: &str) -> String {
    format!(
        "NOT EXISTS (SELECT 1 FROM a.note nr WHERE nr.root = {n}.root AND nr.commit_seq > {n}.commit_seq)"
    )
}

pub fn withdrawn_note(n: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM a.note nw JOIN a.note_target nwt ON nwt.note_id = nw.id \
         WHERE nw.kind = 'withdrawal' AND nwt.ref = 'note:' || {n}.root \
         AND NOT EXISTS (SELECT 1 FROM a.note nw2 WHERE nw2.root = nw.root AND nw2.commit_seq > nw.commit_seq))"
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
        "WITH RECURSIVE tr(pk) AS (SELECT {s}.pk UNION SELECT l.child_pk FROM current_link l \
         JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr"
    )
}

#[cfg(test)]
mod tests;
