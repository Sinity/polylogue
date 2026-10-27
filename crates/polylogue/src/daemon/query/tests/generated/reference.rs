//! What a generated query means, worked out directly over the generated
//! rows: a value nobody knows makes a condition unknown (neither it nor
//! its `NOT` matches), defaults are bound once from the top-level
//! conditions, and rows come in each unit's default order.

use std::cmp::Reverse;

pub const ORIGINS: [&str; 3] = ["claude-code", "codex", "chatgpt"];
pub const REPOS: [&str; 3] = ["op/morrow", "https://github.com/op/morrow", "op/kestrel"];
pub const REPO_NAMES: [&str; 2] = ["morrow", "kestrel"];
pub const MODELS: [&str; 2] = ["model-a", "model-b"];
pub const WORDS: [&str; 5] = ["parser", "chrono", "walrus", "date", "fix"];
pub const COMMANDS: [&str; 3] = ["cargo test", "cargo build", "git status"];
/// Role and author.
pub const SPEAKERS: [(&str, &str); 4] = [
    ("user", "human"),
    ("assistant", "model"),
    ("user", "harness"),
    ("user", "unknown"),
];
/// Tool and action.
pub const TOOLS: [(&str, &str); 3] = [
    ("Bash", "shell"),
    ("Edit", "file_edit"),
    ("Read", "file_read"),
];
pub const OUTCOMES: [&str; 3] = ["ok", "error", "unknown"];
/// Message costs, in nano-dollars.
pub const COSTS: [i64; 3] = [200_000_000, 700_000_000, 1_300_000_000];
/// Cost thresholds, in tenths of a dollar.
pub const THRESHOLDS: [i64; 3] = [5, 10, 20];
/// 2026-10-01T00:00Z; generated times fall in the three days after it.
pub const FIRST_DAY: i64 = 1_790_812_800_000;
pub const HOUR: i64 = 3_600_000;
const DAY: i64 = 24 * HOUR;

#[derive(Debug, Clone)]
pub struct Rows {
    pub sessions: Vec<SessionRow>,
}

#[derive(Debug, Clone)]
pub struct SessionRow {
    pub origin: &'static str,
    pub subagent: bool,
    pub repo: Option<&'static str>,
    pub messages: Vec<MessageRow>,
}

#[derive(Debug, Clone)]
pub struct MessageRow {
    pub speaker: (&'static str, &'static str),
    pub model: Option<&'static str>,
    pub cost: Option<i64>,
    pub at: Option<i64>,
    pub text: Option<String>,
    /// On the active path: `Some(false)` is an abandoned branch.
    pub path: Option<bool>,
    pub calls: Vec<CallRow>,
}

#[derive(Debug, Clone)]
pub struct CallRow {
    pub tool: (&'static str, &'static str),
    pub command: Option<&'static str>,
    pub outcome: &'static str,
    pub exit_code: Option<i64>,
    pub at: Option<i64>,
}

pub fn session_id(rows: &Rows, i: usize) -> String {
    format!("{}:g{i:02}", rows.sessions[i].origin)
}

pub fn message_id(i: usize, j: usize) -> String {
    format!("g{i:02}m{j:02}")
}

/// A tool call as its result row shows it: its message, tool, command,
/// outcome and exit code. Calls alike in all of these are alike to every
/// condition the generator writes.
pub fn call_key(
    message: &str,
    tool: &str,
    command: Option<&str>,
    outcome: &str,
    exit_code: Option<i64>,
) -> String {
    format!("{message} {tool} {command:?} {outcome} {exit_code:?}")
}

impl SessionRow {
    pub fn started(&self) -> Option<i64> {
        self.messages.iter().filter_map(|m| m.at).min()
    }

    pub fn last_active(&self) -> Option<i64> {
        self.messages.iter().filter_map(|m| m.at).max()
    }

    /// Messages not on an abandoned branch: the session's own.
    fn own(&self) -> impl Iterator<Item = &MessageRow> {
        self.messages.iter().filter(|m| m.path != Some(false))
    }

    fn own_calls(&self) -> impl Iterator<Item = &CallRow> {
        self.own().flat_map(|m| m.calls.iter())
    }

    /// The sum of the known costs of own messages; unknown when none is known.
    pub fn cost(&self) -> Option<i64> {
        let known: Vec<i64> = self.own().filter_map(|m| m.cost).collect();
        (!known.is_empty()).then(|| known.iter().sum())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truth {
    Yes,
    No,
    Unknown,
}

impl Truth {
    fn of(b: bool) -> Self {
        if b { Truth::Yes } else { Truth::No }
    }

    fn known(value: Option<bool>) -> Self {
        value.map_or(Truth::Unknown, Truth::of)
    }

    fn not(self) -> Self {
        match self {
            Truth::Yes => Truth::No,
            Truth::No => Truth::Yes,
            Truth::Unknown => Truth::Unknown,
        }
    }

    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Truth::No, _) | (_, Truth::No) => Truth::No,
            (Truth::Yes, Truth::Yes) => Truth::Yes,
            _ => Truth::Unknown,
        }
    }

    fn or(self, other: Self) -> Self {
        self.not().and(other.not()).not()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Sessions,
    Messages,
    ToolCalls,
}

impl Unit {
    pub const ALL: [Unit; 3] = [Unit::Sessions, Unit::Messages, Unit::ToolCalls];

    pub fn name(self) -> &'static str {
        match self {
            Unit::Sessions => "sessions",
            Unit::Messages => "messages",
            Unit::ToolCalls => "tool_calls",
        }
    }

    /// A top-level condition that names every default field of the unit,
    /// so conditions combined after it are compared over one source.
    pub fn every_row(self) -> &'static str {
        match self {
            Unit::Sessions => "kind:*",
            Unit::Messages => "active:*",
            Unit::ToolCalls => "action:*",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Cond {
    Leaf(Leaf),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Not(Box<Cond>),
}

/// One field condition. Session conditions first, then message
/// conditions, then tool call conditions.
#[derive(Debug, Clone)]
pub enum Leaf {
    Origin(&'static str),
    EitherOrigin(&'static str, &'static str),
    Subagent(bool),
    Repo(&'static str),
    RepoUnknown,
    Model(&'static str),
    ModelUnknown,
    CostAbove(i64),
    CostBelow(i64),
    CostUnknown,
    MessagesAtLeast(usize),
    ToolCallsAbove(usize),
    ErrorsAtLeast(usize),
    Has(&'static str),
    StartedBefore(i64),
    StartedFrom(i64),
    StartedUnknown,
    ExistsCall(Box<Cond>),

    Role(&'static str),
    WrittenBy(&'static str),
    MessageModel(&'static str),
    MessageModelUnknown,
    MessageCostAbove(i64),
    MessageCostUnknown,
    AtBefore(i64),
    AtUnknown,
    WordsAtLeast(usize),
    Word(&'static str),
    Abandoned,
    SessionOrigin(&'static str),

    Tool(&'static str),
    Action(&'static str),
    Outcome(&'static str),
    ExitZero,
    ExitNonzero,
    ExitUnknown,
    Command(&'static str),
    CommandUnknown,
    CallAtBefore(i64),
    CallAtUnknown,
    CallSessionOrigin(&'static str),
}

fn dollars(tenths: i64) -> String {
    format!("{}.{}", tenths / 10, tenths % 10)
}

/// Day `d` of October 2026, as the query writes it.
fn day(d: i64) -> String {
    format!("2026-10-{d:02}")
}

fn day_start(d: i64) -> i64 {
    FIRST_DAY + (d - 1) * DAY
}

impl Leaf {
    pub fn text(&self) -> String {
        match self {
            Leaf::Origin(o) => format!("origin:{o}"),
            Leaf::SessionOrigin(o) | Leaf::CallSessionOrigin(o) => format!("session.origin:{o}"),
            Leaf::EitherOrigin(a, b) => format!("origin:({a}|{b})"),
            Leaf::Subagent(true) => "kind:subagent".into(),
            Leaf::Subagent(false) => "kind:main".into(),
            Leaf::Repo(r) => format!("repo:{r}"),
            Leaf::RepoUnknown => "repo:unknown".into(),
            Leaf::Model(m) | Leaf::MessageModel(m) => format!("model:{m}"),
            Leaf::ModelUnknown | Leaf::MessageModelUnknown => "model:unknown".into(),
            Leaf::CostAbove(t) | Leaf::MessageCostAbove(t) => format!("cost > {}", dollars(*t)),
            Leaf::CostBelow(t) => format!("cost < {}", dollars(*t)),
            Leaf::CostUnknown | Leaf::MessageCostUnknown => "cost:unknown".into(),
            Leaf::MessagesAtLeast(n) => format!("messages >= {n}"),
            Leaf::ToolCallsAbove(n) => format!("tool_calls > {n}"),
            Leaf::ErrorsAtLeast(n) => format!("errors >= {n}"),
            Leaf::Has(h) => format!("has:{h}"),
            Leaf::StartedBefore(d) => format!("started < {}", day(*d)),
            Leaf::StartedFrom(d) => format!("started >= {}", day(*d)),
            Leaf::StartedUnknown => "started:unknown".into(),
            Leaf::ExistsCall(c) => format!("exists tool_call({})", c.text()),
            Leaf::Role(r) => format!("role:{r}"),
            Leaf::WrittenBy(w) => format!("written_by:{w}"),
            Leaf::AtBefore(d) | Leaf::CallAtBefore(d) => format!("at < {}", day(*d)),
            Leaf::AtUnknown | Leaf::CallAtUnknown => "at:unknown".into(),
            Leaf::WordsAtLeast(n) => format!("words >= {n}"),
            Leaf::Word(w) => (*w).to_owned(),
            Leaf::Abandoned => "active:false".into(),
            Leaf::Tool(t) => format!("tool:{t}"),
            Leaf::Action(a) => format!("action:{a}"),
            Leaf::Outcome(o) => format!("outcome:{o}"),
            Leaf::ExitZero => "exit_code = 0".into(),
            Leaf::ExitNonzero => "exit_code != 0".into(),
            Leaf::ExitUnknown => "exit_code:unknown".into(),
            Leaf::Command(c) => format!("command:\"{c}\""),
            Leaf::CommandUnknown => "command:unknown".into(),
        }
    }

    /// The condition selecting the rows this one is unknown for, when the
    /// field has one.
    pub fn unknown_form(&self) -> Option<Leaf> {
        Some(match self {
            Leaf::Repo(_) => Leaf::RepoUnknown,
            Leaf::Model(_) => Leaf::ModelUnknown,
            Leaf::CostAbove(_) | Leaf::CostBelow(_) => Leaf::CostUnknown,
            Leaf::StartedBefore(_) | Leaf::StartedFrom(_) => Leaf::StartedUnknown,
            Leaf::WrittenBy(w) if *w != "unknown" => Leaf::WrittenBy("unknown"),
            Leaf::MessageModel(_) => Leaf::MessageModelUnknown,
            Leaf::MessageCostAbove(_) => Leaf::MessageCostUnknown,
            Leaf::AtBefore(_) => Leaf::AtUnknown,
            Leaf::Outcome(o) if *o != "unknown" => Leaf::Outcome("unknown"),
            Leaf::ExitZero | Leaf::ExitNonzero => Leaf::ExitUnknown,
            Leaf::Command(_) => Leaf::CommandUnknown,
            Leaf::CallAtBefore(_) => Leaf::CallAtUnknown,
            _ => return None,
        })
    }
}

impl Cond {
    pub fn text(&self) -> String {
        match self {
            Cond::Leaf(l) => l.text(),
            Cond::And(a, b) => format!("({} AND {})", a.text(), b.text()),
            Cond::Or(a, b) => format!("({} OR {})", a.text(), b.text()),
            Cond::Not(a) => format!("NOT ({})", a.text()),
        }
    }

    /// The conditions joined to the rest by AND, parentheses flattened.
    fn top_level(&self) -> Vec<&Leaf> {
        match self {
            Cond::Leaf(l) => vec![l],
            Cond::And(a, b) => [a.top_level(), b.top_level()].concat(),
            _ => vec![],
        }
    }

    /// A search word among the top-level conditions: the rows are then
    /// ranked by relevance.
    pub fn ranks(&self) -> bool {
        self.top_level().iter().any(|l| matches!(l, Leaf::Word(_)))
    }

    fn eval(&self, rows: &Rows, row: Row) -> Truth {
        match self {
            Cond::Leaf(l) => l.eval(rows, row),
            Cond::And(a, b) => a.eval(rows, row).and(b.eval(rows, row)),
            Cond::Or(a, b) => a.eval(rows, row).or(b.eval(rows, row)),
            Cond::Not(a) => a.eval(rows, row).not(),
        }
    }
}

/// A row of one unit, by its place in the generated rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row {
    Session(usize),
    Message(usize, usize),
    Call(usize, usize, usize),
}

impl Leaf {
    fn eval(&self, rows: &Rows, row: Row) -> Truth {
        let session = match row {
            Row::Session(i) | Row::Message(i, _) | Row::Call(i, _, _) => &rows.sessions[i],
        };
        let message = match row {
            Row::Message(i, j) | Row::Call(i, j, _) => Some(&rows.sessions[i].messages[j]),
            Row::Session(_) => None,
        };
        let call = match row {
            Row::Call(i, j, k) => Some(&rows.sessions[i].messages[j].calls[k]),
            _ => None,
        };
        let m = || message.expect("a message condition on a message row");
        let c = || call.expect("a tool call condition on a tool call row");
        let any_model = session.messages.iter().any(|m| m.model.is_some());
        match self {
            Leaf::Origin(o) | Leaf::SessionOrigin(o) | Leaf::CallSessionOrigin(o) => {
                Truth::of(session.origin == *o)
            }
            Leaf::EitherOrigin(a, b) => Truth::of(session.origin == *a || session.origin == *b),
            Leaf::Subagent(s) => Truth::of(session.subagent == *s),
            Leaf::Repo(name) => Truth::known(session.repo.map(|r| {
                let r = r.trim_end_matches('/').to_lowercase();
                r == *name
                    || r.ends_with(&format!("/{name}"))
                    || r.ends_with(&format!("/{name}.git"))
            })),
            Leaf::RepoUnknown => Truth::of(session.repo.is_none()),
            Leaf::Model(name) => Truth::known(
                any_model.then(|| session.messages.iter().any(|m| m.model == Some(*name))),
            ),
            Leaf::ModelUnknown => Truth::of(!any_model),
            Leaf::CostAbove(t) => Truth::known(session.cost().map(|c| c > t * 100_000_000)),
            Leaf::CostBelow(t) => Truth::known(session.cost().map(|c| c < t * 100_000_000)),
            Leaf::CostUnknown => Truth::of(session.cost().is_none()),
            Leaf::MessagesAtLeast(n) => Truth::of(session.own().count() >= *n),
            Leaf::ToolCallsAbove(n) => Truth::of(session.own_calls().count() > *n),
            Leaf::ErrorsAtLeast(n) => {
                Truth::of(session.own_calls().filter(|c| c.outcome == "error").count() >= *n)
            }
            Leaf::Has(what) => {
                let mut calls = session.own().flat_map(|m| m.calls.iter());
                Truth::of(match *what {
                    "errors" => calls.any(|c| c.outcome == "error"),
                    "tool_calls" => calls.next().is_some(),
                    "branches" => session.messages.iter().any(|m| m.path == Some(false)),
                    other => unreachable!("has:{other} is not generated"),
                })
            }
            Leaf::StartedBefore(d) => Truth::known(session.started().map(|t| t < day_start(*d))),
            Leaf::StartedFrom(d) => Truth::known(session.started().map(|t| t >= day_start(*d))),
            Leaf::StartedUnknown => Truth::of(session.started().is_none()),
            Leaf::ExistsCall(inner) => {
                let Row::Session(i) = row else {
                    unreachable!("exists is generated on sessions")
                };
                let mut found = Truth::No;
                for (j, m) in session.messages.iter().enumerate() {
                    if m.path == Some(false) {
                        continue;
                    }
                    for k in 0..m.calls.len() {
                        found = found.or(inner.eval(rows, Row::Call(i, j, k)));
                    }
                }
                found
            }
            Leaf::Role(r) => Truth::of(m().speaker.0 == *r),
            Leaf::WrittenBy(w) => {
                let stored = m().speaker.1;
                if *w == "unknown" {
                    Truth::of(stored == "unknown")
                } else if stored == "unknown" {
                    Truth::Unknown
                } else {
                    Truth::of(stored == *w)
                }
            }
            Leaf::MessageModel(name) => Truth::known(m().model.map(|x| x == *name)),
            Leaf::MessageModelUnknown => Truth::of(m().model.is_none()),
            Leaf::MessageCostAbove(t) => Truth::known(m().cost.map(|c| c > t * 100_000_000)),
            Leaf::MessageCostUnknown => Truth::of(m().cost.is_none()),
            Leaf::AtBefore(d) => Truth::known(m().at.map(|t| t < day_start(*d))),
            Leaf::AtUnknown => Truth::of(m().at.is_none()),
            Leaf::WordsAtLeast(n) => Truth::of(
                m().text
                    .as_deref()
                    .map_or(0, |t| t.split_whitespace().count())
                    >= *n,
            ),
            // Words are searched in prose: what a person or a model wrote.
            Leaf::Word(w) => Truth::of(
                matches!(m().speaker.1, "human" | "model")
                    && m()
                        .text
                        .as_deref()
                        .is_some_and(|t| t.split_whitespace().any(|x| x == *w)),
            ),
            // An unknown path is read as on the path.
            Leaf::Abandoned => Truth::of(m().path == Some(false)),
            Leaf::Tool(t) => Truth::of(c().tool.0 == *t),
            Leaf::Action(a) => Truth::of(c().tool.1 == *a),
            Leaf::Outcome(o) => {
                let stored = c().outcome;
                if *o == "unknown" {
                    Truth::of(stored == "unknown")
                } else if stored == "unknown" {
                    Truth::Unknown
                } else {
                    Truth::of(stored == *o)
                }
            }
            Leaf::ExitZero => Truth::known(c().exit_code.map(|e| e == 0)),
            Leaf::ExitNonzero => Truth::known(c().exit_code.map(|e| e != 0)),
            Leaf::ExitUnknown => Truth::of(c().exit_code.is_none()),
            Leaf::Command(part) => Truth::known(c().command.map(|x| x.contains(part))),
            Leaf::CommandUnknown => Truth::of(c().command.is_none()),
            Leaf::CallAtBefore(d) => Truth::known(c().at.map(|t| t < day_start(*d))),
            Leaf::CallAtUnknown => Truth::of(c().at.is_none()),
        }
    }
}

/// Every row of the unit the unit's defaults keep, given the top-level
/// conditions, in the unit's default order.
fn source(rows: &Rows, unit: Unit, cond: Option<&Cond>) -> Vec<Row> {
    let top = cond.map(Cond::top_level).unwrap_or_default();
    let names_kind = top.iter().any(|l| matches!(l, Leaf::Subagent(_)));
    let names_active = top.iter().any(|l| matches!(l, Leaf::Abandoned));
    let sessions = rows.sessions.iter().enumerate();
    match unit {
        Unit::Sessions => {
            let mut out: Vec<(usize, &SessionRow)> = sessions
                .filter(|(_, s)| names_kind || !s.subagent)
                .collect();
            // Newest activity first, unknown last, then the ID, highest first.
            out.sort_by_key(|(i, s)| {
                (
                    s.last_active().is_none(),
                    Reverse(s.last_active()),
                    Reverse(session_id(rows, *i)),
                )
            });
            out.into_iter().map(|(i, _)| Row::Session(i)).collect()
        }
        Unit::Messages => {
            let mut out = vec![];
            for (i, s) in sessions {
                for (j, m) in s.messages.iter().enumerate() {
                    if names_active || m.path != Some(false) {
                        out.push((m.at, Row::Message(i, j)));
                    }
                }
            }
            // Oldest first, unknown last, then session order (the order
            // they were written in).
            out.sort_by_key(|(at, row)| {
                let Row::Message(i, j) = row else {
                    unreachable!()
                };
                (at.is_none(), *at, *i, *j)
            });
            out.into_iter().map(|(_, row)| row).collect()
        }
        Unit::ToolCalls => {
            let mut out = vec![];
            for (i, s) in sessions {
                for (j, m) in s.messages.iter().enumerate() {
                    if m.path == Some(false) {
                        continue;
                    }
                    for (k, c) in m.calls.iter().enumerate() {
                        out.push((c.at, Row::Call(i, j, k)));
                    }
                }
            }
            // Oldest first, unknown last, then the message ID and the
            // call's place in it.
            out.sort_by_key(|(at, row)| {
                let Row::Call(i, j, k) = row else {
                    unreachable!()
                };
                (at.is_none(), *at, message_id(*i, *j), *k)
            });
            out.into_iter().map(|(_, row)| row).collect()
        }
    }
}

/// The rows `unit where cond` returns, in order.
pub fn answer(rows: &Rows, unit: Unit, cond: &Cond) -> Vec<Row> {
    source(rows, unit, Some(cond))
        .into_iter()
        .filter(|row| cond.eval(rows, *row) == Truth::Yes)
        .collect()
}

/// Whether `cond` holds for one row.
pub fn truth(rows: &Rows, cond: &Cond, row: Row) -> Truth {
    cond.eval(rows, row)
}

/// A row as the result rows name it: a session or message ID, or a
/// tool call's [`call_key`].
pub fn key(rows: &Rows, row: Row) -> String {
    match row {
        Row::Session(i) => session_id(rows, i),
        Row::Message(i, j) => message_id(i, j),
        Row::Call(i, j, k) => {
            let c = &rows.sessions[i].messages[j].calls[k];
            call_key(
                &message_id(i, j),
                c.tool.0,
                c.command,
                c.outcome,
                c.exit_code,
            )
        }
    }
}
