//! The parsed query. Every condition keeps the byte range it came from, so
//! an error can point at it and a correction can replace exactly it.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Query {
    pub source: Source,
    pub attach: Vec<Child>,
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    /// `UNIT [where COND]`; a bare unit selects all its rows.
    Unit { unit: Unit, cond: Option<Cond> },
    /// Conditions without a unit: a session query.
    Sessions { cond: Cond },
    /// The empty query: every session, by the defaults.
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Sessions,
    Messages,
    Blocks,
    ToolCalls,
    Files,
    Events,
    Notes,
    Links,
    Turns,
    Runs,
    Deliveries,
    Processes,
    Work,
}

impl Unit {
    pub const ALL: [Unit; 13] = [
        Unit::Sessions,
        Unit::Messages,
        Unit::Blocks,
        Unit::ToolCalls,
        Unit::Files,
        Unit::Events,
        Unit::Notes,
        Unit::Links,
        Unit::Turns,
        Unit::Runs,
        Unit::Deliveries,
        Unit::Processes,
        Unit::Work,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Unit::Sessions => "sessions",
            Unit::Messages => "messages",
            Unit::Blocks => "blocks",
            Unit::ToolCalls => "tool_calls",
            Unit::Files => "files",
            Unit::Events => "events",
            Unit::Notes => "notes",
            Unit::Links => "links",
            Unit::Turns => "turns",
            Unit::Runs => "runs",
            Unit::Deliveries => "deliveries",
            Unit::Processes => "processes",
            Unit::Work => "work",
        }
    }

    /// The singular form `exists` and `seq` use.
    pub fn singular(self) -> &'static str {
        match self {
            Unit::Sessions => "session",
            Unit::Messages => "message",
            Unit::Blocks => "block",
            Unit::ToolCalls => "tool_call",
            Unit::Files => "file",
            Unit::Events => "event",
            Unit::Notes => "note",
            Unit::Links => "link",
            Unit::Turns => "turn",
            Unit::Runs => "run",
            Unit::Deliveries => "delivery",
            Unit::Processes => "process",
            Unit::Work => "work",
        }
    }

    pub fn from_name(name: &str) -> Option<Unit> {
        let name = name.to_ascii_lowercase();
        Unit::ALL
            .into_iter()
            .find(|u| u.name() == name || u.singular() == name)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cond {
    pub span: Span,
    #[serde(flatten)]
    pub kind: CondKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CondKind {
    And {
        all: Vec<Cond>,
    },
    Or {
        any: Vec<Cond>,
    },
    Not {
        cond: Box<Cond>,
    },
    /// `field:value`.
    Match {
        field: String,
        value: Value,
    },
    /// `field OP value`, and the compact `field:>=value`.
    Compare {
        field: String,
        op: Op,
        value: String,
    },
    Between {
        field: String,
        from: String,
        to: String,
    },
    /// A full-text term: a bare word, or a quoted phrase.
    Text {
        text: String,
        phrase: bool,
    },
    /// A pasted URL: a reference.
    Url {
        url: String,
    },
    /// `@N` or `@N.M`: a position in the last listing.
    Position {
        text: String,
    },
    Exists {
        unit: Unit,
        cond: Box<Cond>,
    },
    /// `seq(step -> [gap] step …)`: tool calls in session order.
    Seq {
        steps: Vec<Cond>,
        gaps: Vec<Gap>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "value", rename_all = "snake_case")]
pub enum Value {
    One {
        text: String,
        quoted: bool,
    },
    Alternatives {
        any: Vec<String>,
    },
    /// `*`: any value, absent included.
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    pub fn parse(text: &str) -> Option<Op> {
        Some(match text {
            "=" => Op::Eq,
            "!=" => Op::Ne,
            "<" => Op::Lt,
            "<=" => Op::Le,
            ">" => Op::Gt,
            ">=" => Op::Ge,
            _ => return None,
        })
    }

    pub fn sql(self) -> &'static str {
        match self {
            Op::Eq => "=",
            Op::Ne => "<>",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
        }
    }
}

/// What may lie between two steps of a `seq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "gap", rename_all = "snake_case")]
pub enum Gap {
    /// Any later call in the same session.
    Later,
    /// The very next call.
    Next,
    /// A later call that started at most this long after.
    Within { ms: i64 },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Child {
    pub span: Span,
    pub unit: Unit,
    pub fields: Vec<String>,
    pub windows: Vec<Window>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "window", rename_all = "snake_case")]
pub enum Window {
    First { n: u64 },
    Last { n: u64 },
    Cond { cond: Cond },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stage {
    pub span: Span,
    #[serde(flatten)]
    pub kind: StageKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum StageKind {
    SortBy {
        key: SortKey,
        descending: bool,
    },
    GroupBy {
        keys: Vec<GroupKey>,
    },
    Agg {
        fns: Vec<AggFn>,
    },
    Count,
    Limit {
        n: u64,
    },
    Offset {
        n: u64,
    },
    Where {
        cond: Cond,
    },
    Distinct,
    Matches,
    /// `| UNIT [where COND]`: rows of another unit within the matched
    /// sessions.
    Scope {
        unit: Unit,
        cond: Option<Cond>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "key", rename_all = "snake_case")]
pub enum SortKey {
    Field { field: String },
    Random { seed: Option<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GroupKey {
    pub field: String,
    pub bucket: Option<Bucket>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bucket {
    Day,
    Week,
    Month,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "fn", rename_all = "snake_case")]
pub enum AggFn {
    Count,
    Sum {
        field: String,
    },
    Avg {
        field: String,
    },
    Min {
        field: String,
    },
    Max {
        field: String,
    },
    /// `p90:latency`.
    Percentile {
        p: u8,
        field: String,
    },
    CountDistinct {
        field: String,
    },
}

impl AggFn {
    /// The output column: `count`, `sum_cost`, `p90_latency`.
    pub fn column(&self) -> String {
        match self {
            AggFn::Count => "count".into(),
            AggFn::Sum { field } => format!("sum_{}", flat(field)),
            AggFn::Avg { field } => format!("avg_{}", flat(field)),
            AggFn::Min { field } => format!("min_{}", flat(field)),
            AggFn::Max { field } => format!("max_{}", flat(field)),
            AggFn::Percentile { p, field } => format!("p{p}_{}", flat(field)),
            AggFn::CountDistinct { field } => format!("count_distinct_{}", flat(field)),
        }
    }

    pub fn field(&self) -> Option<&str> {
        match self {
            AggFn::Count => None,
            AggFn::Sum { field }
            | AggFn::Avg { field }
            | AggFn::Min { field }
            | AggFn::Max { field }
            | AggFn::Percentile { field, .. }
            | AggFn::CountDistinct { field } => Some(field),
        }
    }
}

/// A field name as a column name: `session.origin` becomes `session_origin`.
pub fn flat(field: &str) -> String {
    field.replace(['.', '-'], "_")
}
