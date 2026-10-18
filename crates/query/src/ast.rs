//! The parsed query.

#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub source: Source,
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// A predicate without a unit; implicitly `sessions where`.
    Sessions(Predicate),
    Unit {
        unit: Unit,
        predicate: Predicate,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Sessions,
    Messages,
    Blocks,
    Actions,
    Files,
    Links,
    Notes,
    Assertions,
    Work,
    Deliveries,
    Processes,
    Runs,
    Events,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
    Field {
        field: String,
        op: Op,
        values: Vec<String>,
    },
    Range {
        field: String,
        from: String,
        to: String,
    },
    /// A search term: a bare word or a quoted phrase.
    Text(String),
    Exists {
        unit: String,
        predicate: Box<Predicate>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    GroupBy(Vec<String>),
    Count,
    SortBy {
        field: String,
        descending: bool,
    },
    Limit(u64),
    Offset(u64),
    Select(Vec<String>),
    /// An explicit fusion stage: merge rows by a field.
    Fuse(String),
    /// A sub-selection over the previous stage's rows.
    Where {
        unit: Unit,
        predicate: Predicate,
    },
}
