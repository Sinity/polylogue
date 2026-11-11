//! The unit registry: everything about a query unit, declared once. The
//! parser resolves unit names here; the compiler reads a unit's table, key,
//! default columns, order, defaults, time window, relations, text search
//! and parent rows from here; the fields each unit accepts are in
//! [`crate::query::fields::FIELDS`]. Adding a unit is adding its [`Unit`] variant
//! and one [`UnitSpec`].
//!
//! Templates name the unit's aliases as `{s}` session, `{m}` message, `{b}`
//! block, `{t}` tool call, `{cb}` its call block and `{r}` the unit's own
//! row (see [`Aliases`]). A relation's templates name the related unit's
//! aliases with an `n` prefix (`{ns}`, `{nt}`, `{nr}`, …), its FROM clause as
//! `{from}` and the condition on it as `{cond}`.

use crate::query::compile::ColKind;
use crate::query::fields::{self, Aliases, Expr};
use polylogue_model::SessionKind;
pub use polylogue_query::defaults::DefaultId;
pub use polylogue_query::units::Unit;

pub struct UnitSpec {
    pub from: Expr,
    /// The row key, `_pk` in the output. It can change when the index is
    /// rebuilt.
    pub key: &'static str,
    /// The terms that name one row, unchanged by an index rebuild; the last
    /// terms of every order, each compared by its own type, so the order is
    /// total and pages continue after a rebuild. Never the row key, and
    /// never bytes, which a cursor cannot carry.
    pub identity: &'static [&'static str],
    /// The default columns: name, SQL template, kind.
    pub columns: &'static [(&'static str, Expr, ColKind)],
    /// The time rows are ordered by.
    pub time: &'static str,
    pub newest_first: bool,
    /// Rows at the same time keep session order: the ordering input, the
    /// record's offset in it, then the message ID.
    pub session_order: bool,
    /// `since:` and `until:`: a condition on `{v}`, the resolved instant.
    pub since: Option<&'static str>,
    pub until: Option<&'static str>,
    /// The expression naming the row's session key (`session.pk`).
    pub session_pk: &'static str,
    /// For a unit not in one session: its rows related to the sessions
    /// `{sessions}` selects.
    pub session_scope: Option<&'static str>,
    /// How words search this unit's rows.
    pub search: Search,
    pub relations: &'static [Relation],
    /// The column tying a row to a parent row of another unit, for
    /// `exists` and attached rows: a block's or tool call's message. Rows
    /// of a unit not listed relate through their session.
    pub parents: &'static [(Unit, &'static str)],
}

/// Which full-text index a term searches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Index {
    /// `text` blocks written by a person or a model.
    Prose,
    /// `thinking` blocks.
    Thinking,
    /// Tool output.
    Output,
}

/// How words search a unit's rows.
pub enum Search {
    /// Words do not search this unit.
    None,
    /// Full-text hits related to the row, as a condition and a rank (lower
    /// is better), or `None` when the index does not apply to the unit.
    /// Which words search it is the query unit's declaration.
    Hits {
        rows: fn(&Aliases, &Hits<'_>) -> Option<HitRelation>,
    },
    /// Each word is a substring of this column: notes are few and live in
    /// archive.db, outside the full-text index.
    Substring(&'static str),
}

/// The full-text hits of one search: the CTE of `(text_id, rank)` and the
/// index searched.
pub struct Hits<'a> {
    pub cte: &'a str,
    pub index: Index,
}

impl Hits<'_> {
    /// The condition that block alias `b` is of the index's kind.
    pub fn block_kind(&self, b: &str) -> String {
        let kinds = match self.index {
            Index::Prose => "'text'",
            Index::Thinking => "'thinking'",
            Index::Output => "'tool_result','tool_call'",
        };
        format!("{b}.kind IN ({kinds})")
    }

    /// For prose, the condition that message alias `m` was written by a
    /// person or a model, after ` AND `; empty for the other indexes.
    pub fn writer(&self, m: &str) -> String {
        match self.index {
            Index::Prose => format!(" AND {m}.written_by IN ('human', 'model')"),
            _ => String::new(),
        }
    }
}

/// The membership and rank read from shared hit relations.
pub struct HitRelation {
    pub ctes: Vec<String>,
    pub condition: String,
    pub rank: String,
}

fn documents_hits(a: &Aliases, h: &Hits<'_>) -> Option<HitRelation> {
    if h.index != Index::Prose {
        return None;
    }
    let from = format!(
        "FROM {} h JOIN document_block db ON db.text_id=h.text_id WHERE db.document_pk={}.pk AND db.kind='text'",
        h.cte, a.row
    );
    Some(HitRelation {
        ctes: vec![],
        condition: format!("EXISTS(SELECT 1 {from})"),
        rank: format!("(SELECT min(h.rank) {from})"),
    })
}

fn sessions_hits(a: &Aliases, h: &Hits<'_>) -> Option<HitRelation> {
    let sessions = format!("{}_sessions", h.cte);
    // Find the owning sessions from the hits, then their subagent ancestors.
    // UNION over (session, rank) bounds cycles and shared descendants. A
    // session's own best rank suffices for every ancestor's best rank.
    let rows = format!(
        "{sessions} AS MATERIALIZED (WITH RECURSIVE \
         matched(session_pk, rank) AS MATERIALIZED (\
           SELECT xm.session_pk, min(h.rank) FROM {texts} h \
           CROSS JOIN block xb ON xb.text_id=h.text_id \
           CROSS JOIN message xm ON xm.pk=xb.message_pk \
           WHERE {}{} AND {} GROUP BY xm.session_pk), \
         ancestors(session_pk, rank) AS (\
           SELECT session_pk, rank FROM matched UNION \
           SELECT parent.pk, child.rank FROM ancestors child \
           CROSS JOIN link l ON l.child_pk=child.session_pk AND l.kind='subagent' \
           CROSS JOIN session parent ON parent.id=l.parent_id) \
         SELECT session_pk, min(rank) AS rank FROM ancestors GROUP BY session_pk)",
        h.block_kind("xb"),
        h.writer("xm"),
        fields::own_messages("xm", None),
        texts = h.cte
    );
    Some(HitRelation {
        ctes: vec![rows],
        // A missing related session has no hits, including under NOT.
        condition: format!(
            "({s}.pk IS NOT NULL AND {s}.pk IN (SELECT session_pk FROM {sessions}))",
            s = a.s
        ),
        rank: format!(
            "(SELECT rank FROM {sessions} h WHERE h.session_pk={}.pk)",
            a.s
        ),
    })
}

fn messages_hits(a: &Aliases, h: &Hits<'_>) -> Option<HitRelation> {
    let from = format!(
        "FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
         WHERE xb.message_pk = {m}.pk AND {}",
        h.block_kind("xb"),
        cte = h.cte,
        m = a.m
    );
    let holding = format!(
        "{m}.pk IN (SELECT xb.message_pk FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
         WHERE {})",
        h.block_kind("xb"),
        cte = h.cte,
        m = a.m
    );
    Some(HitRelation {
        ctes: vec![],
        condition: format!("({holding}{})", h.writer(&a.m)),
        rank: format!("(SELECT min(h.rank) {from})"),
    })
}

fn blocks_hits(a: &Aliases, h: &Hits<'_>) -> Option<HitRelation> {
    Some(HitRelation {
        ctes: vec![],
        condition: format!(
            "({b}.text_id IN (SELECT text_id FROM {cte}) AND {})",
            h.block_kind(&a.b),
            b = a.b,
            cte = h.cte
        ),
        rank: format!(
            "(SELECT h.rank FROM {cte} h WHERE h.text_id={b}.text_id)",
            b = a.b,
            cte = h.cte
        ),
    })
}

fn tool_call_hits(a: &Aliases, h: &Hits<'_>) -> Option<HitRelation> {
    if h.index != Index::Output {
        return None;
    }
    let from = format!(
        "FROM {cte} h JOIN block xb ON xb.text_id = h.text_id WHERE xb.pk IN ({t}.call_block_pk, {t}.result_block_pk)",
        cte = h.cte,
        t = a.t
    );
    Some(HitRelation {
        ctes: vec![],
        condition: format!("EXISTS (SELECT 1 {from})"),
        rank: format!("(SELECT min(h.rank) {from})"),
    })
}

/// A `prefix.FIELD` path from one unit to another.
pub struct Relation {
    pub prefix: &'static str,
    pub target: Unit,
    pub via: Via,
}

pub enum Via {
    /// The target is read through the unit's own aliases.
    Same,
    /// A session joined under another alias (`{r}p`, a link's parent).
    Session(&'static str),
    /// A correlated subquery over the target: the condition template, and
    /// the value template (`{e}` the target's expression) when the related
    /// field can be grouped or sorted by.
    Exists {
        cond: &'static str,
        value: Option<&'static str>,
    },
}

impl UnitSpec {
    pub fn from(&self, a: &Aliases) -> String {
        self.from.render(a, 0)
    }

    pub fn relation(&self, prefix: &str) -> Option<&'static Relation> {
        self.relations.iter().find(|r| r.prefix == prefix)
    }
}

/// A relation template over the outer aliases `a` and the target's `n`.
pub fn expand_pair(template: &str, a: &Aliases, n: &Aliases) -> String {
    let inner = template
        .replace("{ns}", &n.s)
        .replace("{nm}", &n.m)
        .replace("{ncb}", &n.cb)
        .replace("{nb}", &n.b)
        .replace("{nt}", &n.t)
        .replace("{nr}", &n.row);
    a.expand(&inner)
}

/// How the store reads each unit; exhaustive, so a declared unit fails the
/// build until it has rows here.
pub fn spec(unit: Unit) -> &'static UnitSpec {
    match unit {
        Unit::Sessions => &SESSIONS,
        Unit::Documents => &DOCUMENTS,
        Unit::Messages => &MESSAGES,
        Unit::Blocks => &BLOCKS,
        Unit::ToolCalls => &TOOL_CALLS,
        Unit::Files => &FILES,
        Unit::Events => &EVENTS,
        Unit::Assertions => &ASSERTIONS,
        Unit::Links => &LINKS,
        Unit::Turns => &TURNS,
        Unit::Deliveries => &DELIVERIES,
        Unit::LiveIntervals => &LIVE_INTERVALS,
        Unit::Processes => &PROCESSES,
        Unit::Changes => &CHANGES,
    }
}

use ColKind::*;
use Expr::Template as T;

fn display_title(a: &Aliases) -> String {
    crate::session_sql::display_title(&a.s)
}

const fn rel(prefix: &'static str, target: Unit, via: Via) -> Relation {
    Relation {
        prefix,
        target,
        via,
    }
}

fn not_hidden(a: &Aliases) -> String {
    format!("NOT {}", fields::hidden(&a.s))
}

fn maybe_session_visible(a: &Aliases) -> String {
    format!("({s}.pk IS NULL OR NOT {})", fields::hidden(&a.s), s = a.s)
}

fn counts_as_action(a: &Aliases) -> String {
    fields::counts_as_action(&a.t)
}

fn current_assertion(a: &Aliases) -> String {
    fields::current_assertion(&a.row)
}

fn not_withdrawn(a: &Aliases) -> String {
    format!("NOT {}", fields::withdrawn_assertion(&a.row))
}

fn turns_from(a: &Aliases) -> String {
    format!(
        "{} {r} JOIN session {s} ON {s}.pk = {r}.session_pk",
        turns(),
        r = a.row,
        s = a.s
    )
}

/// Sessions another session spawned (subagents, sidechains, compactions)
/// stay out of a sessions list unless a condition asks about structure.
fn main_sessions(a: &Aliases) -> String {
    format!("{}.kind = '{}'", a.s, SessionKind::Main.as_str())
}

/// Adjacent counted calls use the same complete session order as `seq`.
pub fn adjacent_call(a: &Aliases, previous: bool) -> String {
    let cmp = if previous { "<" } else { ">" };
    let dir = if previous { " DESC" } else { "" };
    let order = ["qm.session_order", "qm.id", "qb.idx"];
    format!(
        "SELECT tq.pk FROM tool_call tq JOIN block qb ON qb.pk = tq.call_block_pk JOIN message qm ON qm.pk = qb.message_pk WHERE tq.session_pk = {}.session_pk AND {} AND {} AND ({}) {cmp} ({}) ORDER BY {} LIMIT 1",
        a.t,
        fields::own_messages("qm", None),
        fields::counts_as_action("tq"),
        order.join(", "),
        [
            format!("{}.session_order", a.m),
            format!("{}.id", a.m),
            format!("{}.idx", a.cb)
        ]
        .join(", "),
        order
            .iter()
            .map(|term| format!("{term}{dir}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Position in the complete counted-call scope, before query filtering.
pub fn call_position(a: &Aliases) -> String {
    format!(
        "(SELECT count(*) FROM tool_call pt JOIN block pb ON pb.pk=pt.call_block_pk JOIN message pm ON pm.pk=pb.message_pk WHERE pt.session_pk={}.session_pk AND {} AND {} AND (pm.session_order,pm.id,pb.idx) <= ({}.session_order,{}.id,{}.idx))",
        a.t,
        fields::own_messages("pm", None),
        fields::counts_as_action("pt"),
        a.m,
        a.m,
        a.cb
    )
}

const MESSAGE_RELATIONS: &[Relation] = &[rel("session", Unit::Sessions, Via::Same)];

static SESSIONS: UnitSpec = UnitSpec {
    from: Expr::Template("session {s}"),
    key: "{s}.pk",
    identity: &["{s}.id"],
    columns: &[
        ("id", Expr::Template("{s}.id"), Text),
        ("origin", T("{s}.origin"), Text),
        ("kind", T("{s}.kind"), Text),
        ("title", Expr::Fn(display_title), Text),
        ("started", T("{s}.started_at"), Time),
        ("last_active", T("{s}.last_activity_at"), Time),
        ("cwd", T("{s}.cwd"), Text),
        ("repo", T("{s}.repo"), Text),
    ],
    time: "{s}.last_activity_at",
    newest_first: true,
    session_order: false,
    since: Some("{s}.last_activity_at >= {v}"),
    until: Some("{s}.started_at < {v}"),
    session_pk: "{r}.session_pk",
    session_scope: None,
    search: Search::Hits {
        rows: sessions_hits,
    },
    relations: &[],
    parents: &[],
};

static DOCUMENTS: UnitSpec = UnitSpec {
    from: Expr::Template("document {r}"),
    key: "{r}.pk",
    identity: &["{r}.id"],
    columns: &[
        ("id", Expr::Template("{r}.id"), Text),
        ("origin", T("{r}.origin"), Text),
        ("purpose", T("{r}.purpose"), Text),
        ("title", T("{r}.title"), Text),
        ("source_time", T("{r}.source_time"), Time),
        ("authorship", T("{r}.authorship"), Text),
    ],
    time: "{r}.source_time",
    newest_first: true,
    session_order: false,
    since: Some("{r}.source_time >= {v}"),
    until: Some("{r}.source_time < {v}"),
    session_pk: "",
    session_scope: None,
    search: Search::Hits {
        rows: documents_hits,
    },
    relations: &[],
    parents: &[],
};

static MESSAGES: UnitSpec = UnitSpec {
    from: Expr::Template("message {m} JOIN session {s} ON {s}.pk = {m}.session_pk"),
    key: "{m}.pk",
    identity: &["{m}.id"],
    columns: &[
        ("id", Expr::Template("{m}.id"), Text),
        ("session", Expr::Template("{s}.id"), Text),
        ("role", T("{m}.role"), Text),
        ("written_by", T("{m}.written_by"), Text),
        ("at", T("{m}.at"), Time),
        ("model", T("{m}.model"), Text),
    ],
    time: "{m}.at",
    newest_first: false,
    session_order: true,
    since: Some("{m}.at >= {v}"),
    until: Some("{m}.at < {v}"),
    session_pk: "{m}.session_pk",
    session_scope: None,
    search: Search::Hits {
        rows: messages_hits,
    },
    relations: MESSAGE_RELATIONS,
    parents: &[],
};

static BLOCKS: UnitSpec = UnitSpec {
    from: Expr::Template(
        "block {b} JOIN message {m} ON {m}.pk = {b}.message_pk JOIN session {s} ON {s}.pk = {b}.session_pk",
    ),
    key: "{b}.pk",
    identity: &["{m}.id", "{b}.idx"],
    columns: &[
        ("session", Expr::Template("{s}.id"), Text),
        ("message", T("{m}.id"), Text),
        ("kind", T("{b}.kind"), Text),
        ("idx", T("{b}.idx"), Number),
        ("at", T("{m}.at"), Time),
    ],
    time: "{m}.at",
    newest_first: false,
    session_order: true,
    since: Some("{m}.at >= {v}"),
    until: Some("{m}.at < {v}"),
    session_pk: "{b}.session_pk",
    session_scope: None,
    search: Search::Hits { rows: blocks_hits },
    relations: &[
        rel("session", Unit::Sessions, Via::Same),
        rel("message", Unit::Messages, Via::Same),
        rel(
            "call",
            Unit::ToolCalls,
            Via::Exists {
                cond: "EXISTS (SELECT 1 FROM {from} WHERE ({nt}.call_block_pk = {b}.pk OR {nt}.result_block_pk = {b}.pk) AND {cond})",
                value: None,
            },
        ),
    ],
    parents: &[(Unit::Messages, "{b}.message_pk")],
};

static TOOL_CALLS: UnitSpec = UnitSpec {
    from: Expr::Template(
        "tool_call {t} JOIN block {cb} ON {cb}.pk = {t}.call_block_pk \
         JOIN message {m} ON {m}.pk = {cb}.message_pk JOIN session {s} ON {s}.pk = {t}.session_pk",
    ),
    key: "{t}.pk",
    identity: &["{m}.id", "{cb}.idx"],
    columns: &[
        ("session", Expr::Template("{s}.id"), Text),
        ("message", T("{m}.id"), Text),
        ("tool", T("{t}.tool"), Text),
        ("action", T("{t}.action"), Text),
        ("command", T("{t}.command"), Text),
        ("path", T("{t}.path"), Text),
        ("outcome", T("{t}.outcome"), Text),
        ("exit_code", T("{t}.exit_code"), Number),
        ("at", T("{t}.at"), Time),
    ],
    time: "{t}.at",
    newest_first: false,
    session_order: false,
    since: Some("({t}.at >= {v} OR {t}.result_block_pk IS NULL)"),
    until: Some("{t}.at < {v}"),
    session_pk: "{t}.session_pk",
    session_scope: None,
    search: Search::Hits {
        rows: tool_call_hits,
    },
    relations: &[
        rel("session", Unit::Sessions, Via::Same),
        rel("message", Unit::Messages, Via::Same),
        rel(
            "child",
            Unit::Sessions,
            Via::Exists {
                cond: "EXISTS (SELECT 1 FROM link lc JOIN {from} ON {ns}.pk = lc.child_pk \
                       WHERE lc.kind = 'subagent' AND lc.parent_id = {s}.id \
                       AND lc.spawning_call = {t}.call_id AND {cond})",
                value: None,
            },
        ),
        rel(
            "next",
            Unit::ToolCalls,
            Via::Exists {
                cond: "EXISTS (SELECT 1 FROM {from} WHERE {nt}.pk = ({next}) AND {cond})",
                value: Some("(SELECT {e} FROM {from} WHERE {nt}.pk = ({next}))"),
            },
        ),
        rel(
            "previous",
            Unit::ToolCalls,
            Via::Exists {
                cond: "EXISTS (SELECT 1 FROM {from} WHERE {nt}.pk = ({previous}) AND {cond})",
                value: Some("(SELECT {e} FROM {from} WHERE {nt}.pk = ({previous}))"),
            },
        ),
    ],
    parents: &[(Unit::Messages, "{cb}.message_pk")],
};

static FILES: UnitSpec = UnitSpec {
    from: Expr::Template(
        "(SELECT ft.session_pk AS session_pk, ft.path AS path, min(ft.pk) AS pk, \
         sum(ft.action = 'file_read') AS reads, sum(ft.action = 'file_write') AS writes, \
         sum(ft.action = 'file_edit') AS edits, min(ft.at) AS first_at, max(ft.at) AS last_at \
         FROM tool_call ft WHERE ft.path IS NOT NULL \
         AND ft.action IN ('file_read', 'file_write', 'file_edit') GROUP BY ft.session_pk, ft.path) {r} \
         JOIN session {s} ON {s}.pk = {r}.session_pk",
    ),
    key: "{r}.pk",
    identity: &["{s}.id", "{r}.path"],
    columns: &[
        ("session", Expr::Template("{s}.id"), Text),
        ("path", T("{r}.path"), Text),
        ("reads", T("{r}.reads"), Number),
        ("writes", T("{r}.writes"), Number),
        ("edits", T("{r}.edits"), Number),
        ("first_at", T("{r}.first_at"), Time),
        ("last_at", T("{r}.last_at"), Time),
    ],
    time: "{r}.first_at",
    newest_first: false,
    session_order: false,
    since: None,
    until: None,
    session_pk: "{r}.session_pk",
    session_scope: None,
    search: Search::None,
    relations: MESSAGE_RELATIONS,
    parents: &[],
};

static EVENTS: UnitSpec = UnitSpec {
    from: Expr::Template(
        "event {r} LEFT JOIN session {s} ON {s}.pk = {r}.session_pk JOIN input_ref {r}i ON {r}i.pk={r}.input_pk",
    ),
    key: "{r}.pk",
    identity: &["{r}.id"],
    columns: &[
        ("id", T("{r}.id"), Text),
        ("session", Expr::Template("{s}.id"), Text),
        ("input", T("{r}i.id"), Text),
        ("source", T("{r}i.format"), Text),
        ("kind", T("{r}.kind"), Text),
        ("at", T("{r}.at"), Time),
        (
            "data",
            T("pl_block({r}.data)"),
            crate::query::compile::ColKind::Json,
        ),
    ],
    time: "{r}.at",
    newest_first: false,
    session_order: false,
    since: Some("{r}.at >= {v}"),
    until: Some("{r}.at < {v}"),
    session_pk: "{r}.session_pk",
    session_scope: None,
    search: Search::None,
    relations: MESSAGE_RELATIONS,
    parents: &[],
};

static ASSERTIONS: UnitSpec = UnitSpec {
    from: Expr::Template("source_assertion {r}"),
    key: "{r}.rowid",
    identity: &["{r}.id", "{r}.root"],
    columns: &[
        ("id", T("{r}.id"), Text),
        (
            "reference",
            Expr::Template("pl_assertion_ref({r}.root,{r}.id)"),
            Text,
        ),
        ("kind", T("{r}.kind"), Text),
        ("key", T("{r}.key"), Text),
        (
            "value",
            T("{r}.value"),
            crate::query::compile::ColKind::Json,
        ),
        ("author", T("{r}.author"), Text),
        ("created_at", T("{r}.created_at"), Time),
        ("root", T("{r}.root"), Text),
        (
            "about",
            T(
                "(SELECT group_concat(ref, ' ') FROM (SELECT nt.ref FROM source_link nt WHERE nt.record = {r}.id AND nt.role = 'about' ORDER BY nt.ref))",
            ),
            Text,
        ),
    ],
    time: "{r}.created_at",
    newest_first: true,
    session_order: false,
    since: Some("{r}.created_at >= {v}"),
    until: Some("{r}.created_at < {v}"),
    session_pk: "{r}.session_pk",
    session_scope: Some(
        "EXISTS (SELECT 1 FROM source_link snt JOIN session ss ON 'session:' || ss.id = snt.ref \
         WHERE snt.record = {r}.id AND snt.role = 'about' AND ss.pk IN ({sessions}))",
    ),
    search: Search::Substring("{r}.value"),
    relations: &[],
    parents: &[],
};

static LINKS: UnitSpec = UnitSpec {
    from: Expr::Template(
        "link {r} LEFT JOIN session {r}p ON {r}p.id = {r}.parent_id LEFT JOIN session {r}c ON {r}c.pk = {r}.child_pk",
    ),
    key: "{r}.pk",
    identity: &["{r}.parent_id", "{r}.child_id", "{r}.kind", "{r}.how"],
    columns: &[
        ("parent", Expr::Template("{r}.parent_id"), Text),
        ("child", Expr::Template("{r}.child_id"), Text),
        ("kind", T("{r}.kind"), Text),
        ("how", T("{r}.how"), Text),
        ("evidence", T("{r}.evidence"), Text),
        ("pending", T("({r}p.pk IS NULL)"), Bool),
    ],
    time: "{r}p.last_activity_at",
    newest_first: false,
    session_order: false,
    since: None,
    until: None,
    session_pk: "{r}.session_pk",
    session_scope: Some("({r}.child_pk IN ({sessions}) OR {r}p.pk IN ({sessions}))"),
    search: Search::None,
    relations: &[
        rel("parent", Unit::Sessions, Via::Session("{r}p")),
        rel("child", Unit::Sessions, Via::Session("{r}c")),
    ],
    parents: &[],
};

static TURNS: UnitSpec = UnitSpec {
    from: Expr::Fn(turns_from),
    key: "{r}.pk",
    identity: &["{r}.id"],
    columns: &[
        ("id", T("{r}.id"), Text),
        ("session", Expr::Template("{s}.id"), Text),
        ("start", T("{r}.start"), Time),
        ("end", T("{r}.end"), Time),
        ("burst", T("{r}.burst"), Number),
        ("prompts", T("{r}.prompts"), Text),
        ("slash_command", T("{r}.slash_command"), Bool),
        ("answer", T("{r}.answer"), Text),
        ("attempts", T("{r}.attempts"), Number),
        ("path", T("{r}.path"), Text),
        ("latency", T("{r}.latency"), Duration),
        ("tool_calls", T("{r}.tool_calls"), Number),
        ("errors", T("{r}.errors"), Number),
        ("input_tokens", T("{r}.input_tokens"), Number),
        ("output_tokens", T("{r}.output_tokens"), Number),
        ("cost", T("{r}.cost_nusd"), Money),
        ("abandoned", T("{r}.abandoned"), Bool),
    ],
    time: "{r}.start",
    newest_first: false,
    session_order: false,
    since: None,
    until: None,
    session_pk: "{r}.session_pk",
    session_scope: None,
    search: Search::None,
    relations: MESSAGE_RELATIONS,
    parents: &[],
};

static DELIVERIES: UnitSpec = UnitSpec {
    from: Expr::Template("source_delivery {r} LEFT JOIN session {s} ON {s}.id = {r}.recipient_id"),
    key: "{r}.rowid",
    identity: &["{r}.id"],
    columns: &[
        ("id", T("{r}.id"), Text),
        ("recipient", T("{r}.recipient_id"), Text),
        ("event", T("{r}.event"), Text),
        (
            "outcome",
            Expr::Fn(crate::query::fields::deliveries_outcome),
            Text,
        ),
        ("at", T("{r}.at"), Time),
        ("bytes", T("{r}.bytes"), Number),
    ],
    time: "{r}.at",
    newest_first: false,
    session_order: false,
    since: None,
    until: None,
    session_pk: "{s}.pk",
    session_scope: None,
    search: Search::None,
    relations: &[
        rel("session", Unit::Sessions, Via::Same),
        rel("recipient", Unit::Sessions, Via::Same),
    ],
    parents: &[],
};

static LIVE_INTERVALS: UnitSpec = UnitSpec {
    from: Expr::Template("live_interval {r} JOIN session {s} ON {s}.pk = {r}.session_pk"),
    key: "{r}.rowid",
    identity: &["{s}.id", "{r}.aspect", "{r}.start_at"],
    columns: &[
        ("session", Expr::Template("{s}.id"), Text),
        ("aspect", T("{r}.aspect"), Text),
        ("value", T("{r}.value"), Text),
        ("start", T("{r}.start_at"), Time),
        ("end", T("{r}.end_at"), Time),
        ("duration_ms", T("({r}.end_at - {r}.start_at)"), Number),
        ("closed_by", T("{r}.closed_by"), Text),
    ],
    time: "{r}.start_at",
    newest_first: false,
    session_order: false,
    since: Some("{r}.end_at >= {v}"),
    until: Some("{r}.start_at < {v}"),
    session_pk: "{r}.session_pk",
    session_scope: None,
    search: Search::None,
    relations: MESSAGE_RELATIONS,
    parents: &[],
};

static PROCESSES: UnitSpec = UnitSpec {
    from: Expr::Template("process {r} JOIN session {s} ON {s}.pk = {r}.session_pk"),
    key: "{r}.rowid",
    identity: &["{s}.id", "{r}.boot_id", "{r}.pid", "{r}.start"],
    columns: &[
        ("session", Expr::Template("{s}.id"), Text),
        ("boot_id", T("{r}.boot_id"), Text),
        ("pid", T("{r}.pid"), Number),
        ("start", T("{r}.start"), Number),
        ("parent_pid", T("{r}.parent_pid"), Number),
        ("comm", T("{r}.comm"), Text),
        ("argv", T("{r}.argv"), Text),
        ("cwd", T("{r}.cwd"), Text),
        ("started_at", T("{r}.started_at"), Time),
        ("ended_at", T("{r}.ended_at"), Time),
        ("role", T("{r}.role"), Text),
        (
            "tool_call",
            Expr::Fn(crate::query::fields::processes_tool_call),
            Text,
        ),
    ],
    time: "{r}.started_at",
    newest_first: false,
    session_order: false,
    since: Some("{r}.started_at >= {v}"),
    until: Some("{r}.started_at < {v}"),
    session_pk: "{r}.session_pk",
    session_scope: None,
    search: Search::None,
    relations: MESSAGE_RELATIONS,
    parents: &[],
};

static CHANGES: UnitSpec = UnitSpec {
    from: Expr::Template(
        "(SELECT 'session' AS kind, cs.id AS ref, cs.pk AS session_pk, \
           MAX(cs.changed, COALESCE(cs.live_changed, 0), COALESCE(cs.derived_changed, 0)) AS seq, \
           cs.changed AS changed, cs.live_changed AS live_changed, cs.derived_changed AS derived_changed, \
           NULL AS assertion_kind FROM session cs \
         UNION ALL SELECT 'removed', rs.session_id, NULL, rs.changed, rs.changed, NULL, NULL, NULL \
           FROM removed_session rs \
         UNION ALL SELECT 'assertion', cn.id, NULL, cn.commit_seq, NULL, NULL, NULL, cn.kind FROM source_assertion cn \
) {r}",
    ),
    key: "({r}.kind || ':' || {r}.ref)",
    identity: &["{r}.kind", "{r}.ref"],
    columns: &[
        ("kind", T("{r}.kind"), Text),
        ("ref", T("{r}.ref"), Text),
        ("seq", T("{r}.seq"), Number),
        ("live_changed", T("{r}.live_changed"), Number),
        ("assertion_kind", T("{r}.assertion_kind"), Text),
    ],
    time: "{r}.seq",
    newest_first: false,
    session_order: false,
    since: None,
    until: None,
    session_pk: "{r}.session_pk",
    session_scope: Some("({r}.kind = 'session' AND {r}.session_pk IN ({sessions}))"),
    search: Search::None,
    relations: &[],
    parents: &[],
};

/// One row per turn: along the active path (or file
/// order when it is unknown), a turn starts at a human prompt that does not
/// follow another prompt, or at a typed command, and runs until the next
/// such start; a session with no prompt is one turn. Duplicates are left out.
/// Every turn of every session, as a subquery: the `turns` unit's rows.
pub fn turns() -> String {
    format!(
        "(SELECT g.*, \
     (SELECT count(*) FROM message ax WHERE ax.parent_id = g.last_prompt \
        AND ax.role = 'assistant' AND ax.written_by = 'model') AS attempts, \
     (SELECT min(lx.at) FROM message lx WHERE lx.session_pk = g.session_pk \
        AND lx.role = 'assistant' AND lx.written_by = 'model' AND lx.at >= g.last_prompt_at) - g.last_prompt_at AS latency, \
     (g.answer IS NULL AND g.turn = max(g.turn) OVER (PARTITION BY g.session_pk)) AS abandoned \
   FROM (SELECT tm.session_pk AS session_pk, tm.turn AS turn, min(tm.pk) AS pk, \
       substr(min(CASE WHEN tm.hp THEN printf('%012d ', tm.ord) || tm.id END), 14) AS first_prompt, \
       substr(min(printf('%012d ', tm.ord) || tm.id), 14) AS first_id, \
       COALESCE(substr(min(CASE WHEN tm.hp THEN printf('%012d ', tm.ord) || tm.id END), 14), \
                substr(min(printf('%012d ', tm.ord) || tm.id), 14)) AS id, \
       substr(max(CASE WHEN tm.hp THEN printf('%012d ', tm.ord) || tm.id END), 14) AS last_prompt, \
       max(CASE WHEN tm.hp THEN tm.at END) AS last_prompt_at, \
       min(tm.at) AS start, max(tm.at) AS \"end\", \
       sum(tm.hp) AS burst, \
       json_group_array(tm.id ORDER BY tm.ord) FILTER (WHERE tm.hp) AS prompts, \
       substr(min(printf('%012d ', tm.ord) || COALESCE(tm.prompt_kind, '')), 14) = 'command' AS slash_command, \
       substr(max(CASE WHEN tm.role = 'assistant' AND tm.written_by = 'model' THEN printf('%012d ', tm.ord) || tm.id END), 14) AS answer, \
       CASE WHEN min(COALESCE(tm.on_active_path, -1)) < 0 THEN 'unknown' ELSE 'active' END AS path, \
       sum(tm.tc) AS tool_calls, sum(tm.te) AS errors, \
       sum(tm.input_tokens) AS input_tokens, sum(tm.output_tokens) AS output_tokens, \
       sum(tm.cost_nusd) AS cost_nusd \
     FROM (SELECT o.*, \
         sum(CASE WHEN o.hp AND (o.prompt_kind = 'command' OR NOT o.prev_hp) THEN 1 ELSE 0 END) \
           OVER (PARTITION BY o.session_pk ORDER BY o.session_order, o.id \
                 ROWS UNBOUNDED PRECEDING) AS turn, \
         row_number() OVER (PARTITION BY o.session_pk ORDER BY o.session_order, o.id) AS ord \
       FROM (SELECT m.pk, m.id, m.session_pk, m.session_order, m.role, m.written_by, m.prompt_kind, \
             m.at, m.on_active_path, m.input_tokens, m.output_tokens, m.cost_nusd, \
             (m.written_by = 'human') AS hp, \
             COALESCE(lag(m.written_by = 'human') OVER (PARTITION BY m.session_pk \
               ORDER BY m.session_order, m.id), 0) AS prev_hp, \
             (SELECT count(*) FROM tool_call c JOIN block cb ON cb.pk = c.call_block_pk \
               WHERE cb.message_pk = m.pk) AS tc, \
             (SELECT count(*) FROM tool_call c JOIN block cb ON cb.pk = c.call_block_pk \
               WHERE cb.message_pk = m.pk AND c.outcome = 'error') AS te \
           FROM message m WHERE {own_messages} \
             AND m.written_by IN ('human', 'model')) o) tm \
     GROUP BY tm.session_pk, tm.turn) g)",
        own_messages = fields::own_messages("m", None)
    )
}

#[cfg(test)]
mod tests;

/// The SQL of a default filter over a unit's aliases.
pub fn default_sql(id: DefaultId, a: &Aliases) -> String {
    match id {
        DefaultId::MainSessions => main_sessions(a),
        DefaultId::NotStub => a.expand("{s}.stub = 0"),
        DefaultId::SessionsVisible | DefaultId::Visible => not_hidden(a),
        DefaultId::NotDuplicate => a.expand("{m}.duplicate_of IS NULL"),
        DefaultId::OnActivePath => a.expand("COALESCE({m}.on_active_path, 1) = 1"),
        DefaultId::CountedCalls => counts_as_action(a),
        DefaultId::NotWithdrawal => a.expand("{r}.kind <> 'withdrawal'"),
        DefaultId::CurrentAssertion => current_assertion(a),
        DefaultId::NotWithdrawn => not_withdrawn(a),
        DefaultId::VisibleIfSession => maybe_session_visible(a),
    }
}
