//! The unit registry: everything about a query unit, declared once. The
//! compiler reads a unit's table, key, default columns, order, defaults,
//! time window and relations from here; the fields each unit accepts are in
//! [`crate::fields::FIELDS`]. Adding a unit is adding one [`UnitSpec`].
//!
//! Templates name the unit's aliases as `{s}` session, `{m}` message, `{b}`
//! block, `{t}` tool call, `{cb}` its call block and `{r}` the unit's own
//! row (see [`Aliases`]). A relation's templates name the related unit's
//! aliases with an `n` prefix (`{ns}`, `{nt}`, `{nr}`, …), its FROM clause as
//! `{from}` and the condition on it as `{cond}`.

use crate::ast::Unit;
use crate::compile::ColKind;
use crate::fields::{self, Aliases, Expr};

pub struct UnitSpec {
    pub unit: Unit,
    pub name: &'static str,
    /// The singular form `exists` and `seq` use.
    pub singular: &'static str,
    pub from: Expr,
    /// The row key, `_pk` in the output.
    pub key: &'static str,
    /// The default columns: name, SQL template, kind.
    pub columns: &'static [(&'static str, Expr, ColKind)],
    /// The time rows are ordered by.
    pub time: &'static str,
    pub newest_first: bool,
    /// Rows at the same time keep session order (message order, then seq).
    pub session_order: bool,
    /// `since:` and `until:`: a condition on `{v}`, the resolved instant.
    pub since: Option<&'static str>,
    pub until: Option<&'static str>,
    /// The row names one session through the `s` alias, so session fields
    /// and `session.FIELD` apply to it.
    pub in_session: bool,
    /// The expression naming the row's session key (`session.pk`).
    pub session_pk: &'static str,
    /// For a unit not in one session: its rows related to the sessions
    /// `{sessions}` selects.
    pub session_scope: Option<&'static str>,
    /// Plain words search this unit's text.
    pub text: bool,
    /// Filters every query of the unit applies (spec query.md, Defaults).
    pub defaults: &'static [Default],
    pub relations: &'static [Relation],
}

/// A default filter, dropped when a top-level condition names one of
/// `unless`, or (with `unless_reference`) when the query names rows by
/// reference.
pub struct Default {
    pub sql: Expr,
    pub unless: &'static [&'static str],
    pub unless_reference: bool,
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

pub fn spec(unit: Unit) -> &'static UnitSpec {
    UNITS
        .iter()
        .find(|u| u.unit == unit)
        .expect("every unit is declared")
}

use ColKind::*;
use Expr::Template as T;

const fn d(sql: Expr, unless: &'static [&'static str], unless_reference: bool) -> Default {
    Default {
        sql,
        unless,
        unless_reference,
    }
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

fn not_excised(a: &Aliases) -> String {
    format!("NOT {}", fields::excised(&a.s))
}

fn maybe_session_visible(a: &Aliases) -> String {
    format!(
        "({s}.pk IS NULL OR (NOT {} AND NOT {}))",
        fields::hidden(&a.s),
        fields::excised(&a.s),
        s = a.s
    )
}

fn counted_call(a: &Aliases) -> String {
    fields::counted_call(&a.t)
}

fn current_note(a: &Aliases) -> String {
    fields::current_note(&a.row)
}

fn not_withdrawn(a: &Aliases) -> String {
    format!("NOT {}", fields::withdrawn_note(&a.row))
}

fn work_goal(a: &Aliases) -> String {
    fields::work_goal(&a.row)
}

fn turns_from(a: &Aliases) -> String {
    format!(
        "{TURNS} {r} JOIN session {s} ON {s}.pk = {r}.session_pk",
        r = a.row,
        s = a.s
    )
}

/// Hidden and excised sessions stay out (spec query.md, Defaults).
const SESSION_VISIBLE: &[Default] = &[
    d(Expr::Fn(not_hidden), &["hidden"], true),
    d(Expr::Fn(not_excised), &[], false),
];

/// Own, active-path messages, and the session defaults.
const MESSAGE_DEFAULTS: &[Default] = &[
    d(
        Expr::Template("{m}.inherited_from IS NULL"),
        &["inherited"],
        true,
    ),
    d(
        Expr::Template("COALESCE({m}.on_active_path, 1) = 1"),
        &["active"],
        true,
    ),
    d(Expr::Fn(not_hidden), &["hidden"], true),
    d(Expr::Fn(not_excised), &[], false),
];

/// The session's next counted tool call after `{t}`, in session order.
pub const NEXT_CALL: &str = "SELECT tq.pk FROM tool_call tq JOIN block qb ON qb.pk = tq.call_block_pk \
     JOIN message qm ON qm.pk = qb.message_pk WHERE tq.session_pk = {t}.session_pk \
     AND (qm.order_input_pk, qm.seq, qb.idx) > ({m}.order_input_pk, {m}.seq, {cb}.idx) \
     ORDER BY qm.order_input_pk, qm.seq, qb.idx LIMIT 1";

const MESSAGE_RELATIONS: &[Relation] = &[rel("session", Unit::Sessions, Via::Same)];

pub static UNITS: &[UnitSpec] = &[
    UnitSpec {
        unit: Unit::Sessions,
        name: "sessions",
        singular: "session",
        from: Expr::Template("session {s}"),
        key: "{s}.pk",
        columns: &[
            ("id", T("{s}.id"), Text),
            ("origin", T("{s}.origin"), Text),
            ("kind", T("{s}.kind"), Text),
            ("title", T("{s}.title"), Text),
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
        in_session: false,
        session_pk: "{r}.session_pk",
        session_scope: None,
        text: true,
        defaults: &[
            d(
                Expr::Template("{s}.kind <> 'subagent'"),
                &[
                    "kind",
                    "parent",
                    "root",
                    "depth",
                    "lineage",
                    "link",
                    "team",
                    "agent_name",
                ],
                true,
            ),
            d(Expr::Template("{s}.stub = 0"), &["source"], false),
            d(Expr::Fn(not_hidden), &["hidden"], true),
            d(Expr::Fn(not_excised), &[], false),
        ],
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Messages,
        name: "messages",
        singular: "message",
        from: Expr::Template("message {m} JOIN session {s} ON {s}.pk = {m}.session_pk"),
        key: "{m}.pk",
        columns: &[
            ("id", T("{m}.id"), Text),
            ("session", T("{s}.id"), Text),
            ("role", T("{m}.role"), Text),
            ("written_by", T("{m}.written_by"), Text),
            ("at", T("{m}.at"), Time),
            ("model", T("{m}.model"), Text),
            ("inherited", T("({m}.inherited_from IS NOT NULL)"), Bool),
        ],
        time: "{m}.at",
        newest_first: false,
        session_order: true,
        since: Some("{m}.at >= {v}"),
        until: Some("{m}.at < {v}"),
        in_session: true,
        session_pk: "{m}.session_pk",
        session_scope: None,
        text: true,
        defaults: MESSAGE_DEFAULTS,
        relations: MESSAGE_RELATIONS,
    },
    UnitSpec {
        unit: Unit::Blocks,
        name: "blocks",
        singular: "block",
        from: Expr::Template(
            "block {b} JOIN message {m} ON {m}.pk = {b}.message_pk JOIN session {s} ON {s}.pk = {b}.session_pk",
        ),
        key: "{b}.pk",
        columns: &[
            ("session", T("{s}.id"), Text),
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
        in_session: true,
        session_pk: "{b}.session_pk",
        session_scope: None,
        text: true,
        defaults: MESSAGE_DEFAULTS,
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
    },
    UnitSpec {
        unit: Unit::ToolCalls,
        name: "tool_calls",
        singular: "tool_call",
        from: Expr::Template(
            "tool_call {t} JOIN block {cb} ON {cb}.pk = {t}.call_block_pk \
             JOIN message {m} ON {m}.pk = {cb}.message_pk JOIN session {s} ON {s}.pk = {t}.session_pk",
        ),
        key: "{t}.pk",
        columns: &[
            ("session", T("{s}.id"), Text),
            ("message", T("{m}.id"), Text),
            ("tool", T("{t}.tool"), Text),
            ("action", T("{t}.action"), Text),
            ("command", T("{t}.command"), Text),
            ("path", T("{t}.path"), Text),
            ("outcome", T("{t}.outcome"), Text),
            ("exit_code", T("{t}.exit_code"), Number),
            ("at", T("{t}.at"), Time),
            ("duration", T("{t}.duration_ms"), Duration),
        ],
        time: "{t}.at",
        newest_first: false,
        session_order: false,
        since: Some(
            "(COALESCE({t}.at + {t}.duration_ms, {t}.at) >= {v} OR {t}.result_block_pk IS NULL)",
        ),
        until: Some("{t}.at < {v}"),
        in_session: true,
        session_pk: "{t}.session_pk",
        session_scope: None,
        text: false,
        defaults: &[
            d(
                Expr::Template("{m}.inherited_from IS NULL"),
                &["inherited"],
                true,
            ),
            d(
                Expr::Template("COALESCE({m}.on_active_path, 1) = 1"),
                &["active"],
                true,
            ),
            d(Expr::Fn(counted_call), &["action"], false),
            d(Expr::Fn(not_hidden), &["hidden"], true),
            d(Expr::Fn(not_excised), &[], false),
        ],
        relations: &[
            rel("session", Unit::Sessions, Via::Same),
            rel("message", Unit::Messages, Via::Same),
            rel(
                "child",
                Unit::Sessions,
                Via::Exists {
                    cond: "EXISTS (SELECT 1 FROM link lc JOIN {from} ON {ns}.pk = lc.child_pk \
                           WHERE lc.kind = 'subagent' AND lc.parent_id = {s}.id \
                           AND (lc.spawning_call = {t}.call_id OR lc.spawned_by = {cb}.pk) AND {cond})",
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
        ],
    },
    UnitSpec {
        unit: Unit::Files,
        name: "files",
        singular: "file",
        from: Expr::Template(
            "(SELECT ft.session_pk AS session_pk, ft.path AS path, min(ft.pk) AS pk, \
             sum(ft.action = 'file_read') AS reads, sum(ft.action = 'file_write') AS writes, \
             sum(ft.action = 'file_edit') AS edits, min(ft.at) AS first_at, max(ft.at) AS last_at \
             FROM tool_call ft WHERE ft.path IS NOT NULL \
             AND ft.action IN ('file_read', 'file_write', 'file_edit') GROUP BY ft.session_pk, ft.path) {r} \
             JOIN session {s} ON {s}.pk = {r}.session_pk",
        ),
        key: "{r}.pk",
        columns: &[
            ("session", T("{s}.id"), Text),
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
        in_session: true,
        session_pk: "{r}.session_pk",
        session_scope: None,
        text: false,
        defaults: SESSION_VISIBLE,
        relations: MESSAGE_RELATIONS,
    },
    UnitSpec {
        unit: Unit::Events,
        name: "events",
        singular: "event",
        from: Expr::Template("event {r} JOIN session {s} ON {s}.pk = {r}.session_pk"),
        key: "{r}.pk",
        columns: &[
            ("id", T("{r}.id"), Text),
            ("session", T("{s}.id"), Text),
            ("kind", T("{r}.kind"), Text),
            ("at", T("{r}.at"), Time),
        ],
        time: "{r}.at",
        newest_first: false,
        session_order: false,
        since: Some("{r}.at >= {v}"),
        until: Some("{r}.at < {v}"),
        in_session: true,
        session_pk: "{r}.session_pk",
        session_scope: None,
        text: false,
        defaults: SESSION_VISIBLE,
        relations: MESSAGE_RELATIONS,
    },
    UnitSpec {
        unit: Unit::Notes,
        name: "notes",
        singular: "note",
        from: Expr::Template("a.note {r}"),
        key: "{r}.rowid",
        columns: &[
            ("id", T("{r}.id"), Text),
            ("kind", T("{r}.kind"), Text),
            ("key", T("{r}.key"), Text),
            ("value", T("{r}.value"), Text),
            ("author", T("{r}.author"), Text),
            ("created_at", T("{r}.created_at"), Time),
            ("root", T("{r}.root"), Text),
            (
                "target",
                T(
                    "(SELECT group_concat(nt.ref, ' ') FROM a.note_target nt WHERE nt.note_id = {r}.id)",
                ),
                Text,
            ),
        ],
        time: "{r}.created_at",
        newest_first: true,
        session_order: false,
        since: Some("{r}.created_at >= {v}"),
        until: Some("{r}.created_at < {v}"),
        in_session: false,
        session_pk: "{r}.session_pk",
        session_scope: Some(
            "EXISTS (SELECT 1 FROM a.note_target snt JOIN session ss ON 'session:' || ss.id = snt.ref \
             WHERE snt.note_id = {r}.id AND ss.pk IN ({sessions}))",
        ),
        text: true,
        defaults: &[
            d(
                Expr::Fn(current_note),
                &["current", "root", "withdrawn"],
                false,
            ),
            d(Expr::Fn(not_withdrawn), &["withdrawn"], false),
        ],
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Links,
        name: "links",
        singular: "link",
        from: Expr::Template(
            "link {r} LEFT JOIN session {r}p ON {r}p.id = {r}.parent_id LEFT JOIN session {r}c ON {r}c.pk = {r}.child_pk",
        ),
        key: "{r}.pk",
        columns: &[
            ("parent", T("{r}.parent_id"), Text),
            ("child", T("{r}.child_id"), Text),
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
        in_session: false,
        session_pk: "{r}.session_pk",
        session_scope: Some("({r}.child_pk IN ({sessions}) OR {r}p.pk IN ({sessions}))"),
        text: false,
        defaults: &[],
        relations: &[
            rel("parent", Unit::Sessions, Via::Session("{r}p")),
            rel("child", Unit::Sessions, Via::Session("{r}c")),
        ],
    },
    UnitSpec {
        unit: Unit::Turns,
        name: "turns",
        singular: "turn",
        from: Expr::Fn(turns_from),
        key: "{r}.pk",
        columns: &[
            ("id", T("{r}.id"), Text),
            ("session", T("{s}.id"), Text),
            ("start", T("{r}.start"), Time),
            ("end", T("{r}.end"), Time),
            ("burst", T("{r}.burst"), Number),
            ("prompts", T("{r}.prompts"), Text),
            ("slash_command", T("{r}.slash_command"), Number),
            ("answer", T("{r}.answer"), Text),
            ("attempts", T("{r}.attempts"), Number),
            ("path", T("{r}.path"), Text),
            ("latency", T("{r}.latency"), Number),
            ("tool_calls", T("{r}.tool_calls"), Number),
            ("errors", T("{r}.errors"), Number),
            ("input_tokens", T("{r}.input_tokens"), Number),
            ("output_tokens", T("{r}.output_tokens"), Number),
            ("cost", T("{r}.cost_nusd"), Number),
            ("abandoned", T("{r}.abandoned"), Number),
        ],
        time: "{r}.start",
        newest_first: false,
        session_order: false,
        since: None,
        until: None,
        in_session: true,
        session_pk: "{r}.session_pk",
        session_scope: None,
        text: false,
        defaults: SESSION_VISIBLE,
        relations: MESSAGE_RELATIONS,
    },
    UnitSpec {
        unit: Unit::Runs,
        name: "runs",
        singular: "run",
        from: Expr::Template("run {r}"),
        key: "{r}.pk",
        columns: &[
            ("id", T("{r}.id"), Text),
            ("kind", T("{r}.kind"), Text),
            ("state", T("{r}.state"), Text),
            ("instruction", T("{r}.instruction"), Text),
            ("started", T("{r}.started_at"), Time),
            ("ended", T("{r}.ended_at"), Time),
        ],
        time: "{r}.started_at",
        newest_first: false,
        session_order: false,
        since: None,
        until: None,
        in_session: false,
        session_pk: "{r}.session_pk",
        session_scope: Some(
            "EXISTS (SELECT 1 FROM run_session rs WHERE rs.run_pk = {r}.pk AND rs.session_pk IN ({sessions}))",
        ),
        text: false,
        defaults: &[],
        relations: &[
            rel(
                "session",
                Unit::Sessions,
                Via::Exists {
                    cond: "EXISTS (SELECT 1 FROM run_session rs JOIN {from} ON {ns}.pk = rs.session_pk \
                           WHERE rs.run_pk = {r}.pk AND {cond})",
                    value: None,
                },
            ),
            rel(
                "task",
                Unit::Runs,
                Via::Exists {
                    cond: "EXISTS (SELECT 1 FROM {from} WHERE {nr}.id = {r}.task_id AND {cond})",
                    value: None,
                },
            ),
        ],
    },
    UnitSpec {
        unit: Unit::Deliveries,
        name: "deliveries",
        singular: "delivery",
        from: Expr::Template("a.delivery {r} LEFT JOIN session {s} ON {s}.id = {r}.recipient_id"),
        key: "{r}.rowid",
        columns: &[
            ("id", T("{r}.id"), Text),
            ("recipient", T("{r}.recipient_id"), Text),
            ("event", T("{r}.event"), Text),
            ("outcome", Expr::Fn(crate::fields::deliveries_outcome), Text),
            ("at", T("{r}.at"), Time),
            ("bytes", T("{r}.bytes"), Number),
        ],
        time: "{r}.at",
        newest_first: false,
        session_order: false,
        since: None,
        until: None,
        in_session: true,
        session_pk: "{s}.pk",
        session_scope: None,
        text: false,
        defaults: &[d(Expr::Fn(maybe_session_visible), &[], false)],
        relations: &[
            rel("session", Unit::Sessions, Via::Same),
            rel("recipient", Unit::Sessions, Via::Same),
        ],
    },
    UnitSpec {
        unit: Unit::Processes,
        name: "processes",
        singular: "process",
        from: Expr::Template("process {r}"),
        key: "{r}.pk",
        columns: &[
            ("role", T("{r}.role"), Text),
            ("comm", T("{r}.comm"), Text),
            ("pid", T("{r}.pid"), Number),
            ("started_at", T("{r}.started_at"), Time),
            ("ended_at", T("{r}.ended_at"), Time),
        ],
        time: "{r}.started_at",
        newest_first: false,
        session_order: false,
        since: None,
        until: None,
        in_session: false,
        session_pk: "{r}.session_pk",
        session_scope: Some(
            "EXISTS (SELECT 1 FROM process_session ps WHERE ps.process_pk = {r}.pk AND ps.session_pk IN ({sessions}))",
        ),
        text: false,
        defaults: &[],
        relations: &[
            rel(
                "session",
                Unit::Sessions,
                Via::Exists {
                    cond: "EXISTS (SELECT 1 FROM process_session rs JOIN {from} ON {ns}.pk = rs.session_pk \
                           WHERE rs.process_pk = {r}.pk AND {cond})",
                    value: None,
                },
            ),
            rel(
                "tool_call",
                Unit::ToolCalls,
                Via::Exists {
                    cond: "EXISTS (SELECT 1 FROM {from} WHERE {nt}.pk = {r}.tool_call_pk AND {cond})",
                    value: None,
                },
            ),
        ],
    },
    UnitSpec {
        unit: Unit::Work,
        name: "work",
        singular: "work",
        from: Expr::Template("work_node {r} LEFT JOIN session {s} ON {s}.pk = {r}.session_pk"),
        key: "{r}.pk",
        columns: &[
            ("kind", T("{r}.kind"), Text),
            ("title", T("{r}.title"), Text),
            ("state", T("{r}.state"), Text),
            ("session", T("{s}.id"), Text),
            ("at", T("{r}.at"), Time),
            ("goal", Expr::Fn(work_goal), Text),
        ],
        time: "{r}.at",
        newest_first: false,
        session_order: false,
        since: None,
        until: None,
        in_session: true,
        session_pk: "{r}.session_pk",
        session_scope: Some(
            "({r}.session_pk IN ({sessions}) OR EXISTS (SELECT 1 FROM work_edge we \
             JOIN work_node wa ON wa.pk = we.from_pk \
             WHERE we.to_pk = {r}.pk AND wa.session_pk IN ({sessions})))",
        ),
        text: false,
        defaults: &[d(Expr::Fn(maybe_session_visible), &[], false)],
        relations: &[rel("session", Unit::Sessions, Via::Same)],
    },
    UnitSpec {
        unit: Unit::Changes,
        name: "changes",
        singular: "change",
        from: Expr::Template(
            "(SELECT 'session' AS kind, cs.id AS ref, cs.pk AS session_pk, \
               MAX(cs.changed, COALESCE(cs.live_changed, 0)) AS seq, cs.changed AS changed, \
               cs.live_changed AS live_changed, NULL AS note_kind FROM session cs \
             UNION ALL SELECT 'removed', rs.session_id, NULL, rs.changed, rs.changed, NULL, NULL \
               FROM removed_session rs \
             UNION ALL SELECT 'note', cn.id, NULL, cn.commit_seq, NULL, NULL, cn.kind FROM a.note cn) {r}",
        ),
        key: "({r}.kind || ':' || {r}.ref)",
        columns: &[
            ("kind", T("{r}.kind"), Text),
            ("ref", T("{r}.ref"), Text),
            ("seq", T("{r}.seq"), Number),
            ("live_changed", T("{r}.live_changed"), Number),
            ("note_kind", T("{r}.note_kind"), Text),
        ],
        time: "{r}.seq",
        newest_first: false,
        session_order: false,
        since: None,
        until: None,
        in_session: false,
        session_pk: "{r}.session_pk",
        session_scope: Some("({r}.kind = 'session' AND {r}.session_pk IN ({sessions}))"),
        text: false,
        defaults: &[],
        relations: &[],
    },
];

/// One row per turn (spec query.md, turns): along the active path (or file
/// order when it is unknown), a turn starts at a human prompt that does not
/// follow another prompt, or at a typed command, and runs until the next
/// such start; a session with no prompt is one turn. Copies are counted in
/// the session that originated them, so inherited messages are left out.
const TURNS: &str = "(SELECT g.*, \
     (SELECT count(*) FROM message ax WHERE ax.parent_id = g.last_prompt \
        AND ax.written_by = 'model') AS attempts, \
     (SELECT min(lx.at) FROM message lx WHERE lx.session_pk = g.session_pk \
        AND lx.written_by = 'model' AND lx.at >= g.last_prompt_at) - g.last_prompt_at AS latency, \
     (g.answer IS NULL AND g.turn = max(g.turn) OVER (PARTITION BY g.session_pk)) AS abandoned \
   FROM (SELECT tm.session_pk AS session_pk, tm.turn AS turn, min(tm.pk) AS pk, \
       substr(min(CASE WHEN tm.hp THEN printf('%012d ', tm.ord) || tm.id END), 14) AS first_prompt, \
       substr(min(printf('%012d ', tm.ord) || tm.id), 14) AS first_id, \
       COALESCE(substr(min(CASE WHEN tm.hp THEN printf('%012d ', tm.ord) || tm.id END), 14), \
                substr(min(printf('%012d ', tm.ord) || tm.id), 14)) AS id, \
       substr(max(CASE WHEN tm.hp THEN printf('%012d ', tm.ord) || tm.id END), 14) AS last_prompt, \
       max(CASE WHEN tm.hp THEN tm.at END) AS last_prompt_at, \
       min(tm.at) AS start, max(COALESCE(tm.completed_at, tm.at)) AS \"end\", \
       sum(tm.hp) AS burst, \
       json_group_array(tm.id ORDER BY tm.ord) FILTER (WHERE tm.hp) AS prompts, \
       substr(min(printf('%012d ', tm.ord) || COALESCE(tm.prompt_kind, '')), 14) = 'command' AS slash_command, \
       substr(max(CASE WHEN tm.written_by = 'model' THEN printf('%012d ', tm.ord) || tm.id END), 14) AS answer, \
       CASE WHEN min(COALESCE(tm.on_active_path, -1)) < 0 THEN 'unknown' ELSE 'active' END AS path, \
       sum(tm.tc) AS tool_calls, sum(tm.te) AS errors, \
       sum(tm.input_tokens) AS input_tokens, sum(tm.output_tokens) AS output_tokens, \
       sum(tm.cost_nusd) AS cost_nusd \
     FROM (SELECT o.*, \
         sum(CASE WHEN o.hp AND (o.prompt_kind = 'command' OR NOT o.prev_hp) THEN 1 ELSE 0 END) \
           OVER (PARTITION BY o.session_pk ORDER BY o.order_input_pk, o.seq \
                 ROWS UNBOUNDED PRECEDING) AS turn, \
         row_number() OVER (PARTITION BY o.session_pk ORDER BY o.order_input_pk, o.seq) AS ord \
       FROM (SELECT m.pk, m.id, m.session_pk, m.order_input_pk, m.seq, m.written_by, m.prompt_kind, \
             m.at, m.completed_at, m.on_active_path, m.input_tokens, m.output_tokens, m.cost_nusd, \
             (m.written_by = 'human') AS hp, \
             COALESCE(lag(m.written_by = 'human') OVER (PARTITION BY m.session_pk \
               ORDER BY m.order_input_pk, m.seq), 0) AS prev_hp, \
             (SELECT count(*) FROM tool_call c JOIN block cb ON cb.pk = c.call_block_pk \
               WHERE cb.message_pk = m.pk) AS tc, \
             (SELECT count(*) FROM tool_call c JOIN block cb ON cb.pk = c.call_block_pk \
               WHERE cb.message_pk = m.pk AND c.outcome = 'error') AS te \
           FROM message m WHERE COALESCE(m.on_active_path, 1) = 1 AND m.inherited_from IS NULL \
             AND m.written_by IN ('human', 'model')) o) tm \
     GROUP BY tm.session_pk, tm.turn) g)";

#[cfg(test)]
mod tests;
