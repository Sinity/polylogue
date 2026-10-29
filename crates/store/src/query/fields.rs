//! Field expressions and product-specific predicate rules.
use super::ast::Unit;
use super::compile::custom_fields::{self as custom, Custom};
use super::compile::{ColKind, ColumnReference};
use Kind::*;
use Unit::*;
use polylogue_model::{BlockKind, CallOutcome, Role, SessionKind, WrittenBy};
pub use polylogue_query::fields::{
    Kind, SESSION_OUTCOMES, construct, has_session, names, relation,
};
pub struct Field {
    pub unit: Unit,
    pub name: &'static str,
    pub sql: Option<Expr>,
    pub(crate) compile: Option<Custom>,
}
impl std::ops::Deref for Field {
    type Target = polylogue_query::fields::Field;
    fn deref(&self) -> &Self::Target {
        polylogue_query::fields::lookup(self.unit, self.name)
            .expect("backend field must have a catalog declaration")
    }
}
const fn f(unit: Unit, name: &'static str) -> Field {
    Field {
        unit,
        name,
        sql: None,
        compile: None,
    }
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
    const fn sql(mut self, t: &'static str) -> Self {
        self.sql = Some(Expr::Template(t));
        self
    }
    const fn sql_ref(mut self, t: &'static str, r: ColumnReference) -> Self {
        self.sql = Some(Expr::Reference(t, r));
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
    const fn custom(mut self, f: Custom) -> Self {
        self.compile = Some(f);
        self
    }
    pub fn expr(&self, a: &Aliases, now: i64) -> Option<String> {
        self.sql.map(|e| e.render(a, now))
    }
}
/// One value of a `has:` field and the condition it stands for, over the
/// row's alias (`s` for sessions, `m` for messages).
pub struct Has {
    pub name: &'static str,
    pub sql: fn(&str) -> String,
}

pub const SESSION_HAS: [Has; 8] = [
    Has {
        name: "thinking",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM block xb JOIN message xm ON xm.pk = xb.message_pk WHERE {} AND xb.kind = 'thinking')",
                own_messages("xm", Some(&format!("= {s}.pk")))
            )
        },
    },
    Has {
        name: "tool_calls",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM tool_call xt JOIN block xb ON xb.pk = xt.call_block_pk JOIN message xm ON xm.pk = xb.message_pk WHERE {})",
                own_messages("xm", Some(&format!("= {s}.pk")))
            )
        },
    },
    Has {
        name: "errors",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM tool_call xt JOIN block xb ON xb.pk = xt.call_block_pk JOIN message xm ON xm.pk = xb.message_pk WHERE {} AND xt.outcome = 'error')",
                own_messages("xm", Some(&format!("= {s}.pk")))
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
                "EXISTS (SELECT 1 FROM event xe WHERE xe.session_pk = {s}.pk AND xe.kind = 'compaction')"
            )
        },
    },
    Has {
        name: "branches",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM message xm WHERE xm.session_pk = {s}.pk AND xm.on_active_path = 0)"
            )
        },
    },
    Has {
        name: "note",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM source_note hn JOIN source_note_target ht ON ht.note_id = hn.id \
                 WHERE (ht.ref = 'session:' || {s}.id OR ht.ref IN (SELECT 'message:' || xm.id FROM message xm WHERE xm.session_pk = {s}.pk)) \
                 AND {})",
                live_note("hn")
            )
        },
    },
    Has {
        name: "conflict",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM source_note hn JOIN source_note_target ht ON ht.note_id = hn.id \
                 WHERE ht.ref = 'session:' || {s}.id AND {} \
                 GROUP BY hn.first_note HAVING count(*) > 1)",
                current_note("hn")
            )
        },
    },
];

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

pub static FIELDS:&[Field]=&[
f(Documents,"id").sql_ref("{r}.id", ColumnReference::Document),
f(Documents,"origin").sql("{r}.origin"),
f(Documents,"purpose").sql("{r}.purpose"),
f(Documents,"authorship").sql("{r}.authorship"),
f(Documents,"title").sql("{r}.title"),
f(Documents,"source_time").sql("{r}.source_time"),
f(Documents,"versions").sql("(SELECT count(DISTINCT ds.version_hash) FROM document_source ds WHERE ds.document_pk={r}.pk)"),
f(Sessions,"id").sql_ref("{s}.id", ColumnReference::Session),
f(Sessions,"origin").sql("{s}.origin"),
f(Sessions,"kind").sql("{s}.kind"),
f(Sessions,"title").sql("COALESCE({s}.title, {s}.derived_title)"),
f(Sessions,"repo").custom(custom::repo).sql("{s}.repo"),
f(Sessions,"branch").sql("{s}.branch"),
f(Sessions,"commit").custom(custom::commit),
f(Sessions,"pr").custom(custom::pr),
f(Sessions,"outcome").sql_fn(sessions_outcome),
f(Sessions,"hidden").sql_fn(sessions_hidden),
f(Sessions,"cwd").custom(custom::cwd).sql("{s}.cwd"),
f(Sessions,"source").sql("CASE WHEN {s}.stub = 1 THEN 'missing' ELSE 'present' END"),
f(Sessions,"live").custom(custom::live).sql_clock(sessions_live),
f(Sessions,"waiting").custom(custom::waiting).sql_clock(sessions_waiting),
f(Sessions,"running").custom(custom::running).sql_clock(sessions_running),
f(Sessions,"waiting_on").sql("{s}.live_waiting_on"),
f(Sessions,"task").sql("{s}.live_task"),
f(Sessions,"live_until").sql_clock(sessions_live_until),
f(Sessions,"last_event").sql("{s}.live_last_event"),
f(Sessions,"last_event_at").sql("{s}.live_last_event_at"),
f(Sessions,"waiting_since").sql("{s}.live_waiting_since"),
f(Sessions,"todo").sql("{s}.live_todo"),
f(Sessions,"live_rank").sql_clock(sessions_live_rank),
f(Sessions,"peers").sql_clock(sessions_peers),
f(Sessions,"repo_root").sql("{s}.repo_root"),
f(Sessions,"harness").sql("{s}.harness"),
f(Sessions,"model").custom(custom::model).sql(
        "(SELECT x.model FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL GROUP BY x.model ORDER BY count(*) DESC LIMIT 1)",
    ),
f(Sessions,"tag").custom(custom::tag),
f(Sessions,"has").custom(custom::has),
f(Sessions,"parent").custom(custom::parent),
f(Sessions,"root").custom(custom::root),
f(Sessions,"depth").custom(custom::depth),
f(Sessions,"link").custom(custom::link),
f(Sessions,"continues").custom(custom::continues),
f(Sessions,"continued_by").custom(custom::continued_by),
f(Sessions,"connected").custom(custom::connected),
f(Sessions,"path").custom(custom::session_tool_call),
f(Sessions,"tool").custom(custom::session_tool_call),
f(Sessions,"action").custom(custom::session_tool_call),
f(Sessions,"started").sql("{s}.started_at"),
f(Sessions,"first_seen").sql_fn(session_first_seen),
f(Sessions,"evidence_until").custom(custom::evidence_until),
f(Sessions,"last_active").sql("{s}.last_activity_at"),
f(Sessions,"since").custom(custom::since),
f(Sessions,"until").custom(custom::until),
f(Sessions,"messages").sql_fn(|a| crate::session_sql::count(&a.s,"message_count")),
f(Sessions,"human_prompts").sql_fn(|a| crate::session_sql::count(&a.s,"human_prompt_count")),
f(Sessions,"human_commands").sql_fn(|a| crate::session_sql::count(&a.s,"human_command_count")),
f(Sessions,"words").sql_fn(|a| crate::session_sql::count(&a.s,"word_count")),
f(Sessions,"tool_calls").sql_fn(|a| crate::session_sql::count(&a.s,"tool_call_count")),
f(Sessions,"errors").sql_fn(|a| crate::session_sql::count(&a.s,"error_count")),
f(Sessions,"input_tokens").sql_fn(sessions_input_tokens),
f(Sessions,"output_tokens").sql_fn(sessions_output_tokens),
f(Sessions,"cache_read_tokens").sql_fn(sessions_cache_read_tokens),
f(Sessions,"cache_write_tokens").sql_fn(sessions_cache_write_tokens),
f(Sessions,"cost").sql_fn(sessions_cost),
f(Sessions,"tree_cost").sql_fn(sessions_tree_cost),
f(Sessions,"wall_span").sql_fn(|a| crate::session_sql::span(&a.s,"wall_ms")),
f(Sessions,"clustered_time").sql_fn(|a| crate::session_sql::span(&a.s,"clustered_ms")),
f(Sessions,"tool_time").sql_fn(|a| crate::session_sql::span(&a.s,"tool_sum_ms")),
f(Sessions,"tool_time_union").sql_fn(|a| crate::session_sql::span(&a.s,"tool_union_ms")),
f(Sessions,"tool_untimed").sql_fn(|a| crate::session_sql::span(&a.s,"tool_untimed")),
f(Sessions,"ends_with").sql_fn(|a| crate::session_sql::span(&a.s,"ends_with")),
f(Sessions,"last_stop_reason").sql_fn(|a| crate::session_sql::span(&a.s,"last_stop_reason")),
f(Sessions,"time_source").sql_fn(|a| crate::session_sql::span(&a.s,"time_source")),
f(Sessions,"waiting_model").sql_fn(|a| format!("(SELECT ms FROM session_time st WHERE st.session_pk={s}.pk AND st.lane='model' AND st.key='' AND {})",crate::session_sql::current(&a.s,"session_span"),s=a.s)),
f(Sessions,"waiting_human").sql_fn(|a| format!("(SELECT ms FROM session_time st WHERE st.session_pk={s}.pk AND st.lane='human' AND st.key='' AND {})",crate::session_sql::current(&a.s,"session_span"),s=a.s)),
f(Sessions,"human_words").sql_fn(|a| crate::session_sql::count(&a.s,"human_word_count")),
f(Sessions,"model_words").sql_fn(|a| crate::session_sql::count(&a.s,"model_word_count")),
f(Sessions,"thinking_messages").sql_fn(|a| crate::session_sql::count(&a.s,"thinking_count")),
f(Sessions,"compactions").sql_fn(|a| crate::session_sql::count(&a.s,"compaction_count")),
f(Sessions,"similar").custom(custom::near),
f(Sessions,"output").custom(custom::output),
f(Sessions,"thinking").custom(custom::thinking),
f(Messages,"id").sql_ref("{m}.id", ColumnReference::Message),
f(Messages,"role").sql("{m}.role"),
f(Messages,"written_by").sql("{m}.written_by"),
f(Messages,"prompt_kind").sql("{m}.prompt_kind"),
f(Messages,"model").sql("{m}.model"),
f(Messages,"at").sql("{m}.at"),
f(Messages,"first_seen").sql_fn(message_first_seen),
f(Messages,"evidence_until").custom(custom::evidence_until),
f(Messages,"duplicate").sql("({m}.duplicate_of IS NOT NULL)"),
f(Messages,"active").sql("COALESCE({m}.on_active_path, 1)"),
f(Messages,"has").custom(custom::has),
f(Messages,"words").sql("{m}.words"),
f(Messages,"input_tokens").sql("{m}.input_tokens"),
f(Messages,"output_tokens").sql("{m}.output_tokens"),
f(Messages,"cache_read_tokens").sql("{m}.cache_read_tokens"),
f(Messages,"cache_write_tokens").sql("{m}.cache_write_tokens"),
f(Messages,"cost").sql("{m}.cost_nusd"),
f(Messages,"stop_reason").sql("{m}.stop_reason"),
f(Messages,"since").custom(custom::since),
f(Messages,"until").custom(custom::until),
f(Messages,"similar").custom(custom::near),
f(Messages,"output").custom(custom::output),
f(Messages,"thinking").custom(custom::thinking),
f(Blocks,"kind").sql("{b}.kind"),
f(Blocks,"media_type").sql("{b}.media_type"),
f(Blocks,"file_name").sql("{b}.file_name"),
f(Blocks,"at").sql("{m}.at"),
f(Blocks,"first_seen").sql_fn(message_first_seen),
f(Blocks,"evidence_until").custom(custom::evidence_until),
f(Blocks,"since").custom(custom::since),
f(Blocks,"until").custom(custom::until),
f(ToolCalls,"tool").sql("{t}.tool"),
f(ToolCalls,"mcp_server").custom(custom::mcp_server).sql(
        "CASE WHEN {t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' THEN substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1) END",
    ),
f(ToolCalls,"action").sql("{t}.action"),
f(ToolCalls,"command").custom(custom::tool_call_text).sql("{t}.command"),
f(ToolCalls,"command_norm").sql("{t}.command_norm"),
f(ToolCalls,"input").custom(custom::input),
f(ToolCalls,"output").custom(custom::output),
f(ToolCalls,"path").custom(custom::tool_call_text).sql("{t}.path"),
f(ToolCalls,"outcome").sql(
        "CASE WHEN {t}.result_block_pk IS NULL AND {t}.outcome = 'unknown' AND {s}.live_open_tool_pk = {t}.pk \
             THEN 'pending' ELSE {t}.outcome END",
    ),
f(ToolCalls,"exit_code").sql("{t}.exit_code"),
f(ToolCalls,"at").sql("{t}.at"),
f(ToolCalls,"result_at").sql("{t}.result_at"),
f(ToolCalls,"reported_duration").sql("{t}.reported_duration_ms"),
f(ToolCalls,"first_seen").sql_fn(message_first_seen),
f(ToolCalls,"evidence_until").custom(custom::evidence_until),
f(ToolCalls,"since").custom(custom::since),
f(ToolCalls,"until").custom(custom::until),
f(ToolCalls,"child").custom(custom::spawned_child),
f(ToolCalls,"next").custom(custom::next_call),
f(Files,"path").sql("{r}.path"),
f(Files,"reads").sql("{r}.reads"),
f(Files,"writes").sql("{r}.writes"),
f(Files,"edits").sql("{r}.edits"),
f(Files,"first_at").sql("{r}.first_at"),
f(Files,"last_at").sql("{r}.last_at"),
f(Events,"id").sql("{r}.id"),
f(Events,"text").custom(custom::event_text),
f(Events,"input").sql("{r}i.id"),
f(Events,"source").sql("{r}i.format"),
f(Events,"origin").sql("COALESCE({s}.origin,{r}i.format)"),
f(Events,"data").sql("pl_block({r}.data)"),
f(Events,"source_chain").sql("{r}.source_chain"),
f(Events,"source_start").sql("{r}.source_start"),
f(Events,"source_end").sql("{r}.source_end"),
f(Events,"kind").sql("{r}.kind"),
f(Events,"at").sql("{r}.at"),
f(Events,"first_seen").sql_fn(event_first_seen),
f(Events,"evidence_until").custom(custom::evidence_until),
f(Events,"since").custom(custom::since),
f(Events,"until").custom(custom::until),
f(Links,"kind").sql("{r}.kind"),
f(Links,"how").sql("{r}.how"),
f(Links,"parent").custom(custom::link_parent).sql_ref("{r}.parent_id", ColumnReference::Session),
f(Links,"child").custom(custom::link_child).sql_ref("{r}c.id", ColumnReference::Session),
f(Links,"pending").sql("(NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = {r}.parent_id))"),
f(Links,"evidence").sql("{r}.evidence"),
f(Links,"branch_point").sql("{r}.branch_point"),
f(Notes,"id").sql_ref("{r}.id", ColumnReference::Note),
f(Notes,"kind").sql("{r}.kind"),
f(Notes,"key").sql("{r}.key"),
f(Notes,"value").sql("{r}.value"),
f(Notes,"author").sql("{r}.author"),
f(Notes,"agent").sql("({r}.author = 'agent' OR {r}.author LIKE 'agent:%')"),
f(Notes,"judged").sql_fn(notes_judged),
f(Notes,"target").custom(custom::note_target).sql("(SELECT group_concat(ref, ' ') FROM (SELECT nt.ref FROM source_note_target nt WHERE nt.note_id = {r}.id ORDER BY nt.ref))"),
f(Notes,"write_counter").sql("{r}.commit_seq"),
f(Notes,"created_at").sql("{r}.created_at"),
f(Notes,"first_note").sql("{r}.first_note"),
f(Notes,"current").sql_fn(notes_current),
f(Notes,"withdrawn").sql_fn(notes_withdrawn),
f(Notes,"since").custom(custom::since),
f(Notes,"evidence_until").custom(custom::evidence_until),
f(Notes,"until").custom(custom::until),
f(Deliveries,"id").sql("{r}.id"),
f(Deliveries,"recipient").custom(custom::recipient).sql("{r}.recipient_id"),
f(Deliveries,"event").sql("{r}.event"),
f(Deliveries,"sender").sql("{r}.sender"),
f(Deliveries,"outcome").sql_fn(deliveries_outcome),
f(Deliveries,"detail").sql_fn(deliveries_detail),
f(Deliveries,"text").sql("pl_text({r}.body)"),
f(Deliveries,"refs").custom(custom::refs),
f(Deliveries,"left_out").custom(custom::left_out),
f(Deliveries,"at").sql("{r}.at"),
f(Deliveries,"bytes").sql("{r}.bytes"),
f(Changes,"kind").sql("{r}.kind"),
f(Changes,"ref").sql("{r}.ref"),
f(Changes,"seq").sql("{r}.seq"),
f(Changes,"note_kind").sql("{r}.note_kind"),
f(Changes,"after").custom(custom::changes_after),
f(Changes,"sessions").custom(custom::selection),
f(Changes,"messages").custom(custom::selection),
f(Turns,"id").sql("{r}.id"),
f(Turns,"start").sql("{r}.start"),
f(Turns,"end").sql("{r}.end"),
f(Turns,"burst").sql("{r}.burst"),
f(Turns,"prompts").sql("{r}.prompts"),
f(Turns,"slash_command").sql("{r}.slash_command"),
f(Turns,"answer").sql("{r}.answer"),
f(Turns,"attempts").sql("{r}.attempts"),
f(Turns,"path").sql("{r}.path"),
f(Turns,"latency").sql("{r}.latency"),
f(Turns,"tool_calls").sql("{r}.tool_calls"),
f(Turns,"errors").sql("{r}.errors"),
f(Turns,"input_tokens").sql("{r}.input_tokens"),
f(Turns,"output_tokens").sql("{r}.output_tokens"),
f(Turns,"cost").sql("{r}.cost_nusd"),
f(Turns,"abandoned").sql("{r}.abandoned"),
f(Turns,"corrected").sql_fn(turns_corrected),
f(Work,"kind").sql("{r}.kind"),
f(Work,"title").sql("{r}.title"),
f(Work,"key").sql("{r}.key"),
f(Work,"state").sql("{r}.state"),
f(Work,"at").sql("{r}.at"),
f(Work,"goal").sql_fn(work_goal_field),
f(Work,"checks_passed").sql("json_extract({r}.attrs_json, '$.checks_passed')"),
f(Work,"checks_failed").sql("json_extract({r}.attrs_json, '$.checks_failed')"),
f(Work,"checks_unknown").sql("json_extract({r}.attrs_json, '$.checks_unknown')"),
f(Work,"artifacts").sql("json_extract({r}.attrs_json, '$.artifacts')")
];
pub fn lookup(unit: Unit, name: &str) -> Option<&'static Field> {
    FIELDS
        .iter()
        .find(|f| f.unit == unit && f.name.eq_ignore_ascii_case(name))
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

pub use crate::session_sql::{counts_as_action, own_messages};

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

/// The live states.
pub use polylogue_query::fields::LIVE_STATES;

/// The live window: hook evidence this recent makes a session live.
pub const LIVE_WINDOW_MS: i64 = 15 * 60 * 1000;
/// Evidence older than this, with nothing newer, means the session ended.
pub const LIVE_DAY_MS: i64 = 24 * 60 * 60 * 1000;

/// The live state of session alias `s` at `now`: computed when asked from
/// the stored live facts and the time, never written by a clock. Rules apply in order and the result is total.
/// A fact dated after `now` counts as at `now`.
pub fn live_state(s: &str, now: i64) -> String {
    let w = LIVE_WINDOW_MS;
    let d = LIVE_DAY_MS;
    let ev = format!("MIN({s}.live_last_event_at, {now})");
    let grew = format!("MIN({s}.last_activity_at, {now})");
    let live = format!("({s}.live_last_event_at IS NOT NULL AND {now} - {ev} <= {w})");
    format!(
        "(CASE \
         WHEN {s}.live_last_event = 'hook_session_end' THEN 'ended' \
         WHEN {live} AND {s}.live_waiting_on IS NOT NULL THEN 'waiting' \
         WHEN {live} AND {s}.live_last_event IN ('hook_stop', 'hook_notification') THEN 'idle' \
         WHEN {live} THEN 'working' \
         WHEN {s}.live_last_event IS NULL \
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
         WHEN 'stale' THEN {last} + {d} \
         WHEN 'unknown' THEN {grew} + {w} \
         ELSE {ev} + {w} END)",
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
        "(SELECT sum(x.{col}) FROM message x WHERE {})",
        own_messages("x", Some(&format!("= {s}.pk")))
    )
}

fn sessions_hidden(a: &Aliases) -> String {
    hidden(&a.s)
}

fn sessions_outcome(a: &Aliases) -> String {
    format!(
        "COALESCE((SELECT lower(n.value) FROM source_note n JOIN source_note_target nt ON nt.note_id = n.id \
         WHERE n.kind = 'outcome' AND nt.ref = 'session:' || {s}.id AND {} ORDER BY n.commit_seq DESC LIMIT 1), 'none')",
        live_note("n"),
        s = a.s
    )
}

/// The latest delivery event's `col` for delivery `r`.
fn latest_delivery_event(r: &str, col: &str) -> String {
    format!(
        "(SELECT e.{col} FROM source_delivery_event e WHERE e.delivery = {r}.id ORDER BY e.at DESC LIMIT 1)"
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
        "EXISTS (SELECT 1 FROM source_note nc JOIN source_note_target nct ON nct.note_id = nc.id \
         WHERE nc.kind = 'correction' AND nct.ref = 'message:' || {r}.answer \
         AND {} AND NOT {})",
        current_note("nc"),
        withdrawn_note("nc"),
        r = a.row
    )
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
        "(SELECT sum(x.cost_nusd) FROM message x WHERE {})",
        own_messages("x", Some(&format!("IN ({})", subtree(&a.s))))
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

/// Peers: the other sessions working, waiting or idle that
/// share a repository, or, when either has no git evidence, a
/// working directory where one lies at or below the other, by whole path
/// components. Symmetric by construction.
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
         OR ((pp.repo_root IS NULL OR {s}.repo_root IS NULL) \
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

/// The key of the operator's latest live `judgment` note on this exact
/// revision: `accepted` or `rejected`.
fn notes_judged(a: &Aliases) -> String {
    format!(
        "(SELECT j.key FROM source_note j JOIN source_note_target jt ON jt.note_id = j.id \
         WHERE jt.ref = 'note:' || {r}.id AND j.kind = 'judgment' AND j.author = 'operator' \
         AND {live} ORDER BY j.commit_seq DESC LIMIT 1)",
        r = a.row,
        live = live_note("j"),
    )
}

fn notes_withdrawn(a: &Aliases) -> String {
    format!("({})", withdrawn_note(&a.row))
}

fn work_goal_field(a: &Aliases) -> String {
    work_goal(&a.row)
}

pub use crate::note_sql::{current_note, live_note, withdrawn_note};

/// When Polylogue first had message `m`'s bytes: the earliest observation,
/// over every chain whose records hold a version of it, of a revision of
/// that chain long enough to cover those records.
pub fn first_seen_of(m: &str) -> String {
    format!(
        "(SELECT min(fo.seen_at) FROM message_source fs JOIN input_ref fi ON fi.pk = fs.input_pk \
         JOIN source_revision fr ON fr.input = fi.id AND fr.chain = fs.chain \
         AND fr.length > fs.source_start AND fr.length >= fs.source_end \
         JOIN source_observation fo ON fo.input = fr.input AND fo.revision = fr.number \
         WHERE fs.message_pk = {m}.pk)"
    )
}

fn message_first_seen(a: &Aliases) -> String {
    first_seen_of(&a.m)
}

fn session_first_seen(a: &Aliases) -> String {
    format!(
        "(SELECT min(fo.seen_at) FROM message fm JOIN message_source fs ON fs.message_pk = fm.pk \
         JOIN input_ref fi ON fi.pk = fs.input_pk \
         JOIN source_revision fr ON fr.input = fi.id AND fr.chain = fs.chain \
         AND fr.length > fs.source_start AND fr.length >= fs.source_end \
         JOIN source_observation fo ON fo.input = fr.input AND fo.revision = fr.number \
         WHERE fm.session_pk = {}.pk)",
        a.s
    )
}

/// An event's first observation, as for a message, over its one chain.
pub fn event_first_seen(a: &Aliases) -> String {
    format!(
        "(SELECT min(fo.seen_at) FROM input_ref fi \
         JOIN source_revision fr ON fr.input = fi.id AND fr.chain = {r}.source_chain \
         AND fr.length > {r}.source_start AND fr.length >= coalesce({r}.source_end, {r}.source_start) \
         JOIN source_observation fo ON fo.input = fr.input AND fo.revision = fr.number \
         WHERE fi.pk = {r}.input_pk)",
        r = a.row
    )
}

/// A current `hidden` note targets the session.
pub fn hidden(s: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM source_note hn JOIN source_note_target ht ON ht.note_id = hn.id \
         WHERE hn.kind = 'hidden' AND ht.ref = 'session:' || {s}.id AND {})",
        live_note("hn")
    )
}

/// The session and every session below it in its subagent tree.
pub fn subtree(s: &str) -> String {
    format!(
        "WITH RECURSIVE tr(pk) AS (SELECT {s}.pk UNION SELECT l.child_pk FROM link l \
         JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr"
    )
}
