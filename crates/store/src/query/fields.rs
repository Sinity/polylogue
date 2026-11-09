//! Field expressions and product-specific predicate rules.
use super::ast::Unit;
use super::compile::ColumnReference;
use super::compile::custom_fields::{self as custom, Custom};
use Unit::*;
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
    Fn(fn(&Aliases) -> String),
    Clock(fn(&Aliases, i64) -> String),
    NativeTime {
        unit: Unit,
        field: &'static str,
        attribute: NativeTimeAttribute,
    },
}

impl Expr {
    pub fn render(&self, a: &Aliases, now: i64) -> String {
        match self {
            Expr::Template(t) => a.expand(t),
            Expr::Fn(f) => f(a),
            Expr::Clock(f) => f(a, now),
            Expr::NativeTime {
                unit,
                field,
                attribute,
            } => native_time(*unit, field, a)
                .map_or_else(|| "NULL".into(), |native| native.attribute(*attribute)),
        }
    }
}

impl Field {
    const fn sql(mut self, t: &'static str) -> Self {
        self.sql = Some(Expr::Template(t));
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
    const fn time_attribute(mut self, field: &'static str, attribute: NativeTimeAttribute) -> Self {
        self.sql = Some(Expr::NativeTime {
            unit: self.unit,
            field,
            attribute,
        });
        self
    }
    pub fn reference(&self) -> Option<ColumnReference> {
        polylogue_query::fields::column_reference(self.unit, self.name)
    }

    pub fn expr(&self, a: &Aliases, now: i64) -> Option<String> {
        query_language::depth::deeper(|| self.sql.map(|e| e.render(a, now)))
    }
}
/// Canonical native evidence behind a temporal field. The integer cache is
/// used only for broad candidates; the blob decides the exact predicate.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeTimeStorage {
    Tuple,
    Event,
    Latest,
}
impl NativeTimeStorage {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::Tuple => "tuple",
            Self::Event => "event",
            Self::Latest => "latest",
        }
    }
    pub(crate) fn from_code(code: &str) -> Option<Self> {
        match code {
            "tuple" => Some(Self::Tuple),
            "event" => Some(Self::Event),
            "latest" => Some(Self::Latest),
            _ => None,
        }
    }
}
#[derive(Clone, Copy)]
pub enum NativeTimeAttribute {
    Quality,
    Precision,
}
impl NativeTimeAttribute {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Precision => "precision",
        }
    }
    pub(crate) fn from_code(code: &str) -> Option<Self> {
        match code {
            "quality" => Some(Self::Quality),
            "precision" => Some(Self::Precision),
            _ => None,
        }
    }
}
pub(crate) struct NativeTime {
    pub cache: String,
    pub bytes: String,
    pub storage: NativeTimeStorage,
}
impl NativeTime {
    fn attribute(&self, attribute: NativeTimeAttribute) -> String {
        format!(
            "pl_time_attribute({}, '{}', '{}')",
            self.bytes,
            self.storage.code(),
            attribute.code()
        )
    }
}
pub(crate) fn native_time(unit: Unit, field: &str, a: &Aliases) -> Option<NativeTime> {
    let (cache, bytes, storage) = match (unit, field) {
        (Messages | Blocks, "at") => ("{m}.at", "{m}.at_time", NativeTimeStorage::Tuple),
        (ToolCalls, "at") => ("{m}.at", "{m}.at_time", NativeTimeStorage::Tuple),
        (Sessions, "started") => (
            "{s}.started_at",
            "{s}.started_time",
            NativeTimeStorage::Tuple,
        ),
        (Sessions, "last_active") => (
            "{s}.last_activity_at",
            "(SELECT pl_time_latest({s}_activity.at_time) FROM message {s}_activity WHERE {s}_activity.session_pk={s}.pk AND {s}_activity.at IS NOT NULL)",
            NativeTimeStorage::Latest,
        ),
        (Events, "at") => ("{r}.at", "{r}.fields", NativeTimeStorage::Event),
        (Documents, "source_time") => (
            "{r}.source_time",
            "(SELECT {r}_time.time FROM document_source {r}_time WHERE {r}_time.document_pk={r}.pk AND {r}_time.input_pk={r}.input_pk AND {r}_time.chain={r}.source_chain AND {r}_time.shown=1)",
            NativeTimeStorage::Tuple,
        ),
        _ => return None,
    };
    Some(NativeTime {
        cache: a.expand(cache),
        bytes: a.expand(bytes),
        storage,
    })
}
pub(crate) fn native_window(unit: Unit, since: bool, a: &Aliases) -> Option<NativeTime> {
    let field = if unit == Sessions && !since {
        "started"
    } else {
        polylogue_query::units::spec(unit).main_time?
    };
    native_time(unit, field, a)
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
                "EXISTS (SELECT 1 FROM source_assertion hn JOIN source_link ht ON ht.record = hn.id AND ht.role = 'about' \
                 WHERE (ht.ref = 'session:' || {s}.id OR ht.ref IN (SELECT 'message:' || xm.id FROM message xm WHERE xm.session_pk = {s}.pk)) \
                 AND {})",
                live_assertion("hn")
            )
        },
    },
    Has {
        name: "conflict",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM source_assertion hn JOIN source_link ht ON ht.record = hn.id AND ht.role = 'about' \
                 WHERE ht.ref = 'session:' || {s}.id AND {} \
                 GROUP BY hn.root HAVING count(*) > 1)",
                current_assertion("hn")
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
f(Messages,"at.time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Messages,"at.precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Blocks,"at.time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Blocks,"at.precision").time_attribute("at",NativeTimeAttribute::Precision),
f(ToolCalls,"at.time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(ToolCalls,"at.precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Events,"at.time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Events,"at.precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Deliveries,"at.time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Deliveries,"at.precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Sessions,"started_at.time_quality").time_attribute("started",NativeTimeAttribute::Quality),
f(Sessions,"started_at.precision").time_attribute("started",NativeTimeAttribute::Precision),
f(Documents,"source_time.time_quality").time_attribute("source_time",NativeTimeAttribute::Quality),
f(Documents,"source_time.precision").time_attribute("source_time",NativeTimeAttribute::Precision),
f(Sessions,"last_active.time_quality").time_attribute("last_active",NativeTimeAttribute::Quality),
f(Sessions,"last_active.precision").time_attribute("last_active",NativeTimeAttribute::Precision),
f(ToolCalls,"result_at.time_quality").time_attribute("result_at",NativeTimeAttribute::Quality),
f(ToolCalls,"result_at.precision").time_attribute("result_at",NativeTimeAttribute::Precision),
f(Documents,"time_quality").time_attribute("source_time",NativeTimeAttribute::Quality),
f(Documents,"precision").time_attribute("source_time",NativeTimeAttribute::Precision),
f(Messages,"time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Messages,"precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Blocks,"time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Blocks,"precision").time_attribute("at",NativeTimeAttribute::Precision),
f(ToolCalls,"time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(ToolCalls,"precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Events,"time_quality").time_attribute("at",NativeTimeAttribute::Quality),
f(Events,"precision").time_attribute("at",NativeTimeAttribute::Precision),
f(Documents,"id").sql("{r}.id"),
f(Documents,"origin").sql("{r}.origin"),
f(Documents,"purpose").sql("{r}.purpose"),
f(Documents,"authorship").sql("{r}.authorship"),
f(Documents,"title").sql("{r}.title"),
f(Documents,"source_time").sql("{r}.source_time"),
f(Documents,"since").custom(custom::since),
f(Documents,"until").custom(custom::until),
f(Documents,"versions").sql("(SELECT count(DISTINCT ds.version_hash) FROM document_source ds WHERE ds.document_pk={r}.pk)"),
f(Sessions,"id").sql("{s}.id"),
f(Sessions,"origin").sql("{s}.origin"),
f(Sessions,"kind").sql("{s}.kind"),
f(Sessions,"title").sql_fn(sessions_title),
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
f(Sessions,"waiting_on").sql_clock(sessions_waiting_on),
f(Sessions,"stall").sql_clock(sessions_stall),
f(Sessions,"process_alive").sql_clock(sessions_process_alive),
f(Sessions,"harness_status").sql("json_extract({s}.live_facts, '$.harness_status.word')"),
f(Sessions,"open_tool").sql("json_extract({s}.live_facts, '$.open_call.tool')"),
f(Sessions,"footprint").sql("(SELECT group_concat(fp.value, char(10)) FROM json_each({s}.live_facts, '$.footprint') fp)"),
f(Sessions,"task").sql("json_extract({s}.live_facts, '$.task')"),
f(Sessions,"live_until").sql_clock(sessions_live_until),
f(Sessions,"live_since").sql_clock(sessions_live_since),
f(Sessions,"last_event_at").sql("{s}.live_last_at"),
f(Sessions,"todo").sql("json_extract({s}.live_facts, '$.todo')"),
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
f(Messages,"id").sql("{m}.id"),
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
f(Blocks,"message").sql("{m}.id"),
f(Blocks,"kind").sql("{b}.kind"),
f(Blocks,"media_type").sql("{b}.media_type"),
f(Blocks,"file_name").sql("{b}.file_name"),
f(Blocks,"at").sql("{m}.at"),
f(Blocks,"first_seen").sql_fn(message_first_seen),
f(Blocks,"evidence_until").custom(custom::evidence_until),
f(Blocks,"since").custom(custom::since),
f(Blocks,"until").custom(custom::until),
f(ToolCalls,"message").sql("{m}.id"),
f(ToolCalls,"idx").sql("{cb}.idx"),
f(ToolCalls,"position").sql_fn(super::units::call_position),
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
f(ToolCalls,"outcome").custom(custom::tool_call_outcome).sql_fn(call_outcome),
f(ToolCalls,"exit_code").sql("{t}.exit_code"),
f(ToolCalls,"at").sql("{t}.at"),
f(ToolCalls,"result_at").sql("{t}.result_at"),
f(ToolCalls,"reported_duration").sql("{t}.reported_duration_ms"),
f(ToolCalls,"first_seen").sql_fn(message_first_seen),
f(ToolCalls,"evidence_until").custom(custom::evidence_until),
f(ToolCalls,"since").custom(custom::since),
f(ToolCalls,"until").custom(custom::until),
f(ToolCalls,"child").custom(custom::spawned_child),
f(ToolCalls,"next").custom(custom::adjacent_call),
f(ToolCalls,"previous").custom(custom::adjacent_call),
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
f(Links,"parent").custom(custom::link_parent).sql("{r}.parent_id"),
f(Links,"child").custom(custom::link_child).sql("{r}.child_id"),
f(Links,"pending").sql("(NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = {r}.parent_id))"),
f(Links,"evidence").sql("{r}.evidence"),
f(Links,"branch_point").sql("{r}.branch_point"),
f(Assertions,"id").sql("{r}.id"),
f(Assertions,"reference").sql("pl_assertion_ref({r}.root,{r}.id)"),
f(Assertions,"kind").sql("{r}.kind"),
f(Assertions,"key").sql("{r}.key"),
f(Assertions,"value").sql("{r}.value"),
f(Assertions,"author").sql("{r}.author"),
f(Assertions,"agent").sql("({r}.author = 'agent' OR {r}.author LIKE 'agent:%')"),
f(Assertions,"judged").sql_fn(notes_judged),
f(Assertions,"about").custom(custom::assertion_about).sql("(SELECT group_concat(ref, ' ') FROM (SELECT nt.ref FROM source_link nt WHERE nt.record = {r}.id AND nt.role = 'about' ORDER BY nt.ref))"),
f(Assertions,"write_counter").sql("{r}.commit_seq"),
f(Assertions,"created_at").sql("{r}.created_at"),
f(Assertions,"root").sql("{r}.root"),
f(Assertions,"current").sql_fn(notes_current),
f(Assertions,"conflict").sql("(SELECT count(*) FROM source_revision_state ac WHERE ac.root={r}.root AND ac.current=1)>1"),
f(Assertions,"withdrawn").sql_fn(notes_withdrawn),
f(Assertions,"since").custom(custom::since),
f(Assertions,"evidence_until").custom(custom::evidence_until),
f(Assertions,"until").custom(custom::until),
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
f(LiveIntervals,"session").custom(custom::live_session).sql("{s}.id"),
f(LiveIntervals,"aspect").sql("{r}.aspect"),
f(LiveIntervals,"value").sql("{r}.value"),
f(LiveIntervals,"start").sql("{r}.start_at"),
f(LiveIntervals,"end").sql("{r}.end_at"),
f(LiveIntervals,"duration_ms").sql("({r}.end_at - {r}.start_at)"),
f(LiveIntervals,"closed_by").sql("{r}.closed_by"),
f(Processes,"session").custom(custom::live_session).sql("{s}.id"),
f(Processes,"boot_id").sql("{r}.boot_id"),
f(Processes,"pid").sql("{r}.pid"),
f(Processes,"start").sql("{r}.start"),
f(Processes,"parent_pid").sql("{r}.parent_pid"),
f(Processes,"comm").sql("{r}.comm"),
f(Processes,"argv").sql("{r}.argv"),
f(Processes,"cwd").sql("{r}.cwd"),
f(Processes,"started_at").sql("{r}.started_at"),
f(Processes,"ended_at").sql("{r}.ended_at"),
f(Processes,"role").sql("{r}.role"),
f(Processes,"tool_call").sql_fn(processes_tool_call),
f(Changes,"kind").sql("{r}.kind"),
f(Changes,"ref").sql("{r}.ref"),
f(Changes,"seq").sql("{r}.seq"),
f(Changes,"assertion_kind").sql("{r}.assertion_kind"),
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

/// The live states.
pub use polylogue_query::fields::LIVE_STATES;

/// One part of session alias `s`'s live view at `now`, computed by
/// [`polylogue_model::live::view`] through the `pl_live` SQL function:
/// `state`, `until`, `since`, `waiting_on`, `stall`, `process_alive` or
/// `rank`.
pub fn live_part(s: &str, part: &str, now: i64) -> String {
    format!("pl_live({s}.live_facts, {s}.last_activity_at, {now}, '{part}')")
}

/// The live state of session alias `s` at `now`.
pub fn live_state(s: &str, now: i64) -> String {
    live_part(s, "state", now)
}

/// When `s`'s live state or stall changes with no new evidence; NULL when
/// only new evidence can change them.
pub fn live_until(s: &str, now: i64) -> String {
    live_part(s, "until", now)
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

fn sessions_title(a: &Aliases) -> String {
    crate::session_sql::display_title(&a.s)
}

fn sessions_outcome(a: &Aliases) -> String {
    format!(
        "(SELECT CASE WHEN count(*)=0 THEN 'none' WHEN count(*)=1 THEN min(lower(CASE json_type(n.value) WHEN 'text' THEN json_extract(n.value,'$') ELSE json_extract(n.value,'$.text') END)) ELSE NULL END FROM source_assertion n JOIN source_link nt ON nt.record=n.id AND nt.role='about' WHERE n.kind='outcome' AND nt.ref='session:' || {s}.id AND {live})",
        s = a.s,
        live = live_assertion("n"),
    )
}

/// The tool call a child process started during: the session's call whose
/// start is at or before the process's first sighting and whose result
/// came after it (or has not come). Never from command text.
pub(crate) fn processes_tool_call(a: &Aliases) -> String {
    format!(
        "(CASE WHEN {r}.role = 'child' THEN (SELECT tc.call_id FROM tool_call tc \
         WHERE tc.session_pk = {r}.session_pk AND tc.at <= {r}.started_at \
         AND (tc.result_at IS NULL OR tc.result_at >= {r}.started_at) \
         ORDER BY tc.at DESC LIMIT 1) END)",
        r = a.row
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
        "EXISTS (SELECT 1 FROM source_assertion nc JOIN source_link nct ON nct.record = nc.id AND nct.role = 'about' \
         WHERE nc.kind = 'correction' AND nct.ref = 'message:' || {r}.answer \
         AND {} AND NOT {})",
        current_assertion("nc"),
        withdrawn_assertion("nc"),
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

fn sessions_waiting_on(a: &Aliases, now: i64) -> String {
    live_part(&a.s, "waiting_on", now)
}

fn sessions_stall(a: &Aliases, now: i64) -> String {
    live_part(&a.s, "stall", now)
}

fn sessions_process_alive(a: &Aliases, now: i64) -> String {
    live_part(&a.s, "process_alive", now)
}

fn sessions_live_until(a: &Aliases, now: i64) -> String {
    live_until(&a.s, now)
}

fn sessions_live_since(a: &Aliases, now: i64) -> String {
    live_part(&a.s, "since", now)
}

fn sessions_live_rank(a: &Aliases, now: i64) -> String {
    live_part(&a.s, "rank", now)
}

/// Peers: the other sessions working, waiting or idle in the same
/// repository (by its identity) or, when either has no repository
/// identity, in the same working directory. Symmetric by construction.
pub(crate) fn sessions_peers(a: &Aliases, now: i64) -> String {
    let s = &a.s;
    let active = |x: &str| format!("{} IN ('working', 'waiting', 'idle')", live_state(x, now));
    format!(
        "(SELECT group_concat(pe.id, ' ') FROM (SELECT pp.id FROM session pp WHERE pp.pk <> {s}.pk \
         AND pp.live_last_at IS NOT NULL AND {mine} AND {theirs} \
         AND ((pp.repo_root IS NOT NULL AND pp.repo_root = {s}.repo_root) \
         OR ((pp.repo_root IS NULL OR {s}.repo_root IS NULL) AND pp.cwd = {s}.cwd)) \
         ORDER BY pp.id) pe)",
        mine = active(s),
        theirs = active("pp"),
    )
}

fn notes_current(a: &Aliases) -> String {
    format!("({})", current_assertion(&a.row))
}

/// The operator's judgment of this exact revision: the key of the latest
/// non-withdrawn operator `judgment` about it, latest by `(created_at, id)`.
fn notes_judged(a: &Aliases) -> String {
    format!(
        "(SELECT j.key FROM source_assertion j JOIN source_link jt ON jt.record=j.id AND jt.role='about' WHERE jt.ref='assertion:' || {r}.root || '@' || {r}.id AND j.kind='judgment' AND j.author='operator' AND {live} ORDER BY j.created_at DESC, j.id DESC LIMIT 1)",
        r = a.row,
        live = live_assertion("j"),
    )
}

fn notes_withdrawn(a: &Aliases) -> String {
    format!("({})", withdrawn_assertion(&a.row))
}

pub use crate::assertions::sql::{current_assertion, live_assertion, withdrawn_assertion};

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

/// A stored `unknown` call with no result that the session's live state
/// names as its open call: the `pending` outcome, decided when read.
pub(crate) const OPEN_CALL: &str = "({t}.result_block_pk IS NULL AND {t}.outcome = 'unknown' \
     AND coalesce(json_extract({s}.live_facts, '$.open_call.call_id') = {t}.call_id, 0))";

fn call_outcome(a: &Aliases) -> String {
    format!(
        "CASE WHEN {} THEN 'pending' ELSE {}.outcome END",
        a.expand(OPEN_CALL),
        a.t
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
        "EXISTS (SELECT 1 FROM source_assertion hn JOIN source_link ht ON ht.record = hn.id AND ht.role = 'about' \
         WHERE hn.kind = 'hidden' AND ht.ref = 'session:' || {s}.id AND {})",
        live_assertion("hn")
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

/// Every native working directory, with the currently selected value retained.
pub(crate) fn working_directories(s: &str) -> String {
    format!("json_each(json_insert(COALESCE({s}.cwd_json,'[]'),'$[#]',{s}.cwd))")
}
#[cfg(test)]
mod tests;
