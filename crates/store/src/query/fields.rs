//! Field expressions and product-specific predicate rules.
use super::ast::Unit;
use super::compile::custom_fields::{self as custom, Custom};
use Unit::*;

/// Link kinds that join a logical session; `reference` never does.
pub(crate) const STRUCTURAL_KINDS: &str = "'spawn','fork','continuation','resume'";
pub use polylogue_query::fields::{FIELDS, Field, FieldId};
pub use polylogue_query::fields::{
    Kind, SESSION_OUTCOMES, construct, has_session, names, relation,
};

/// How the store reads one field: the SQL of its value, and a compile rule
/// for a field whose conditions say more than comparing that value.
#[derive(Clone, Copy)]
pub struct Rule {
    pub sql: Option<Expr>,
    pub(crate) compile: Option<Custom>,
}
/// SQL over a unit's aliases: a template whose `{s}`, `{m}`, `{b}`, `{t}`,
/// `{cb}` and `{r}` name them, or a function for what a template cannot
/// say; `Clock` also takes the query's instant.
#[derive(Clone, Copy)]
pub enum Expr {
    Template(&'static str),
    Fn(fn(&Aliases) -> String),
    Clock(fn(&Aliases, i64) -> String),
    StoredTime {
        unit: Unit,
        field: &'static str,
        attribute: TimeAttribute,
    },
}

impl Expr {
    pub fn render(&self, a: &Aliases, now: i64) -> String {
        match self {
            Expr::Template(t) => a.expand(t),
            Expr::Fn(f) => f(a),
            Expr::Clock(f) => f(a, now),
            Expr::StoredTime {
                unit,
                field,
                attribute,
            } => stored_time(*unit, field, a)
                .map_or_else(|| "NULL".into(), |stored| stored.attribute(*attribute)),
        }
    }
}

impl Rule {
    const NONE: Rule = Rule {
        sql: None,
        compile: None,
    };
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
    const fn time_attribute(
        mut self,
        unit: Unit,
        field: &'static str,
        attribute: TimeAttribute,
    ) -> Self {
        self.sql = Some(Expr::StoredTime {
            unit,
            field,
            attribute,
        });
        self
    }

    pub fn expr(&self, a: &Aliases, now: i64) -> Option<String> {
        query_language::depth::deeper(|| self.sql.map(|e| e.render(a, now)))
    }
}
/// How the source time behind a temporal field is stored. The integer cache is
/// used only for broad candidates; the blob decides the exact predicate.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeStorage {
    Tuple,
    Event,
    Latest,
}
polylogue_model::stored_names!(TimeStorage { Tuple => "tuple", Event => "event", Latest => "latest" });
#[derive(Clone, Copy)]
pub enum TimeAttribute {
    Quality,
    Precision,
}
polylogue_model::stored_names!(TimeAttribute { Quality => "quality", Precision => "precision" });
pub(crate) struct StoredTime {
    pub cache: String,
    pub bytes: String,
    pub storage: TimeStorage,
}
impl StoredTime {
    fn attribute(&self, attribute: TimeAttribute) -> String {
        format!(
            "pl_time_attribute({}, '{}', '{}')",
            self.bytes,
            self.storage.as_str(),
            attribute.as_str()
        )
    }
}
pub(crate) fn stored_time(unit: Unit, field: &str, a: &Aliases) -> Option<StoredTime> {
    let (cache, bytes, storage) = match (unit, field) {
        (Messages | Blocks, "at") => ("{m}.at", "{m}.at_time", TimeStorage::Tuple),
        (ToolCalls, "at") => ("{m}.at", "{m}.at_time", TimeStorage::Tuple),
        (Sessions, "started") => ("{s}.started_at", "{s}.started_time", TimeStorage::Tuple),
        (Sessions, "last_active") => (
            "{s}.last_activity_at",
            "(SELECT pl_time_latest({s}_activity.at_time) FROM message {s}_activity WHERE {s}_activity.session_pk={s}.pk AND {s}_activity.at IS NOT NULL)",
            TimeStorage::Latest,
        ),
        (Events, "at") => ("{r}.at", "{r}.fields", TimeStorage::Event),
        (Documents, "source_time") => (
            "{r}.source_time",
            "(SELECT {r}_time.time FROM document_source {r}_time WHERE {r}_time.document_pk={r}.pk AND {r}_time.input_pk={r}.input_pk AND {r}_time.chain={r}.source_chain AND {r}_time.shown=1)",
            TimeStorage::Tuple,
        ),
        _ => return None,
    };
    Some(StoredTime {
        cache: a.expand(cache),
        bytes: a.expand(bytes),
        storage,
    })
}
pub(crate) fn time_window(unit: Unit, since: bool, a: &Aliases) -> Option<StoredTime> {
    let field = if unit == Sessions && !since {
        "started"
    } else {
        polylogue_query::units::spec(unit).main_time?
    };
    stored_time(unit, field, a)
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
                 WHERE ht.ref = 'session:' || {s}.id AND {} AND {})",
                current_assertion("hn"),
                crate::assertions::sql::in_conflict("hn")
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
                "EXISTS (SELECT 1 FROM block xb JOIN text tx ON tx.id = xb.text_id \
                 JOIN text_group xg ON xg.id = tx.group_id WHERE xb.message_pk = {m}.pk \
                 AND xb.kind = 'text' AND instr(pl_text_at(xg.body, tx.start, tx.length), char(96, 96, 96)) > 0)"
            )
        },
    },
];

/// How the store reads each declared field. The match is exhaustive, so a
/// field declared in the catalog fails the build until it has a rule here.
pub fn rule(id: FieldId) -> Rule {
    use FieldId::*;
    match id {
        SessionsEndsOpen => Rule::NONE.sql_fn(|a| crate::session_sql::ends_open(&a.s)),
        SessionsTriageScore => Rule::NONE.sql_fn(|a| crate::session_sql::triage_score(&a.s)),
        SessionsUnrecoveredFailures => Rule::NONE.sql_fn(|a| crate::session_sql::unrecovered_failures(&a.s)),
        SessionsFailureStreak => Rule::NONE.sql_fn(|a| crate::session_sql::failure_streak(&a.s)),
        SessionsOutlier => Rule::NONE.sql_fn(|a| crate::session_sql::outlier(&a.s)),
        SessionsCorrections => Rule::NONE.sql_fn(|a| crate::session_sql::corrections(&a.s)),
        SessionsOpenItems => Rule::NONE.sql_fn(|a| crate::session_sql::open_items(&a.s)),
        SessionsSubagentCost => Rule::NONE.sql_fn(|a| crate::measure::cost_of(&format!("SELECT child.pk FROM session child WHERE child.pk IN ({}) AND child.pk<>{}.pk",descendants(&a.s),a.s))),
        SessionsCostComplete => Rule::NONE.sql_fn(|a| crate::session_sql::cost_complete(&a.s)),
        SessionsCostWithParents => Rule::NONE.sql_fn(|a| crate::session_sql::cost_with_parents(&a.s)),
        MessagesLang => Rule::NONE.custom(custom::alternate_lang),
        MessagesAlternate => Rule::NONE.sql_fn(message_alternate),
        AnnotationsAt => Rule::NONE.sql("{r}.created_at"),
        AnnotationsRequest => Rule::NONE.sql("{r}.id").custom(custom::request),
        AnnotationsLinksTo => Rule::NONE.custom(custom::links_to),
        AnnotationsAnswered => Rule::NONE.sql_fn(|a| crate::session_sql::answered(&a.row)),
        AnnotationsOpen => Rule::NONE.sql_fn(|a| crate::session_sql::open_annotation(&a.row)),
        SessionsMonthDay => Rule::NONE.sql("pl_month_day({s}.last_activity_at)"),
        DocumentsMonthDay => Rule::NONE.sql("pl_month_day({r}.source_time)"),
        MessagesMonthDay => Rule::NONE.sql("pl_month_day({m}.at)"),
        BlocksMonthDay => Rule::NONE.sql("pl_month_day({m}.at)"),
        ToolCallsMonthDay => Rule::NONE.sql("pl_month_day({m}.at)"),
        EventsMonthDay => Rule::NONE.sql("pl_month_day({r}.at)"),
        AnnotationsMonthDay => Rule::NONE.sql("pl_month_day({r}.created_at)"),
        DeliveriesMonthDay => Rule::NONE.sql("pl_month_day({r}.at)"),
        TurnsMonthDay => Rule::NONE.sql("pl_month_day({r}.start)"),
        LiveIntervalsMonthDay => Rule::NONE.sql("pl_month_day({r}.start_at)"),
        ProcessesMonthDay => Rule::NONE.sql("pl_month_day({r}.started_at)"),
        AlternatesMonthDay => Rule::NONE.sql("pl_month_day(CAST(unixepoch({r}.created_at,'subsec')*1000 AS INTEGER))"),
        QueriesMonthDay => Rule::NONE.sql("pl_month_day({r}.saved_at)"),
        RunsMonthDay => Rule::NONE.sql("pl_month_day((CAST({r}.at AS INTEGER)/1000000))"),
        ResultsMonthDay => Rule::NONE.sql("pl_month_day({r}.ran_at)"),
        AlternatesId => Rule::NONE.sql("{r}.id"),
        AlternatesRoot => Rule::NONE.sql("{r}.root"),
        AlternatesBase => Rule::NONE.sql("{r}.base"),
        AlternatesSource => Rule::NONE.sql("{r}.source"),
        AlternatesKind => Rule::NONE.sql("{r}.kind"),
        AlternatesFromLang => Rule::NONE.sql("{r}.from_lang"),
        AlternatesToLang => Rule::NONE.sql("{r}.to_lang"),
        AlternatesLang => Rule::NONE.sql("{r}.to_lang"),
        AlternatesAuthor => Rule::NONE.sql("{r}.author"),
        AlternatesVia => Rule::NONE.sql("{r}.via"),
        AlternatesMethod => Rule::NONE.sql("{r}.method"),
        AlternatesCreatedAt => Rule::NONE.sql("CAST(unixepoch({r}.created_at,'subsec')*1000 AS INTEGER)"),
        AlternatesCurrent => Rule::NONE.sql_fn(|a| crate::variants::current_sql(&a.row)),
        AlternatesWithdrawn => Rule::NONE.sql_fn(|a| crate::variants::withdrawn_sql(&a.row)),
        AlternatesSince => Rule::NONE.custom(custom::since),
        AlternatesUntil => Rule::NONE.custom(custom::until),
        QueriesQuery => Rule::NONE.sql("'query:'||{r}.hash"),
        QueriesName => Rule::NONE.sql("{r}.name"),
        QueriesRevision => Rule::NONE.sql("{r}.revision"),
        QueriesText => Rule::NONE.sql("{r}.text"),
        QueriesSavedAt => Rule::NONE.sql("{r}.saved_at"),
        QueriesKept => Rule::NONE.sql("(SELECT 'result:'||k.id FROM source_kept_result k WHERE k.name={r}.name ORDER BY k.id DESC LIMIT 1)"),
        QueriesWatched => Rule::NONE.sql("EXISTS(SELECT 1 FROM source_assertion w JOIN source_revision_state ws ON ws.id=w.id AND ws.current=1 AND ws.withdrawn=0 WHERE w.kind='signal' AND w.key={r}.name)"),
        QueriesSince => Rule::NONE.custom(custom::since),
        QueriesUntil => Rule::NONE.custom(custom::until),
        RunsId => Rule::NONE.sql("'run:'||{r}.id"),
        RunsQuery => Rule::NONE.sql("'query:'||{r}.query"),
        RunsAt => Rule::NONE.sql("(CAST({r}.at AS INTEGER)/1000000)"),
        RunsReadAt => Rule::NONE.sql("{r}.read_at"),
        RunsAuthor => Rule::NONE.sql("json_extract({r}.author,'$')"),
        RunsVia => Rule::NONE.sql("{r}.via"),
        RunsCount => Rule::NONE.sql("CAST({r}.count AS INTEGER)"),
        RunsExact => Rule::NONE.sql("{r}.exact"),
        RunsOutcome => Rule::NONE.sql("{r}.outcome").custom(custom::run_outcome),
        RunsSince => Rule::NONE.custom(custom::since),
        RunsUntil => Rule::NONE.custom(custom::until),
        ResultsId => Rule::NONE.sql("'result:'||{r}.id"),
        ResultsName => Rule::NONE.sql("{r}.name"),
        ResultsQuery => Rule::NONE.sql("{r}.query"),
        ResultsRanAt => Rule::NONE.sql("{r}.ran_at"),
        ResultsRows => Rule::NONE.sql("coalesce(json_array_length({r}.rows_json,'$.rows'),0)"),
        ResultsSince => Rule::NONE.custom(custom::since),
        ResultsUntil => Rule::NONE.custom(custom::until),
        MessagesAtTimeQuality => Rule::NONE.time_attribute(Messages, "at",TimeAttribute::Quality),
        MessagesAtPrecision => Rule::NONE.time_attribute(Messages, "at",TimeAttribute::Precision),
        BlocksAtTimeQuality => Rule::NONE.time_attribute(Blocks, "at",TimeAttribute::Quality),
        BlocksAtPrecision => Rule::NONE.time_attribute(Blocks, "at",TimeAttribute::Precision),
        ToolCallsAtTimeQuality => Rule::NONE.time_attribute(ToolCalls, "at",TimeAttribute::Quality),
        ToolCallsAtPrecision => Rule::NONE.time_attribute(ToolCalls, "at",TimeAttribute::Precision),
        EventsAtTimeQuality => Rule::NONE.time_attribute(Events, "at",TimeAttribute::Quality),
        EventsAtPrecision => Rule::NONE.time_attribute(Events, "at",TimeAttribute::Precision),
        DeliveriesAtTimeQuality => Rule::NONE.time_attribute(Deliveries, "at",TimeAttribute::Quality),
        DeliveriesAtPrecision => Rule::NONE.time_attribute(Deliveries, "at",TimeAttribute::Precision),
        SessionsStartedAtTimeQuality => Rule::NONE.time_attribute(Sessions, "started",TimeAttribute::Quality),
        SessionsStartedAtPrecision => Rule::NONE.time_attribute(Sessions, "started",TimeAttribute::Precision),
        DocumentsSourceTimeTimeQuality => Rule::NONE.time_attribute(Documents, "source_time",TimeAttribute::Quality),
        DocumentsSourceTimePrecision => Rule::NONE.time_attribute(Documents, "source_time",TimeAttribute::Precision),
        SessionsLastActiveTimeQuality => Rule::NONE.time_attribute(Sessions, "last_active",TimeAttribute::Quality),
        SessionsLastActivePrecision => Rule::NONE.time_attribute(Sessions, "last_active",TimeAttribute::Precision),
        ToolCallsResultAtTimeQuality => Rule::NONE.time_attribute(ToolCalls, "result_at",TimeAttribute::Quality),
        ToolCallsResultAtPrecision => Rule::NONE.time_attribute(ToolCalls, "result_at",TimeAttribute::Precision),
        DocumentsTimeQuality => Rule::NONE.time_attribute(Documents, "source_time",TimeAttribute::Quality),
        DocumentsPrecision => Rule::NONE.time_attribute(Documents, "source_time",TimeAttribute::Precision),
        MessagesTimeQuality => Rule::NONE.time_attribute(Messages, "at",TimeAttribute::Quality),
        MessagesPrecision => Rule::NONE.time_attribute(Messages, "at",TimeAttribute::Precision),
        BlocksTimeQuality => Rule::NONE.time_attribute(Blocks, "at",TimeAttribute::Quality),
        BlocksPrecision => Rule::NONE.time_attribute(Blocks, "at",TimeAttribute::Precision),
        ToolCallsTimeQuality => Rule::NONE.time_attribute(ToolCalls, "at",TimeAttribute::Quality),
        ToolCallsPrecision => Rule::NONE.time_attribute(ToolCalls, "at",TimeAttribute::Precision),
        EventsTimeQuality => Rule::NONE.time_attribute(Events, "at",TimeAttribute::Quality),
        EventsPrecision => Rule::NONE.time_attribute(Events, "at",TimeAttribute::Precision),
        DocumentsId => Rule::NONE.sql("{r}.id"),
        DocumentsOrigin => Rule::NONE.sql("{r}.origin"),
        DocumentsPurpose => Rule::NONE.sql("{r}.purpose"),
        DocumentsAuthorship => Rule::NONE.sql("{r}.authorship"),
        DocumentsTitle => Rule::NONE.sql("{r}.title"),
        DocumentsSourceTime => Rule::NONE.sql("{r}.source_time"),
        DocumentsSince => Rule::NONE.custom(custom::since),
        DocumentsUntil => Rule::NONE.custom(custom::until),
        DocumentsVersions => Rule::NONE.sql("(SELECT count(DISTINCT ds.version_hash) FROM document_source ds WHERE ds.document_pk={r}.pk)"),
        SessionsId => Rule::NONE.sql("{s}.id"),
        SessionsOrigin => Rule::NONE.sql("{s}.origin"),
        SessionsPhysical => Rule::NONE.sql("1").custom(custom::physical),
        SessionsKind => Rule::NONE.sql("{s}.kind"),
        SessionsTitle => Rule::NONE.sql_fn(sessions_title),
        SessionsRepo => Rule::NONE.custom(custom::repo).sql("{s}.repo"),
        SessionsBranch => Rule::NONE.sql("{s}.branch"),
        SessionsCommit => Rule::NONE.custom(custom::commit),
        SessionsPr => Rule::NONE.custom(custom::pr),
        SessionsOutcome => Rule::NONE.sql_fn(sessions_outcome),
        SessionsHidden => Rule::NONE.sql_fn(sessions_hidden),
        SessionsCwd => Rule::NONE.custom(custom::cwd).sql("{s}.cwd"),
        // Gone: an export that lists every conversation of the session's account
        // (one input, or every input of its import) holds messages newer than the
        // session's last activity and does not hold the session.
        SessionsSource => Rule::NONE.sql("CASE WHEN {s}.stub = 1 THEN 'missing' WHEN EXISTS(SELECT 1 FROM export_reading {s}_e JOIN input_ref {s}_ei ON {s}_ei.pk={s}_e.input_pk WHERE {s}_e.origin={s}.origin AND {s}_e.latest_at>{s}.last_activity_at AND NOT EXISTS(SELECT 1 FROM parsed {s}_p JOIN input_ref {s}_pi ON {s}_pi.pk={s}_p.input_pk WHERE {s}_p.session_pk={s}.pk AND ({s}_pi.pk={s}_e.input_pk OR {s}_pi.id IN (SELECT {s}_m.input FROM import_member {s}_m JOIN import_member {s}_x ON {s}_x.import_id={s}_m.import_id WHERE {s}_x.input={s}_ei.id)))) THEN 'gone' ELSE 'present' END"),
        SessionsLive => Rule::NONE.custom(custom::live).sql_clock(sessions_live),
        SessionsWaiting => Rule::NONE.custom(custom::waiting).sql_clock(sessions_waiting),
        SessionsRunning => Rule::NONE.custom(custom::running).sql_clock(sessions_running),
        SessionsWaitingOn => Rule::NONE.sql_clock(sessions_waiting_on),
        SessionsStall => Rule::NONE.sql_clock(sessions_stall),
        SessionsProcessAlive => Rule::NONE.sql_clock(sessions_process_alive),
        SessionsHarnessStatus => Rule::NONE.sql("json_extract({s}.live_facts, '$.harness_status.word')"),
        SessionsOpenTool => Rule::NONE.sql("json_extract({s}.live_facts, '$.open_call.tool')"),
        SessionsEditedFiles => Rule::NONE.sql("(SELECT group_concat(fp.value, char(10)) FROM json_each({s}.live_facts, '$.edited_files') fp)"),
        SessionsTask => Rule::NONE.sql("json_extract({s}.live_facts, '$.task')"),
        SessionsLiveUntil => Rule::NONE.sql_clock(sessions_live_until),
        SessionsLiveSince => Rule::NONE.sql_clock(sessions_live_since),
        SessionsLastEventAt => Rule::NONE.sql("{s}.live_last_at"),
        SessionsTodo => Rule::NONE.sql("json_extract({s}.live_facts, '$.todo')"),
        SessionsLiveRank => Rule::NONE.sql_clock(sessions_live_rank),
        SessionsPeers => Rule::NONE.sql_clock(sessions_peers),
        SessionsRepoRoot => Rule::NONE.sql("{s}.repo_root"),
        SessionsHarness => Rule::NONE.sql("{s}.harness"),
        SessionsModel => Rule::NONE.custom(custom::model).sql(
        "(SELECT x.model FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL GROUP BY x.model ORDER BY count(*) DESC LIMIT 1)",
    ),
        SessionsTag => Rule::NONE.custom(custom::tag),
        SessionsHas => Rule::NONE.custom(custom::has),
        SessionsParent => Rule::NONE.custom(custom::parent),
        SessionsRoot => Rule::NONE.custom(custom::root).sql_fn(sessions_root),
        SessionsDepth => Rule::NONE.sql_fn(sessions_depth),
        SessionsLink => Rule::NONE.custom(custom::link),
        SessionsContinues => Rule::NONE.custom(custom::continues),
        SessionsContinuedBy => Rule::NONE.custom(custom::continued_by),
        SessionsConnected => Rule::NONE.custom(custom::connected),
        SessionsPath => Rule::NONE.custom(custom::session_tool_call),
        SessionsTool => Rule::NONE.custom(custom::session_tool_call),
        SessionsAction => Rule::NONE.custom(custom::session_tool_call),
        SessionsStarted => Rule::NONE.sql("{s}.started_at"),
        SessionsFirstSeen => Rule::NONE.sql_fn(session_first_seen),
        SessionsSeenBy => Rule::NONE.custom(custom::seen_by),
        SessionsLastActive => Rule::NONE.sql("{s}.last_activity_at"),
        SessionsSince => Rule::NONE.custom(custom::since),
        SessionsUntil => Rule::NONE.custom(custom::until),
        SessionsMessages => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"message_count")),
        SessionsHumanPrompts => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"human_prompt_count")),
        SessionsHumanCommands => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"human_command_count")),
        SessionsWords => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"word_count")),
        SessionsToolCalls => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"tool_call_count")),
        SessionsErrors => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"error_count")),
        SessionsInputTokens => Rule::NONE.sql_fn(sessions_input_tokens),
        SessionsOutputTokens => Rule::NONE.sql_fn(sessions_output_tokens),
        SessionsCacheReadTokens => Rule::NONE.sql_fn(sessions_cache_read_tokens),
        SessionsCacheWriteTokens => Rule::NONE.sql_fn(sessions_cache_write_tokens),
        SessionsCost => Rule::NONE.sql_fn(sessions_cost),
        SessionsTreeCost => Rule::NONE.sql_fn(sessions_tree_cost),
        SessionsWallSpan => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"wall_ms")),
        SessionsClusteredTime => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"clustered_ms")),
        SessionsToolTime => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"tool_sum_ms")),
        SessionsToolTimeUnion => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"tool_union_ms")),
        SessionsToolUntimed => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"tool_untimed")),
        SessionsEndsWith => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"ends_with")),
        SessionsLastStopReason => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"last_stop_reason")),
        SessionsTimeSource => Rule::NONE.sql_fn(|a| crate::session_sql::span(&a.s,"time_source")),
        SessionsWaitingModel => Rule::NONE.sql_fn(|a| format!("(SELECT ms FROM session_time st WHERE st.session_pk={s}.pk AND st.party='model' AND st.key='' AND {})",crate::session_sql::current(&a.s,"session_span"),s=a.s)),
        SessionsWaitingHuman => Rule::NONE.sql_fn(|a| format!("(SELECT ms FROM session_time st WHERE st.session_pk={s}.pk AND st.party='human' AND st.key='' AND {})",crate::session_sql::current(&a.s,"session_span"),s=a.s)),
        SessionsHumanWords => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"human_word_count")),
        SessionsModelWords => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"model_word_count")),
        SessionsThinkingMessages => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"thinking_count")),
        SessionsCompactions => Rule::NONE.sql_fn(|a| crate::session_sql::count(&a.s,"compaction_count")),
        SessionsSimilar => Rule::NONE.custom(custom::near),
        SessionsOutput => Rule::NONE.custom(custom::output),
        SessionsThinking => Rule::NONE.custom(custom::thinking),
        MessagesId => Rule::NONE.sql("{m}.id"),
        MessagesRole => Rule::NONE.sql("{m}.role"),
        MessagesWrittenBy => Rule::NONE.sql("{m}.written_by"),
        MessagesPromptKind => Rule::NONE.sql("{m}.prompt_kind"),
        MessagesModel => Rule::NONE.sql("{m}.model"),
        MessagesAt => Rule::NONE.sql("{m}.at"),
        MessagesFirstSeen => Rule::NONE.sql_fn(message_first_seen),
        MessagesSeenBy => Rule::NONE.custom(custom::seen_by),
        MessagesDuplicate => Rule::NONE.sql("({m}.duplicate_of IS NOT NULL)"),
        MessagesActive => Rule::NONE.sql("{m}.on_active_path"),
        MessagesHas => Rule::NONE.custom(custom::has),
        MessagesWords => Rule::NONE.sql("{m}.words"),
        MessagesInputTokens => Rule::NONE.sql("{m}.input_tokens"),
        MessagesOutputTokens => Rule::NONE.sql("{m}.output_tokens"),
        MessagesCacheReadTokens => Rule::NONE.sql("{m}.cache_read_tokens"),
        MessagesCacheWriteTokens => Rule::NONE.sql("{m}.cache_write_tokens"),
        MessagesCost => Rule::NONE.sql("{m}.cost_nusd"),
        MessagesStopReason => Rule::NONE.sql("{m}.stop_reason"),
        MessagesSince => Rule::NONE.custom(custom::since),
        MessagesUntil => Rule::NONE.custom(custom::until),
        MessagesSimilar => Rule::NONE.custom(custom::near),
        MessagesOutput => Rule::NONE.custom(custom::output),
        MessagesThinking => Rule::NONE.custom(custom::thinking),
        BlocksMessage => Rule::NONE.sql("{m}.id"),
        BlocksKind => Rule::NONE.sql("{b}.kind"),
        BlocksMediaType => Rule::NONE.sql("{b}.media_type"),
        BlocksFileName => Rule::NONE.sql("{b}.file_name"),
        BlocksAt => Rule::NONE.sql("{m}.at"),
        BlocksFirstSeen => Rule::NONE.sql_fn(message_first_seen),
        BlocksSeenBy => Rule::NONE.custom(custom::seen_by),
        BlocksSince => Rule::NONE.custom(custom::since),
        BlocksUntil => Rule::NONE.custom(custom::until),
        ToolCallsMessage => Rule::NONE.sql("{m}.id"),
        ToolCallsIdx => Rule::NONE.sql("{cb}.idx"),
        ToolCallsPosition => Rule::NONE.sql_fn(super::units::call_position),
        ToolCallsTool => Rule::NONE.sql("{t}.tool"),
        ToolCallsMcpServer => Rule::NONE.custom(custom::mcp_server).sql(
        "CASE WHEN {t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' THEN substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1) END",
    ),
        ToolCallsAction => Rule::NONE.sql("{t}.action"),
        ToolCallsCommand => Rule::NONE.custom(custom::tool_call_text).sql("{t}.command"),
        ToolCallsCommandNorm => Rule::NONE.sql("{t}.command_norm"),
        ToolCallsInput => Rule::NONE.custom(custom::input),
        ToolCallsOutput => Rule::NONE.custom(custom::output),
        ToolCallsPath => Rule::NONE.custom(custom::tool_call_text).sql("{t}.path"),
        ToolCallsOutcome => Rule::NONE.custom(custom::tool_call_outcome).sql_fn(call_outcome),
        ToolCallsExitCode => Rule::NONE.sql("{t}.exit_code"),
        ToolCallsAt => Rule::NONE.sql("{t}.at"),
        ToolCallsResultAt => Rule::NONE.sql("{t}.result_at"),
        ToolCallsReportedDuration => Rule::NONE.sql("{t}.reported_duration_ms"),
        ToolCallsFirstSeen => Rule::NONE.sql_fn(message_first_seen),
        ToolCallsSeenBy => Rule::NONE.custom(custom::seen_by),
        ToolCallsSince => Rule::NONE.custom(custom::since),
        ToolCallsUntil => Rule::NONE.custom(custom::until),
        ToolCallsChild => Rule::NONE.custom(custom::spawned_child),
        ToolCallsNext => Rule::NONE.custom(custom::adjacent_call),
        ToolCallsPrevious => Rule::NONE.custom(custom::adjacent_call),
        FilesPath => Rule::NONE.sql("{r}.path"),
        FilesReads => Rule::NONE.sql("{r}.reads"),
        FilesWrites => Rule::NONE.sql("{r}.writes"),
        FilesEdits => Rule::NONE.sql("{r}.edits"),
        FilesFirstAt => Rule::NONE.sql("{r}.first_at"),
        FilesLastAt => Rule::NONE.sql("{r}.last_at"),
        EventsId => Rule::NONE.sql("{r}.id"),
        EventsText => Rule::NONE.custom(custom::event_text),
        EventsInput => Rule::NONE.sql("{r}i.id"),
        EventsSource => Rule::NONE.sql("{r}i.format"),
        EventsOrigin => Rule::NONE.sql("COALESCE({s}.origin,{r}i.format)"),
        EventsData => Rule::NONE.sql("pl_block({r}.data)"),
        EventsSourceChain => Rule::NONE.sql("{r}.source_chain"),
        EventsSourceStart => Rule::NONE.sql("{r}.source_start"),
        EventsSourceEnd => Rule::NONE.sql("{r}.source_end"),
        EventsKind => Rule::NONE.sql("{r}.kind"),
        EventsAt => Rule::NONE.sql("{r}.at"),
        EventsFirstSeen => Rule::NONE.sql_fn(event_first_seen),
        EventsSeenBy => Rule::NONE.custom(custom::seen_by),
        EventsSince => Rule::NONE.custom(custom::since),
        EventsUntil => Rule::NONE.custom(custom::until),
        SessionLinksKind => Rule::NONE.sql("{r}.kind"),
        SessionLinksHow => Rule::NONE.sql("{r}.how"),
        SessionLinksCopiesPrefix => Rule::NONE.sql("{r}.copies_prefix"),
        SessionLinksParent => Rule::NONE.custom(custom::link_parent).sql("{r}.parent_id"),
        SessionLinksChild => Rule::NONE.custom(custom::link_child).sql("{r}.child_id"),
        SessionLinksPending => Rule::NONE.sql("(NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = {r}.parent_id))"),
        SessionLinksEvidence => Rule::NONE.sql("{r}.evidence"),
        SessionLinksBranchPoint => Rule::NONE.sql("{r}.branch_point"),
        AnnotationsId => Rule::NONE.sql("{r}.id"),
        AnnotationsReference => Rule::NONE.sql("pl_assertion_ref({r}.root,{r}.id)"),
        AnnotationsKind => Rule::NONE.sql("{r}.kind"),
        AnnotationsKey => Rule::NONE.sql("{r}.key"),
        AnnotationsValue => Rule::NONE.sql("{r}.value"),
        AnnotationsAuthor => Rule::NONE.sql("{r}.author"),
        AnnotationsAgent => Rule::NONE.sql("({r}.author = 'agent' OR {r}.author LIKE 'agent:%')"),
        AnnotationsJudged => Rule::NONE.sql_fn(notes_judged),
        AnnotationsAbout => Rule::NONE.custom(custom::assertion_about).sql("(SELECT group_concat(ref, ' ') FROM (SELECT nt.ref FROM source_link nt WHERE nt.record = {r}.id AND nt.role = 'about' ORDER BY nt.ref))"),
        AnnotationsWriteCounter => Rule::NONE.sql("{r}.commit_seq"),
        AnnotationsCreatedAt => Rule::NONE.sql("{r}.created_at"),
        AnnotationsRoot => Rule::NONE.sql("{r}.root"),
        AnnotationsCurrent => Rule::NONE.sql_fn(notes_current),
        AnnotationsConflict => Rule::NONE.sql_fn(|a| crate::assertions::sql::in_conflict(&a.row)),
        AnnotationsWithdrawn => Rule::NONE.sql_fn(notes_withdrawn),
        AnnotationsSince => Rule::NONE.custom(custom::since),
        AnnotationsSeenBy => Rule::NONE.custom(custom::seen_by),
        AnnotationsUntil => Rule::NONE.custom(custom::until),
        DeliveriesId => Rule::NONE.sql("{r}.id"),
        DeliveriesRecipient => Rule::NONE.custom(custom::recipient).sql("{r}.recipient_id"),
        DeliveriesEvent => Rule::NONE.sql("{r}.event"),
        DeliveriesSender => Rule::NONE.sql("{r}.sender"),
        DeliveriesOutcome => Rule::NONE.sql_fn(deliveries_outcome),
        DeliveriesDetail => Rule::NONE.sql_fn(deliveries_detail),
        DeliveriesText => Rule::NONE.sql("pl_text({r}.body)"),
        DeliveriesRefs => Rule::NONE.custom(custom::refs),
        DeliveriesLeftOut => Rule::NONE.custom(custom::left_out),
        DeliveriesAt => Rule::NONE.sql("{r}.at"),
        DeliveriesBytes => Rule::NONE.sql("{r}.bytes"),
        LiveIntervalsSession => Rule::NONE.custom(custom::live_session).sql("{s}.id"),
        LiveIntervalsAspect => Rule::NONE.sql("{r}.aspect"),
        LiveIntervalsValue => Rule::NONE.sql("{r}.value"),
        LiveIntervalsStart => Rule::NONE.sql("{r}.start_at"),
        LiveIntervalsEnd => Rule::NONE.sql("{r}.end_at"),
        LiveIntervalsDurationMs => Rule::NONE.sql("({r}.end_at - {r}.start_at)"),
        LiveIntervalsClosedBy => Rule::NONE.sql("{r}.closed_by"),
        ProcessesSession => Rule::NONE.custom(custom::live_session).sql("{s}.id"),
        ProcessesBootId => Rule::NONE.sql("{r}.boot_id"),
        ProcessesPid => Rule::NONE.sql("{r}.pid"),
        ProcessesStart => Rule::NONE.sql("{r}.start"),
        ProcessesParentPid => Rule::NONE.sql("{r}.parent_pid"),
        ProcessesComm => Rule::NONE.sql("{r}.comm"),
        ProcessesArgv => Rule::NONE.sql("{r}.argv"),
        ProcessesCwd => Rule::NONE.sql("{r}.cwd"),
        ProcessesStartedAt => Rule::NONE.sql("{r}.started_at"),
        ProcessesEndedAt => Rule::NONE.sql("{r}.ended_at"),
        ProcessesRole => Rule::NONE.sql("{r}.role"),
        ProcessesToolCall => Rule::NONE.sql_fn(processes_tool_call),
        FeedKind => Rule::NONE.sql("{r}.kind"),
        FeedRef => Rule::NONE.sql("{r}.ref"),
        FeedSeq => Rule::NONE.sql("{r}.seq"),
        FeedAssertionKind => Rule::NONE.sql("{r}.assertion_kind"),
        FeedAfter => Rule::NONE.custom(custom::changes_after),
        FeedSessions => Rule::NONE.custom(custom::selection),
        FeedMessages => Rule::NONE.custom(custom::selection),
        TurnsId => Rule::NONE.sql("{r}.id"),
        TurnsStart => Rule::NONE.sql("{r}.start"),
        TurnsEnd => Rule::NONE.sql("{r}.end"),
        TurnsBurst => Rule::NONE.sql("{r}.burst"),
        TurnsPrompts => Rule::NONE.sql("{r}.prompts"),
        TurnsSlashCommand => Rule::NONE.sql("{r}.slash_command"),
        TurnsAnswer => Rule::NONE.sql("{r}.answer"),
        TurnsAttempts => Rule::NONE.sql("{r}.attempts"),
        TurnsPath => Rule::NONE.sql("{r}.path"),
        TurnsLatency => Rule::NONE.sql("{r}.latency"),
        TurnsToolCalls => Rule::NONE.sql("{r}.tool_calls"),
        TurnsErrors => Rule::NONE.sql("{r}.errors"),
        TurnsInputTokens => Rule::NONE.sql("{r}.input_tokens"),
        TurnsOutputTokens => Rule::NONE.sql("{r}.output_tokens"),
        TurnsCost => Rule::NONE.sql("{r}.cost_nusd"),
        TurnsAbandoned => Rule::NONE.sql("{r}.abandoned"),
        TurnsCorrected => Rule::NONE.sql_fn(turns_corrected),
    }
}
pub use polylogue_query::fields::lookup;

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

/// When `s`'s live state or stall changes with no new live events; NULL when
/// only new live events can change them.
pub fn live_until(s: &str, now: i64) -> String {
    live_part(s, "until", now)
}

/// The value of a field over `a` at `now`.
pub fn sql(unit: Unit, name: &str, a: &Aliases, now: i64) -> Option<String> {
    lookup(unit, name).and_then(|id| rule(id).expr(a, now))
}

/// The sum of a column over session `s`'s own messages.
/// A session's token kind over its own responses, the population every
/// usage figure sums.
fn own_sum(s: &str, col: &str) -> String {
    crate::measure::own_responses_sum(&format!("{s}.pk"), col)
}

fn sessions_hidden(a: &Aliases) -> String {
    hidden(&a.s)
}

fn sessions_root(a: &Aliases) -> String {
    crate::lineage::value_sql(&format!("{}.pk", a.s), "root")
}

fn sessions_depth(a: &Aliases) -> String {
    crate::lineage::value_sql(&format!("{}.pk", a.s), "depth")
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
    crate::measure::cost_of(&format!("{}.pk", a.s))
}

fn sessions_tree_cost(a: &Aliases) -> String {
    crate::measure::cost_of(&descendants(&a.s))
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

/// The operator's judgment of this exact revision: the key of the current
/// operator `judgment` about it. Judging again revises the judgment, so one
/// family holds it; several current values that differ are `conflict`.
fn notes_judged(a: &Aliases) -> String {
    format!(
        "(SELECT CASE WHEN count(DISTINCT j.key) > 1 THEN 'conflict' ELSE min(j.key) END FROM source_assertion j JOIN source_link jt ON jt.record=j.id AND jt.role='about' WHERE jt.ref='assertion:' || {r}.root || '@' || {r}.id AND j.kind='judgment' AND j.author='operator' AND {live})",
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

pub(crate) fn call_outcome(a: &Aliases) -> String {
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

/// The session and every session below it through structural links.
pub fn descendants(s: &str) -> String {
    format!(
        "WITH RECURSIVE tr(pk) AS (SELECT {s}.pk UNION SELECT l.child_pk FROM session_link l \
         JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         WHERE l.kind IN ({STRUCTURAL_KINDS}) AND l.child_pk IS NOT NULL) SELECT pk FROM tr"
    )
}

/// The known logical root, or the session itself when its ancestry is unresolved.
pub fn logical_root(s: &str) -> String {
    format!(
        "(SELECT r.pk FROM session r WHERE r.id = coalesce({}, {s}.id))",
        crate::lineage::value_sql(&format!("{s}.pk"), "root")
    )
}

#[cfg(test)]
mod tests;

pub(crate) fn message_variant_match(variant: &str, part: &str, message: &str) -> String {
    format!(
        "({variant}.source='message:'||{message}.id OR ({variant}.source>'message:'||{message}.id||'#' AND {variant}.source<'message:'||{message}.id||'$') OR {part}.of='message:'||{message}.id OR ({part}.of>'block:'||{message}.id||'#' AND {part}.of<'block:'||{message}.id||'$'))"
    )
}
fn message_alternate(a: &Aliases) -> String {
    format!(
        "EXISTS(SELECT 1 FROM source_variant_part vp JOIN source_variant v ON v.id=vp.variant WHERE {} AND {})",
        crate::variants::current_sql("v"),
        message_variant_match("v", "vp", &a.m)
    )
}
