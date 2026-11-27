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
polylogue_model::stored_names! {
    /// How the source time behind a temporal field is stored. The integer cache is
    /// used only for broad candidates; the blob decides the exact predicate.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum TimeStorage {
        Tuple => "tuple",
        Event => "event",
        Latest => "latest",
    }
}
polylogue_model::stored_names! {
    #[derive(Clone, Copy)]
    pub enum TimeAttribute {
        Quality => "quality",
        Precision => "precision",
    }
}
/// A time a condition tests, as the columns that hold it (docs/query.md,
/// times): its start, the exclusive end of its period (NULL for an
/// instant, for an unknown time and for a calendar grain without a bound)
/// and its precision, all in milliseconds. `bytes` and `storage` hold the
/// whole source time, read only for its quality and by order relations.
pub(crate) struct StoredTime {
    pub at: String,
    pub until: String,
    pub precision: String,
    /// The table whose `<table>_period` index gives its longest stored
    /// period, so a lower bound on the end of a period reads a range of
    /// starts; `None` for a table small enough to read whole.
    pub periods: Option<&'static str>,
    pub bytes: String,
    pub storage: TimeStorage,
}

impl StoredTime {
    fn attribute(&self, attribute: TimeAttribute) -> String {
        match attribute {
            TimeAttribute::Precision => self.precision.clone(),
            TimeAttribute::Quality => format!(
                "pl_time_attribute({}, '{}', 'quality')",
                self.bytes,
                self.storage.as_str()
            ),
        }
    }

    /// The last millisecond of the time's period: the instant itself, the
    /// end of a period less one, or NULL when the time or its end is
    /// unknown, so any comparison of it is unknown too.
    pub(crate) fn last(&self) -> String {
        format!(
            "coalesce({} - 1, iif({} = 'instant', {}, NULL))",
            self.until, self.precision, self.at
        )
    }

    /// The longest stored period, read by one seek in its index.
    pub(crate) fn longest(&self) -> Option<String> {
        self.periods.map(|table| {
            format!(
                "(SELECT coalesce(max(at_until - at), 0) FROM {table} \
                 INDEXED BY {table}_period WHERE at_until IS NOT NULL)"
            )
        })
    }
}
pub(crate) fn stored_time(unit: Unit, field: &str, a: &Aliases) -> Option<StoredTime> {
    let (at, until, precision, periods, bytes, storage) = match (unit, field) {
        (Messages | Blocks | ToolCalls, "at") => (
            "{m}.at",
            "{m}.at_until",
            "{m}.at_precision",
            Some("message"),
            "{m}.at_time".to_owned(),
            TimeStorage::Tuple,
        ),
        (Sessions, "started") => (
            "{s}.started_at",
            "{s}.started_until",
            "{s}.started_precision",
            None,
            "{s}.started_time".to_owned(),
            TimeStorage::Tuple,
        ),
        // The latest of the session's message times, which the session
        // summary keeps; its quality is read from the messages.
        (Sessions, "last_active") => (
            "{s}.last_active_at",
            "{s}.last_active_until",
            "{s}.last_active_precision",
            None,
            "(SELECT pl_time_latest({s}_activity.at_time) FROM message {s}_activity \
              WHERE {s}_activity.session_pk={s}.pk AND {s}_activity.at IS NOT NULL)"
                .to_owned(),
            TimeStorage::Latest,
        ),
        (Events, "at") => (
            "{r}.at",
            "{r}.at_until",
            "{r}.at_precision",
            Some("event"),
            crate::content::BODY_BYTES_SQL.to_owned(),
            TimeStorage::Event,
        ),
        (Documents, "source_time") => (
            "{r}.at",
            "{r}.at_until",
            "{r}.at_precision",
            None,
            "(SELECT {r}_time.at_details FROM document_parse {r}_time WHERE {r}_time.document_id={r}.id AND {r}_time.input_pk={r}.input_pk AND {r}_time.chain={r}.source_chain)".to_owned(),
            TimeStorage::Tuple,
        ),
        _ => return None,
    };
    Some(StoredTime {
        at: a.expand(at),
        until: a.expand(until),
        precision: a.expand(precision),
        periods,
        bytes: a.expand(&bytes),
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
                "EXISTS (SELECT 1 FROM source_annotation hn JOIN source_annotation_link ht ON ht.annotation = hn.id AND ht.role = 'about' \
                 WHERE (ht.ref = 'session:' || {s}.id OR ht.ref IN (SELECT 'message:' || xm.id FROM message xm WHERE xm.session_pk = {s}.pk)) \
                 AND {})",
                current_annotation("hn")
            )
        },
    },
    Has {
        name: "conflict",
        sql: |s| {
            format!(
                "EXISTS (SELECT 1 FROM source_annotation hn JOIN source_annotation_link ht ON ht.annotation = hn.id AND ht.role = 'about' \
                 WHERE ht.ref = 'session:' || {s}.id AND {} AND {})",
                current_annotation("hn"),
                crate::annotations::sql::in_conflict("hn")
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
                 JOIN content_group xg ON xg.id = tx.group_id WHERE xb.message_pk = {m}.pk \
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
        SessionsSubagentCost => Rule::NONE.sql_fn(subagent_cost),
        SessionsCostComplete => Rule::NONE.sql_fn(|a| crate::session_sql::cost_complete(&a.s)),
        SessionsCostWithParents => Rule::NONE.sql_fn(|a| crate::session_sql::cost_with_parents(&a.s)),
        MessagesLang => Rule::NONE.custom(custom::alternate_lang),
        MessagesAlternate => Rule::NONE.sql_fn(message_alternate),
        AnnotationsAt => Rule::NONE.sql("{r}.created_at"),
        AnnotationsRequest => Rule::NONE.sql("{r}.id").custom(custom::request),
        AnnotationsLinksTo => Rule::NONE.custom(custom::links_to),
        AnnotationsAnswered => Rule::NONE.sql_fn(|a| crate::session_sql::answered(&a.row)),
        AnnotationsOpen => Rule::NONE.sql_fn(|a| crate::session_sql::open_annotation(&a.row)),
        SessionsMonthDay => Rule::NONE.sql("pl_month_day({s}.last_active_at)"),
        DocumentsMonthDay => Rule::NONE.sql("pl_month_day({r}.at)"),
        MessagesMonthDay => Rule::NONE.sql("pl_month_day({m}.at)"),
        BlocksMonthDay => Rule::NONE.sql("pl_month_day({m}.at)"),
        ToolCallsMonthDay => Rule::NONE.sql("pl_month_day({m}.at)"),
        EventsMonthDay => Rule::NONE.sql("pl_month_day({r}.at)"),
        AnnotationsMonthDay => Rule::NONE.sql("pl_month_day({r}.created_at)"),
        DeliveriesMonthDay => Rule::NONE.sql("pl_month_day({r}.at)"),
        TurnsMonthDay => Rule::NONE.sql("pl_month_day({r}.start)"),
        LiveIntervalsMonthDay => Rule::NONE.sql("pl_month_day({r}.started_at)"),
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
        AlternatesCurrent => Rule::NONE.sql_fn(|a| crate::alternates::current_sql(&a.row)),
        AlternatesWithdrawn => Rule::NONE.sql_fn(|a| crate::alternates::withdrawn_sql(&a.row)),
        AlternatesSince => Rule::NONE.custom(custom::since),
        AlternatesUntil => Rule::NONE.custom(custom::until),
        QueriesQuery => Rule::NONE.sql("'query:'||{r}.hash"),
        QueriesName => Rule::NONE.sql("{r}.name"),
        QueriesRevision => Rule::NONE.sql("{r}.revision"),
        QueriesText => Rule::NONE.sql("{r}.text"),
        QueriesSavedAt => Rule::NONE.sql("{r}.saved_at"),
        QueriesKept => Rule::NONE.sql("(SELECT 'result:'||k.id FROM source_saved_rows k WHERE k.name={r}.name ORDER BY k.id DESC LIMIT 1)"),
        QueriesWatched => Rule::NONE.sql("EXISTS(SELECT 1 FROM source_annotation w JOIN source_revision_state ws ON ws.id=w.id AND ws.current=1 AND ws.withdrawn=0 WHERE w.kind='signal' AND w.key={r}.name)"),
        QueriesSince => Rule::NONE.custom(custom::since),
        QueriesUntil => Rule::NONE.custom(custom::until),
        RunsId => Rule::NONE.sql("'run:'||{r}.id"),
        RunsQuery => Rule::NONE.sql("'query:'||{r}.query"),
        RunsAt => Rule::NONE.sql("(CAST({r}.at AS INTEGER)/1000000)"),
        RunsReadAt => Rule::NONE.sql("{r}.read_position"),
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
        ImportJobsId => Rule::NONE.sql("{r}.id"),
        ImportJobsProvider => Rule::NONE.sql("{r}.provider"),
        ImportJobsChangedSince => Rule::NONE.sql("{r}.since"),
        ImportJobsState => Rule::NONE.sql_clock(super::units::import_job_state),
        ImportJobsListed => Rule::NONE.sql(super::units::import_job_count!("listed")),
        ImportJobsCaptured => Rule::NONE.sql(super::units::import_job_count!("captured")),
        ImportJobsUnchanged => Rule::NONE.sql(super::units::import_job_count!("unchanged")),
        ImportJobsFailed => Rule::NONE.sql(super::units::import_job_count!("failed")),
        ImportJobsPosition => Rule::NONE.sql(super::units::IMPORT_JOB_POSITION),
        ImportJobsDetail => Rule::NONE.sql(super::units::import_job_count!("detail")),
        ImportJobsCreatedAt => Rule::NONE.sql("{r}.created_at"),
        ImportJobsUpdatedAt => Rule::NONE.sql(super::units::IMPORT_JOB_UPDATED),
        ImportJobsClaimedBy => Rule::NONE.sql_clock(super::units::import_job_holder),
        ImportJobsSince => Rule::NONE.custom(custom::since),
        ImportJobsUntil => Rule::NONE.custom(custom::until),
        ImportJobsMonthDay => Rule::NONE.sql("pl_month_day({r}.created_at)"),
        ChatActionsId => Rule::NONE.sql("{r}.id"),
        ChatActionsProvider => Rule::NONE.sql("json_extract({r}.request_json,'$.provider')"),
        ChatActionsConversation => Rule::NONE.sql("json_extract({r}.request_json,'$.conversation')"),
        ChatActionsModel => Rule::NONE.sql("json_extract({r}.request_json,'$.model')"),
        ChatActionsText => Rule::NONE.sql("json_extract({r}.request_json,'$.text')"),
        ChatActionsKind => Rule::NONE.sql(super::units::CHAT_ACTION_KIND),
        ChatActionsState => Rule::NONE.sql_clock(super::units::chat_action_state),
        ChatActionsPhase => Rule::NONE.sql(super::units::CHAT_ACTION_PHASE),
        ChatActionsNeedsApproval => Rule::NONE.sql("{r}.needs_approval"),
        ChatActionsCreatedAt => Rule::NONE.sql("{r}.created_at"),
        ChatActionsUpdatedAt => Rule::NONE.sql(super::units::CHAT_ACTION_UPDATED),
        ChatActionsClaimedBy => Rule::NONE.sql_clock(super::units::chat_action_holder),
        ChatActionsDetail => Rule::NONE.sql(super::units::CHAT_ACTION_DETAIL),
        ChatActionsSince => Rule::NONE.custom(custom::since),
        ChatActionsUntil => Rule::NONE.custom(custom::until),
        ChatActionsMonthDay => Rule::NONE.sql("pl_month_day({r}.created_at)"),
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
        DocumentsSourceTime => Rule::NONE.sql("{r}.at"),
        DocumentsSince => Rule::NONE.custom(custom::since),
        DocumentsUntil => Rule::NONE.custom(custom::until),
        DocumentsVersions => Rule::NONE.sql("(SELECT count(DISTINCT ds.hash) FROM document_parse ds WHERE ds.document_id={r}.id)"),
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
        // session's last activity and does not hold the session; or the newest
        // chain of an input that states the session states it gone (the Drive
        // collector's listing of a removed prompt).
        SessionsSource => Rule::NONE.sql("CASE WHEN {s}.has_transcript = 0 THEN 'missing' WHEN EXISTS(SELECT 1 FROM session_parse {s}_g WHERE {s}_g.session_id={s}.id AND json_extract({s}_g.facts,'$.gone')=1 AND {s}_g.chain=(SELECT max({s}_n.chain) FROM session_parse {s}_n WHERE {s}_n.input_pk={s}_g.input_pk AND {s}_n.session_id={s}_g.session_id)) THEN 'gone' WHEN EXISTS(SELECT 1 FROM export_parse {s}_e JOIN input_ref {s}_ei ON {s}_ei.pk={s}_e.input_pk WHERE {s}_e.origin={s}.origin AND {s}_e.latest_at>{s}.last_active_at AND NOT EXISTS(SELECT 1 FROM session_parse {s}_p JOIN input_ref {s}_pi ON {s}_pi.pk={s}_p.input_pk WHERE {s}_p.session_id={s}.id AND ({s}_pi.pk={s}_e.input_pk OR {s}_pi.id IN (SELECT {s}_m.input FROM import_member {s}_m JOIN import_member {s}_x ON {s}_x.import_id={s}_m.import_id WHERE {s}_x.input={s}_ei.id)))) THEN 'gone' ELSE 'present' END"),
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
        SessionsLastActive => Rule::NONE.sql("{s}.last_active_at"),
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
        ToolCallsIdx => Rule::NONE.sql("{cb}.position"),
        ToolCallsPosition => Rule::NONE.sql_fn(super::units::call_position),
        ToolCallsTool => Rule::NONE.sql("{t}.tool"),
        ToolCallsMcpServer => Rule::NONE.custom(custom::mcp_server).sql(
        "CASE WHEN {t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' AND instr(substr({t}.tool, 6), '__') > 0 THEN substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1) END",
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
        EventsSource => Rule::NONE.sql("{r}i.origin"),
        EventsOrigin => Rule::NONE.sql("COALESCE({s}.origin,{r}i.origin)"),
        EventsData => Rule::NONE.sql(crate::content::DATA_SQL),
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
        SessionLinksReason => Rule::NONE.sql("{r}.reason"),
        SessionLinksCopiesPrefix => Rule::NONE.sql("({r}.copies_prefix OR EXISTS(SELECT 1 FROM shared_prefix {r}x WHERE {r}x.parent_id = {r}.parent_id AND {r}x.child_id = {r}.child_id))"),
        SessionLinksParent => Rule::NONE.custom(custom::link_parent).sql("{r}.parent_id"),
        SessionLinksChild => Rule::NONE.custom(custom::link_child).sql("{r}.child_id"),
        SessionLinksPending => Rule::NONE.sql("(NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = {r}.parent_id))"),
        SessionLinksEvidence => Rule::NONE.sql("{r}.evidence"),
        SessionLinksBranchPoint => Rule::NONE.sql("{r}.branch_point"),
        AnnotationsId => Rule::NONE.sql("{r}.id"),
        AnnotationsReference => Rule::NONE.sql("pl_annotation_ref({r}.root,{r}.id)"),
        AnnotationsKind => Rule::NONE.sql("{r}.kind"),
        AnnotationsKey => Rule::NONE.sql("{r}.key"),
        AnnotationsValue => Rule::NONE.sql("{r}.value"),
        AnnotationsAuthor => Rule::NONE.sql("{r}.author"),
        AnnotationsAgent => Rule::NONE.sql("({r}.author = 'agent' OR {r}.author LIKE 'agent:%')"),
        AnnotationsVia => Rule::NONE.sql("{r}.via"),
        AnnotationsJudged => Rule::NONE.sql_fn(notes_judged),
        AnnotationsAbout => Rule::NONE.custom(custom::annotation_about).sql("(SELECT group_concat(ref, ' ') FROM (SELECT nt.ref FROM source_annotation_link nt WHERE nt.annotation = {r}.id AND nt.role = 'about' ORDER BY nt.ref))"),
        AnnotationsWriteCounter => Rule::NONE.sql("{r}.commit_seq"),
        AnnotationsCreatedAt => Rule::NONE.sql("{r}.created_at"),
        AnnotationsRoot => Rule::NONE.sql("{r}.root"),
        AnnotationsCurrent => Rule::NONE.sql_fn(notes_current),
        AnnotationsConflict => Rule::NONE.sql_fn(|a| crate::annotations::sql::in_conflict(&a.row)),
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
        LiveIntervalsStart => Rule::NONE.sql("{r}.started_at"),
        LiveIntervalsEnd => Rule::NONE.sql("{r}.ended_at"),
        LiveIntervalsDurationMs => Rule::NONE.sql("({r}.ended_at - {r}.started_at)"),
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
        FeedAnnotationKind => Rule::NONE.sql("{r}.annotation_kind"),
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
    format!("pl_live({s}.live_facts, {s}.last_active_at, {now}, '{part}')")
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

/// The sum of a column over session `s`'s own responses, the population
/// every usage figure sums.
fn own_sum(s: &str, col: &str) -> String {
    crate::measure::own_responses_sum(&format!("{s}.pk"), col)
}

fn sessions_hidden(a: &Aliases) -> String {
    hidden(&a.s)
}

fn sessions_root(a: &Aliases) -> String {
    crate::lineage::lineage_column_sql(&format!("{}.pk", a.s), "root")
}

fn sessions_depth(a: &Aliases) -> String {
    crate::lineage::lineage_column_sql(&format!("{}.pk", a.s), "depth")
}

fn sessions_title(a: &Aliases) -> String {
    crate::session_sql::display_title(&a.s)
}

fn sessions_outcome(a: &Aliases) -> String {
    format!(
        "(SELECT CASE WHEN count(*)=0 THEN 'none' WHEN count(*)=1 THEN min(lower(CASE json_type(n.value) WHEN 'text' THEN json_extract(n.value,'$') ELSE json_extract(n.value,'$.text') END)) ELSE NULL END FROM source_annotation n JOIN source_annotation_link nt ON nt.annotation=n.id AND nt.role='about' WHERE n.kind='outcome' AND nt.ref='session:' || {s}.id AND {live})",
        s = a.s,
        live = current_annotation("n"),
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
        latest_delivery_event(&a.row, "outcome")
    )
}

fn deliveries_detail(a: &Aliases) -> String {
    latest_delivery_event(&a.row, "detail")
}

fn turns_corrected(a: &Aliases) -> String {
    format!(
        "EXISTS (SELECT 1 FROM source_annotation nc JOIN source_annotation_link nct ON nct.annotation = nc.id AND nct.role = 'about' \
         WHERE nc.kind = 'correction' AND nct.ref = 'message:' || {r}.answer \
         AND {} AND NOT {})",
        current_annotation("nc"),
        withdrawn_annotation("nc"),
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
    crate::measure::cost_of(&spawn_tree(&a.s))
}

fn sessions_live(a: &Aliases, now: i64) -> String {
    live_state(&a.s, now)
}

fn sessions_waiting(a: &Aliases, now: i64) -> String {
    format!("({} = 'waiting')", live_state(&a.s, now))
}

fn sessions_running(a: &Aliases, now: i64) -> String {
    format!("({} IN {})", live_state(&a.s, now), running_states_sql())
}

/// The live states that count as running, as a SQL list; the model's
/// `LiveState::running` is the one definition.
pub(crate) fn running_states_sql() -> String {
    let names: Vec<String> = polylogue_model::live::LiveState::ALL
        .iter()
        .filter(|state| state.running())
        .map(|state| format!("'{}'", state.as_str()))
        .collect();
    format!("({})", names.join(", "))
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
    format!("({})", current_annotation(&a.row))
}

/// The operator's judgment of this exact revision: the key of the current
/// operator `judgment` about it. Judging again revises the judgment, so one
/// family holds it; several current values that differ are `conflict`.
fn notes_judged(a: &Aliases) -> String {
    format!(
        "(SELECT CASE WHEN count(DISTINCT j.key) > 1 THEN 'conflict' ELSE min(j.key) END FROM source_annotation j JOIN source_annotation_link jt ON jt.annotation=j.id AND jt.role='about' WHERE jt.ref='annotation:' || {r}.root || '@' || {r}.id AND j.kind='judgment' AND j.author='operator' AND {live})",
        r = a.row,
        live = current_annotation("j"),
    )
}

fn notes_withdrawn(a: &Aliases) -> String {
    format!("({})", withdrawn_annotation(&a.row))
}

pub use crate::annotations::sql::{current_annotation, withdrawn_annotation};

/// When Polylogue first had this byte-framed record, over all revisions
/// long enough to hold it. The chain and any others that agree through the
/// record's end form a connected part of the agreement tree.
pub fn record_first_seen(input: &str, route: &str, chain: &str, start: &str, end: &str) -> String {
    format!(
        "(WITH RECURSIVE held(chain) AS (SELECT {chain} \
           UNION SELECT hc.begins_like FROM held JOIN source_chain hc \
             ON hc.input = {input} AND hc.number = held.chain \
             WHERE {route} NOT IN ('copy', 'export') AND {end} > {start} \
               AND hc.begins_like IS NOT NULL AND hc.agrees_to >= {end} \
           UNION SELECT hc.number FROM held JOIN source_chain hc \
             ON hc.input = {input} AND hc.begins_like = held.chain \
             WHERE {route} NOT IN ('copy', 'export') AND {end} > {start} \
               AND hc.agrees_to >= {end}) \
         SELECT min(hr.seen_at) FROM held CROSS JOIN source_revision hr \
         WHERE hr.input = {input} AND hr.chain = held.chain \
           AND hr.length > {start} AND hr.length >= {end})"
    )
}

/// When Polylogue first had message `m`'s bytes: the earliest over every
/// reading that holds a version of it.
pub fn first_seen_of(m: &str) -> String {
    format!(
        "(SELECT min({seen}) FROM assembled_message_parse fs JOIN input_ref fi ON fi.pk = fs.input_pk \
         WHERE fs.message_pk = {m}.pk)",
        seen = record_first_seen(
            "fi.id",
            "fi.route",
            "fs.chain",
            "fs.source_start",
            "fs.source_start + fs.source_length"
        )
    )
}

/// A stored `unknown` call with no result that the session's live state
/// names as its open call: the `pending` outcome, decided when read.
pub(crate) const OPEN_CALL: &str = "({t}.result_message_id IS NULL AND {t}.outcome = 'unknown' \
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
        "(SELECT min({seen}) FROM message fm JOIN assembled_message_parse fs ON fs.message_pk = fm.pk \
         JOIN input_ref fi ON fi.pk = fs.input_pk WHERE fm.session_pk = {}.pk)",
        a.s,
        seen = record_first_seen(
            "fi.id",
            "fi.route",
            "fs.chain",
            "fs.source_start",
            "fs.source_start + fs.source_length"
        )
    )
}

/// An event's first observation, as for a message, over its chain and the
/// chains that hold its record.
pub fn event_first_seen(a: &Aliases) -> String {
    format!(
        "(SELECT {seen} FROM input_ref fi WHERE fi.pk = {r}.input_pk)",
        r = a.row,
        seen = record_first_seen(
            "fi.id",
            "fi.route",
            &format!("{}.source_chain", a.row),
            &format!("{}.source_start", a.row),
            &format!("coalesce({r}.source_end, {r}.source_start)", r = a.row)
        )
    )
}

/// A current `hidden` note targets the session.
pub fn hidden(s: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM source_annotation hn JOIN source_annotation_link ht ON ht.annotation = hn.id AND ht.role = 'about' \
         WHERE hn.kind = 'hidden' AND ht.ref = 'session:' || {s}.id AND {})",
        current_annotation("hn")
    )
}

/// The IDs of the hidden sessions, read from their annotations: the few
/// sessions [`hidden`] holds for, without visiting every session.
pub fn hidden_session_ids() -> String {
    format!(
        "SELECT substr(ht.ref, 9) FROM source_annotation hn JOIN source_annotation_link ht ON ht.annotation = hn.id AND ht.role = 'about' \
         WHERE hn.kind = 'hidden' AND ht.ref LIKE 'session:%' AND {}",
        current_annotation("hn")
    )
}

/// The session and every session below it through `spawn` links: its tree
/// (docs/measures.md, session cost). Continuations, forks and resumes are
/// not trees, so a child belongs to exactly one.
pub fn spawn_tree(s: &str) -> String {
    format!(
        "WITH RECURSIVE tr(pk) AS (SELECT {s}.pk UNION SELECT lc.pk FROM session_link l \
         INDEXED BY session_link_key \
         JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         JOIN session lc ON lc.id = l.child_id \
         WHERE l.kind = 'spawn') SELECT pk FROM tr"
    )
}

/// The priced responses of spawned sessions. A priced session with no spawn
/// links has zero subagent cost; an unresolved or unpriced subtree stays unknown.
fn subagent_cost(a: &Aliases) -> String {
    let s = &a.s;
    let tree = spawn_tree(s);
    let children = crate::measure::cost_of(&format!("SELECT pk FROM ({tree}) WHERE pk <> {s}.pk"));
    let own = crate::measure::cost_of(&format!("SELECT {s}.pk"));
    format!(
        "(CASE WHEN EXISTS (SELECT 1 FROM session_link l WHERE l.parent_id = {s}.id AND l.kind='spawn')
          THEN {children} ELSE 0 * {own} END)"
    )
}

/// The session and every session below it through structural links.
pub fn descendants(s: &str) -> String {
    format!(
        "WITH RECURSIVE tr(pk) AS (SELECT {s}.pk UNION SELECT lc.pk FROM session_link l \
         JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         JOIN session lc ON lc.id = l.child_id \
         WHERE l.kind IN ({STRUCTURAL_KINDS})) SELECT pk FROM tr"
    )
}

/// The known logical root, or the session itself when its ancestry is unresolved.
pub fn logical_root(s: &str) -> String {
    format!(
        "(SELECT r.pk FROM session r WHERE r.id = coalesce({}, {s}.id))",
        crate::lineage::lineage_column_sql(&format!("{s}.pk"), "root")
    )
}

#[cfg(test)]
mod tests;

pub(crate) fn message_alternate_match(alternate: &str, part: &str, message: &str) -> String {
    format!(
        "({alternate}.source='message:'||{message}.id OR ({alternate}.source>'message:'||{message}.id||'#' AND {alternate}.source<'message:'||{message}.id||'$') OR {part}.source='message:'||{message}.id OR ({part}.source>'block:'||{message}.id||'#' AND {part}.source<'block:'||{message}.id||'$'))"
    )
}
fn message_alternate(a: &Aliases) -> String {
    format!(
        "EXISTS(SELECT 1 FROM source_alternate_part vp JOIN source_alternate v ON v.id=vp.alternate WHERE {} AND {})",
        crate::alternates::current_sql("v"),
        message_alternate_match("v", "vp", &a.m)
    )
}
