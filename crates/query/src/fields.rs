//! Typed product field declarations. SQL expressions are owned by Store.
use crate::{ColKind, ast::Unit};
use polylogue_model::{
    AlternateKind, BlockKind, CallOutcome, DELIVERY_STATES, DeliveryEvent, DeliverySender,
    DocumentAuthorship, DocumentPurpose, LINK_HOW_NAMES, LINK_KIND_NAMES, PromptKind, Role,
    SessionKind, WrittenBy,
    user::{ChatStatus, ImportJobStatus},
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
    /// Product-specific matching rules.
    Special,
}

impl Kind {
    /// The stored form of one comparison value for a number, amount or
    /// duration field, or why the value cannot be one. Preparation refuses a
    /// value by this rule and compilation stores the same normalised form.
    /// Kinds other than amounts and durations compare as numbers.
    pub fn scalar(self, field: &str, text: &str) -> Result<String, String> {
        match self {
            Kind::Money => crate::money::money_nusd(text)
                .map(|n| n.to_string())
                .ok_or_else(|| format!("{field} is an amount in dollars such as 2.90")),
            Kind::Duration => crate::time::duration_ms(text)
                .map(|n| n.to_string())
                .ok_or_else(|| format!("{field} is a duration such as 90s, 5m or 2h")),
            _ => {
                let digits = text.strip_prefix(['-', '+']).unwrap_or(text);
                if let Ok(n) = text.parse::<i64>() {
                    Ok(n.to_string())
                } else if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                    // Rounding it through f64 would compare a different value.
                    Err(format!("{field} is out of range for a whole number"))
                } else if let Ok(f) = text.parse::<f64>()
                    && f.is_finite()
                {
                    Ok(format!("{f}"))
                } else {
                    Err(format!("{field} is a number"))
                }
            }
        }
    }
}

pub struct Field {
    pub unit: Unit,
    pub name: &'static str,
    pub kind: Kind,
    /// The closed set of values, when there is one.
    pub values: Option<&'static [&'static str]>,
    /// The stored word representing an unknown value, when present.
    pub stored_unknown: Option<&'static str>,
    pub help: &'static str,
    /// The column kind of the value when the matching kind does not say.
    pub value_kind: Option<ColKind>,
    /// The entity named by this value (or each member of an ID list).
    /// Opaque provider IDs and mixed-reference fields have no fixed target.
    pub reference_target: Option<Unit>,
    /// This matcher takes a complete natural reference, rather than an opaque ID.
    pub reference_argument: bool,
}

impl Field {
    const fn reference_argument(mut self) -> Self {
        self.reference_argument = true;
        self
    }

    const fn reference(mut self, target: Unit) -> Self {
        self.reference_target = Some(target);
        self
    }

    const fn stored_unknown(mut self, word: &'static str) -> Self {
        self.stored_unknown = Some(word);
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

    const fn json(mut self) -> Self {
        self.value_kind = Some(ColKind::Json);
        self
    }

    /// The entity a value of this field names, as column metadata.
    pub fn column_reference(&self) -> Option<crate::ColumnReference> {
        column_reference(self.unit, self.name)
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
        stored_unknown: None,
        help,
        value_kind: None,
        reference_target: None,
        reference_argument: false,
    }
}

/// A field with product-specific matching rules.
const fn sp(unit: Unit, name: &'static str, help: &'static str) -> Field {
    f(unit, name, Kind::Special, help)
}

const fn v(
    unit: Unit,
    name: &'static str,
    values: &'static [&'static str],
    help: &'static str,
) -> Field {
    Field {
        values: Some(values),
        ..f(unit, name, Kind::Match, help)
    }
}

/// A field with a closed set of values and its own compile rule.
const fn sv(
    unit: Unit,
    name: &'static str,
    values: &'static [&'static str],
    help: &'static str,
) -> Field {
    Field {
        values: Some(values),
        ..sp(unit, name, help)
    }
}

use Kind::*;
use Unit::*;

const SESSION_HAS_NAMES: [&str; 8] = [
    "thinking",
    "tool_calls",
    "errors",
    "attachments",
    "compaction",
    "branches",
    "note",
    "conflict",
];
const MESSAGE_HAS_NAMES: [&str; 4] = ["thinking", "tool_calls", "attachments", "code"];
pub const LIVE_STATES: &[&str] = polylogue_model::live::LiveState::NAMES;
pub const SESSION_OUTCOMES: &[&str] = &["solved", "partial", "abandoned", "none"];
// The shared enums own the vocabulary; the catalog borrows their spellings.
const MONTH_DAY_HELP: &str = "Logical month and day in the request day zone";
const SOURCE_TIME_QUALITY_HELP: &str = "source time quality; unknown when the source does not say";
const SOURCE_TIME_PRECISION_HELP: &str =
    "source time precision; unknown when the source does not say";

const TIME_QUALITIES: [&str; query_language::time::TimeQuality::ALL.len()] = {
    let mut names = [""; query_language::time::TimeQuality::ALL.len()];
    let mut index = 0;
    while index < names.len() {
        names[index] = query_language::time::TimeQuality::ALL[index].as_str();
        index += 1;
    }
    names
};
const TIME_PRECISIONS: [&str; query_language::time::Precision::ALL.len()] = {
    let mut names = [""; query_language::time::Precision::ALL.len()];
    let mut index = 0;
    while index < names.len() {
        names[index] = query_language::time::Precision::ALL[index].as_str();
        index += 1;
    }
    names
};

/// Declares every field once: its identity and its meaning. The list
/// emits [`FieldId`], one variant per field, and [`FIELDS`], indexed by it,
/// so a field's lowering is an exhaustive `match` on its identity.
macro_rules! fields {
    ($($(#[$doc:meta])* $id:ident = $declaration:expr,)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum FieldId {
            $($(#[$doc])* $id,)*
        }
        impl FieldId {
            pub const ALL: &'static [FieldId] = &[$(FieldId::$id,)*];
            /// This field's one declaration.
            pub fn declaration(self) -> &'static Field {
                &FIELDS[self as usize]
            }
        }
        pub static FIELDS: &[Field] = &[$($declaration,)*];
    };
}

fields! {
    SessionsEndsOpen = f(Sessions,"ends_open",Bool,""),
    SessionsTriageScore = f(Sessions,"triage_score",Number,""),
    SessionsUnrecoveredFailures = f(Sessions,"unrecovered_failures",Number,""),
    SessionsFailureStreak = f(Sessions,"failure_streak",Number,""),
    SessionsOutlier = f(Sessions,"outlier",Bool,""),
    SessionsCorrections = f(Sessions,"corrections",Number,""),
    SessionsOpenItems = f(Sessions,"open_items",Number,""),
    SessionsSubagentCost = f(Sessions,"subagent_cost",Money,"Cost of descendant sessions"),
    SessionsCostComplete = f(Sessions,"cost_complete",Bool,"Every own model response has a known cost"),
    SessionsCostWithParents = f(Sessions,"cost_with_parents",Money,""),
    MessagesLang = sp(Messages,"lang","A current alternate in this language"),
    MessagesAlternate = f(Messages,"alternate",Bool,"A current alternate holds text for this message"),
    AnnotationsAt = f(Annotations,"at",Time,"When this annotation was authored"),
    AnnotationsRequest = f(Annotations,"request",Prefix,""),
    AnnotationsLinksTo = f(Annotations,"links_to",Special,"").reference_argument(),
    AnnotationsAnswered = f(Annotations,"answered",Bool,""),
    AnnotationsOpen = f(Annotations,"open",Bool,""),
    SessionsMonthDay = f(Sessions,"month_day",Match,MONTH_DAY_HELP),
    DocumentsMonthDay = f(Documents,"month_day",Match,MONTH_DAY_HELP),
    MessagesMonthDay = f(Messages,"month_day",Match,MONTH_DAY_HELP),
    BlocksMonthDay = f(Blocks,"month_day",Match,MONTH_DAY_HELP),
    ToolCallsMonthDay = f(ToolCalls,"month_day",Match,MONTH_DAY_HELP),
    EventsMonthDay = f(Events,"month_day",Match,MONTH_DAY_HELP),
    AnnotationsMonthDay = f(Annotations,"month_day",Match,MONTH_DAY_HELP),
    DeliveriesMonthDay = f(Deliveries,"month_day",Match,MONTH_DAY_HELP),
    TurnsMonthDay = f(Turns,"month_day",Match,MONTH_DAY_HELP),
    LiveIntervalsMonthDay = f(LiveIntervals,"month_day",Match,MONTH_DAY_HELP),
    ProcessesMonthDay = f(Processes,"month_day",Match,MONTH_DAY_HELP),
    AlternatesMonthDay = f(Alternates,"month_day",Match,MONTH_DAY_HELP),
    QueriesMonthDay = f(Queries,"month_day",Match,MONTH_DAY_HELP),
    RunsMonthDay = f(Runs,"month_day",Match,MONTH_DAY_HELP),
    ResultsMonthDay = f(Results,"month_day",Match,MONTH_DAY_HELP),
    AlternatesId = f(Alternates,"id",Prefix,""),
    AlternatesRoot = f(Alternates,"root",Prefix,""),
    AlternatesBase = f(Alternates,"base",Prefix,""),
    AlternatesSource = f(Alternates,"source",Match,"").reference_argument(),
    AlternatesKind = v(Alternates, "kind", AlternateKind::NAMES, "the kind of alternate"),
    AlternatesFromLang = f(Alternates,"from_lang",Match,""),
    AlternatesToLang = f(Alternates,"to_lang",Match,""),
    AlternatesLang = f(Alternates,"lang",Match,""),
    AlternatesAuthor = f(Alternates,"author",Match,""),
    AlternatesVia = f(Alternates,"via",Match,""),
    AlternatesMethod = f(Alternates,"method",Match,""),
    AlternatesCreatedAt = f(Alternates,"created_at",Time,""),
    AlternatesCurrent = f(Alternates,"current",Bool,""),
    AlternatesWithdrawn = f(Alternates,"withdrawn",Bool,""),
    AlternatesSince = sp(Alternates,"since","Time of this row"),
    AlternatesUntil = sp(Alternates,"until","Time of this row"),
    QueriesQuery = f(Queries,"query",Prefix,""),
    QueriesName = f(Queries,"name",Match,""),
    QueriesRevision = f(Queries,"revision",Prefix,""),
    QueriesText = f(Queries,"text",Substring,""),
    QueriesSavedAt = f(Queries,"saved_at",Time,""),
    QueriesKept = f(Queries,"kept",Match,""),
    QueriesWatched = f(Queries,"watched",Bool,""),
    QueriesSince = sp(Queries,"since","Time of this row"),
    QueriesUntil = sp(Queries,"until","Time of this row"),
    RunsId = f(Runs,"id",Prefix,""),
    RunsQuery = f(Runs,"query",Match,""),
    RunsAt = f(Runs,"at",Time,""),
    RunsReadAt = f(Runs,"read_at",Match,""),
    RunsAuthor = f(Runs,"author",Match,""),
    RunsVia = f(Runs,"via",Match,""),
    RunsCount = f(Runs,"count",Number,""),
    RunsExact = f(Runs,"exact",Bool,""),
    RunsOutcome = sp(Runs,"outcome","The run outcome state").json(),
    RunsSince = sp(Runs,"since","Time of this row"),
    RunsUntil = sp(Runs,"until","Time of this row"),
    ResultsId = f(Results,"id",Prefix,""),
    ResultsName = f(Results,"name",Match,""),
    ResultsQuery = f(Results,"query",Match,""),
    ResultsRanAt = f(Results,"ran_at",Time,""),
    ResultsRows = f(Results,"rows",Number,""),
    ResultsSince = sp(Results,"since","Time of this row"),
    ResultsUntil = sp(Results,"until","Time of this row"),
    ImportJobsId = f(ImportJobs,"id",Prefix,"import job ID"),
    ImportJobsProvider = f(ImportJobs,"provider",Match,"the web chat, by the extension's name for it"),
    ImportJobsChangedSince = f(ImportJobs,"changed_since",Time,"only conversations the provider changed at or after this time"),
    ImportJobsState = v(ImportJobs,"state",ImportJobStatus::NAMES,"where the job stands now; a claim held past its lifetime reads as waiting"),
    ImportJobsListed = f(ImportJobs,"listed",Number,"conversations the provider listed"),
    ImportJobsCaptured = f(ImportJobs,"captured",Number,"conversations captured and stored"),
    ImportJobsUnchanged = f(ImportJobs,"unchanged",Number,"conversations already in the archive as the provider has them"),
    ImportJobsFailed = f(ImportJobs,"failed",Number,"conversations that could not be captured"),
    ImportJobsPosition = f(ImportJobs,"position",Match,"where in the provider's list the next claim continues"),
    ImportJobsDetail = f(ImportJobs,"detail",Substring,"the latest event's detail"),
    ImportJobsCreatedAt = f(ImportJobs,"created_at",Time,"when the job was started"),
    ImportJobsUpdatedAt = f(ImportJobs,"updated_at",Time,"when its latest event was recorded"),
    ImportJobsClaimedBy = f(ImportJobs,"claimed_by",Match,"the extension instance holding an unexpired claim"),
    ImportJobsSince = sp(ImportJobs,"since","Time of this row"),
    ImportJobsUntil = sp(ImportJobs,"until","Time of this row"),
    ImportJobsMonthDay = f(ImportJobs,"month_day",Match,MONTH_DAY_HELP),
    ChatActionsId = f(ChatActions,"id",Prefix,"chat action ID"),
    ChatActionsProvider = f(ChatActions,"provider",Match,"the web chat, by the extension's name for it"),
    ChatActionsConversation = f(ChatActions,"conversation",Match,"the provider's conversation it replies in; unknown for a new one"),
    ChatActionsModel = f(ChatActions,"model",Match,"the model as the provider's menu names it"),
    ChatActionsText = f(ChatActions,"text",Substring,"the turn's text"),
    ChatActionsKind = v(ChatActions,"kind",&["draft","submit"],"draft fills the composer and stops; submit sends the turn"),
    ChatActionsState = v(ChatActions,"state",ChatStatus::NAMES,"where the action stands now; a claim held past its lifetime reads as queued, or outcome_unknown once sending began"),
    ChatActionsPhase = f(ChatActions,"phase",Match,"the extension's latest reported step"),
    ChatActionsNeedsApproval = f(ChatActions,"needs_approval",Bool,"a submit into an existing conversation, which waits for the operator"),
    ChatActionsCreatedAt = f(ChatActions,"created_at",Time,"when the action was queued"),
    ChatActionsUpdatedAt = f(ChatActions,"updated_at",Time,"when its latest event was recorded"),
    ChatActionsClaimedBy = f(ChatActions,"claimed_by",Match,"the extension instance holding an unexpired claim"),
    ChatActionsDetail = f(ChatActions,"detail",Substring,"the latest event's detail"),
    ChatActionsSince = sp(ChatActions,"since","Time of this row"),
    ChatActionsUntil = sp(ChatActions,"until","Time of this row"),
    ChatActionsMonthDay = f(ChatActions,"month_day",Match,MONTH_DAY_HELP),
    MessagesAtTimeQuality = v(
        Messages,
        "at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    MessagesAtPrecision = v(
        Messages,
        "at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    BlocksAtTimeQuality = v(
        Blocks,
        "at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    BlocksAtPrecision = v(
        Blocks,
        "at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    ToolCallsAtTimeQuality = v(
        ToolCalls,
        "at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    ToolCallsAtPrecision = v(
        ToolCalls,
        "at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    EventsAtTimeQuality = v(
        Events,
        "at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    EventsAtPrecision = v(
        Events,
        "at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    DeliveriesAtTimeQuality = v(
        Deliveries,
        "at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    DeliveriesAtPrecision = v(
        Deliveries,
        "at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    SessionsStartedAtTimeQuality = v(
        Sessions,
        "started_at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    SessionsStartedAtPrecision = v(
        Sessions,
        "started_at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    DocumentsSourceTimeTimeQuality = v(
        Documents,
        "source_time.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    DocumentsSourceTimePrecision = v(
        Documents,
        "source_time.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    SessionsLastActiveTimeQuality = v(
        Sessions,
        "last_active.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    SessionsLastActivePrecision = v(
        Sessions,
        "last_active.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    ToolCallsResultAtTimeQuality = v(
        ToolCalls,
        "result_at.time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    ToolCallsResultAtPrecision = v(
        ToolCalls,
        "result_at.precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    DocumentsTimeQuality = v(
        Documents,
        "time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    DocumentsPrecision = v(
        Documents,
        "precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    MessagesTimeQuality = v(
        Messages,
        "time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    MessagesPrecision = v(
        Messages,
        "precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    BlocksTimeQuality = v(
        Blocks,
        "time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    BlocksPrecision = v(
        Blocks,
        "precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    ToolCallsTimeQuality = v(
        ToolCalls,
        "time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    ToolCallsPrecision = v(
        ToolCalls,
        "precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    EventsTimeQuality = v(
        Events,
        "time_quality",
        &TIME_QUALITIES,
        SOURCE_TIME_QUALITY_HELP,
    ),
    EventsPrecision = v(
        Events,
        "precision",
        &TIME_PRECISIONS,
        SOURCE_TIME_PRECISION_HELP,
    ),
    DocumentsId = f(Documents, "id", Prefix, "document ID or a unique prefix").reference(Documents),
    DocumentsOrigin = f(Documents, "origin", Match, "the provider export family"),
    DocumentsPurpose = v(
        Documents,
        "purpose",
        DocumentPurpose::NAMES,
        "instruction, context, memory or transcript",
    ),
    DocumentsAuthorship = v(
        Documents,
        "authorship",
        DocumentAuthorship::NAMES,
        "who wrote the text; unknown when the source does not say",
    )
    .stored_unknown("unknown"),
    DocumentsTitle = f(Documents, "title", Substring, "the supplied source title"),
    DocumentsSince = sp(
        Documents,
        "since",
        "source-record time on or after the time",
    ),
    DocumentsUntil = sp(
        Documents,
        "until",
        "source-record time before the instant or through the logical day",
    ),
    DocumentsSourceTime = f(
        Documents,
        "source_time",
        Time,
        "the provider's source-record time",
    ),
    DocumentsVersions = f(
        Documents,
        "versions",
        Number,
        "the number of distinct versions its sources hold",
    ),
    SessionsId = f(Sessions, "id", Prefix, "session ID or a unique prefix").reference(Sessions),
    SessionsOrigin = f(Sessions, "origin", Match, "claude-code, codex, chatgpt, …"),
    SessionsPhysical = f(
        Sessions,
        "physical",
        Bool,
        "list physical sessions one by one instead of logical sessions",
    ),
    SessionsKind = v(
        Sessions,
        "kind",
        SessionKind::NAMES,
        "who started the session: main, or one another session spawned",
    ),
    SessionsTitle = f(Sessions, "title", Substring, "the displayed title"),
    SessionsRepo = sp(Sessions, "repo", "the repository: name, owner/name or URL"),
    SessionsBranch = f(Sessions, "branch", Match, "the git branch, as recorded"),
    SessionsCommit = sp(
        Sessions,
        "commit",
        "a commit recorded or linked (prefix, 7+ hex digits)",
    ),
    SessionsPr = sp(Sessions, "pr", "a pull request linked to the session"),
    SessionsOutcome = v(
        Sessions,
        "outcome",
        SESSION_OUTCOMES,
        "the session's outcome note",
    ),
    SessionsHidden = f(Sessions, "hidden", Bool, "has a hidden note"),
    SessionsCwd = sp(
        Sessions,
        "cwd",
        "working directory, and the directories below it",
    ),
    SessionsSource = v(
        Sessions,
        "source",
        &["present", "missing", "gone"],
        "whether a transcript was stored, and whether its source (a newer export of its account, the Drive folder) still holds it",
    ),
    SessionsLive = sv(
        Sessions,
        "live",
        LIVE_STATES,
        "the live state now, from hook events, processes and the registry",
    ),
    SessionsWaiting = sp(Sessions, "waiting", "the live state is waiting").boolean(),
    SessionsRunning = sp(
        Sessions,
        "running",
        "the live state is working, waiting, idle or unknown",
    )
    .boolean(),
    SessionsWaitingOn = v(
        Sessions,
        "waiting_on",
        polylogue_model::live::WaitingOn::NAMES,
        "what a waiting session waits for from the operator",
    ),
    SessionsStall = v(
        Sessions,
        "stall",
        polylogue_model::live::Stall::NAMES,
        "what a stalled session waits for that is not the operator",
    ),
    SessionsProcessAlive = f(
        Sessions,
        "process_alive",
        Bool,
        "whether the session's harness process is alive; empty when no process was observed",
    ),
    SessionsHarnessStatus = f(
        Sessions,
        "harness_status",
        Match,
        "the harness registry's own word for the session's status",
    ),
    SessionsOpenTool = f(
        Sessions,
        "open_tool",
        Match,
        "the tool of the call running now",
    ),
    SessionsEditedFiles = f(
        Sessions,
        "edited_files",
        Substring,
        "files the live session changed through edit tools and has not committed",
    ),
    SessionsTask = f(
        Sessions,
        "task",
        Match,
        "the live task, when a task is stated explicitly",
    ),
    SessionsLiveUntil = f(
        Sessions,
        "live_until",
        Time,
        "when the live state or stall changes with no new events; empty when only new events can change it",
    ),
    SessionsLiveSince = f(
        Sessions,
        "live_since",
        Time,
        "when the current live activity began",
    ),
    SessionsLastEventAt = f(
        Sessions,
        "last_event_at",
        Time,
        "when the latest live events happened",
    ),
    SessionsTodo = f(
        Sessions,
        "todo",
        Substring,
        "the harness's in-progress todo item, as recorded",
    ),
    SessionsLiveRank = f(
        Sessions,
        "live_rank",
        Number,
        "0 waiting, 1 working, 2 idle, 3 unknown, 4 stale, 5 ended: the order the running list uses",
    ),
    SessionsPeers = f(
        Sessions,
        "peers",
        Substring,
        "other working, waiting or idle sessions in the same repository, or the same working directory without one",
    ),
    SessionsRepoRoot = f(
        Sessions,
        "repo_root",
        Prefix,
        "the repository's identity: its root commit hash, when found",
    ),
    SessionsHarness = f(
        Sessions,
        "harness",
        Match,
        "claude-code, codex-tui, codex-exec, …",
    ),
    SessionsModel = sp(Sessions, "model", "a model that wrote at least one message"),
    SessionsTag = sp(
        Sessions,
        "tag",
        "a tag note; tag:key:value for key:value tags",
    ),
    SessionsHas = sv(
        Sessions,
        "has",
        &SESSION_HAS_NAMES,
        "the session has thinking, tool calls, errors, …",
    ),
    SessionsParent = sp(Sessions, "parent", "sessions linked from the given session"),
    SessionsRoot = sp(
        Sessions,
        "root",
        "sessions whose explicit ancestry has the given root",
    ).reference(Sessions),
    SessionsDepth = f(Sessions, "depth", Number, "depth in the explicit provider lineage"),
    SessionsLink = sp(
        Sessions,
        "link",
        "the kind of an incoming link; link:fork, link:resume",
    ),
    SessionsContinues = sp(
        Sessions,
        "continues",
        "sessions continuing the given session",
    ),
    SessionsContinuedBy = sp(
        Sessions,
        "continued_by",
        "sessions the given session continues",
    ),
    SessionsConnected = sp(
        Sessions,
        "connected",
        "every session connected to the given one through links",
    ),
    SessionsPath = sp(
        Sessions,
        "path",
        "a file path the session read, wrote or edited",
    ),
    SessionsTool = sp(Sessions, "tool", "the session made such a tool call"),
    SessionsAction = sp(Sessions, "action", "the session made such a tool call"),
    SessionsStarted = f(Sessions, "started", Time, "first own message or event"),
    SessionsFirstSeen = f(
        Sessions,
        "first_seen",
        Time,
        "when Polylogue first had any of its messages",
    ),
    SessionsSeenBy = sp(
        Sessions,
        "seen_by",
        "what Polylogue had seen by the time (asked of messages, blocks, tool calls, events and notes)",
    ),
    SessionsLastActive = f(Sessions, "last_active", Time, "last own message or event"),
    SessionsSince = sp(Sessions, "since", "active on or after the time"),
    SessionsUntil = sp(
        Sessions,
        "until",
        "begun before the instant or through the logical day",
    ),
    SessionsMessages = f(Sessions, "messages", Number, "own messages"),
    SessionsHumanPrompts = f(Sessions, "human_prompts", Number, "own prose prompts"),
    SessionsHumanCommands = f(Sessions, "human_commands", Number, "own typed commands"),
    SessionsWords = f(Sessions, "words", Number, "words in own prose"),
    SessionsToolCalls = f(Sessions, "tool_calls", Number, "own tool calls"),
    SessionsErrors = f(
        Sessions,
        "errors",
        Number,
        "own tool calls with outcome error",
    ),
    SessionsInputTokens = f(Sessions, "input_tokens", Number, "summed over own model responses; unknown when none reports it"),
    SessionsOutputTokens = f(
        Sessions,
        "output_tokens",
        Number,
        "summed over own model responses; unknown when none reports it",
    ),
    SessionsCacheReadTokens = f(
        Sessions,
        "cache_read_tokens",
        Number,
        "summed over own model responses; unknown when none reports it",
    ),
    SessionsCacheWriteTokens = f(
        Sessions,
        "cache_write_tokens",
        Number,
        "summed over own model responses; unknown when none reports it",
    ),
    SessionsCost = f(
        Sessions,
        "cost",
        Money,
        "own cost; unknown when nothing is priced",
    ),
    SessionsTreeCost = f(
        Sessions,
        "tree_cost",
        Money,
        "own cost plus every session below in the subagent tree",
    ),
    SessionsWallSpan = f(
        Sessions,
        "wall_span",
        Duration,
        "Elapsed time between first and last own recorded time",
    ),
    SessionsClusteredTime = f(
        Sessions,
        "clustered_time",
        Duration,
        "Union of consecutive message gaps at most 5 minutes, extended across paired tools",
    ),
    SessionsToolTime = f(
        Sessions,
        "tool_time",
        Duration,
        "Sum of strictly positive paired tool intervals; untimed calls counted separately",
    ),
    SessionsToolTimeUnion = f(
        Sessions,
        "tool_time_union",
        Duration,
        "Union of strictly positive paired tool intervals",
    ),
    SessionsToolUntimed = f(
        Sessions,
        "tool_untimed",
        Number,
        "Own tool calls whose durations are unknown; shown beside tool-time sums",
    ),
    SessionsEndsWith = f(
        Sessions,
        "ends_with",
        Match,
        "Kind of the last recorded own item",
    )
    .stored_unknown("unknown"),
    SessionsLastStopReason = f(
        Sessions,
        "last_stop_reason",
        Match,
        "Last model stop reason as recorded",
    ),
    SessionsTimeSource = f(
        Sessions,
        "time_source",
        Match,
        "Messages, or events when there are no messages",
    )
    .stored_unknown("absent"),
    SessionsWaitingModel = f(
        Sessions,
        "waiting_model",
        Duration,
        "Wall time waiting for the model",
    ),
    SessionsWaitingHuman = f(
        Sessions,
        "waiting_human",
        Duration,
        "Wall time waiting for the human",
    ),
    SessionsHumanWords = f(Sessions, "human_words", Number, "Own human words"),
    SessionsModelWords = f(Sessions, "model_words", Number, "Own model words"),
    SessionsThinkingMessages = f(
        Sessions,
        "thinking_messages",
        Number,
        "Own thinking messages",
    ),
    SessionsCompactions = f(Sessions, "compactions", Number, "Own compactions"),
    SessionsSimilar = sp(Sessions, "similar", "semantic similarity (a ranking)"),
    SessionsOutput = sp(Sessions, "output", "words in tool output"),
    SessionsThinking = sp(Sessions, "thinking", "words in thinking"),
    MessagesId = f(Messages, "id", Prefix, "message ID").reference(Messages),
    MessagesRole = v(Messages, "role", Role::NAMES, "who the source says spoke"),
    MessagesWrittenBy = v(Messages, "written_by", WrittenBy::NAMES, "who wrote it").stored_unknown("unknown"),
    MessagesPromptKind = v(
        Messages,
        "prompt_kind",
        PromptKind::NAMES,
        "for user messages: prose, command, or text the harness or another model wrote",
    ),
    MessagesModel = f(Messages, "model", Match, "the model that wrote it"),
    MessagesAt = f(Messages, "at", Time, "provider time"),
    MessagesFirstSeen = f(
        Messages,
        "first_seen",
        Time,
        "when Polylogue first had its bytes",
    ),
    MessagesSeenBy = sp(
        Messages,
        "seen_by",
        "first seen by the time: what Polylogue had seen then",
    ),
    MessagesDuplicate = f(
        Messages,
        "duplicate",
        Bool,
        "a turn another input of the session also holds, shown from that input",
    ),
    MessagesActive = f(Messages, "active", Bool, "on the active path"),
    MessagesHas = sv(
        Messages,
        "has",
        &MESSAGE_HAS_NAMES,
        "the message has thinking, tool calls, attachments or code",
    ),
    MessagesWords = f(Messages, "words", Number, "words in its text"),
    MessagesInputTokens = f(Messages, "input_tokens", Number, ""),
    MessagesOutputTokens = f(Messages, "output_tokens", Number, ""),
    MessagesCacheReadTokens = f(Messages, "cache_read_tokens", Number, ""),
    MessagesCacheWriteTokens = f(Messages, "cache_write_tokens", Number, ""),
    MessagesCost = f(
        Messages,
        "cost",
        Money,
        "per message; unknown when unpriced",
    ),
    MessagesStopReason = f(Messages, "stop_reason", Match, ""),
    MessagesSince = sp(Messages, "since", "at or after the time"),
    MessagesUntil = sp(
        Messages,
        "until",
        "before the instant or through the logical day",
    ),
    MessagesSimilar = sp(Messages, "similar", "semantic similarity (a ranking)"),
    MessagesOutput = sp(Messages, "output", "words in tool output"),
    MessagesThinking = sp(Messages, "thinking", "words in thinking"),
    BlocksMessage = f(
        Blocks,
        "message",
        Prefix,
        "the message containing the block",
    )
    .reference(Messages),
    BlocksKind = v(Blocks, "kind", BlockKind::NAMES, "the block kind"),
    BlocksMediaType = f(Blocks, "media_type", Match, ""),
    BlocksFileName = f(Blocks, "file_name", Substring, ""),
    BlocksAt = f(Blocks, "at", Time, "the message's time"),
    BlocksFirstSeen = f(
        Blocks,
        "first_seen",
        Time,
        "when Polylogue first had its message's bytes",
    ),
    BlocksSeenBy = sp(
        Blocks,
        "seen_by",
        "its message first seen by the time",
    ),
    BlocksSince = sp(Blocks, "since", "at or after the time"),
    BlocksUntil = sp(
        Blocks,
        "until",
        "before the instant or through the logical day",
    ),
    ToolCallsMessage = f(
        ToolCalls,
        "message",
        Prefix,
        "the message containing the call",
    )
    .reference(Messages),
    ToolCallsTool = f(ToolCalls, "tool", Match, "the tool name as written"),
    ToolCallsMcpServer = sp(
        ToolCalls,
        "mcp_server",
        "for MCP tools, the server part of the name",
    ),
    ToolCallsAction = v(
        ToolCalls,
        "action",
        polylogue_model::ToolAction::NAMES,
        "shell, file_read, file_write, file_edit, search, web, …",
    ),
    ToolCallsCommand = sp(ToolCalls, "command", "the shell command (substring)"),
    ToolCallsCommandNorm = f(
        ToolCalls,
        "command_norm",
        Match,
        "the normalized command: cargo test, git commit",
    ),
    ToolCallsInput = sp(ToolCalls, "input", "text inside the arguments (substring)"),
    ToolCallsOutput = sp(ToolCalls, "output", "words in the result"),
    ToolCallsPath = sp(
        ToolCalls,
        "path",
        "a file path in the arguments (substring)",
    ),
    ToolCallsOutcome = v(
        ToolCalls,
        "outcome",
        CallOutcome::NAMES,
        "ok, error, pending or unknown",
    )
    .stored_unknown("unknown"),
    ToolCallsIdx = f(
        ToolCalls,
        "idx",
        Number,
        "the call block position in its message",
    ),
    ToolCallsPosition = f(
        ToolCalls,
        "position",
        Number,
        "the call position in the complete session call order",
    ),
    ToolCallsExitCode = f(ToolCalls, "exit_code", Number, ""),
    ToolCallsAt = f(ToolCalls, "at", Time, ""),
    ToolCallsResultAt = f(
        ToolCalls,
        "result_at",
        Time,
        "source time of the message or notification reporting the result",
    ),
    ToolCallsReportedDuration = f(
        ToolCalls,
        "reported_duration",
        Duration,
        "provider-reported tool duration in milliseconds; unknown when absent",
    ),
    ToolCallsFirstSeen = f(
        ToolCalls,
        "first_seen",
        Time,
        "when Polylogue first had the bytes of the message that made the call",
    ),
    ToolCallsSeenBy = sp(
        ToolCalls,
        "seen_by",
        "its call first seen by the time",
    ),
    ToolCallsSince = sp(
        ToolCalls,
        "since",
        "ended at or after the time, or still open",
    ),
    ToolCallsUntil = sp(
        ToolCalls,
        "until",
        "started before the instant or through the logical day",
    ),
    ToolCallsChild = sp(ToolCalls, "child", "for subagent calls: the child session"),
    ToolCallsNext = sp(ToolCalls, "next", "the session's next tool call"),
    ToolCallsPrevious = sp(ToolCalls, "previous", "the session's previous tool call"),
    FilesPath = f(Files, "path", Substring, "the file path"),
    FilesReads = f(Files, "reads", Number, ""),
    FilesWrites = f(Files, "writes", Number, ""),
    FilesEdits = f(Files, "edits", Number, ""),
    FilesFirstAt = f(Files, "first_at", Time, ""),
    FilesLastAt = f(Files, "last_at", Time, ""),
    EventsId = f(Events, "id", Prefix, "the observation ID or its prefix").reference(Events),
    EventsText = sp(
        Events,
        "text",
        "a substring in one textual observation value",
    ),
    EventsInput = f(
        Events,
        "input",
        Match,
        "the input this observation came from",
    ),
    EventsSource = f(Events, "source", Match, "the input's source parser"),
    EventsOrigin = f(
        Events,
        "origin",
        Match,
        "conversation origin, or source format when unbound",
    ),
    EventsData = f(Events, "data", Substring, "the observation payload").json(),
    EventsSourceChain = f(Events, "source_chain", Number, "the source revision chain"),
    EventsSourceStart = f(Events, "source_start", Number, "record byte start"),
    EventsSourceEnd = f(Events, "source_end", Number, "record byte end"),
    EventsKind = f(
        Events,
        "kind",
        Match,
        "compaction, model_switch, permission, …",
    ),
    EventsAt = f(Events, "at", Time, ""),
    EventsFirstSeen = f(
        Events,
        "first_seen",
        Time,
        "when Polylogue first had its bytes",
    ),
    EventsSeenBy = sp(Events, "seen_by", "first seen by the time"),
    EventsSince = sp(Events, "since", "at or after the time"),
    EventsUntil = sp(
        Events,
        "until",
        "before the instant or through the logical day",
    ),
    SessionLinksKind = v(
        SessionLinks,
        "kind",
        LINK_KIND_NAMES,
        "the recorded session relationship",
    ),
    SessionLinksHow = v(SessionLinks, "reason", LINK_HOW_NAMES, "why the sessions reference each other"),
    SessionLinksCopiesPrefix = f(SessionLinks, "copies_prefix", Bool, "the child holds a copy of the parent's prefix"),
    SessionLinksParent = sp(SessionLinks, "parent", "the parent session").reference(Sessions),
    SessionLinksChild = sp(SessionLinks, "child", "the child session").reference(Sessions),
    SessionLinksPending = f(SessionLinks, "pending", Bool, "the parent is not in the archive"),
    SessionLinksEvidence = f(
        SessionLinks,
        "evidence",
        Match,
        "the input field that established it",
    ),
    SessionLinksBranchPoint = f(
        SessionLinks,
        "branch_point",
        Prefix,
        "the message where the child branched off",
    )
    .reference(Messages),
    AnnotationsId = f(
        Annotations,
        "id",
        Prefix,
        "annotation revision ID or a unique prefix",
    ),
    AnnotationsReference = f(
        Annotations,
        "reference",
        Match,
        "the natural reference",
    )
    .reference(Annotations).reference_argument(),
    AnnotationsKind = f(
        Annotations,
        "kind",
        Match,
        "tag, note, title, decision, claim, …",
    ),
    AnnotationsKey = f(Annotations, "key", Match, ""),
    AnnotationsValue = f(Annotations, "value", Substring, "").json(),
    AnnotationsAuthor = f(
        Annotations,
        "author",
        Match,
        "operator, a session ID, agent or script",
    ),
    AnnotationsAgent = f(Annotations, "agent", Bool, "an agent wrote it"),
    AnnotationsVia = f(Annotations, "via", Match, "the channel it was written through"),
    AnnotationsJudged = f(
        Annotations,
        "judged",
        Match,
        "the operator's judgment of this revision: accepted, rejected, deferred or conflict",
    ).stored_unknown("unknown"),
    AnnotationsAbout = sp(
        Annotations,
        "about",
        "a reference it is about: session:ID, repo:NAME, cwd:PATH, …",
    ).reference_argument(),
    AnnotationsWriteCounter = f(
        Annotations,
        "write_counter",
        Number,
        "the archive's write counter when it was written",
    ),
    AnnotationsCreatedAt = f(Annotations, "created_at", Time, ""),
    AnnotationsRoot = f(
        Annotations,
        "root",
        Prefix,
        "the first note of its revisions",
    ).reference_argument(),
    AnnotationsCurrent = f(
        Annotations,
        "current",
        Bool,
        "a current head of the revision graph",
    ),
    AnnotationsConflict = f(
        Annotations,
        "conflict",
        Bool,
        "the family has multiple current heads",
    ),
    AnnotationsWithdrawn = f(Annotations, "withdrawn", Bool, "a withdrawal names it"),
    AnnotationsSince = sp(Annotations, "since", "created at or after the time"),
    AnnotationsSeenBy = sp(
        Annotations,
        "seen_by",
        "the notes stored by the time, each family at its revision then",
    ),
    AnnotationsUntil = sp(
        Annotations,
        "until",
        "created before the instant or through the logical day",
    ),
    DeliveriesId = f(Deliveries, "id", Prefix, "").reference(Deliveries),
    DeliveriesRecipient = sp(Deliveries, "recipient", "the receiving session").reference(Sessions),
    DeliveriesEvent = v(
        Deliveries,
        "event",
        DeliveryEvent::NAMES,
        "what the delivery answered",
    ),
    DeliveriesSender = v(
        Deliveries,
        "sender",
        DeliverySender::NAMES,
        "what prepared it; signal notices have sender signal",
    ),
    DeliveriesOutcome = v(
        Deliveries,
        "outcome",
        DELIVERY_STATES,
        "what last happened to it; prepared until a delivery event says more",
    ),
    DeliveriesDetail = f(
        Deliveries,
        "detail",
        Substring,
        "the latest event's detail: a failure's reason, or what used it",
    ),
    DeliveriesText = f(Deliveries, "text", Substring, "the delivered text"),
    DeliveriesRefs = sp(Deliveries, "refs", "a reference the delivery included").reference_argument(),
    DeliveriesLeftOut = sp(Deliveries, "left_out", "a reference left out for budget").reference_argument(),
    DeliveriesAt = f(Deliveries, "at", Time, ""),
    DeliveriesBytes = f(Deliveries, "bytes", Number, ""),
    LiveIntervalsSession = sp(
        LiveIntervals,
        "session",
        "the session the interval belongs to",
    ),
    LiveIntervalsAspect = v(
        LiveIntervals,
        "aspect",
        polylogue_model::live::Aspect::NAMES,
        "what the interval measures: activity, waiting_on or stall",
    ),
    LiveIntervalsValue = f(
        LiveIntervals,
        "value",
        Match,
        "the activity, wait or stall during the interval",
    ),
    LiveIntervalsStart = f(LiveIntervals, "start", Time, "when the interval began"),
    LiveIntervalsEnd = f(LiveIntervals, "end", Time, "when an event closed it"),
    LiveIntervalsDurationMs = f(
        LiveIntervals,
        "duration_ms",
        Number,
        "its length in milliseconds",
    ),
    LiveIntervalsClosedBy = f(
        LiveIntervals,
        "closed_by",
        Match,
        "the kind of event that closed it",
    ),
    ProcessesSession = sp(
        Processes,
        "session",
        "the session whose process tree it is in",
    ),
    ProcessesBootId = f(Processes, "boot_id", Match, "the boot it ran in"),
    ProcessesPid = f(Processes, "pid", Number, ""),
    ProcessesStart = f(
        Processes,
        "start",
        Number,
        "its start time in clock ticks after boot",
    ),
    ProcessesParentPid = f(Processes, "parent_pid", Number, ""),
    ProcessesComm = f(
        Processes,
        "comm",
        Match,
        "the kernel's name for its command",
    ),
    ProcessesArgv = f(
        Processes,
        "argv",
        Substring,
        "its arguments, as a JSON array",
    ),
    ProcessesCwd = f(
        Processes,
        "cwd",
        Subtree,
        "its working directory, and the directories below it",
    ),
    ProcessesStartedAt = f(
        Processes,
        "started_at",
        Time,
        "when the watcher first saw it",
    ),
    ProcessesEndedAt = f(Processes, "ended_at", Time, "when the watcher saw it exit"),
    ProcessesRole = v(
        Processes,
        "role",
        &["harness", "child"],
        "the harness root or one of its descendants",
    ),
    ProcessesToolCall = f(
        Processes,
        "tool_call",
        Prefix,
        "the tool call it started during: between that call's start and its result",
    ),
    FeedKind = v(
        Feed,
        "kind",
        &["session", "removed", "annotation"],
        "what changed",
    ),
    FeedRef = f(Feed, "ref", Prefix, "the session or annotation ID").reference_argument(),
    FeedSeq = f(
        Feed,
        "seq",
        Number,
        "the write counter of the change: index.db's for sessions, archive.db's for annotations",
    ),
    FeedAnnotationKind = f(Feed, "annotation_kind", Match, "an annotation's kind"),
    FeedAfter = sp(
        Feed,
        "after",
        "changes after a feed position `GENERATION:INDEX:ARCHIVE`: the `view` of an earlier answer",
    ),
    FeedSessions = sp(
        Feed,
        "sessions",
        "selection array, available as output only",
    )
    .ids()
    .reference(Sessions),
    FeedMessages = sp(
        Feed,
        "messages",
        "selection array, available as output only",
    )
    .ids()
    .reference(Messages),
    TurnsId = f(Turns, "id", Prefix, "its first prompt's message ID").reference(Messages),
    TurnsStart = f(Turns, "start", Time, "the first message's time"),
    TurnsEnd = f(
        Turns,
        "end",
        Time,
        "the latest time of any message in the turn",
    ),
    TurnsBurst = f(Turns, "burst", Number, "how many prompts it has"),
    TurnsPrompts = f(
        Turns,
        "prompts",
        Substring,
        "the human prompt messages of the turn, in order (`prompts:ID` holds that message)",
    ),
    TurnsSlashCommand = f(Turns, "slash_command", Bool, "starts with a typed command"),
    TurnsAnswer = f(
        Turns,
        "answer",
        Prefix,
        "the model's final message in the turn",
    )
    .reference(Messages),
    TurnsAttempts = f(
        Turns,
        "attempts",
        Number,
        "replies under the turn's last prompt, abandoned branches included",
    ),
    TurnsPath = v(
        Turns,
        "path",
        &["active", "unknown"],
        "unknown when the session's active path is unknown",
    )
    .stored_unknown("unknown"),
    TurnsLatency = f(
        Turns,
        "latency",
        Duration,
        "from the last prompt to the first model message",
    ),
    TurnsToolCalls = f(Turns, "tool_calls", Number, ""),
    TurnsErrors = f(Turns, "errors", Number, "tool calls with outcome error"),
    TurnsInputTokens = f(Turns, "input_tokens", Number, ""),
    TurnsOutputTokens = f(Turns, "output_tokens", Number, ""),
    TurnsCost = f(Turns, "cost", Money, ""),
    TurnsAbandoned = f(
        Turns,
        "abandoned",
        Bool,
        "the turn ended the session with no answer",
    ),
    TurnsCorrected = f(
        Turns,
        "corrected",
        Bool,
        "a current correction note targets the turn's answer",
    ),
}
/// The field a unit declares under `name`, ignoring case.
pub fn lookup(unit: Unit, name: &str) -> Option<FieldId> {
    FieldId::ALL.iter().copied().find(|id| {
        let field = id.declaration();
        field.unit == unit && field.name.eq_ignore_ascii_case(name)
    })
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

/// The declared construct and unknown rule of fields backed by session products.
pub fn construct(field: &Field) -> Option<polylogue_model::derived::Construct> {
    use polylogue_model::derived;
    if field.unit != Unit::Sessions {
        return None;
    }
    if polylogue_model::derived::COUNT_FIELDS
        .iter()
        .any(|(name, _)| *name == field.name)
    {
        return Some(derived::SESSION_COUNTS);
    }
    match field.name {
        "root" | "depth" => Some(derived::SESSION_LINEAGE),
        "wall_span" | "clustered_time" | "tool_time" | "tool_time_union" | "tool_untimed"
        | "ends_with" | "last_stop_reason" | "time_source" => Some(derived::SESSION_SPAN),
        "waiting_model" | "waiting_human" => Some(derived::SESSION_TIME),
        _ => None,
    }
}

/// Resolve logical reference targets independently of SQL aliases and storage.
/// `session` is the containing-session reference declared by `in_session`.
fn reference_value(unit: Unit, name: &str) -> (Option<Unit>, ColKind) {
    if name.eq_ignore_ascii_case("session") && has_session(unit) {
        return (Some(Unit::Sessions), ColKind::Text);
    }
    if let Some((prefix, field)) = name.split_once('.')
        && let Some(target) = relation(unit, prefix)
    {
        return reference_value(target, field);
    }
    if let Some(field) = lookup(unit, name).map(FieldId::declaration) {
        return (field.reference_target, field.column_kind());
    }
    if has_session(unit) {
        return reference_value(Unit::Sessions, name);
    }
    (None, ColKind::Text)
}

pub fn reference_target(unit: Unit, name: &str) -> Option<Unit> {
    reference_value(unit, name).0
}

/// Scalar wire metadata never describes an ID container as one entity.
pub fn column_reference(unit: Unit, name: &str) -> Option<crate::ColumnReference> {
    let (target, kind) = reference_value(unit, name);
    if kind != ColKind::Text {
        return None;
    }
    target.and_then(|unit| crate::ColumnReference::from_unit(unit.name()))
}

#[cfg(test)]
mod tests;
