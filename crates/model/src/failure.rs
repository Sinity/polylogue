//! Product failure codes and refusal reasons, with the wire spelling
//! declared once.
stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Code {
        Empty => "empty",
        ModelCallsOff => "model_calls_off",
        Query => "query",
        Sql => "sql",
        InvalidQuery => "invalid_query",
        UnknownField => "unknown_field",
        BadValue => "bad_value",
        NotAQuery => "not_a_query",
        BadRequest => "bad_request",
        Invalid => "invalid",
        IncompleteParents => "incomplete_parents",
        UnsupportedProvider => "unsupported_provider",
        EmptyText => "empty_text",
        ModelRequired => "model_required",
        AttachmentNameNotABasename => "attachment_name_not_a_basename",
        InvalidAction => "invalid_action",
        SubmissionReceiptRequired => "submission_receipt_required",
        SubmissionConversationMismatch => "submission_conversation_mismatch",
        SinceIsADateOrTime => "since_is_a_date_or_time",
        UnknownOutcome => "unknown_outcome",
        UnknownAttachment => "unknown_attachment",
        UnknownReport => "unknown_report",
        Ambiguous => "ambiguous",
        NotAVerb => "not_a_verb",
        NeedsAll => "needs_all",
        RowNumbersOffTerminal => "row_numbers_off_terminal",
        StaleCursor => "stale_cursor",
        NotBuilt => "not_built",
        NotYetAvailable => "not_yet_available",
        IndexBuilding => "index_building",
        Degraded => "degraded",
        PrimaryContextMissing => "primary_context_missing",
        PrimaryContextPending => "primary_context_pending",
        PrimaryContextStale => "primary_context_stale",
        Refused => "refused",
        OperatorRequired => "operator_required",
        EmbeddingsOff => "embeddings_off",
        Confirm => "confirm",
        Rejected => "rejected",
        PrimaryContextRefused => "primary_context_refused",
        NotFound => "not_found",
        UnknownChatAction => "unknown_chat_action",
        UnknownChatActionAttachment => "unknown_chat_action_attachment",
        AttachmentMissing => "attachment_missing",
        AttachmentUnavailable => "attachment_unavailable",
        UnknownImportJob => "unknown_import_job",
        Conflict => "conflict",
        Stale => "stale",
        ScopeChanged => "scope_changed",
        ResyncRequired => "resync_required",
        IdempotencyKeyReused => "idempotency_key_reused",
        ActionFinished => "action_finished",
        NotTheClaimant => "not_the_claimant",
        StageOnlyActionCannotSubmit => "stage_only_action_cannot_submit",
        NotAwaitingApproval => "not_awaiting_approval",
        JobEnded => "job_ended",
        BrowserActionsOff => "browser_actions_off",
        StoppedEarly => "stopped_early",
        NotRunning => "not_running",
        DaemonRequired => "daemon_required",
        ShuttingDown => "shutting_down",
        DaemonProtocolMismatch => "daemon_protocol_mismatch",
        DeadlineExceeded => "deadline_exceeded",
        Cancelled => "cancelled",
        Demo => "demo",
        AttachmentOutput => "attachment_output",
        ReaderUnavailable => "reader_unavailable",
        Schema => "schema",
        ReadOutput => "read_output",
        Io => "io",
        BlobCorrupt => "blob_corrupt",
        Busy => "busy",
        Store => "store",
        Internal => "internal",
        Import => "import",
        Embed => "embed",
        SinexUnavailable => "sinex_unavailable",
        PrimaryContextUnavailable => "primary_context_unavailable",
        PrimaryContextInvalid => "primary_context_invalid",
    }
}
impl From<query_language::ErrorKind> for Code {
    fn from(kind: query_language::ErrorKind) -> Self {
        match kind {
            query_language::ErrorKind::InvalidQuery => Self::InvalidQuery,
            query_language::ErrorKind::UnknownField => Self::UnknownField,
            query_language::ErrorKind::BadValue => Self::BadValue,
            query_language::ErrorKind::NotAQuery => Self::NotAQuery,
            query_language::ErrorKind::Ambiguous => Self::Ambiguous,
            query_language::ErrorKind::StaleCursor => Self::StaleCursor,
        }
    }
}

/// Names joined with commas, as a reason's field.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Names(pub Vec<String>);

impl std::fmt::Display for Names {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.join(", "))
    }
}

impl<T: ToString> FromIterator<T> for Names {
    fn from_iter<I: IntoIterator<Item = T>>(names: I) -> Self {
        Self(names.into_iter().map(|name| name.to_string()).collect())
    }
}

/// Declares the closed reason set: each reason's wire name, its failure
/// code, its fields and its one sentence. A reason without a code does not
/// parse.
macro_rules! reasons {
    ($(
        $(#[doc = $doc:literal])*
        $variant:ident $name:literal => $code:ident { $($field:ident : $ty:ty),* $(,)? } $text:literal;
    )+) => {
        /// Why a request was refused, as a caller can act on it: the
        /// reason's name and fields are compared, its sentence is shown.
        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        #[serde(tag = "name", content = "fields")]
        pub enum Reason {
            $(
                $(#[doc = $doc])*
                #[serde(rename = $name)]
                $variant { $($field: $ty),* },
            )+
        }

        impl Reason {
            pub const NAMES: &'static [&'static str] = &[$($name),+];

            pub fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant { .. } => $name,)+
                }
            }

            pub fn code(&self) -> Code {
                match self {
                    $(Self::$variant { .. } => Code::$code,)+
                }
            }
        }

        impl std::fmt::Display for Reason {
            #[allow(unused_variables)]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant { $($field),* } => write!(f, $text),)+
                }
            }
        }
    };
}

reasons! {
    UnknownMeasure "unknown_measure" => BadValue { value: String, known: Names }
        "unknown measurement `{value}`; choose one of {known}";
    EmbeddingsOff "embeddings_off" => EmbeddingsOff {}
        "semantic search is off; enable it with `polylogue set embed.provider local`";
    EmbeddingStore "embedding_store" => Embed { problem: String }
        "the embeddings store cannot be read: {problem}";
    EmbeddingConfiguration "embedding_configuration" => BadValue { problem: String }
        "semantic search configuration is invalid: {problem}";
    NoVectors "no_vectors" => Embed { model: String }
        "no vectors are stored for model `{model}`; preview their cost with `polylogue embed`";
    EmbeddingProvider "embedding_provider" => Embed { problem: String }
        "the embedding provider failed: {problem}";
    EmbeddingCallsOff "embedding_calls_off" => ModelCallsOff {}
        "semantic search for new text is off (model calls are off); enable it with `polylogue set embed.calls on`";

    /// A write of a single-valued kind over more than one row.
    SingleValuedBulk "single_valued_bulk" => BadRequest { kind: String, rows: usize }
        "`{kind}` holds one value per target and is not written over {rows} rows; label rows with a rubric field (annotation) instead";
    /// Text that does not parse as a reference.
    BadReference "bad_reference" => BadRequest { text: String, problem: String }
        "{text} is not a reference: {problem}";
    RootNamesSession "root_names_session" => BadRequest {}
        "root: names a session, the root of a logical session";
    RangeAcrossSessions "range_across_sessions" => BadRequest {}
        "a message range is read inside one physical session";
    RangeEndsBeforeStart "range_ends_before_start" => BadRequest {}
        "a message range ends before it starts";
    CompactionCount "compaction_count" => BadValue { text: String }
        "compaction:N takes a count, not `{text}`";
    DiffTarget "diff_target" => BadRequest {}
        "diff takes a message or compaction[:N]";
    PhysicalNeedsRoot "physical_needs_root" => BadRequest {}
        "--physical applies to root:REF; a session reference is already one physical session";
    UnknownReadFormat "unknown_read_format" => BadValue { format: String }
        "unknown format `{format}`; formats are terminal, markdown, html, json and jsonl";
    FromSeenNeedsContentView "from_seen_needs_content_view" => BadRequest {}
        "read --from seen needs a session content view";
    FromSeenNeedsOneSession "from_seen_needs_one_session" => BadRequest {}
        "read --from seen needs one session or logical-root reference";
    TraceTakesNoAlternates "trace_takes_no_alternates" => BadRequest {}
        "trace shows original source evidence; alternates are unsupported";
    StreamHasNoContinuation "stream_has_no_continuation" => BadRequest {}
        "streaming reads do not have paged continuations";
    PagedReadNeedsPageDelivery "paged_read_needs_page_delivery" => BadRequest {}
        "paged reads require page delivery";
    StreamNeedsStreamDelivery "stream_needs_stream_delivery" => BadRequest {}
        "streaming reads require stream delivery";
    /// `--at` given a message: the selection names the message to read at.
    AtNamesAMessage "at_names_a_message" => BadRequest { message: String, line: String }
        "--at takes a time or a position, not the message {message}; select the message: `{line}`";
    FollowStreams "follow_streams" => BadRequest {}
        "read --follow streams; it has no single answer";
    ReadNeedsRows "read_needs_rows" => BadRequest {}
        "read needs rows: name a selection";
    ListingNeedsSelection "listing_needs_selection" => BadRequest {}
        "a listing needs a selection";
    StepNeedsRows "step_needs_rows" => BadRequest {}
        "this step needs rows";
    /// A verb that acts on sessions given another row.
    ActsOnSessions "acts_on_sessions" => BadRequest { verb: String, reference: String }
        "{verb} acts on sessions; {reference} is not one";
    PickOnCommandLine "pick_on_command_line" => BadRequest {}
        "pick runs on the command line; send the chosen references";
    BudgetTooLarge "budget_too_large" => BadValue { most: u64 }
        "brief --budget exceeds {most}";
    /// A session word that names no session or several.
    NotOneSession "not_one_session" => Ambiguous { text: String, rows: usize }
        "{text:?} names {rows} rows, not one session";
    MessageNeedsText "message_needs_text" => BadRequest {}
        "a message needs text";
    StartNeedsHarnessOrRole "start_needs_harness_or_role" => BadRequest {}
        "start takes --in or --role";
    RoleRunsAnotherHarness "role_runs_another_harness" => BadValue { role: String, runs: String, asked: String }
        "the role {role} runs {runs}, not {asked}";
    NotADirectory "not_a_directory" => BadValue { path: String }
        "{path} is not a directory";
    ResumeInNeedsSession "resume_in_needs_session" => BadRequest {}
        "resume --in needs the session whose work it continues";
    ResumeOptionsNeedIn "resume_options_need_in" => BadRequest {}
        "resume --at, --cwd and --stop-source apply only with --in HARNESS";
    /// A session whose harness has no resume command of its own.
    NoHarnessResume "no_harness_resume" => BadRequest { session: String }
        "{session} has no harness resume command; use resume --in HARNESS";
    /// A session whose directory is not known, with the line that continues
    /// it with context instead.
    ResumeDirectoryUnknown "resume_directory_unknown" => BadRequest { line: String }
        "directory_unknown: the session recorded no directory to resume in; continue with context instead: {line}";
    /// An option other than those an attachment's bytes take.
    AttachmentReadOptions "attachment_read_options" => BadRequest {}
        "an attachment reads as its bytes; it takes only --input, --budget, --cursor and --format bytes";
    AttachmentReadFormat "attachment_read_format" => BadValue {}
        "an attachment reads only in the bytes format";
    InputNeedsAttachment "input_needs_attachment" => BadRequest {}
        "--input chooses the stored copy of an attachment; read attachment:MESSAGE#BLOCK";
    /// A number larger than this machine can address.
    TooLarge "too_large" => BadValue { field: String }
        "{field} exceeds address space";
    /// A time that does not resolve.
    BadTime "bad_time" => BadRequest { text: String, problem: String }
        "`{text}` is not a time: {problem}";
    HistoricalContextTakesNoAround "historical_context_takes_no_around" => BadRequest {}
        "historical context does not accept around; select its context message";
    PositionFromAnotherGeneration "position_from_another_generation" => BadRequest {}
        "read --at position belongs to another index generation";
    PositionAhead "position_ahead" => BadRequest {}
        "read --at position is ahead of this reader";
    HistoricalReadTarget "historical_read_target" => BadRequest {}
        "historical read --at takes a session or message";
    HistoricalContextTarget "historical_context_target" => BadRequest {}
        "historical context takes one message or a session; select its context message";
    /// A page budget too small to hold the reading's own framing.
    BudgetBelowFraming "budget_below_framing" => BadRequest {}
        "the page budget cannot hold the reading's metadata; raise the budget";
    /// An attachment slot the selected block does not have.
    NoSuchAttachment "no_such_attachment" => NotFound { message: String, block: usize, attachment: usize }
        "{message} block {block} has no attachment {attachment}";
    /// An attachment descriptor the caller stated that differs from the one
    /// the selected source records.
    AttachmentDescriptorMismatch "attachment_descriptor_mismatch" => Conflict { message: String, block: u64, attachment: u64 }
        "the attachment descriptor for {message} block {block} attachment {attachment} differs from the one its source records";
    /// A time or date outside the range the calendar represents.
    OutsideCalendar "outside_calendar" => BadValue { problem: String }
        "the time is outside the representable calendar: {problem}";
    UnknownSetting "unknown_setting" => BadValue { key: String }
        "unknown runtime setting `{key}`";
    BlankSetting "blank_setting" => BadValue { key: String }
        "runtime setting `{key}` cannot be blank";
    BadSettingValue "bad_setting_value" => BadValue { key: String, expected: String }
        "invalid value for runtime setting `{key}`: expected {expected}";
    SettingOwnedBySinex "setting_owned_by_sinex" => Refused { key: String }
        "runtime setting `{key}` is owned by Sinex in primary mode";
    SettingConflict "setting_conflict" => Ambiguous { key: String, revisions: Names }
        "runtime setting `{key}` has conflicting current revisions: {revisions}";
    WatchNeedsName "watch_needs_name" => BadRequest {}
        "a watch needs a saved query name";
    FigureNeedsName "figure_needs_name" => BadRequest {}
        "a figure needs a saved query name";
    WatchDuration "watch_duration" => BadValue { text: String }
        "a watch needs a duration such as 20m, not `{text}`";
    KeptRowsMissing "kept_rows_missing" => BadRequest {}
        "a kept result needs its rows";
    BadSavedName "bad_saved_name" => BadValue { name: String }
        "`{name}` is not a saved query name";
    /// A reference that names an annotation family where one exact
    /// revision is needed.
    NotAnExactRevision "not_an_exact_revision" => BadRequest { reference: String }
        "{reference} is not an exact annotation revision; name one revision";
    /// A reference to a record of another kind than the operation takes.
    WrongKind "wrong_kind" => BadRequest { reference: String, expected: String }
        "{reference} is not a {expected}";
    /// An annotation's value that does not have its kind's shape.
    BadAnnotationValue "bad_value" => BadValue { kind: String, problem: String }
        "the {kind} value is malformed: {problem}";
    /// An annotation's links that break its kind's rules.
    BadAnnotationLinks "bad_annotation_links" => BadRequest { problem: String }
        "the annotation's links are refused: {problem}";
    RubricValueNeedsRubric "rubric_value_needs_rubric" => BadRequest {}
        "a rubric value needs its exact derived_from rubric";
    /// A rubric value its rubric does not allow: the part is the field
    /// value, an abstention, a confidence or a rationale.
    NotAllowedByRubric "not_allowed_by_rubric" => BadRequest { key: String, part: String }
        "the rubric does not allow {part} for `{key}`";
    PassNotByAuthor "pass_not_by_author" => BadRequest {}
        "a rubric value's origin must be a pass opened by its author";
    PassForAnotherRubric "pass_for_another_rubric" => BadRequest {}
        "the rubric value's pass names another rubric";
    PassRequired "pass_required" => BadRequest {}
        "a rubric value needs an exact pass origin";
    BasesSpanFamilies "bases_span_families" => BadRequest {}
        "one revision must name bases of one annotation family, or of families holding one slot";
    /// A revision that would take a uniqueness slot other records hold.
    SlotTaken "slot_taken" => Conflict { kind: String, holders: Names }
        "the revised {kind} would take a slot also held by {holders}; a revision naming every head of every holder resolves it";
    NoStatements "no_statements" => BadRequest {}
        "assert needs at least one statement";
    RevisionChangesKind "revision_changes_kind" => BadRequest {}
        "a revision cannot change the annotation's kind";
    NoExactBase "no_exact_base" => BadRequest {}
        "revise needs at least one exact base";
    WithdrawalRevised "withdrawal_revised" => BadRequest {}
        "a withdrawal cannot be revised; withdraw it to restore what it withdrew";
    BasesOfSeveralKinds "bases_of_several_kinds" => BadRequest {}
        "one revision must name bases of one kind";
    BasesInLinkPatch "bases_in_link_patch" => BadRequest {}
        "revision bases belong in bases, not the link patch";
    QueryAnnotationAboutLinks "query_annotation_about_links" => BadRequest {}
        "a query annotation cannot supply its own about links";
    /// A query annotation whose text cannot be read as an annotation.
    BadAnnotation "bad_annotation" => BadRequest { problem: String }
        "the annotation is refused: {problem}";
    /// A value outside a closed set of choices.
    UnknownChoice "unknown_choice" => BadValue { field: String, value: String, known: Names }
        "{field} takes one of {known}, not `{value}`";
    MeasureTakesNoTimeBounds "measure_takes_no_time_bounds" => BadRequest {}
        "per-session measures take no time bounds";
    AlternateSource "alternate_source" => BadRequest {}
        "an alternate's source must name captured text or an exact authored revision";
    AlternateRevisionChanges "alternate_revision_changes" => BadRequest {}
        "an alternate revision must keep its source revision, kind and languages";
    /// Span coordinates that do not fit the text they name.
    BadSpan "bad_span" => BadRequest { problem: String }
        "the span is refused: {problem}";
    /// A message or seen item outside the session it is read in.
    NotInSession "not_in_session" => BadRequest { item: String, session: String }
        "{item} is not in {session}";
    ReadStartOutsideSession "read_start_outside_session" => BadRequest {}
        "the read starts at a message the session does not hold";
    TraceTarget "trace_target" => BadRequest {}
        "trace needs a message, block, tool call or text span";
    /// A count or size that must be at least one.
    NotPositive "not_positive" => BadValue { field: String }
        "{field} must be positive";
    /// A configured value the daemon cannot run with.
    BadConfiguration "bad_configuration" => BadValue { problem: String }
        "the configuration is refused: {problem}";
    /// Import options that cannot be used together or with what they name.
    ImportOptions "import_options" => BadRequest { problem: String }
        "the import is refused: {problem}";
    NotAFile "not_a_file" => BadRequest { path: String }
        "{path} is not a file: name one of Python Polylogue's vector stores";
    NotAVectorStore "not_a_vector_store" => BadRequest { path: String }
        "{path} is not one of Python Polylogue's vector stores: it has no message_embeddings tables in either layout";
    PathHasNoFileName "path_has_no_file_name" => BadRequest { path: String }
        "{path} has no file name";
    ArchiveModeMismatch "archive_mode_mismatch" => Refused { archive: String, recorded: String, requested: String }
        "archive {archive} is {recorded}, but configuration requests {requested}; select its recorded mode or another archive";
    ArchiveModeMissing "archive_mode_missing" => Refused { archive: String }
        "archive {archive} has no recorded mode; fix its configuration before creating it";
    /// A storage mode this version of Polylogue does not build.
    ModeNotBuilt "mode_not_built" => NotBuilt { mode: String }
        "{mode} mode is not built in this version of Polylogue";
    BadQueryHash "bad_query_hash" => BadValue { prefix: String }
        "a query hash takes 8 to 64 hexadecimal characters, not `{prefix}`";
    SeenMarkNeedsItem "seen_mark_needs_item" => BadRequest {}
        "a seen mark on a session names a message or block";
    SeenFromAnotherGeneration "seen_from_another_generation" => Conflict {}
        "the seen mark belongs to another index generation; read the session again";
    BadCoverage "bad_coverage" => BadRequest { problem: String }
        "the capture coverage is refused: {problem}";
    /// A context budget too small for a required part of the brief.
    BudgetTooSmall "budget_too_small" => BadRequest { part: String }
        "the context budget cannot fit {part}; increase budget_tokens";
    WithdrawalCycle "withdrawal_cycle" => BadRequest { revision: String }
        "withdrawals of {revision} form a cycle";
    /// A kept result whose stored body does not have the kept shape.
    BadKeptResult "bad_kept_result" => BadRequest { id: String, field: String, problem: String }
        "kept result `{id}` field {field}: {problem}";
    /// The daemon did not answer a request a surface forwarded to it.
    DaemonDidNotAnswer "daemon_did_not_answer" => DaemonRequired {}
        "the Polylogue daemon did not answer; it is stopped or busy";
    /// A save the daemon did not answer: it may or may not have been written.
    SaveOutcomeUnknown "save_outcome_unknown" => DaemonRequired {}
        "the note may or may not be saved; retry it with the same ID";
    CaptureNotJson "capture_not_json" => BadRequest {}
        "a capture is one JSON object";
    CaptureNamesNoConversation "capture_names_no_conversation" => BadRequest {}
        "a capture names session.provider and session.provider_session_id";
    CaptureHasNoConversation "capture_has_no_conversation" => BadRequest {}
        "a capture holds the provider's conversation JSON (raw_provider_payload) or session.turns";
    CaptureUnreadable "capture_unreadable" => BadRequest { problem: String }
        "the capture cannot be read: {problem}";
    /// An attachment whose stored bytes changed after it was selected.
    AttachmentChanged "attachment_changed" => Conflict { input: String }
        "the attachment's stored bytes in {input} changed; select it again";
}

impl std::error::Error for Reason {}

#[cfg(test)]
mod tests;
