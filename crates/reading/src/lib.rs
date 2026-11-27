//! Reading stored sessions and records as pages and streams.
mod error;
pub mod execution;
pub mod operation;
use command_surface::Deadline;
pub use error::Error;
use operation::Context;
use polylogue_model::Reason;
use polylogue_model::bundle::ReadSnapshot;
use polylogue_model::{MessageId, freshness::ReadFreshness, read::*};
use polylogue_render::pager::{PageBuilder, PageError, Push};
use polylogue_store::{Reader, WriterSender, selection::NearFn};
use polylogue_store::{selection::PreparedSelection, writer::HistorySlot};
use query_language::cursor::OrderPosition;
use std::{collections::BTreeMap, io::Write, ops::ControlFlow, sync::Arc};

mod attachment_facts;
mod authored;
mod blobs;
mod documents;
mod layout_facts;
mod links;
mod live;
mod object;
mod output;
mod records;
mod rows;
mod saved_query;
mod session;
mod trace;

/// A caller's start, resolved once against its authenticated read.
#[derive(Debug, Clone, Default)]
pub enum ReadStart {
    #[default]
    Beginning,
    Seen,
    Bound(Box<SessionRead>),
}
impl ReadStart {
    pub fn uses_seen(&self) -> bool {
        match self {
            Self::Beginning => false,
            Self::Seen => true,
            Self::Bound(bound) => bound.from_seen,
        }
    }
}

/// One read: what it reads (the target, or the fixed rows a line resolved),
/// how (the options the caller wrote), where it starts, and how it is
/// delivered and paged.
#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub target: String,
    pub references: Option<Vec<polylogue_model::Reference>>,
    pub options: ReadOptions,
    pub start: ReadStart,
    pub delivery: ReadDelivery,
    pub format: Format,
    pub budget: ReadBudget,
    pub cursor: Option<String>,
    /// The alternates the reading renders, resolved once by
    /// [`with_alternates`]; empty before.
    pub alternates: Vec<String>,
}
impl ReadRequest {
    /// The first page of `target` read with `options`, as JSON; `--from
    /// seen` starts where this reader stopped.
    pub fn new(target: impl Into<String>, options: ReadOptions) -> Self {
        Self {
            start: if options.from.as_deref() == Some("seen") {
                ReadStart::Seen
            } else {
                ReadStart::Beginning
            },
            target: target.into(),
            references: None,
            options,
            delivery: ReadDelivery::Page,
            format: Format::Json,
            budget: ReadBudget::Default,
            cursor: None,
            alternates: Vec::new(),
        }
    }
}
/// Resolve the authenticated session start before accepting a page or stream.
/// Composition captures logical parts' marks against this same pinned reader.
pub fn resolve_start(
    reader: &Reader<'_>,
    operation: &Context,
    request: &mut ReadRequest,
) -> Result<(), Error> {
    let from_seen = request.start.uses_seen();
    if from_seen
        && !matches!(
            request.options.view,
            View::Transcript | View::Dialogue | View::Context | View::Raw | View::Changes
        )
    {
        return Err(Reason::FromSeenNeedsContentView {}.into());
    }
    if let Some(cursor) = &request.cursor {
        let opened = PageCursor::open(cursor, &cursor_seal(reader)?).map_err(cursor_failure)?;
        if let Some(bound) = opened.session_read {
            if bound.from_seen != from_seen {
                return Err(Error::invalid_query(
                    "the cursor belongs to another read start",
                ));
            }
            request.start = ReadStart::Bound(Box::new(bound));
            return Ok(());
        }
        if from_seen {
            return Err(Error::invalid_query(
                "the cursor has no accepted seen start",
            ));
        }
    }
    // Fixed populations retain their grains and order. Their joined display
    // question is not a single session or message selector.
    if request
        .references
        .as_ref()
        .is_some_and(|references| references.len() != 1)
    {
        return if from_seen {
            Err(Reason::FromSeenNeedsOneSession {}.into())
        } else {
            Ok(())
        };
    }
    if let Some(scope) =
        session::scope(reader, &request.target)?.filter(|scope| scope.range.is_none())
    {
        let logical = scope.logical && !request.options.physical;
        let after = if from_seen && !logical {
            let target = polylogue_model::Reference::Session(scope.session.clone());
            let mut marks = reader.seen_marks(
                &operation.authorship.author.to_string(),
                std::slice::from_ref(&target),
            )?;
            if marks.len() > 1 {
                return Err(Error::conflict(format!(
                    "multiple current seen marks for {target}"
                )));
            }
            marks.pop().and_then(|mark| mark.item)
        } else {
            None
        };
        request.start = ReadStart::Bound(Box::new(SessionRead {
            from_seen,
            after,
            ..SessionRead::beginning(scope.session, logical)
        }));
    } else if from_seen {
        return Err(Reason::FromSeenNeedsOneSession {}.into());
    }
    Ok(())
}

pub trait ReadServices {
    fn requests(&self) -> &polylogue_model::user::RequestMetadata;
    fn freshness(&self, reader: &Reader<'_>) -> polylogue_store::Result<ReadFreshness>;
    /// Compile options for `target` with this archive's ranking.
    fn compiled<T>(
        &self,
        reader: &Reader<'_>,
        operation: &Arc<Context>,
        target: &str,
        run: impl FnOnce(Compile<'_>) -> Result<T, Error>,
    ) -> Result<T, Error>;
    fn continuation(
        &self,
        request: &ReadRequest,
        cursor: Option<String>,
        deadline: Deadline,
    ) -> Result<ReadyCall, Error>;
    fn attachment_call(&self, at: &AttachmentPlace) -> Result<ReadyCall, Error>;
    fn writer(&self) -> Option<&WriterSender>;
}
pub struct Compile<'a> {
    pub options: polylogue_store::query::Options,
    pub ranking_identity: Option<String>,
    pub near: &'a NearFn<'a>,
}
pub struct AttachmentPlace {
    pub message: MessageId,
    pub block: u64,
    pub attachment: u64,
    pub input: Option<String>,
}

/// The read with the alternates it renders resolved once: the one the
/// caller names (`none` for the originals), else those its cursor's first
/// page rendered, else the `reading.alternates` setting.
pub fn with_alternates(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    mut request: ReadRequest,
) -> Result<ReadRequest, Error> {
    let named = request.options.alternate.as_deref();
    if request.options.view == View::Trace {
        if named.is_some_and(|alternate| alternate != "none") {
            return Err(Reason::TraceTakesNoAlternates {}.into());
        }
        request.alternates.clear();
        return Ok(request);
    }
    request.alternates = match (named, &request.cursor) {
        (Some("none"), _) => Vec::new(),
        (Some(alternate), _) => vec![alternate.to_owned()],
        (None, Some(text)) => {
            PageCursor::open(text, &cursor_seal(reader)?)
                .map_err(cursor_failure)?
                .alternates
        }
        (None, None) => reader
            .settings_snapshot(services.requests().machine())?
            .reading_alternates()
            .map_err(Error::conflict)?
            .to_vec(),
    };
    Ok(request)
}

/// The format, page limits and options digest a continuation cursor of a
/// session or links page must match.
fn page_terms(request: &ReadRequest) -> Result<(Format, PageBudget, String), Error> {
    let format = request.format;
    let limits = PageBudget::default().with_tokens(request.budget);
    let digest = options_digest(&(&request.options, format));
    Ok((format, limits, digest))
}

fn cursor_seal(reader: &polylogue_store::Reader<'_>) -> Result<CursorSeal, Error> {
    Ok(CursorSeal {
        key: reader.cursor_key()?,
        index_instance: reader.change_position()?.index_instance,
    })
}

/// A cursor of an earlier generation is `stale_cursor`; a broken one, or
/// one of another reading, is `invalid_query` (docs/query.md, Paging).
fn cursor_failure(error: CursorError) -> Error {
    match error {
        CursorError::Stale => {
            Error::stale_cursor("the index was rebuilt since this page was issued")
        }
        CursorError::EarlierBuild => {
            Error::stale_cursor("Polylogue was updated since this page was issued")
        }
        CursorError::Invalid | CursorError::OtherReading(_) => {
            Error::InvalidQuery(error.to_string())
        }
    }
}

/// The order values of each row a page visited, by the row's number on
/// that page, so a cursor names its row by key.
#[derive(Default)]
struct RowKeys {
    keys: BTreeMap<u64, OrderPosition>,
}

/// The resolved reading of a row-path request.
struct Reading {
    options: ReadOptions,
    alternates: Vec<String>,
    format: Format,
    budget: Option<PageBudget>,
    target: String,
    references: Option<Vec<polylogue_model::Reference>>,
    digest: String,
}

fn reading(request: &ReadRequest) -> Result<(Reading, Option<&str>), Error> {
    let ReadRequest {
        target,
        options,
        delivery,
        format,
        budget,
        cursor,
        ..
    } = request;
    options.validate().map_err(Error::Layout)?;
    let format = *format;
    let budget = if *delivery == ReadDelivery::Page {
        Some(PageBudget::default().with_tokens(*budget))
    } else {
        None
    };
    if cursor.is_some() && budget.is_none() {
        return Err(Reason::StreamHasNoContinuation {}.into());
    }
    Ok((
        Reading {
            digest: options_digest(&(options, format, &request.references)),
            references: request.references.clone(),
            options: options.clone(),
            alternates: request.alternates.clone(),
            format,
            budget,
            target: target.clone(),
        },
        cursor.as_deref(),
    ))
}

/// A reading's target as compiled against the reader: the selection the
/// reading visits, and what its run record is built from.
struct Compiled<'a> {
    selection: &'a PreparedSelection<'a>,
    options: &'a polylogue_store::query::Options,
    ranking_identity: Option<String>,
    history: Option<&'a saved_query::SavedQuery>,
}

fn prepared<T>(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    operation: &Arc<Context>,
    reading: &Reading,
    run: impl FnOnce(&Compiled<'_>, ReadSnapshot, ReadHeader, Option<&NearFn<'_>>) -> Result<T, Error>,
) -> Result<T, Error> {
    services.compiled(reader, operation, &reading.target, |compile| {
        polylogue_store::read::limit::ReadLimit::set_stage("read preparation");
        let mut selection = match &reading.references {
            Some(references) => {
                reader.prepare_fixed_selection(&reading.target, references, &compile.options)?
            }
            None => reader.prepare_selection(&reading.target, &compile.options)?,
        };
        let similarity = selection.ranking(reader, Some(compile.near))?;
        let near = |_: &str| {
            similarity.clone().ok_or_else(|| {
                polylogue_store::StoreError::from(polylogue_model::Reason::EmbeddingsOff {})
            })
        };
        let snapshot = ReadSnapshot {
            query: format!("query:{}", selection.question_id()),
            query_text: reading.target.clone(),
            ran_at: operation.ran_at,
            read_position: reader.change_position()?,
            freshness: services.freshness(reader)?,
            enabled_alternates: reading.alternates.clone(),
            author: operation.authorship.author.clone(),
            notes: selection.program().notes.clone(),
            incomplete: selection.program().incomplete.clone(),
        };
        let (title, mut kind) = match selection.direct_target() {
            Some(ReadTarget::Session(id)) => {
                let session = reader
                    .session(id)?
                    .ok_or_else(|| Error::not_found(format!("no session {id}")))?;
                let title = reader
                    .display_title(id)?
                    .map(ValueState::known)
                    .unwrap_or_default();
                (
                    title,
                    HeaderKind::Session {
                        session: Box::new(session),
                    },
                )
            }
            Some(target) => (
                ValueState::NotApplicable,
                HeaderKind::Record {
                    record_type: target
                        .to_string()
                        .split_once(':')
                        .map_or("record", |(kind, _)| kind)
                        .into(),
                },
            ),
            None => (
                ValueState::NotApplicable,
                HeaderKind::Record {
                    record_type: "query".into(),
                },
            ),
        };
        let history = saved_query::prepare(reader, &reading.target, &selection, operation)?;
        if let Some(saved) = &history {
            kind = HeaderKind::SavedQuery(Box::new(saved.header.clone()));
        }
        let header = ReadHeader {
            target: selection
                .direct_target()
                .map_or_else(|| snapshot.query.clone(), ToString::to_string),
            view: reading.options.view,
            title,
            kind,
            figures: Vec::new(),
            zone: Some(operation.clock.day.zone_name().to_owned()),
            snapshot: None,
            options: None,
        };
        polylogue_store::read::limit::ReadLimit::set_stage("read records");
        run(
            &Compiled {
                selection: &selection,
                options: &compile.options,
                ranking_identity: compile.ranking_identity,
                history: history.as_ref(),
            },
            snapshot,
            header,
            Some(&near),
        )
    })
}

type Emit<'a> = dyn FnMut(&ReadRecord, SelectionLocator, Option<u64>) -> Result<bool, Error> + 'a;

fn selected_source(
    reader: &Reader<'_>,
    selection: &PreparedSelection<'_>,
    target: Option<&ReadTarget>,
    options: &ReadOptions,
    near: Option<&polylogue_store::selection::NearFn<'_>>,
    source: &Source,
) -> Result<bool, Error> {
    if let (Some(ReadTarget::Session(session)), Some(message)) = (target, &source.message) {
        return Ok(reader
            .select_messages(session, options.inherited())?
            .locate(Some(message))?
            .is_some());
    }
    let reference =
        polylogue_model::Reference::parse(&source.reference).map_err(Error::not_yet_available)?;
    reader
        .selection_contains(selection, &reference, near)
        .map_err(Error::from)
}

struct Rendering<'a> {
    at: polylogue_model::Timestamp,
    target: &'a str,
    options: &'a ReadOptions,
    near: Option<&'a polylogue_store::selection::NearFn<'a>>,
    ready_attachment: &'a attachment_facts::ReadyAttachment<'a>,
    page: bool,
    object_facts: Option<&'a LayoutFacts>,
    history_seed: Option<polylogue_model::BlobHash>,
}

/// A paged read has a budget; only page delivery sets one.
fn require_page_delivery(reading: &Reading) -> Result<(), Error> {
    if reading.budget.is_none() {
        return Err(Reason::PagedReadNeedsPageDelivery {}.into());
    }
    Ok(())
}

fn attachment_call(
    services: &impl ReadServices,
    message: &MessageId,
    block: u64,
    attachment: u64,
    input: Option<&str>,
) -> Result<ReadyCall, Error> {
    services.attachment_call(&AttachmentPlace {
        message: message.clone(),
        block,
        attachment,
        input: input.map(str::to_owned),
    })
}

/// First pages queue their run after reading; an answer stands without its run record.
fn record_run(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    operation: &Arc<Context>,
    request: &ReadRequest,
    reading: &Reading,
    history: Option<HistorySlot>,
) -> Result<(), Error> {
    if history.is_none() {
        return Ok(());
    }
    services.compiled(reader, operation, &reading.target, |compile| {
        let target = reading
            .target
            .strip_prefix("root:")
            .unwrap_or(&reading.target);
        let selection = reader.prepare_selection(target, &compile.options)?;
        record_compiled(
            reader,
            services,
            operation,
            request,
            reading,
            &Compiled {
                selection: &selection,
                options: &compile.options,
                ranking_identity: compile.ranking_identity,
                history: None,
            },
            history,
        )
    })
}

/// Record a reading's run from its compiled target, as [`record_run`] says.
fn record_compiled(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    operation: &Arc<crate::operation::Context>,
    request: &ReadRequest,
    reading: &Reading,
    compiled: &Compiled<'_>,
    history: Option<polylogue_store::writer::HistorySlot>,
) -> Result<(), Error> {
    use polylogue_model::query_objects::{
        ExecutionPresentation, QueryRun, QueryRunOutcome, RunBinding, RunOutcome,
    };
    let ReadRequest { delivery, .. } = request;
    let Some(history) = history else {
        return Ok(());
    };
    let Compiled {
        selection,
        options,
        ranking_identity,
        history: _,
    } = compiled;
    let object = selection.query_object()?;
    let query_options = polylogue_store::query::execute::options_snapshot(
        options,
        false,
        ranking_identity.clone(),
    )?;
    let accepted = operation
        .accepted_request()
        .map(serde_json::to_value)
        .transpose()
        .map_err(Error::internal)?;
    let mut binding = selection.execution_binding(
        query_options,
        ExecutionPresentation::Read {
            options: Box::new(reading.options.clone()),
            delivery: *delivery,
            format: reading.format,
            page_budget: reading.budget,
            direct_source: None,
        },
        accepted,
    )?;
    binding
        .replay
        .context
        .versions
        .extend(operation.cursor_versions());
    let run = QueryRun {
        id: crate::operation::new_run_id(),
        query_id: Some(selection.question_id()),
        at: operation.ran_at,
        read_position: reader.change_position()?,
        machine: operation.machine.clone(),
        authorship: operation.authorship.clone(),
        binding: RunBinding::Original {
            execution: Box::new(binding),
        },
    };
    let outcome = QueryRunOutcome {
        run: run.id.clone(),
        binding_hash: run.binding_hash().map_err(Error::internal)?,
        run_hash: run.record_hash().map_err(Error::internal)?,
        at: services.requests().now(),
        outcome: RunOutcome::Ok,
        count: None,
        exact: false,
        next: None,
    };
    history.record(object, run, outcome).map_err(Error::from)
}

/// A reading's seed is its target's, so a sampled or shuffled reading
/// keeps its order across pages. `ReadServices::compiled` implementations
/// compile with this seed.
pub fn seed(target: &str) -> u64 {
    u64::from_le_bytes(
        blake3::hash(target.as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("eight bytes"),
    )
}

/// The deadline the caller asked for, carried to every continuation.
fn deadline(operation: &crate::operation::Context) -> command_surface::Deadline {
    operation
        .accepted_request()
        .map_or(command_surface::Deadline::Default, |request| {
            request.deadline
        })
}

fn next_call(
    services: &impl ReadServices,
    request: &ReadRequest,
    cursor: String,
    deadline: Deadline,
) -> Result<ReadyCall, Error> {
    services.continuation(request, Some(cursor), deadline)
}

pub fn is_object_read(request: &ReadRequest) -> bool {
    object::target(request).is_some()
}

/// Resolve against the accepted query clock before sealing a page cursor. Ready
/// calls carry the absolute instant, so a relative time cannot drift on replay.
/// A word that names a message is refused with the line that selects it: the
/// selection, not `--at`, names the message a context is read at.
fn resolve_at<'a>(
    request: &'a ReadRequest,
    operation: &Context,
) -> Result<std::borrow::Cow<'a, ReadRequest>, Error> {
    let text = match request.options.at.as_ref() {
        Some(ReadAt::TimeExpression(text)) => text.clone(),
        Some(ReadAt::Time(at)) => at.to_string(),
        _ => return Ok(std::borrow::Cow::Borrowed(request)),
    };
    let resolved = query_language::time::resolve(&text, &operation.clock).map_err(|error| {
        match MessageId::parse(&text) {
            Ok(message) => Error::from(Reason::AtNamesAMessage {
                message: message.to_string(),
                line: message_line(&message, request.options.view),
            }),
            Err(_) => Reason::BadTime {
                text: text.clone(),
                problem: error.to_string(),
            }
            .into(),
        }
    })?;
    // A snapshot names one boundary. Query predicates still match the whole
    // written period; here its resolved lower bound is the chosen instant.
    let at = match resolved.value {
        query_language::time::Time::Instant { at } => at,
        query_language::time::Time::Span { start, .. } => start,
    };
    let mut request = request.clone();
    request.options.at = Some(ReadAt::Time(at.into()));
    Ok(std::borrow::Cow::Owned(request))
}

/// The line that reads `message` in `view`.
fn message_line(message: &MessageId, view: View) -> String {
    let reference = polylogue_model::Reference::Message(message.clone());
    match view {
        View::Transcript => format!("polylogue {reference} read"),
        view => format!("polylogue {reference} read --view {view}"),
    }
}

fn validate_at(reader: &Reader<'_>, request: &ReadRequest) -> Result<(), Error> {
    if request.options.annotation_at().is_none() {
        return Ok(());
    }
    if request.options.view == View::Context && request.options.around.is_some_and(|n| n > 0) {
        return Err(Reason::HistoricalContextTakesNoAround {}.into());
    }
    if let Some(position) = request.options.annotation_position() {
        let current = reader.change_position()?;
        if position.index_instance != current.index_instance {
            return Err(Reason::PositionFromAnotherGeneration {}.into());
        }
        if position.index > current.index || position.archive > current.archive {
            return Err(Reason::PositionAhead {}.into());
        }
    }
    let Some(scope) = session::scope(reader, &request.target)? else {
        return Err(Reason::HistoricalReadTarget {}.into());
    };
    if request.options.view == View::Context && !scope.has_context_anchor() {
        return Err(Reason::HistoricalContextTarget {}.into());
    }
    Ok(())
}

/// What a read reads, decided once from its target, view and delivery: a
/// trace, a stored request or run, a session reading, the links tree of a
/// session, or the rows of a selection.
enum Route {
    Trace,
    Object(polylogue_model::Reference),
    Session(session::Scope),
    Links(session::Scope),
    Rows,
}

fn route(reader: &Reader<'_>, request: &ReadRequest) -> Result<Route, Error> {
    let route = reading_route(reader, request)?;
    if !matches!(route, Route::Session(_)) {
        if request.options.from_position().is_some() {
            return Err(Error::invalid_query("--from needs a session reading"));
        }
        if request.options.until.is_some() {
            return Err(Error::invalid_query("--until needs a session reading"));
        }
    }
    Ok(route)
}

fn reading_route(reader: &Reader<'_>, request: &ReadRequest) -> Result<Route, Error> {
    let view = request.options.view;
    if view == View::Trace {
        return Ok(Route::Trace);
    }
    if let Some(reference) = object::target(request) {
        return Ok(Route::Object(reference?));
    }
    // A line's fixed population of several rows is read as those rows.
    if request
        .references
        .as_ref()
        .is_some_and(|references| references.len() != 1)
    {
        return Ok(Route::Rows);
    }
    let Some(scope) = session::scope(reader, &request.target)? else {
        return Ok(Route::Rows);
    };
    // A bundle holds the selection's records; a context, or a reading from
    // the seen mark, is the composed session reading in a bundle too.
    if request.delivery == ReadDelivery::Bundle
        && view != View::Context
        && !request.start.uses_seen()
    {
        return Ok(Route::Rows);
    }
    Ok(match view {
        View::Transcript | View::Dialogue | View::Context | View::Raw | View::Changes => {
            Route::Session(scope)
        }
        View::Tree => Route::Links(scope),
        View::Overview | View::Live | View::Trace => Route::Rows,
    })
}

/// One page and its completed physical-session progress, from the same
/// pinned reader. The consumer decides whether this delivery acknowledges it.
pub fn page(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    history: Option<HistorySlot>,
) -> Result<(ReadPage, Vec<polylogue_model::user::SeenMark>), Error> {
    let resolved = resolve_at(request, operation)?;
    let request = resolved.as_ref();
    validate_at(reader, request)?;
    let route = match route(reader, request)? {
        Route::Trace => {
            return trace::page(reader, services, request, operation, history)
                .map(|page| (page, vec![]));
        }
        Route::Object(reference) => {
            return object::page(reader, services, request, operation, reference)
                .map(|page| (page, vec![]));
        }
        route => route,
    };
    let (reading, cursor) = reading(request)?;
    require_page_delivery(&reading)?;
    let page = match route {
        Route::Session(scope) => session::page(reader, services, request, operation, scope)?,
        Route::Links(scope) => (
            links::page(reader, services, request, operation, scope)?,
            vec![],
        ),
        Route::Trace | Route::Object(_) | Route::Rows => {
            return prepared(
                reader,
                services,
                operation,
                &reading,
                |compiled, snapshot, header, near| {
                    let page = rows_page(
                        reader,
                        services,
                        request,
                        operation,
                        &reading,
                        &match compiled.history {
                            Some(history) => rows::Rows::SavedQuery(compiled.selection, history),
                            None => rows::Rows::Selection(compiled.selection),
                        },
                        snapshot,
                        header,
                        ReadFraming::Query {
                            query_id: compiled.selection.question_id(),
                        },
                        compiled.history.map_or_else(
                            || Ok(PageCursor::version(&[])),
                            |history| history.version(),
                        )?,
                        near,
                        None,
                        ReadFooter::default(),
                    )?;
                    if cursor.is_none() {
                        let _ = record_compiled(
                            reader, services, operation, request, &reading, compiled, history,
                        );
                    }
                    Ok((page, vec![]))
                },
            );
        }
    };
    if cursor.is_none() {
        let _ = record_run(reader, services, operation, request, &reading, history);
    }
    Ok(page)
}

#[allow(clippy::too_many_arguments)]
fn rows_page(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    reading: &Reading,
    selected: &rows::Rows<'_>,
    snapshot: ReadSnapshot,
    header: ReadHeader,
    framing: ReadFraming,
    versions: polylogue_model::BlobHash,
    near: Option<&NearFn<'_>>,
    object_facts: Option<&LayoutFacts>,
    mut footer: ReadFooter,
) -> Result<ReadPage, Error> {
    let key = cursor_seal(reader)?;
    let opened = request
        .cursor
        .as_deref()
        .map(|text| {
            let opened = PageCursor::open(text, &key).map_err(cursor_failure)?;
            opened
                .check(&reading.target, &reading.digest)
                .map_err(cursor_failure)?;
            if (matches!(selected, rows::Rows::Object(..)) || selected.history().is_some())
                && opened.rewritten(&versions)
            {
                return Err(Error::stale_cursor(
                    "the stored record changed since this page was issued",
                ));
            }
            if opened.direction == ReadDirection::Past {
                return Err(Error::invalid_query(
                    "a row reading continues forward or backward only",
                ));
            }
            match opened.position {
                PagePosition::Record {
                    start_row,
                    place,
                    history_seed,
                } => Ok((start_row, place, opened.direction, history_seed)),
                _ => Err(Error::InvalidQuery(
                    "the cursor belongs to a session reading".into(),
                )),
            }
        })
        .transpose()?;
    let history_seed = if selected.history().is_some() {
        match &opened {
            Some((_, _, _, Some(seed))) => Some(*seed),
            Some(_) => {
                return Err(Error::stale_cursor(
                    "the saved-query history cursor predates this reading layout",
                ));
            }
            None => selected.history().map(|history| history.seed),
        }
    } else {
        if opened
            .as_ref()
            .is_some_and(|(_, _, _, seed)| seed.is_some())
        {
            return Err(Error::InvalidQuery(
                "the cursor belongs to a saved-query history reading".into(),
            ));
        }
        None
    };
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        attachment_call(services, message, block, index, input)
    };
    let rendering = Rendering {
        at: operation.ran_at,
        target: &reading.target,
        options: &reading.options,
        near,
        ready_attachment: &ready_attachment,
        page: true,
        object_facts,
        history_seed,
    };
    let scope = reader.scope_token();
    let mut budget = reading.budget.expect("a paged reading has a budget");
    let snapshot_bytes = serde_json::to_vec(&ReadPageSnapshot::from(&snapshot))
        .map_err(Error::internal)?
        .len();
    budget.bytes = std::num::NonZeroUsize::new(budget.bytes.get().saturating_sub(snapshot_bytes))
        .ok_or(Reason::BudgetBelowFraming {})?;
    let backwards = opened
        .as_ref()
        .is_some_and(|(_, _, direction, _)| *direction == ReadDirection::Before);
    // A forward page numbers its rows from its start; a backward page from
    // its end, the row its cursor names, counting back.
    let page_start = opened.as_ref().map(|(_, place, _, _)| {
        let mut place = *place;
        place.selection.row = 0;
        place
    });
    let mut pager = if backwards {
        PageBuilder::new_before(
            header,
            budget,
            page_start.expect("Before has a page start"),
            framing,
        )
    } else {
        PageBuilder::new(header, budget, page_start, framing)
    }
    .map_err(Error::from)?;
    let mut keys = RowKeys::default();
    let mut emit = |record: &ReadRecord, at: SelectionLocator, ordinal: Option<u64>| {
        let Some(ordinal) = ordinal else {
            return Ok(true);
        };
        let Some(ReadPart::Item(item)) = record.parts().first() else {
            return Err(Error::Internal(
                "a rendered record has no layout item".into(),
            ));
        };
        Ok(pager
            .push_at(at, ordinal, item.clone())
            .map_err(Error::from)?
            != Push::Full)
    };
    let start = rows::RowStart {
        row: opened.as_ref().and_then(|(row, _, _, _)| row.clone()),
        place: page_start,
        direction: if backwards {
            ReadDirection::Before
        } else {
            ReadDirection::After
        },
    };
    rows::visit_rows(
        reader,
        selected,
        &rendering,
        &start,
        &mut keys,
        &mut footer,
        &mut emit,
    )?;
    scope.check()?;
    polylogue_store::read::limit::ReadLimit::set_stage("read page footer");
    let upper = pager.upper_position().copied();
    // The rows of a backward page, renumbered from its first row once built.
    let first_row = backwards
        .then(|| pager.first_position().map(|place| place.selection.row))
        .flatten();
    let continue_at = |place: &ItemPlace, direction| -> Result<ReadyCall, PageError> {
        let row = keys.keys.get(&place.selection.row).cloned();
        let mut place = *place;
        place.selection.row = 0;
        let cursor = PageCursor {
            session_read: None,
            target: reading.target.clone(),
            digest: reading.digest.clone(),
            alternates: reading.alternates.clone(),
            versions,
            position: PagePosition::Record {
                start_row: row,
                place,
                history_seed,
            },
            direction,
        };
        next_call(services, request, cursor.seal(&key), deadline(operation))
            .map_err(|e| PageError::Serialization(e.to_string()))
    };
    let mut document = pager
        .finish_with(|position| {
            let mut footer = footer.clone();
            if let Some(upper) = upper {
                footer.next = Some(continue_at(&upper, ReadDirection::After)?);
                footer.previous = position
                    .map(|position| continue_at(position, ReadDirection::Before))
                    .transpose()?;
            } else {
                footer.next = position
                    .map(|position| continue_at(position, ReadDirection::After))
                    .transpose()?
                    .or_else(|| footer.next.clone());
                footer.previous = page_start
                    .map(|page_start| continue_at(&page_start, ReadDirection::Before))
                    .transpose()?;
            }
            Ok(footer)
        })
        .map_err(Error::from)?;
    if let Some(first_row) = first_row {
        for item in &mut document.items {
            if let ItemKind::Fragment { position, .. } = &mut item.kind {
                position.selection.row = first_row - position.selection.row;
            }
        }
    }
    scope.check()?;
    Ok(ReadPage {
        format: reading.format,
        snapshot: ReadPageSnapshot::from(&snapshot),
        document,
    })
}

pub fn stream(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    output: &mut dyn Write,
) -> Result<ReadFooter, Error> {
    let resolved = resolve_at(request, operation)?;
    let request = resolved.as_ref();
    validate_at(reader, request)?;
    if !request.delivery.streams() {
        return Err(Reason::StreamNeedsStreamDelivery {}.into());
    }
    let route = match route(reader, request)? {
        Route::Trace => return trace::stream(reader, services, request, operation, output),
        Route::Object(reference) => {
            return object::stream(reader, services, request, operation, reference, output);
        }
        route => route,
    };
    let (reading, _) = reading(request)?;
    prepared(
        reader,
        services,
        operation,
        &reading,
        |compiled, snapshot, header, near| match route {
            Route::Session(scope) => session::stream(
                reader, services, request, operation, scope, snapshot, output,
            ),
            Route::Links(scope) => links::stream(
                reader, services, request, operation, scope, snapshot, output,
            ),
            Route::Trace | Route::Object(_) | Route::Rows => rows_stream(
                reader,
                services,
                request,
                &reading,
                &match compiled.history {
                    Some(history) => rows::Rows::SavedQuery(compiled.selection, history),
                    None => rows::Rows::Selection(compiled.selection),
                },
                snapshot,
                header,
                near,
                None,
                ReadFooter::default(),
                output,
            ),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn rows_stream(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    reading: &Reading,
    selected: &rows::Rows<'_>,
    snapshot: ReadSnapshot,
    header: ReadHeader,
    near: Option<&NearFn<'_>>,
    object_facts: Option<&LayoutFacts>,
    mut footer: ReadFooter,
    output: &mut dyn Write,
) -> Result<ReadFooter, Error> {
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        attachment_call(services, message, block, index, input)
    };
    let rendering = Rendering {
        at: snapshot.ran_at,
        target: &reading.target,
        options: &reading.options,
        near,
        ready_attachment: &ready_attachment,
        page: false,
        object_facts,
        history_seed: selected.history().map(|history| history.seed),
    };
    let scope = reader.scope_token();
    let check = || {
        scope.check()?;
        reader.check()?;
        Ok(())
    };
    let header = ReadHeader {
        snapshot: Some(Box::new(snapshot)),
        options: Some(Box::new(reading.options.clone())),
        ..header
    };
    let mut stream = output::Stream::begin(output, &check, &header)?;
    if let Some(text) = layout_facts::annotation_notice(&reading.options) {
        stream.record(&ReadRecord::Presentation {
            parts: vec![ReadPart::Item(ReadItem {
                source: Source::record(&reading.target),
                kind: ItemKind::Text { text },
            })],
        })?;
    }
    let mut blobs = if request.delivery == ReadDelivery::Bundle {
        Some(blobs::Blobs::new(reader)?)
    } else {
        None
    };
    rows::visit_rows(
        reader,
        selected,
        &rendering,
        &rows::RowStart {
            row: None,
            place: None,
            direction: ReadDirection::After,
        },
        &mut RowKeys::default(),
        &mut footer,
        &mut |record, _, _| {
            stream.record(record)?;
            if let Some(blobs) = &mut blobs {
                blobs.record(record, &mut stream, &check)?;
            }
            Ok(true)
        },
    )?;
    check()?;
    stream.finish(footer)
}
