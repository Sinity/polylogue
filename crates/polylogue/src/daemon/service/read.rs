//! Paged reads. A session, a message or a message range is read over the
//! session item index (`session`); documents, events, authored records
//! and query rows are read row by row, resumed by a row's order values.
//! Session and query pages queue their first-page run without waiting for a write.
//! Exact request and raw run reads retain the execution before exposing its reference.
use super::{Daemon, Failure, error};
use command_surface::Answer;
use polylogue_model::{bundle::ReadSnapshot, read::*};
use polylogue_render::pager::{PageBuilder, PageError, Push};
use polylogue_store::{Reader, selection::PreparedSelection};
use query_language::cursor::OrderPosition;
use serde_json::Value;
use std::{collections::BTreeMap, io::Write, num::NonZeroU64, ops::ControlFlow, sync::Arc};

mod attachment_evidence;
mod authored;
mod backward;
mod before;
mod blobs;
mod documents;
mod evidence;
mod live;
mod object;
mod output;
mod records;
mod session;

/// One read as the daemon resolves it from the read operation's arguments.
#[derive(Debug, Clone)]
pub(crate) struct ReadRequest {
    pub target: String,
    pub options: ReadOptions,
    pub delivery: ReadDelivery,
    pub format: String,
    pub budget: Option<usize>,
    pub with_blobs: bool,
    /// A page cursor issued by an earlier page of this same reading.
    pub cursor: Option<String>,
}

impl ReadRequest {
    /// The read the public arguments name: the one parser of read arguments.
    pub(crate) fn from_arguments(
        arguments: crate::protocol::read::Arguments,
        cursor: Option<String>,
    ) -> Result<Self, String> {
        let crate::protocol::read::Arguments {
            target,
            view,
            format,
            delivery,
            page_tokens,
            show,
            hide,
            output_lines,
            for_others,
            variant,
            with_blobs,
            at,
            diff,
            around,
            latest,
            physical,
        } = arguments;
        let mut options = ReadOptions {
            view,
            output_lines,
            for_others,
            variants: variant,
            at: at
                .map(|at| polylogue_model::MessageId::parse(&at))
                .transpose()
                .map_err(|error| error.to_string())?,
            diff,
            around,
            latest,
            physical,
            ..ReadOptions::default()
        };
        options
            .apply_fold(&show, &hide)
            .map_err(|error| error.to_string())?;
        options.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            target,
            options,
            delivery,
            format: format.unwrap_or(Format::Json).name().into(),
            budget: page_tokens
                .map(|tokens| usize::try_from(tokens.get()))
                .transpose()
                .map_err(|_| "page_tokens exceeds address space")?,
            with_blobs,
            cursor,
        })
    }
}

/// Reads one reference or query as a page, or as a stream of records.
pub(crate) struct Read;
impl crate::daemon::handler::Handler for Read {
    type Arguments = crate::protocol::read::Arguments;
    type Answer = crate::protocol::read::Answer;
    const PAGES: bool = true;
    fn ranking_text(arguments: &Self::Arguments) -> Option<&str> {
        Some(&arguments.target)
    }
    fn default_deadline(arguments: &Self::Arguments) -> bool {
        arguments.delivery != ReadDelivery::Stream
    }
    const PRINT: fn(&Value, bool) -> String = crate::cli::print::read;
    fn run(
        arguments: Self::Arguments,
        call: &crate::daemon::handler::Call<'_>,
    ) -> Result<Answer<Value>, Failure> {
        let request = match ReadRequest::from_arguments(arguments, call.cursor.map(str::to_owned)) {
            Ok(request) => request,
            Err(message) => return Ok(error("bad_request", message)),
        };
        let reader = call.reader()?;
        let context = call.context()?;
        let request = with_variants(&reader, call.daemon, request)?;
        Ok(page(&reader, call.daemon, &request, &context))
    }
}

impl crate::daemon::handler::Streams for Read {
    fn streaming(arguments: &Self::Arguments) -> bool {
        arguments.delivery == ReadDelivery::Stream
    }
    fn produce(
        arguments: Self::Arguments,
        call: &crate::daemon::handler::Call<'_>,
        context: &Arc<crate::operation::Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure> {
        let request = ReadRequest::from_arguments(arguments, None)
            .map_err(|message| Failure::new("bad_request", message))?;
        let request = with_variants(&call.reader()?, call.daemon, request)?;
        produce(&request, call.daemon, context, output).unwrap_or_else(|| {
            Err(Failure::new(
                "bad_request",
                "streaming reads require stream delivery",
            ))
        })
    }
}

/// The read with the variants it renders resolved once: the request's own
/// (`none` for the originals), else those its cursor's first page rendered,
/// else the `reading.variants` setting.
pub(crate) fn with_variants(
    reader: &Reader<'_>,
    d: &Daemon,
    mut request: ReadRequest,
) -> Result<ReadRequest, Failure> {
    if request.options.variants == ["none"] {
        request.options.variants.clear();
        return Ok(request);
    }
    if !request.options.variants.is_empty() {
        return Ok(request);
    }
    request.options.variants = match &request.cursor {
        Some(text) => {
            PageCursor::open(text, &cursor_seal(reader)?)
                .map_err(cursor_failure)?
                .variants
        }
        None => reader
            .settings_snapshot(d.config.requests.machine())?
            .reading_variants()
            .map_err(|error| Failure::new("conflict", error))?
            .to_vec(),
    };
    Ok(request)
}

/// What read cursors are sealed with: this index generation's key and the
/// generation.
/// Whether the index has caught up, read from the snapshot of `reader`
/// through the counts the writer keeps, without walking the chains.
fn caught_up(reader: &Reader<'_>) -> polylogue_store::Result<bool> {
    Ok(reader
        .completeness(&polylogue_engine::reading_contracts())?
        .caught_up())
}

fn cursor_seal(reader: &polylogue_store::Reader<'_>) -> Result<CursorSeal, Failure> {
    Ok(CursorSeal {
        key: reader.cursor_key()?,
        generation: reader.change_position()?.generation,
    })
}

/// A cursor of an earlier generation is `stale_cursor`; a broken one, or
/// one of another reading, is `invalid_query` (docs/query.md, Paging).
fn cursor_failure(error: CursorError) -> Failure {
    match error {
        CursorError::Stale => Failure::new("stale_cursor", error),
        CursorError::Invalid | CursorError::OtherReading(_) => Failure::new("invalid_query", error),
    }
}

fn page_error(error: PageError) -> Failure {
    let kind = if matches!(error, PageError::ItemChanged | PageError::InvalidPosition) {
        "stale_cursor"
    } else {
        "bad_request"
    };
    Failure::new(kind, error)
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
    format: Format,
    budget: Option<PageBudget>,
    target: String,
    digest: String,
}

fn reading(request: &ReadRequest) -> Result<(Reading, Option<&str>), Failure> {
    let ReadRequest {
        target,
        options,
        delivery,
        format,
        budget,
        cursor,
        ..
    } = request;
    options
        .validate()
        .map_err(|e| Failure::new("bad_request", e))?;
    let format =
        Format::parse(format).ok_or_else(|| Failure::new("bad_request", "unknown read format"))?;
    let budget = if *delivery == ReadDelivery::Page {
        let mut resolved = PageBudget::default();
        if let Some(tokens) = budget {
            resolved.tokens = NonZeroU64::new(*tokens as u64)
                .ok_or_else(|| Failure::new("bad_request", "read budget must be positive"))?;
        }
        Some(resolved)
    } else {
        None
    };
    if cursor.is_some() && budget.is_none() {
        return Err(Failure::new(
            "bad_request",
            "streaming reads do not have paged continuations",
        ));
    }
    Ok((
        Reading {
            digest: options_digest(&(options, format)),
            options: options.clone(),
            format,
            budget,
            target: target.clone(),
        },
        cursor.as_deref(),
    ))
}

/// Prepare a row reading: compile the target against this reader and run
/// `run` over the selection. The seed is the target's, so a sampled or
/// shuffled reading keeps its order across pages.
fn prepared<T>(
    reader: &Reader<'_>,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    reading: &Reading,
    run: impl FnOnce(
        &PreparedSelection,
        ReadSnapshot,
        ReadHeader,
        Option<&polylogue_store::selection::NearFn<'_>>,
    ) -> Result<T, Failure>,
) -> Result<T, Failure> {
    crate::daemon::query::with_ranking(
        reader,
        &d.warm,
        &d.config,
        |embeddings, ranking_identity, near| {
            polylogue_store::read::limit::ReadLimit::set_stage("read preparation");
            let ask = crate::daemon::query::Ask {
                text: &reading.target,
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                context: Arc::clone(operation),
                embeddings,
                ranking_identity,
                near: Some(near),
                fields: &[],
            };
            let compile = crate::daemon::query::options(reader, &ask, seed(&reading.target))?;
            let selection = reader.prepare_selection(&reading.target, &compile)?;
            let positions = reader.change_position()?;
            let caught_up = caught_up(reader)?;
            let snapshot = ReadSnapshot {
                query: format!("query:{}", selection.question_id()),
                query_text: reading.target.clone(),
                ran_at: operation.ran_at,
                caught_up,
                read_at: positions,
                currency: d.currency(reader)?,
                enabled_variants: reading.options.variants.clone(),
                author: operation.authorship.author.clone(),
                notes: selection.program().notes.clone(),
                incomplete: selection.program().incomplete.clone(),
            };
            let (title, kind) = match selection.direct_target() {
                Some(ReadTarget::Session(id)) => {
                    let session = reader
                        .session(id)?
                        .ok_or_else(|| Failure::new("not_found", format!("no session {id}")))?;
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
            // A direct read names what it reads; a query read names its question.
            let header = ReadHeader {
                target: selection
                    .direct_target()
                    .map_or_else(|| snapshot.query.clone(), ToString::to_string),
                view: reading.options.view,
                title,
                kind,
                figures: Vec::new(),
                zone: Some(operation.clock.day.zone_name().to_owned()),
            };
            polylogue_store::read::limit::ReadLimit::set_stage("read records");
            run(&selection, snapshot, header, Some(near))
        },
    )
}

type Emit<'a> = dyn FnMut(&ReadRecord, SelectionLocator, Option<u64>) -> Result<bool, Failure> + 'a;

fn selected_source(
    reader: &Reader<'_>,
    selection: &PreparedSelection<'_>,
    target: Option<&ReadTarget>,
    options: &ReadOptions,
    near: Option<&polylogue_store::selection::NearFn<'_>>,
    source: &Source,
) -> Result<bool, Failure> {
    if let (Some(ReadTarget::Session(session)), Some(message)) = (target, &source.message) {
        return Ok(reader
            .select_messages(session, options.inherited)?
            .locate(Some(message))?
            .is_some());
    }
    let reference = polylogue_model::Reference::parse(&source.reference)
        .map_err(|e| Failure::new("not_yet_available", e))?;
    reader
        .selection_contains(selection, &reference, near)
        .map_err(Failure::from)
}

struct Rendering<'a> {
    options: &'a ReadOptions,
    near: Option<&'a polylogue_store::selection::NearFn<'a>>,
    ready_attachment: &'a attachment_evidence::ReadyAttachment<'a>,
    page: bool,
}

fn visit(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    rendering: &Rendering<'_>,
    from: Option<(&OrderPosition, ItemPlace)>,
    rows: &mut RowKeys,
    footer: &mut ReadFooter,
    mut emit: impl FnMut(&ReadRecord, SelectionLocator, Option<u64>) -> Result<bool, Failure>,
) -> Result<(), Failure> {
    let start = from.map(|(_, place)| place);
    let options = rendering.options;
    let near = rendering.near;
    let mixed = selection.direct_target().is_none();
    let mut pending = start.map(|p| p.selection);
    let mut row_number = 0u64;
    reader.visit_selection_from_row::<Failure>(
        selection,
        from.map(|(row, _)| row),
        near,
        |row| {
            let row_number = {
                let at = row_number;
                row_number += 1;
                rows.keys.insert(at, row.position.clone());
                at
            };
            let record_start = pending.filter(|p| p.row == row_number).map(|p| p.record);
            let keep = if let Some(target) = row.target {
                if rendering.page
                    && let ReadTarget::Document(id) = &target
                {
                    let selected = |_: &Source| Ok(false);
                    let evidence = ProjectionEvidence::default();
                    let layout = evidence::Layout {
                        options,
                        evidence: &evidence,
                        selected: &selected,
                        ready_attachment: rendering.ready_attachment,
                        backwards: false,
                        anchor: None,
                    };
                    let position = start.filter(|start| start.selection.row == row_number);
                    let keep = if let Some(record) =
                        output::document_omission(&target.to_string(), options, mixed, footer)?
                    {
                        let at = SelectionLocator {
                            row: row_number,
                            record: ReadRecordLocator {
                                kind: ReadRecordKind::Presentation,
                                ordinal: 0,
                            },
                        };
                        if position
                            .is_some_and(|position| position.selection != at || position.item != 0)
                        {
                            return Err(Failure::new(
                                "stale_cursor",
                                "the continued document omission changed",
                            ));
                        }
                        emit(&record, at, Some(0))?
                    } else {
                        documents::visit(reader, id, row_number, position, &layout, &mut emit)?
                    };
                    pending = None;
                    return Ok(if keep {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(())
                    });
                }
                records::reference(
                    reader,
                    &target,
                    options.inherited,
                    record_start.as_ref(),
                    &mut |record, mut evidence, record_position| {
                        if options.view == View::Live
                            && let records::SelectedRecord::Canonical(canonical) = &record
                            && let polylogue_model::bundle::Record::Session(row) =
                                canonical.as_ref()
                        {
                            let now = selection.cursor_context().clock.now.as_millisecond();
                            evidence.cards.push(live::card(
                                reader,
                                &row.session.id.to_string(),
                                now,
                            )?);
                        }
                        polylogue_store::read::limit::ReadLimit::set_stage("read layout");
                        let at = SelectionLocator {
                            row: row_number,
                            record: record_position,
                        };
                        if let Some(expected) = pending {
                            if expected != at {
                                return Err(Failure::new(
                                    "stale_cursor",
                                    "the continued record coordinate changed",
                                ));
                            }
                            pending = None;
                        }
                        let Some(record) =
                            output::project(record, options, &evidence, mixed, footer)?
                        else {
                            return Ok(true);
                        };
                        let selected = |source: &Source| {
                            selected_source(reader, selection, Some(&target), options, near, source)
                        };
                        let layout = evidence::Layout {
                            options,
                            evidence: &evidence,
                            selected: &selected,
                            ready_attachment: rendering.ready_attachment,
                            backwards: false,
                            anchor: start
                                .filter(|start| start.selection == at)
                                .map(|start| start.item),
                        };
                        output::emit_record(reader, &record, at, &layout, &mut emit)
                    },
                )?
            } else {
                let at = SelectionLocator {
                    row: row_number,
                    record: ReadRecordLocator {
                        kind: ReadRecordKind::Presentation,
                        ordinal: 0,
                    },
                };
                if pending.is_some_and(|expected| expected != at) {
                    return Err(Failure::new(
                        "stale_cursor",
                        "the continued result row changed",
                    ));
                }
                pending = None;
                emit(&presentation(selection, &row.values), at, Some(0))?
            };
            Ok(if keep {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            })
        },
    )?;
    if pending.is_some() {
        return Err(Failure::new(
            "stale_cursor",
            "the continued record is no longer selected",
        ));
    }
    Ok(())
}
fn presentation(selection: &PreparedSelection, values: &serde_json::Value) -> ReadRecord {
    ReadRecord::Presentation {
        parts: vec![ReadPart::Item(ReadItem {
            source: Source::record(format!("query:{}", selection.question_id())),
            kind: ItemKind::Card {
                card: Box::new(Card::Fields {
                    title: "Query result".into(),
                    figures: values
                        .as_object()
                        .into_iter()
                        .flat_map(|object| object.iter())
                        .map(|(name, value)| Figure {
                            name: name.clone(),
                            value: ValueState::known(value.clone()),
                            unit: None,
                        })
                        .collect(),
                }),
            },
        })],
    }
}

fn attachment_call(
    d: &Daemon,
    message: &polylogue_model::MessageId,
    block: u64,
    attachment: u64,
    input: Option<&str>,
) -> Result<ReadyCall, Failure> {
    let arguments = serde_json::to_value(crate::protocol::attachment::Arguments {
        message: message.to_string(),
        block,
        attachment,
        input: input.map(str::to_owned),
        length: crate::daemon::attachment::PAGE_BYTES as u64,
    })
    .map_err(|e| Failure::new("internal", e))?;
    let identity = canonical_json::canonical_hash("polylogue-attachment-call", &arguments)
        .map_err(|e| Failure::new("internal", e))?;
    d.ready_call(&command_surface::RunnableRequest {
        operation: "attachment".into(),
        arguments,
        request_id: format!("read-attachment-{}", identity.to_hex()),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    })
    .map_err(Failure::from)
}

/// Record the reading's question and run after the answer, as the
/// operations log does for every query: the records are handed to the
/// writer without waiting, so nothing on the answer path writes. Only the
/// first page of a reading is a run; continuations are pages of it.
fn record_run(
    reader: &Reader<'_>,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    request: &ReadRequest,
    reading: &Reading,
) -> Result<(), Failure> {
    use polylogue_model::query_objects::{
        ExecutionPresentation, QueryRun, QueryRunOutcome, RunBinding, RunOutcome,
    };
    let ReadRequest { delivery, .. } = request;
    let Some(writer) = d.writes.as_ref() else {
        return Ok(());
    };
    crate::daemon::query::with_ranking(
        reader,
        &d.warm,
        &d.config,
        |embeddings, ranking_identity, near| {
            let ask = crate::daemon::query::Ask {
                text: &reading.target,
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                context: Arc::clone(operation),
                embeddings,
                ranking_identity: ranking_identity.clone(),
                near: Some(near),
                fields: &[],
            };
            let compile = crate::daemon::query::options(reader, &ask, seed(&reading.target))?;
            let target = reading
                .target
                .strip_prefix("root:")
                .unwrap_or(&reading.target);
            let selection = reader.prepare_selection(target, &compile)?;
            let object = selection.query_object()?;
            let query_options = polylogue_store::query::execute::options_snapshot(
                &compile,
                false,
                ranking_identity,
            )?;
            let accepted = operation
                .accepted_request()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| Failure::new("internal", e))?;
            let mut binding = selection.execution_binding(
                query_options,
                ExecutionPresentation::Read {
                    options: reading.options.clone(),
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
                question: Some(selection.question_id()),
                at: operation.ran_at,
                read_at: reader.change_position()?,
                machine: operation.machine.clone(),
                authorship: operation.authorship.clone(),
                binding: RunBinding::Original {
                    execution: Box::new(binding),
                },
            };
            let outcome = QueryRunOutcome {
                run: run.id.clone(),
                binding_hash: run
                    .binding_hash()
                    .map_err(|e| Failure::new("internal", e))?,
                run_hash: run.record_hash().map_err(|e| Failure::new("internal", e))?,
                at: d.config.requests.now(),
                outcome: RunOutcome::Ok,
                count: None,
                exact: false,
                next: None,
            };
            writer
                .record_after_answer(object, run, outcome)
                .map_err(Failure::from)
        },
    )
}

/// A reading's seed is its target's, so a sampled or shuffled reading
/// keeps its order across pages.
fn seed(target: &str) -> u64 {
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

/// The ready call that continues this reading at `cursor`.
fn next_call(
    d: &Daemon,
    request: &ReadRequest,
    cursor: String,
    deadline: command_surface::Deadline,
) -> Result<ReadyCall, Failure> {
    let ReadRequest {
        target,
        options,
        delivery,
        format,
        budget,
        with_blobs,
        ..
    } = request;
    let continued = ReadRequest {
        target: target.clone(),
        options: options.clone(),
        delivery: *delivery,
        format: format.clone(),
        budget: *budget,
        with_blobs: *with_blobs,
        cursor: Some(cursor),
    };
    let arguments = crate::protocol::read::Arguments::from_request(&continued)
        .map_err(|e| Failure::new("internal", e))?;
    let arguments = serde_json::to_value(arguments).map_err(|e| Failure::new("internal", e))?;
    let cursor = continued.cursor.clone();
    let identity = canonical_json::canonical_hash("polylogue-read-call", &(&arguments, &cursor))
        .map_err(|e| Failure::new("internal", e))?;
    d.ready_call(&command_surface::RunnableRequest {
        operation: "read".into(),
        arguments,
        request_id: format!("read-page-{}", identity.to_hex()),
        deadline,
        budget_tokens: Default::default(),
        cursor,
        fresh: false,
        format: Default::default(),
    })
    .map_err(Failure::from)
}

pub(crate) fn page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
) -> Answer<Value> {
    if matches!(
        request,
        ReadRequest {
            with_blobs: true,
            ..
        }
    ) {
        return error("bad_request", "with_blobs requires streaming delivery");
    }
    if let Some(response) = self::object::page(reader, d, request, operation) {
        return response;
    }
    let result = (|| {
        let (reading, cursor) = reading(request)?;
        if reading.budget.is_none() {
            return Err(Failure::new(
                "bad_request",
                "streaming reads require stream delivery",
            ));
        }
        let page = match session::scope(reader, &reading.target)? {
            Some(scope)
                if matches!(
                    reading.options.view,
                    View::Transcript | View::Dialogue | View::Context | View::Raw | View::Changes
                ) =>
            {
                session::page(reader, d, request, operation, scope)?
            }
            Some(scope) if reading.options.view == View::Links => {
                session::links_page(reader, d, request, operation, scope)?
            }
            _ => rows_page(reader, d, request, operation, &reading, cursor)?,
        };
        if cursor.is_none() {
            // The answer stands whatever happens to its run record;
            // building it needs this reader, storing it happens after.
            let _ = record_run(reader, d, operation, request, &reading);
        }
        Ok(page)
    })();
    match result {
        Ok(page) => match serde_json::to_value(&page) {
            Ok(value) => crate::protocol::read(value, &page.snapshot.currency),
            Err(problem) => error("internal", problem),
        },
        Err(failure) => failure.answer(),
    }
}

/// The base row number of a backward page's anchor row: rows before it count
/// down from here, so every row of the page has its own number.
const BACKWARD_ANCHOR_ROW: u64 = 1 << 32;

fn rows_page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    reading: &Reading,
    cursor: Option<&str>,
) -> Result<ReadPage, Failure> {
    let key = cursor_seal(reader)?;
    let opened = cursor
        .map(|text| {
            let opened = PageCursor::open(text, &key).map_err(cursor_failure)?;
            opened
                .check(&reading.target, &reading.digest)
                .map_err(cursor_failure)?;
            if opened.direction == ReadDirection::Past {
                return Err(Failure::new(
                    "invalid_query",
                    "a row reading continues forward or backward only",
                ));
            }
            match opened.position {
                PagePosition::Record { anchor_row, place } => {
                    Ok::<_, Failure>((anchor_row, place, opened.direction))
                }
                PagePosition::Entry { .. } | PagePosition::Node { .. } => Err(Failure::new(
                    "invalid_query",
                    "the cursor names a session place, not a row position",
                )),
            }
        })
        .transpose()?;
    prepared(
        reader,
        d,
        operation,
        reading,
        |selection, snapshot, header, near| {
            let ready_attachment =
                |message: &polylogue_model::MessageId, block, index, input: Option<&str>| {
                    attachment_call(d, message, block, index, input)
                };
            let rendering = Rendering {
                options: &reading.options,
                near,
                ready_attachment: &ready_attachment,
                page: true,
            };
            let scope = selection.scope_token();
            let mut budget = reading.budget.expect("a paged reading has a budget");
            let snapshot_bytes = serde_json::to_vec(&ReadPageSnapshot::from(&snapshot))
                .map_err(|e| Failure::new("internal", e))?
                .len();
            budget.bytes =
                std::num::NonZeroUsize::new(budget.bytes.get().saturating_sub(snapshot_bytes))
                    .ok_or_else(|| {
                        Failure::new("bad_request", "reading metadata exceeds page framing")
                    })?;
            let framing = ReadFraming::Query {
                question: selection.question_id(),
            };
            let backwards = opened
                .as_ref()
                .is_some_and(|(_, _, direction)| *direction == ReadDirection::Before);
            // The anchor place, numbered as this page numbers its rows.
            let anchor = opened.as_ref().map(|(_, place, _)| {
                let mut place = *place;
                place.selection.row = if backwards { BACKWARD_ANCHOR_ROW } else { 0 };
                place
            });
            let mut pager = if backwards {
                PageBuilder::new_before(
                    header,
                    budget,
                    anchor.expect("Before has an anchor"),
                    framing,
                )
            } else {
                PageBuilder::new(header, budget, anchor, framing)
            }
            .map_err(page_error)?;
            let mut footer = ReadFooter::default();
            let mut rows = RowKeys::default();
            let mut emit = |record: &ReadRecord, at: SelectionLocator, ordinal: Option<u64>| {
                let Some(ordinal) = ordinal else {
                    return Ok(true);
                };
                let Some(ReadPart::Item(item)) = record.parts().first() else {
                    return Err(Failure::new(
                        "stale_cursor",
                        "continued layout item is unavailable",
                    ));
                };
                Ok(pager
                    .push_at(at, ordinal, item.clone())
                    .map_err(page_error)?
                    != Push::Full)
            };
            let anchor_row = opened.as_ref().and_then(|(row, _, _)| row.clone());
            if backwards {
                before::visit(
                    reader,
                    selection,
                    &rendering,
                    anchor.expect("Before has an anchor"),
                    anchor_row.as_ref(),
                    &mut rows,
                    &mut footer,
                    &mut emit,
                )?;
            } else {
                visit(
                    reader,
                    selection,
                    &rendering,
                    anchor_row.as_ref().zip(anchor),
                    &mut rows,
                    &mut footer,
                    &mut emit,
                )?;
            }
            scope.check()?;
            polylogue_store::read::limit::ReadLimit::set_stage("read page footer");
            let upper = pager.upper_position().copied();
            // A backward page numbers its rows down from the anchor; its
            // fragments are renumbered from the page's first row, as a forward
            // page numbers them, so the same row gives the same positions.
            let first_row = backwards
                .then(|| pager.first_position().map(|place| place.selection.row))
                .flatten();
            let seal = |place: &ItemPlace, direction| -> Result<ReadyCall, PageError> {
                let row = rows.keys.get(&place.selection.row).cloned();
                let mut place = *place;
                place.selection.row = 0;
                let cursor = PageCursor {
                    target: reading.target.clone(),
                    options: reading.digest.clone(),
                    variants: reading.options.variants.clone(),
                    versions: PageCursor::version(&[]),
                    position: PagePosition::Record {
                        anchor_row: row,
                        place,
                    },
                    direction,
                };
                next_call(d, request, cursor.seal(&key), deadline(operation))
                    .map_err(|e| PageError::Serialization(format!("{e:?}")))
            };
            let document = pager
                .finish_with(|position| {
                    let mut footer = footer.clone();
                    if let Some(upper) = upper {
                        footer.next = Some(seal(&upper, ReadDirection::After)?);
                        footer.previous = position
                            .map(|position| seal(position, ReadDirection::Before))
                            .transpose()?;
                    } else {
                        footer.next = position
                            .map(|position| seal(position, ReadDirection::After))
                            .transpose()?;
                        footer.previous = anchor
                            .map(|anchor| seal(&anchor, ReadDirection::Before))
                            .transpose()?;
                    }
                    Ok(footer)
                })
                .map_err(page_error)?;
            let mut document = document;
            if let Some(first_row) = first_row {
                for item in &mut document.items {
                    if let ItemKind::Fragment { position, .. } = &mut item.kind {
                        position.selection.row -= first_row;
                    }
                }
            }
            scope.check()?;
            Ok(ReadPage {
                format: reading.format,
                snapshot: ReadPageSnapshot::from(&snapshot),
                records: vec![],
                document,
            })
        },
    )
}

pub(crate) fn produce(
    request: &ReadRequest,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    output: &mut dyn Write,
) -> Option<Result<(), Failure>> {
    if let Some(result) = self::object::produce(request, d, operation, output) {
        return Some(result);
    }
    let ReadRequest {
        delivery: ReadDelivery::Stream,
        with_blobs,
        ..
    } = request
    else {
        return None;
    };
    Some(super::accepting(d).and_then(|()| {
        let reader = d.reads.reader()?;
        let (reading, _) = reading(request)?;
        prepared(
            &reader,
            d,
            operation,
            &reading,
            |selection, snapshot, header, near| {
                let ready_attachment =
                    |message: &polylogue_model::MessageId, block, index, input: Option<&str>| {
                        attachment_call(d, message, block, index, input)
                    };
                let rendering = Rendering {
                    options: &reading.options,
                    near,
                    ready_attachment: &ready_attachment,
                    page: false,
                };
                let scope = selection.scope_token();
                let check = || {
                    scope.check()?;
                    if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
                        limit.check()?;
                    }
                    Ok(())
                };
                let begin = ReadBegin {
                    format: reading.format,
                    snapshot,
                    header,
                    options: reading.options.clone(),
                };
                let mut stream = output::Stream::begin(output, &check, &begin)?;
                let mut blobs = if *with_blobs {
                    Some(blobs::Blobs::new(&reader)?)
                } else {
                    None
                };
                let mut footer = ReadFooter::default();
                let mut rows = RowKeys::default();
                visit(
                    &reader,
                    selection,
                    &rendering,
                    None,
                    &mut rows,
                    &mut footer,
                    |record, _, _| {
                        stream.record(record)?;
                        if let Some(blobs) = &mut blobs {
                            blobs.record(record, &mut stream, &check)?;
                        }
                        Ok(true)
                    },
                )?;
                check()?;
                stream.finish(footer)?;
                Ok(())
            },
        )
    }))
}
