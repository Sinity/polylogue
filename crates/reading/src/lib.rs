//! Reading stored sessions and records as pages and streams.
mod error;
pub mod execution;
pub mod operation;
use command_surface::Deadline;
pub use error::Error;
use operation::Context;
use polylogue_model::bundle::ReadSnapshot;
use polylogue_model::{MessageId, freshness::ReadFreshness, read::*};
use polylogue_render::pager::{PageBuilder, PageError, Push};
use polylogue_store::{Reader, WriterSender, selection::NearFn};
use polylogue_store::{selection::PreparedSelection, writer::HistorySlot};
use query_language::cursor::OrderPosition;
use std::{collections::BTreeMap, io::Write, num::NonZeroU64, ops::ControlFlow, sync::Arc};

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

#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub target: String,
    pub references: Option<Vec<polylogue_model::Reference>>,
    pub options: ReadOptions,
    pub delivery: ReadDelivery,
    pub format: String,
    pub budget: Option<usize>,
    pub with_blobs: bool,
    pub cursor: Option<String>,
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
        seed: u64,
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

/// The read with the variants it renders resolved once: the request's own
/// (`none` for the originals), else those its cursor's first page rendered,
/// else the `reading.variants` setting.
pub fn with_variants(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    mut request: ReadRequest,
) -> Result<ReadRequest, Error> {
    if request.options.view == View::Trace {
        if !request.options.variants.is_empty() && request.options.variants != ["none"] {
            return Err(Error::bad_request(
                "trace shows original source evidence; alternates are unsupported",
            ));
        }
        request.options.variants.clear();
        return Ok(request);
    }
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
            .settings_snapshot(d.requests().machine())?
            .reading_variants()
            .map_err(Error::conflict)?
            .to_vec(),
    };
    Ok(request)
}

fn cursor_seal(reader: &polylogue_store::Reader<'_>) -> Result<CursorSeal, Error> {
    Ok(CursorSeal {
        key: reader.cursor_key()?,
        generation: reader.change_position()?.generation,
    })
}

/// A cursor of an earlier generation is `stale_cursor`; a broken one, or
/// one of another reading, is `invalid_query` (docs/query.md, Paging).
fn cursor_failure(error: CursorError) -> Error {
    match error {
        CursorError::Stale => {
            Error::stale_cursor("the index was rebuilt since this page was issued")
        }
        CursorError::Invalid | CursorError::OtherReading(_) => {
            Error::InvalidCursor(error.to_string())
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
    options.validate().map_err(Error::bad_request)?;
    let format = Format::parse(format).ok_or_else(|| Error::bad_request("unknown read format"))?;
    let budget = if *delivery == ReadDelivery::Page {
        let mut resolved = PageBudget::default();
        if let Some(tokens) = budget {
            resolved.tokens = NonZeroU64::new(*tokens as u64)
                .ok_or_else(|| Error::bad_request("read budget must be positive"))?;
        }
        Some(resolved)
    } else {
        None
    };
    if cursor.is_some() && budget.is_none() {
        return Err(Error::bad_request(
            "streaming reads do not have paged continuations",
        ));
    }
    Ok((
        Reading {
            digest: options_digest(&(options, format, &request.references)),
            references: request.references.clone(),
            options: options.clone(),
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
}

fn prepared<T>(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    operation: &Arc<Context>,
    reading: &Reading,
    run: impl FnOnce(&Compiled<'_>, ReadSnapshot, ReadHeader, Option<&NearFn<'_>>) -> Result<T, Error>,
) -> Result<T, Error> {
    d.compiled(
        reader,
        operation,
        &reading.target,
        seed(&reading.target),
        |compile| {
            polylogue_store::read::limit::ReadLimit::set_stage("read preparation");
            let selection = match &reading.references {
                Some(references) => {
                    reader.prepare_fixed_selection(&reading.target, references, &compile.options)?
                }
                None => reader.prepare_selection(&reading.target, &compile.options)?,
            };
            let snapshot = ReadSnapshot {
                query: format!("query:{}", selection.question_id()),
                query_text: reading.target.clone(),
                ran_at: operation.ran_at,
                read_at: reader.change_position()?,
                freshness: d.freshness(reader)?,
                enabled_variants: reading.options.variants.clone(),
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
            if let Some(saved) = saved_query::header(reader, &reading.target, &selection)? {
                kind = HeaderKind::SavedQuery(Box::new(saved));
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
            };
            polylogue_store::read::limit::ReadLimit::set_stage("read records");
            run(
                &Compiled {
                    selection: &selection,
                    options: &compile.options,
                    ranking_identity: compile.ranking_identity,
                },
                snapshot,
                header,
                Some(compile.near),
            )
        },
    )
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
            .select_messages(session, options.inherited)?
            .locate(Some(message))?
            .is_some());
    }
    let reference = polylogue_model::Reference::parse_stored(&source.reference)
        .map_err(Error::not_yet_available)?;
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
}

fn attachment_call(
    d: &impl ReadServices,
    message: &MessageId,
    block: u64,
    attachment: u64,
    input: Option<&str>,
) -> Result<ReadyCall, Error> {
    d.attachment_call(&AttachmentPlace {
        message: message.clone(),
        block,
        attachment,
        input: input.map(str::to_owned),
    })
}

/// First pages queue their run after reading; an answer stands without its run record.
fn record_run(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    operation: &Arc<Context>,
    request: &ReadRequest,
    reading: &Reading,
    history: Option<HistorySlot>,
) -> Result<(), Error> {
    if history.is_none() {
        return Ok(());
    }
    d.compiled(
        reader,
        operation,
        &reading.target,
        seed(&reading.target),
        |compile| {
            let target = reading
                .target
                .strip_prefix("root:")
                .unwrap_or(&reading.target);
            let selection = reader.prepare_selection(target, &compile.options)?;
            record_compiled(
                reader,
                d,
                operation,
                request,
                reading,
                &Compiled {
                    selection: &selection,
                    options: &compile.options,
                    ranking_identity: compile.ranking_identity,
                },
                history,
            )
        },
    )
}

/// Record a reading's run from its compiled target, as [`record_run`] says.
fn record_compiled(
    reader: &Reader<'_>,
    d: &impl ReadServices,
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
        binding_hash: run.binding_hash().map_err(Error::internal)?,
        run_hash: run.record_hash().map_err(Error::internal)?,
        at: d.requests().now(),
        outcome: RunOutcome::Ok,
        count: None,
        exact: false,
        next: None,
    };
    history.record(object, run, outcome).map_err(Error::from)
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

fn next_call(
    d: &impl ReadServices,
    request: &ReadRequest,
    cursor: String,
    deadline: Deadline,
) -> Result<ReadyCall, Error> {
    d.continuation(request, Some(cursor), deadline)
}

pub fn is_object_read(request: &ReadRequest) -> bool {
    object::target(request).is_some()
}

pub fn page(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    history: Option<HistorySlot>,
) -> Result<ReadPage, Error> {
    if request.with_blobs {
        return Err(Error::bad_request("with_blobs requires streaming delivery"));
    }
    if request.options.view == View::Trace {
        return trace::page(reader, d, request, operation, history);
    }
    if let Some(response) = object::page(reader, d, request, operation) {
        return response;
    }
    let (reading, cursor) = reading(request)?;
    if reading.budget.is_none() {
        return Err(Error::bad_request(
            "streaming reads require stream delivery",
        ));
    }
    let session_page = match session::scope(reader, &reading.target)? {
        Some(scope)
            if matches!(
                reading.options.view,
                View::Transcript | View::Dialogue | View::Context | View::Raw | View::Changes
            ) =>
        {
            session::page(reader, d, request, operation, scope)?
        }
        Some(scope) if reading.options.view == View::Links => {
            links::page(reader, d, request, operation, scope)?
        }
        _ => {
            return prepared(
                reader,
                d,
                operation,
                &reading,
                |compiled, snapshot, header, near| {
                    let page = rows_page(
                        reader,
                        d,
                        request,
                        operation,
                        &reading,
                        &rows::Rows::Selection(compiled.selection),
                        snapshot,
                        header,
                        ReadFraming::Query {
                            question: compiled.selection.question_id(),
                        },
                        PageCursor::version(&[]),
                        near,
                        None,
                        ReadFooter::default(),
                    )?;
                    if cursor.is_none() {
                        let _ = record_compiled(
                            reader, d, operation, request, &reading, compiled, history,
                        );
                    }
                    Ok(page)
                },
            );
        }
    };
    if cursor.is_none() {
        let _ = record_run(reader, d, operation, request, &reading, history);
    }
    Ok(session_page)
}

/// Rows before a backward page's start count down from here.
const BACKWARD_START_ROW: u64 = 1 << 32;

#[allow(clippy::too_many_arguments)]
fn rows_page(
    reader: &Reader<'_>,
    d: &impl ReadServices,
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
            if matches!(selected, rows::Rows::Object(..)) && opened.rewritten(&versions) {
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
                PagePosition::Record { start_row, place } => {
                    Ok((start_row, place, opened.direction))
                }
                _ => Err(Error::InvalidCursor(
                    "the cursor belongs to a session reading".into(),
                )),
            }
        })
        .transpose()?;
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        attachment_call(d, message, block, index, input)
    };
    let rendering = Rendering {
        at: operation.ran_at,
        target: &reading.target,
        options: &reading.options,
        near,
        ready_attachment: &ready_attachment,
        page: true,
        object_facts,
    };
    let scope = reader.scope_token();
    let mut budget = reading.budget.expect("a paged reading has a budget");
    let snapshot_bytes = serde_json::to_vec(&ReadPageSnapshot::from(&snapshot))
        .map_err(Error::internal)?
        .len();
    budget.bytes = std::num::NonZeroUsize::new(budget.bytes.get().saturating_sub(snapshot_bytes))
        .ok_or_else(|| Error::bad_request("reading metadata exceeds page framing"))?;
    let backwards = opened
        .as_ref()
        .is_some_and(|(_, _, direction)| *direction == ReadDirection::Before);
    let page_start = opened.as_ref().map(|(_, place, _)| {
        let mut place = *place;
        place.selection.row = if backwards { BACKWARD_START_ROW } else { 0 };
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
        row: opened.as_ref().and_then(|(row, _, _)| row.clone()),
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
    let first_row = backwards
        .then(|| pager.first_position().map(|place| place.selection.row))
        .flatten();
    let continue_at = |place: &ItemPlace, direction| -> Result<ReadyCall, PageError> {
        let row = keys.keys.get(&place.selection.row).cloned();
        let mut place = *place;
        place.selection.row = 0;
        let cursor = PageCursor {
            target: reading.target.clone(),
            options: reading.digest.clone(),
            variants: reading.options.variants.clone(),
            versions,
            position: PagePosition::Record {
                start_row: row,
                place,
            },
            direction,
        };
        next_call(d, request, cursor.seal(&key), deadline(operation))
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
}

pub fn stream(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    output: &mut dyn Write,
) -> Result<ReadComplete, Error> {
    if request.delivery != ReadDelivery::Stream {
        return Err(Error::bad_request(
            "streaming reads require stream delivery",
        ));
    }
    if request.options.view == View::Trace {
        return trace::stream(reader, d, request, operation, output);
    }
    if let Some(result) = object::stream(reader, d, request, operation, output) {
        return result;
    }
    let (reading, _) = reading(request)?;
    prepared(
        reader,
        d,
        operation,
        &reading,
        |compiled, snapshot, header, near| {
            rows_stream(
                reader,
                d,
                request,
                &reading,
                &rows::Rows::Selection(compiled.selection),
                snapshot,
                header,
                near,
                None,
                ReadFooter::default(),
                output,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn rows_stream(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    request: &ReadRequest,
    reading: &Reading,
    selected: &rows::Rows<'_>,
    snapshot: ReadSnapshot,
    header: ReadHeader,
    near: Option<&NearFn<'_>>,
    object_facts: Option<&LayoutFacts>,
    mut footer: ReadFooter,
    output: &mut dyn Write,
) -> Result<ReadComplete, Error> {
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        attachment_call(d, message, block, index, input)
    };
    let rendering = Rendering {
        at: snapshot.ran_at,
        target: &reading.target,
        options: &reading.options,
        near,
        ready_attachment: &ready_attachment,
        page: false,
        object_facts,
    };
    let scope = reader.scope_token();
    let check = || {
        scope.check()?;
        reader.check()?;
        Ok(())
    };
    let begin = ReadBegin {
        format: reading.format,
        snapshot,
        header,
        options: reading.options.clone(),
    };
    let mut stream = output::Stream::begin(output, &check, &begin)?;
    let mut blobs = if request.with_blobs {
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
