//! One pinned selection, projected and laid out identically for every delivery.
use super::{Daemon, Response, error, ok, store_error};
use crate::daemon::Request;
use polylogue_model::{BlobHash, RunId, bundle::*, query_objects::*, read::*};
use polylogue_render::pager::{PageBuilder, PageError, Push};
use polylogue_store::{
    Reader,
    selection::{PreparedSelection, RunReplayContext, SelectionError},
};
use query_language::cursor::{Continuation, CursorContext, CursorSeal};
use std::{io::Write, num::NonZeroU64, ops::ControlFlow, sync::Arc};

mod blobs;
mod outline;
mod output;
mod records;

fn failure(error: SelectionError) -> Response {
    match error {
        SelectionError::Store(error) => store_error(&error),
        error => Response::Error {
            kind: error.kind().into(),
            message: error.to_string(),
            detail: error.detail(),
        },
    }
}
impl From<SelectionError> for Response {
    fn from(error: SelectionError) -> Self {
        failure(error)
    }
}
fn language_error(error: query_language::Error) -> Response {
    Response::Error {
        kind: error.kind.code().into(),
        message: error.to_string(),
        detail: serde_json::to_value(error).ok(),
    }
}
fn page_error(error: PageError) -> Response {
    let kind = if matches!(error, PageError::ItemChanged | PageError::InvalidPosition) {
        "stale_cursor"
    } else {
        "bad_request"
    };
    self::error(kind, error)
}
pub(super) fn outline(
    reader: &Reader<'_>,
    id: &polylogue_model::SessionId,
    parents: bool,
    size: usize,
    cursor: Option<&str>,
) -> Response {
    outline::run(reader, id, parents, size, cursor)
}

struct Execution {
    root: RunId,
    hash: BlobHash,
    run: QueryRun,
    context: CursorContext,
    options: ReadOptions,
    format: Format,
    budget: Option<PageBudget>,
    target: String,
    deadline: command_surface::Deadline,
}
fn pending(d: &Daemon) -> Option<&dyn polylogue_store::primary::Sinex> {
    match &d.mode {
        super::ModeState::Standalone => None,
        #[cfg(feature = "sinex")]
        super::ModeState::Primary { connection } => Some(connection.as_ref()),
        #[cfg(feature = "sinex")]
        super::ModeState::Blocked { .. } => None,
    }
}
fn accepted(operation: &crate::operation::Context) -> Result<Option<serde_json::Value>, Response> {
    operation
        .accepted_request()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| error("internal", e))
}
fn new_run() -> RunId {
    uuid::Uuid::now_v7()
        .to_string()
        .parse()
        .expect("generated UUID is a run identity")
}
fn layout_contract(options: &ReadOptions, format: Format) -> Result<String, Response> {
    // This layout uses only the selected canonical record and explicit options.
    // Prepared external evidence must add its own metadata dependency commitment.
    let bytes = serde_json::to_vec(&("canonical-read-layout-v1", options, format))
        .map_err(|e| error("internal", e))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn prepared<T>(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &Request,
    operation: &Arc<crate::operation::Context>,
    run: impl FnOnce(
        &PreparedSelection,
        &Execution,
        Option<ReadPosition>,
        ReadSnapshot,
        ReadHeader,
        Option<&polylogue_store::selection::NearFn<'_>>,
        &Arc<crate::operation::Context>,
    ) -> Result<(T, ReadFooter), Response>,
) -> Result<T, Response> {
    crate::daemon::query::with_ranking(
        reader,
        &d.warm,
        &d.config,
        |embeddings, ranking_identity, near| {
            let writer = d.writes.as_ref().ok_or_else(|| {
                error("not_yet_available", "read execution custody is unavailable")
            })?;
            let versions = operation.cursor_versions();
            let id = new_run();
            let (selection, binding, root, start, budget_override) = match request {
                Request::Read {
                    target,
                    options,
                    delivery,
                    format,
                    budget,
                    ..
                } => {
                    options.validate().map_err(|e| error("bad_request", e))?;
                    let format = Format::parse(format)
                        .ok_or_else(|| error("bad_request", "unknown read format"))?;
                    let logical_budget = if *delivery == ReadDelivery::Page {
                        let mut resolved = PageBudget::default();
                        if let Some(tokens) = budget {
                            resolved.tokens = NonZeroU64::new(*tokens as u64).ok_or_else(|| {
                                error("bad_request", "read budget must be positive")
                            })?;
                        }
                        Some(resolved)
                    } else {
                        None
                    };
                    let ask = crate::daemon::query::Ask {
                        text: target,
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
                    let compile = crate::daemon::query::options(
                        reader,
                        &ask,
                        uuid::Uuid::now_v7().as_u128() as u64,
                    )
                    .map_err(|e| Response::Error {
                        kind: e.kind,
                        message: e.message,
                        detail: e.detail,
                    })?;
                    let selection = reader.prepare_selection(target, &compile)?;
                    let object = selection.query_object()?;
                    let query_options = polylogue_store::query::execute::options_snapshot(
                        &compile,
                        false,
                        ranking_identity.clone(),
                    )?;
                    let mut binding = selection.execution_binding(
                        query_options,
                        ExecutionPresentation::Read {
                            options: options.clone(),
                            delivery: *delivery,
                            format,
                            page_budget: logical_budget,
                            direct_source: None,
                        },
                        accepted(operation)?,
                    )?;
                    binding.replay.context.versions.extend(versions.clone());
                    let content = reader.selection_content_fingerprint(
                        &selection,
                        options.inherited,
                        Some(near),
                    )?;
                    binding
                        .replay
                        .context
                        .versions
                        .insert("read_content".into(), content);
                    binding
                        .replay
                        .context
                        .versions
                        .insert("read_layout".into(), layout_contract(options, format)?);
                    writer.query_object(object).map_err(|e| store_error(&e))?;
                    (selection, binding, id.clone(), None, None)
                }
                Request::ReadResume {
                    root_run,
                    cursor,
                    budget,
                } => {
                    let (selection, position) = reader.prepare_authenticated_run_selection(
                        cursor,
                        root_run,
                        pending(d),
                        RunReplayContext {
                            current_clock: &operation.clock,
                            current_versions: &versions,
                            ranking_identity: ranking_identity.as_deref(),
                            page_size: None,
                            near: Some(near),
                        },
                    )?;
                    let position = ReadPosition::from_order_position(&position)
                        .map_err(|e| error("stale_cursor", e))?;
                    let (root, binding) = selection.authenticated_execution(reader)?;
                    let root = root.clone();
                    let binding = binding.clone();
                    (selection, binding, root, Some(position), *budget)
                }
                _ => return Err(error("bad_request", "a read request is required")),
            };
            let (options, format, mut budget) = match &binding.presentation {
                ExecutionPresentation::Read {
                    options,
                    format,
                    page_budget,
                    ..
                } => (options.clone(), *format, *page_budget),
                ExecutionPresentation::Query => {
                    return Err(error("bad_request", "the root execution is not a read"));
                }
            };
            if binding.replay.context.versions.get("read_layout")
                != Some(&layout_contract(&options, format)?)
            {
                return Err(error("stale_cursor", "the captured read layout changed"));
            }
            if let Some(tokens) = budget_override {
                budget
                    .as_mut()
                    .ok_or_else(|| {
                        error(
                            "bad_request",
                            "streaming reads do not have paged continuations",
                        )
                    })?
                    .tokens = tokens;
            }
            let evaluation =
                Arc::new(operation.replay_clock(selection.cursor_context().clock.clone())?);
            let target = binding
                .query_options
                .direct_reference
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| binding.replay.original_text.clone());
            let deadline = binding
                .accepted_request
                .as_ref()
                .map(|value| {
                    serde_json::from_value::<command_surface::RunnableRequest>(value.clone())
                })
                .transpose()
                .map_err(|e| store_error(&polylogue_store::StoreError::Invalid(e.to_string())))?
                .map_or(command_surface::Deadline::Default, |request| {
                    request.deadline
                });
            let hash = binding.hash().map_err(|e| error("internal", e))?;
            let mut context = selection.cursor_context().clone();
            context.read_positions = binding.replay.context.read_positions.clone();
            context.versions = binding.replay.context.versions.clone();
            context
                .versions
                .insert("read_root_run".into(), root.to_string());
            context
                .versions
                .insert("read_binding".into(), hash.to_string());
            polylogue_store::query::continuation::compact_resolved_times(&mut context)?;
            let run_binding = if let Some(position) = start {
                RunBinding::Continuation {
                    root: root.clone(),
                    binding_hash: hash,
                    position,
                    accepted_request: accepted(operation)?,
                }
            } else {
                RunBinding::Original {
                    execution: Box::new(binding),
                }
            };
            let positions = reader.change_position().map_err(|e| store_error(&e))?;
            let page_run = QueryRun {
                id,
                question: selection.question_id(),
                at: operation.ran_at,
                read_at: positions.clone(),
                machine: operation.machine.clone(),
                authorship: operation.authorship.clone(),
                binding: run_binding,
            };
            writer
                .begin_query_run(page_run.clone())
                .map_err(|e| store_error(&e))?;
            let execution = Execution {
                root,
                hash,
                run: page_run,
                context,
                options,
                format,
                budget,
                target,
                deadline,
            };
            let result = (|| {
                let completeness = reader
                    .completeness(&polylogue_engine::reading_contracts())
                    .map_err(|e| store_error(&e))?;
                let snapshot = ReadSnapshot {
                    query: format!("query:{}", selection.question_id()),
                    query_text: execution.target.clone(),
                    ran_at: operation.ran_at,
                    caught_up: completeness.caught_up(),
                    read_at: positions,
                    currency: d.currency(reader).map_err(|e| store_error(&e))?,
                    enabled_variants: execution.options.variants.clone(),
                    author: operation.authorship.author.clone(),
                    notes: selection.program().notes.clone(),
                    incomplete: selection.program().incomplete.clone(),
                };
                let header = ReadHeader {
                    target: snapshot.query.clone(),
                    view: execution.options.view,
                    title: ValueState::NotApplicable,
                    kind: HeaderKind::Record {
                        record_type: "query".into(),
                    },
                    figures: Vec::new(),
                };
                run(
                    &selection,
                    &execution,
                    start,
                    snapshot,
                    header,
                    Some(near),
                    &evaluation,
                )
            })();
            let outcome = QueryRunOutcome {
                run: execution.run.id.clone(),
                binding_hash: execution.hash,
                run_hash: execution
                    .run
                    .record_hash()
                    .map_err(|e| error("internal", e))?,
                at: d.config.clock.now().into(),
                outcome: match &result {
                    Ok(_) => RunOutcome::Ok,
                    Err(Response::Error { kind, .. }) => RunOutcome::Error { kind: kind.clone() },
                    Err(_) => RunOutcome::Error {
                        kind: "read_failed".into(),
                    },
                },
                count: None,
                exact: false,
                next: result
                    .as_ref()
                    .ok()
                    .and_then(|(_, footer)| footer.next.clone()),
            };
            let completion = writer
                .finish_query_run(outcome)
                .map_err(|e| store_error(&e));
            match result {
                Ok((value, _)) => {
                    drop(completion?);
                    Ok(value)
                }
                Err(error) => {
                    drop(completion);
                    Err(error)
                }
            }
        },
    )
}

fn visit(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    options: &ReadOptions,
    start: Option<ReadPosition>,
    near: Option<&polylogue_store::selection::NearFn<'_>>,
    mixed: bool,
    footer: &mut ReadFooter,
    mut emit: impl FnMut(ReadRecord, SelectionLocator) -> Result<bool, Response>,
) -> Result<(), Response> {
    let start_row = start.map_or(0, |p| p.selection.row);
    let mut pending = start.map(|p| p.selection);
    reader.visit_selection_from::<Response>(selection, start_row, near, |row_number, row| {
        let record_start = pending.filter(|p| p.row == row_number).map(|p| p.record);
        let keep = if let Some(reference) = row.reference {
            records::reference(
                reader,
                &reference,
                options.inherited,
                record_start.as_ref(),
                &mut |record, evidence, record_position| {
                    let at = SelectionLocator {
                        row: row_number,
                        record: record_position,
                    };
                    if let Some(expected) = pending {
                        if expected != at {
                            return Err(error(
                                "stale_cursor",
                                "the continued record coordinate changed",
                            ));
                        }
                        pending = None;
                    }
                    let Some(record) = output::project(record, options, &evidence, mixed, footer)?
                    else {
                        return Ok(true);
                    };
                    emit(record, at)
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
                return Err(error("stale_cursor", "the continued result row changed"));
            }
            pending = None;
            emit(
                ReadRecord::Presentation {
                    parts: vec![ReadPart::Item(ReadItem {
                        source: Source::record(format!("query:{}", selection.question_id())),
                        kind: ItemKind::Card {
                            card: Box::new(Card::Fields {
                                title: "Query result".into(),
                                figures: row
                                    .values
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
                },
                at,
            )?
        };
        Ok(if keep {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    })?;
    if pending.is_some() {
        return Err(error(
            "stale_cursor",
            "the continued record is no longer selected",
        ));
    }
    Ok(())
}
fn next_call(
    reader: &Reader<'_>,
    d: &Daemon,
    execution: &Execution,
    position: ReadPosition,
) -> Result<ReadyCall, Response> {
    let after = position
        .to_order_position()
        .map_err(|e| error("internal", e))?;
    let cursor = CursorSeal::new(reader.cursor_key().map_err(|e| store_error(&e))?)
        .seal(&Continuation {
            context: execution.context.clone(),
            after,
        })
        .map_err(language_error)?;
    let request = Request::ReadResume {
        root_run: execution.root.clone(),
        cursor,
        budget: execution.budget.map(|b| b.tokens),
    };
    let arguments = crate::protocol::read::Arguments::from_request(&request)
        .map_err(|e| error("internal", e))?;
    let arguments = serde_json::to_value(arguments).map_err(|e| error("internal", e))?;
    let identity = serde_json::to_vec(&(execution.hash, position, execution.budget))
        .map_err(|e| error("internal", e))?;
    let call = command_surface::RunnableRequest {
        operation: "read".into(),
        arguments,
        request_id: format!("read-page-{}", blake3::hash(&identity).to_hex()),
        deadline: execution.deadline.clone(),
    };
    d.ready_call(&call).map_err(|e| Response::Error {
        kind: e.kind.code().into(),
        message: e.message,
        detail: None,
    })
}
pub(super) fn page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &Request,
    operation: &Arc<crate::operation::Context>,
) -> Response {
    if matches!(
        request,
        Request::Read {
            with_blobs: true,
            ..
        }
    ) {
        return error("bad_request", "with_blobs requires streaming delivery");
    }
    let result = prepared(
        reader,
        d,
        request,
        operation,
        |selection, execution, start, snapshot, header, near, _evaluation| {
            let scope = selection.scope_token();
            let mut budget = execution
                .budget
                .ok_or_else(|| error("bad_request", "streaming reads require stream delivery"))?;
            let snapshot_bytes = serde_json::to_vec(&ReadPageSnapshot::from(&snapshot))
                .map_err(|e| error("internal", e))?
                .len();
            budget.bytes =
                std::num::NonZeroUsize::new(budget.bytes.get().saturating_sub(snapshot_bytes))
                    .ok_or_else(|| error("bad_request", "reading metadata exceeds page framing"))?;
            let mut pager = PageBuilder::new(
                header,
                budget,
                start,
                ReadFraming {
                    question: selection.question_id(),
                },
            )
            .map_err(page_error)?;
            let mut footer = ReadFooter::default();
            visit(
                reader,
                selection,
                &execution.options,
                start,
                near,
                selection.direct_reference().is_none(),
                &mut footer,
                |record, at| {
                    for (ordinal, part) in record.parts().iter().enumerate() {
                        if let ReadPart::Item(item) = part {
                            if pager
                                .push_at(at, ordinal as u64, item.clone())
                                .map_err(page_error)?
                                == Push::Full
                            {
                                return Ok(false);
                            }
                        }
                    }
                    Ok(true)
                },
            )?;
            scope.check().map_err(|e| store_error(&e))?;
            let document = pager
                .finish_with(|position| {
                    let mut footer = footer.clone();
                    if let Some(position) = position {
                        footer.next = Some(
                            next_call(reader, d, execution, *position)
                                .map_err(|e| PageError::Serialization(format!("{e:?}")))?,
                        );
                    }
                    Ok(footer)
                })
                .map_err(page_error)?;
            scope.check().map_err(|e| store_error(&e))?;
            let footer = document.footer.clone();
            Ok((
                ReadPage {
                    format: execution.format,
                    snapshot: ReadPageSnapshot::from(&snapshot),
                    records: vec![],
                    document,
                },
                footer,
            ))
        },
    );
    match result {
        Ok(page) => ok(page),
        Err(error) => error,
    }
}
pub(super) fn produce(
    request: &Request,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    output: &mut dyn Write,
) -> Option<Result<(), Response>> {
    let Request::Read {
        delivery: ReadDelivery::Stream,
        with_blobs,
        ..
    } = request
    else {
        return None;
    };
    Some(super::accepting(d).and_then(|()| {
        let reader = d.reads.reader().map_err(|e| store_error(&e))?;
        prepared(
            &reader,
            d,
            request,
            operation,
            |selection, execution, start, snapshot, header, near, _evaluation| {
                let scope = selection.scope_token();
                let check = || {
                    scope.check().map_err(|e| store_error(&e))?;
                    if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
                        limit.check().map_err(|e| store_error(&e))?;
                    }
                    Ok(())
                };
                let begin = ReadBegin {
                    format: execution.format,
                    snapshot,
                    header,
                    options: execution.options.clone(),
                };
                let mut stream = output::Stream::begin(output, &check, &begin)?;
                let mut blobs = if *with_blobs {
                    Some(blobs::Blobs::new(&reader)?)
                } else {
                    None
                };
                let mut footer = ReadFooter::default();
                visit(
                    &reader,
                    selection,
                    &execution.options,
                    start,
                    near,
                    selection.direct_reference().is_none(),
                    &mut footer,
                    |record, _| {
                        stream.record(&record)?;
                        if let Some(blobs) = &mut blobs {
                            blobs.record(&record, &mut stream, &check)?;
                        }
                        Ok(true)
                    },
                )?;
                check()?;
                stream.finish(footer.clone())?;
                Ok(((), footer))
            },
        )
    }))
}
