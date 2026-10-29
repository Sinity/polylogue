//! One pinned selection, projected and laid out identically for every delivery.
use super::{Daemon, Response, error, ok, store_error};
use crate::daemon::Request;
use polylogue_model::{Timestamp, authorship::Authorship, bundle::*, read::*};
use polylogue_render::pager::{Budget, PageBuilder, Push};
use polylogue_store::{
    Reader,
    selection::{PreparedSelection, SelectionError},
};
use query_language::cursor::{
    Continuation, CursorContext, CursorSeal, OrderPosition, SealedCursor,
};
use serde::{Deserialize, Serialize};
use std::{io::Write, num::NonZeroU64, ops::ControlFlow};

mod blobs;
mod outline;
mod output;
mod records;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Resume {
    before: Option<OrderPosition>,
    position: ReadPosition,
}
fn failure(error: SelectionError) -> Response {
    Response::Error {
        kind: error.kind().into(),
        message: error.to_string(),
        detail: error.detail(),
    }
}
fn language_error(error: query_language::Error) -> Response {
    Response::Error {
        kind: error.kind.code().into(),
        message: error.to_string(),
        detail: serde_json::to_value(error).ok(),
    }
}
impl From<SelectionError> for Response {
    fn from(error: SelectionError) -> Self {
        failure(error)
    }
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

fn context(
    selection: &PreparedSelection,
    options: &ReadOptions,
) -> Result<CursorContext, Response> {
    let mut context = selection.cursor_context().clone();
    let bytes = serde_json::to_vec(options).map_err(|e| error("internal", e))?;
    context.versions.insert(
        "read_options".into(),
        blake3::hash(&bytes).to_hex().to_string(),
    );
    Ok(context)
}
fn open(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    options: &ReadOptions,
    token: &str,
) -> Result<(Continuation, Resume), Response> {
    let seal = CursorSeal::new(reader.cursor_key().map_err(|e| store_error(&e))?);
    let current = context(selection, options)?;
    let continuation = seal
        .open(
            &SealedCursor::from_wire(token),
            selection.question_id(),
            |zone| {
                if zone == current.clock.day.zone_name() {
                    Ok(current.clock.day.zone().clone())
                } else {
                    Err(query_language::Error::new(
                        query_language::ErrorKind::StaleCursor,
                        "the reading zone changed",
                        0,
                    ))
                }
            },
            |old| old.read_positions == current.read_positions && old.versions == current.versions,
        )
        .map_err(language_error)?;
    if continuation.after.values.len() != 1 {
        return Err(error("bad_request", "invalid read position"));
    }
    let resume: Resume = serde_json::from_value(continuation.after.values[0].clone())
        .map_err(|e| error("bad_request", e))?;
    if resume.position.reference != continuation.after.row_identity {
        return Err(error("bad_request", "read position identity disagrees"));
    }
    Ok((continuation, resume))
}
fn seal(reader: &Reader<'_>, context: CursorContext, resume: Resume) -> Result<String, Response> {
    let continuation = Continuation {
        context,
        after: OrderPosition {
            row_identity: resume.position.reference.clone(),
            values: vec![serde_json::to_value(resume).map_err(|e| error("internal", e))?],
        },
    };
    CursorSeal::new(reader.cursor_key().map_err(|e| store_error(&e))?)
        .seal(&continuation)
        .map(|cursor| cursor.as_str().to_owned())
        .map_err(language_error)
}

fn prepared<T>(
    reader: &Reader<'_>,
    d: &Daemon,
    target: &str,
    options: &ReadOptions,
    cursor: Option<&str>,
    author: &Authorship,
    run: impl FnOnce(
        &PreparedSelection,
        Option<Resume>,
        ReadSnapshot,
        ReadHeader,
        Option<&polylogue_store::selection::NearFn<'_>>,
    ) -> Result<T, Response>,
) -> Result<T, Response> {
    options.validate().map_err(|e| error("bad_request", e))?;
    let clock = d.query_clock()?;
    let ran_at: Timestamp = clock.now.into();
    crate::daemon::query::with_ranking(reader, &d.warm, |embeddings, near| {
        let ask = crate::daemon::query::Ask {
            text: target,
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            clock: clock.clone(),
            embeddings,
            near: Some(near),
            fields: &[],
        };
        let mut compile =
            crate::daemon::query::options(reader, &ask, uuid::Uuid::now_v7().as_u128() as u64)
                .map_err(|e| Response::Error {
                    kind: e.kind,
                    message: e.message,
                    detail: e.detail,
                })?;
        let mut selection = reader
            .prepare_selection(target, &compile)
            .map_err(failure)?;
        let resume = if let Some(token) = cursor {
            let (continuation, resume) = open(reader, &selection, options, token)?;
            compile.clock = continuation.context.clock;
            compile.random_seed = Some(continuation.context.random_seed);
            selection = reader
                .prepare_selection(target, &compile)
                .map_err(failure)?;
            Some(resume)
        } else {
            None
        };
        let position = reader.change_position().map_err(|e| store_error(&e))?;
        let snapshot = ReadSnapshot {
            query: format!("query:{}", selection.question_id()),
            query_text: target.into(),
            ran_at,
            caught_up: position.index == position.archive,
            read_at: position,
            currency: d.currency(reader).map_err(|e| store_error(&e))?,
            enabled_variants: options.variants.clone(),
            author: author.author.clone(),
            notes: selection.program().notes.clone(),
            incomplete: selection.program().incomplete.clone(),
        };
        let header = ReadHeader {
            target: snapshot.query.clone(),
            view: options.view,
            title: ValueState::NotApplicable,
            kind: HeaderKind::Record {
                record_type: "reading".into(),
            },
            figures: vec![],
        };
        run(&selection, resume, snapshot, header, Some(near))
    })
}

fn visit(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    options: &ReadOptions,
    resume: Option<&Resume>,
    near: Option<&polylogue_store::selection::NearFn<'_>>,
    mixed: bool,
    footer: &mut ReadFooter,
    mut emit: impl FnMut(ReadRecord, Option<OrderPosition>, String) -> Result<bool, Response>,
) -> Result<(), Response> {
    let mut before = resume.and_then(|resume| resume.before.clone());
    let mut pending = resume.map(|resume| resume.position.clone());
    let after = before.clone();
    reader.visit_selection::<Response>(selection, after.as_ref(), near, |row| {
        let previous = before.clone();
        let keep = if let Some(reference) = row.reference {
            let start = pending.clone();
            records::reference(
                reader,
                &reference,
                options.inherited,
                start.as_ref(),
                &mut |record, evidence| {
                    if let Some(position) = &pending {
                        if record.reference() != Some(position.reference.as_str()) {
                            return Ok(true);
                        }
                        pending = None;
                    }
                    let anchor = record.reference().ok_or_else(|| error("internal", "canonical reading has no reference"))?.to_owned();
                    let Some(record) = output::project(record, options, &evidence, mixed, footer)?
                    else {
                        return Ok(true);
                    };
                    emit(record, previous.clone(), anchor)
                },
            )?
        } else {
            // This is a transfer anchor for a prepared result row, independent
            // of the source reference displayed by its presentation items.
            let anchor = format!("row:{}", blake3::hash(row.position.row_identity.as_bytes()).to_hex());
            if let Some(position) = &pending {
                if position.reference != anchor {
                    return Err(error("stale_cursor", "the continued result row changed"));
                }
                pending = None;
            }
            let parts = vec![ReadPart::Item(ReadItem {
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
            })];
            emit(ReadRecord::Presentation { parts }, previous, anchor)?
        };
        if !keep {
            return Ok(ControlFlow::Break(()));
        }
        before = Some(row.position);
        Ok(ControlFlow::Continue(()))
    })?;
    if pending.is_some() {
        return Err(error(
            "stale_cursor",
            "the continued record is no longer selected",
        ));
    }
    Ok(())
}

pub(super) fn page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &Request,
    author: &Authorship,
) -> Response {
    let Request::Read {
        target,
        options,
        budget,
        cursor,
        with_blobs,
        ..
    } = request
    else {
        return error("bad_request", "a read request is required");
    };
    if *with_blobs {
        return error("bad_request", "with_blobs requires streaming delivery");
    }
    let result = prepared(
        reader,
        d,
        target,
        options,
        cursor.as_deref(),
        author,
        |selection, resume, snapshot, header, near| {
            let scope = selection.scope_token();
            let mut limit = Budget::default();
            if let Some(tokens) = budget {
                limit.tokens = NonZeroU64::new(*tokens as u64)
                    .ok_or_else(|| error("bad_request", "read budget must be positive"))?;
            }
            let snapshot_bytes = serde_json::to_vec(&ReadPageSnapshot::from(&snapshot))
                .map_err(|e| error("internal", e))?
                .len();
            limit.bytes =
                std::num::NonZeroUsize::new(limit.bytes.get().saturating_sub(snapshot_bytes))
                    .ok_or_else(|| error("bad_request", "reading metadata exceeds page budget"))?;
            let mut pager = PageBuilder::new(
                header,
                limit,
                resume.as_ref().map(|resume| resume.position.clone()),
            )
            .map_err(|e| error("bad_request", e))?;
            let mut footer = ReadFooter::default();
            let mut anchors = std::collections::BTreeMap::new();
            let visited = visit(
                reader,
                selection,
                options,
                resume.as_ref(),
                near,
                !matches!(
                    polylogue_model::Reference::parse(target),
                    Ok(polylogue_model::Reference::Document(_))
                ),
                &mut footer,
                |record, before, reference| {
                    for (ordinal, part) in record.parts().iter().enumerate() {
                        if let ReadPart::Item(item) = part {
                            anchors.entry(reference.clone()).or_insert_with(|| before.clone());
                            if pager
                                .push_at(&reference, ordinal as u64, item.clone())
                                .map_err(|e| error("bad_request", e))?
                                == Push::Full
                            {
                                return Ok(false);
                            }
                        }
                    }
                    Ok(true)
                },
            );
            scope.check().map_err(|e| store_error(&e))?;
            visited?;
            let document = pager.finish_with(|position| {
                let mut page_footer = footer.clone();
                if let Some(position) = position {
                    let next_call = (|| -> Result<ReadyCall, Response> {
                let token = seal(
                    reader,
                    context(selection, options)?,
                    Resume {
                        before: anchors.get(&position.reference).cloned().ok_or_else(|| error("internal", "reading continuation has no selection position"))?,
                        position: position.clone(),
                    },
                )?;
                let mut next = request.clone();
                if let Request::Read { cursor, .. } = &mut next {
                    *cursor = Some(token);
                }
                let arguments = crate::protocol::read::Arguments::from_request(&next)
                    .map_err(|e| error("internal", e))?;
                let call = command_surface::Request {
                    operation: crate::commands::command("read")
                        .ok_or_else(|| error("internal", "read operation missing"))?
                        .name
                        .into(),
                    arguments: serde_json::to_value(arguments).map_err(|e| error("internal", e))?,
                    request_id: uuid::Uuid::now_v7().to_string(),
                    deadline: command_surface::Deadline::Default,
                };
                d.ready_call(&call).map_err(|e| Response::Error {
                    kind: e.code.clone().unwrap_or_else(|| {
                        serde_json::to_value(e.kind)
                            .expect("closed failure kind serializes")
                            .as_str()
                            .expect("failure kind is a string")
                            .to_owned()
                    }),
                    message: e.message.clone(),
                    detail: serde_json::to_value(e).ok(),
                })

                    })().map_err(|e| polylogue_render::pager::PageError::Serialization(format!("{e:?}")))?;
                    page_footer.next = Some(next_call);
                }
                Ok(page_footer)
            }).map_err(|e| error("bad_request", e))?;
            scope.check().map_err(|e| store_error(&e))?;
            Ok(ReadPage {
                snapshot: ReadPageSnapshot::from(&snapshot),
                records: vec![],
                document,
            })
        },
    );
    match result {
        Ok(page) => ok(page),
        Err(error) => error,
    }
}

pub(crate) fn produce(
    request: &Request,
    d: &Daemon,
    author: &Authorship,
    output: &mut dyn Write,
) -> Option<Result<(), Response>> {
    let Request::Read {
        delivery: ReadDelivery::Stream,
        target,
        options,
        cursor,
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
            target,
            options,
            cursor.as_deref(),
            author,
            |selection, resume, snapshot, header, near| {
                let scope = selection.scope_token();
                let check = || {
                    scope.check().map_err(|e| store_error(&e))?;
                    if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
                        limit.check().map_err(|e| store_error(&e))?;
                    }
                    Ok(())
                };
                let begin = ReadBegin {
                    snapshot,
                    header,
                    options: options.clone(),
                };
                let mut stream = output::Stream::begin(output, &check, &begin)?;
                let mut blobs = if *with_blobs {
                    Some(blobs::Blobs::new(&reader)?)
                } else {
                    None
                };
                let mut footer = ReadFooter::default();
                let visited = visit(
                    &reader,
                    selection,
                    options,
                    resume.as_ref(),
                    near,
                    !matches!(
                        polylogue_model::Reference::parse(target),
                        Ok(polylogue_model::Reference::Document(_))
                    ),
                    &mut footer,
                    |record, _, _| {
                        stream.record(&record)?;
                        if let Some(blobs) = &mut blobs {
                            blobs.record(&record, &mut stream, &check)?;
                        }
                        Ok(true)
                    },
                );
                check()?;
                visited?;
                stream.finish(footer)
            },
        )
    }))
}
