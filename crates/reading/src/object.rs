//! Accepted object reads share the stored begin and outcome records and the output codecs.
use super::ReadRequest;
use super::*;
use polylogue_model::Reason;
use polylogue_model::{Reference, RunId, query_objects::*};

fn accepted(operation: &crate::operation::Context) -> Result<Option<serde_json::Value>, Error> {
    operation
        .accepted_request()
        .map(serde_json::to_value)
        .transpose()
        .map_err(Error::internal)
}

pub(super) fn target(request: &ReadRequest) -> Option<Result<Reference, Error>> {
    let target = &request.target;
    (target.starts_with("request:") || target.starts_with("run:")).then(|| {
        Reference::parse(target).map_err(|error| {
            Error::from(Reason::BadReference {
                text: target.clone(),
                problem: error.to_string(),
            })
        })
    })
}

fn execute<T>(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    reference: Reference,
    run: impl FnOnce(
        ReadRecord,
        ReadSnapshot,
        ReadHeader,
        RunId,
        ReadFooter,
        &LayoutFacts,
        polylogue_model::BlobHash,
    ) -> Result<(T, Option<ReadyCall>), Error>,
) -> Result<T, Error> {
    let (reading, _) = reading(request)?;
    let options = &reading.options;
    let format = reading.format;
    let budget = reading.budget;
    let delivery = request.delivery;
    if matches!(&reference, Reference::Run(_))
        && !matches!(options.view, View::Transcript | View::Raw)
    {
        return Err(ReadError::UnsupportedTarget {
            view: options.view,
            target: reference.to_string(),
            accepted_views: vec![View::Transcript, View::Raw],
        }
        .into());
    }
    let prepared = reader.prepare_object_read(&reference)?;
    let content_hash = prepared.content_hash();
    let record = prepared.into_record()?;
    let is_run = matches!(&record, polylogue_store::read::ObjectRecord::Run(_));
    let scope = reader.scope_token();
    let positions = reader.change_position()?;
    let binding = ObjectExecutionBinding {
        reference: reference.clone(),
        content_hash,
        scope_fingerprint: None,
        versions: operation.cursor_versions(),
        presentation: ExecutionPresentation::Read {
            options: options.clone(),
            delivery,
            format,
            page_budget: budget,
            direct_source: None,
        },
        accepted_request: accepted(operation)?,
    };
    binding.validate().map_err(Error::internal)?;
    let id = crate::operation::new_run_id();
    let hash = binding.hash().map_err(Error::internal)?;
    let begin = QueryRun {
        id: id.clone(),
        question: None,
        at: operation.ran_at,
        read_at: positions.clone(),
        machine: operation.machine.clone(),
        authorship: operation.authorship.clone(),
        binding: RunBinding::OriginalObject {
            execution: Box::new(binding),
        },
    };
    let writer = services
        .writer()
        .ok_or_else(|| Error::not_yet_available("the read's run record is unavailable"))?;
    reader.check()?;
    scope.check()?;
    crate::execution::record(
        writer,
        operation.parent_lifetime(),
        begin,
        hash,
        || services.requests().now(),
        || {
            reader.check()?;
            scope.check()?;
            let run_reference = format!("run:{id}");
            let snapshot = ReadSnapshot {
                query: run_reference.clone(),
                query_text: reference.to_string(),
                ran_at: operation.ran_at,
                read_at: positions,
                freshness: services.freshness(reader)?,
                enabled_alternates: vec![],
                author: operation.authorship.author.clone(),
                notes: vec![],
                incomplete: vec![],
            };
            let header = ReadHeader {
                target: if is_run {
                    reference.to_string()
                } else {
                    run_reference
                },
                view: options.view,
                title: ValueState::NotApplicable,
                kind: HeaderKind::Record {
                    record_type: if is_run { "run" } else { "accepted_request" }.into(),
                },
                figures: vec![],
                zone: Some(operation.clock.day.zone_name().to_owned()),
            };
            let facts = LayoutFacts {
                authored: if matches!(&reference, Reference::Request(_)) {
                    Some(AuthoredSelection {
                        selected: reference.clone(),
                        coverage: AuthoredCoverage::WholeRequest,
                    })
                } else {
                    None
                },
                ..Default::default()
            };
            let mut footer = ReadFooter::default();
            // A request the view hides (annotations off) counts as no record.
            let mut hidden = false;
            let projected = match record {
                polylogue_store::read::ObjectRecord::Request(record) => {
                    let shown = output::project(
                        records::SelectedRecord::from(*record),
                        options,
                        &facts,
                        false,
                        &mut footer,
                    )?;
                    hidden = shown.is_none();
                    shown.unwrap_or(ReadRecord::Presentation { parts: vec![] })
                }
                polylogue_store::read::ObjectRecord::Run(record) => {
                    let kind = if options.view == View::Raw {
                        ItemKind::Other {
                            type_name: "run".into(),
                            data: serde_json::to_value(&record).map_err(Error::internal)?,
                        }
                    } else {
                        if let Some(question) = record.run.question
                            && reader.query_object(&question.to_string())?.is_some()
                        {
                            let query = Reference::Query {
                                key: question.to_string(),
                                parameters: vec![],
                            };
                            footer.next = Some(ReadyCall {
                                request: command_surface::RunnableRequest::new(
                                    "rows",
                                    serde_json::json!({"selection":query.to_string()}),
                                    format!("{id}-question"),
                                ),
                            });
                        }
                        ItemKind::Card {
                            card: Box::new(Card::Run {
                                question: record.run.question.map(|question| Reference::Query {
                                    key: question.to_string(),
                                    parameters: vec![],
                                }),
                                read_at: record.run.read_at.clone(),
                                author: record.run.authorship.author.clone(),
                                count: record.outcome.as_ref().and_then(|outcome| outcome.count),
                                exact: record.outcome.as_ref().map(|outcome| outcome.exact),
                                outcome: record
                                    .outcome
                                    .as_ref()
                                    .map(|outcome| outcome.outcome.clone()),
                            }),
                        }
                    };
                    ReadRecord::Presentation {
                        parts: vec![ReadPart::Item(ReadItem {
                            source: Source::record(reference.to_string()),
                            kind,
                        })],
                    }
                }
            };
            let (value, next) = run(
                projected,
                snapshot,
                header,
                id,
                footer,
                &facts,
                content_hash,
            )?;
            reader.check()?;
            scope.check()?;
            Ok(crate::execution::Completed {
                value,
                count: Some(if hidden { 0 } else { 1 }),
                exact: true,
                next: if is_run { next } else { None },
            })
        },
    )
}

fn record_kind(reference: &Reference) -> ReadRecordKind {
    if matches!(reference, Reference::Run(_)) {
        ReadRecordKind::Run
    } else {
        ReadRecordKind::Authored
    }
}

pub(super) fn page(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
) -> Option<Result<ReadPage, Error>> {
    let reference = match target(request)? {
        Ok(reference) => reference,
        Err(error) => return Some(Err(error)),
    };
    let kind = record_kind(&reference);
    Some(execute(
        reader,
        services,
        request,
        operation,
        reference,
        |record, snapshot, header, id, footer, facts, content_hash| {
            let (reading, _) = reading(request)?;
            crate::require_page_delivery(&reading)?;
            let page = rows_page(
                reader,
                services,
                request,
                operation,
                &reading,
                &rows::Rows::Object(&record, kind),
                snapshot,
                header,
                ReadFraming::Object { run: id },
                content_hash,
                None,
                Some(facts),
                footer,
            )?;
            let next = page.document.footer.next.clone();
            Ok((page, next))
        },
    ))
}

pub(super) fn stream(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    output: &mut dyn Write,
) -> Option<Result<ReadComplete, Error>> {
    let reference = match target(request)? {
        Ok(reference) => reference,
        Err(error) => return Some(Err(error)),
    };
    let kind = record_kind(&reference);
    Some(execute(
        reader,
        services,
        request,
        operation,
        reference,
        |record, snapshot, header, _, footer, facts, _| {
            let (reading, _) = reading(request)?;
            let complete = rows_stream(
                reader,
                services,
                request,
                &reading,
                &rows::Rows::Object(&record, kind),
                snapshot,
                header,
                None,
                Some(facts),
                footer,
                output,
            )?;
            let next = complete.footer.next.clone();
            Ok((complete, next))
        },
    ))
}
