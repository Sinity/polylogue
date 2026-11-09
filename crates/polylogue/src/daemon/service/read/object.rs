//! Exact accepted object reads share durable Begin/Outcome and output codecs.
use super::*;
use polylogue_model::{Reference, RunId, query_objects::*};
use std::num::NonZeroU64;

fn accepted(operation: &crate::operation::Context) -> Result<Option<serde_json::Value>, Failure> {
    operation
        .accepted_request()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| Failure::new("internal", e))
}

fn target(request: &Request) -> Option<Result<Reference, Failure>> {
    let Request::Read {
        target, options, ..
    } = request
    else {
        return None;
    };
    (target.starts_with("request:") || options.view == View::Raw && target.starts_with("run:"))
        .then(|| Reference::parse(target).map_err(|error| Failure::new("bad_request", error)))
}

fn execute<T>(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &Request,
    operation: &Arc<crate::operation::Context>,
    reference: Reference,
    run: impl FnOnce(
        ReadRecord,
        ReadSnapshot,
        ReadHeader,
        Format,
        Option<PageBudget>,
        RunId,
        ReadFooter,
        &ProjectionEvidence,
    ) -> Result<T, Failure>,
) -> Result<T, Failure> {
    let Request::Read {
        options,
        format,
        delivery,
        budget,
        ..
    } = request
    else {
        unreachable!()
    };
    options
        .validate()
        .map_err(|error| Failure::new("bad_request", error))?;
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
    let prepared = reader.prepare_object_read(&reference)?;
    let content_hash = prepared.content_hash();
    let record = prepared.into_record()?;
    let scope = reader.scope_token();
    let positions = reader.change_position()?;
    let binding = ObjectExecutionBinding {
        reference: reference.clone(),
        content_hash,
        scope_fingerprint: reader.query_scope_fingerprint()?,
        versions: operation.cursor_versions(),
        presentation: ExecutionPresentation::Read {
            options: options.clone(),
            delivery: *delivery,
            format,
            page_budget: budget,
            direct_source: None,
        },
        accepted_request: accepted(operation)?,
    };
    binding
        .validate()
        .map_err(|error| Failure::new("bad_request", error))?;
    let id = crate::operation::new_run_id();
    let hash = binding
        .hash()
        .map_err(|error| Failure::new("internal", error))?;
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
    let writer = d.writes.as_ref().ok_or_else(|| {
        Failure::new("not_yet_available", "read execution custody is unavailable")
    })?;
    reader.check()?;
    scope.check()?;
    super::super::execution::record(
        writer,
        operation.parent_lifetime(),
        begin,
        hash,
        || d.config.requests.now(),
        || {
            reader.check()?;
            scope.check()?;
            let caught_up = d.warm.caught_up(reader, &positions.to_string())?;
            let run_reference = format!("run:{id}");
            let snapshot = ReadSnapshot {
                query: run_reference.clone(),
                query_text: reference.to_string(),
                ran_at: operation.ran_at,
                read_at: positions,
                caught_up,
                currency: d.currency(reader)?,
                enabled_variants: vec![],
                author: operation.authorship.author.clone(),
                notes: vec![],
                incomplete: vec![],
            };
            let header = ReadHeader {
                target: run_reference,
                view: options.view,
                title: ValueState::NotApplicable,
                kind: HeaderKind::Record {
                    record_type: "accepted_request".into(),
                },
                figures: vec![],
            };
            let evidence = ProjectionEvidence {
                authored: if matches!(&reference, Reference::Request(_)) {
                    Some(AuthoredSelection {
                        selected: reference,
                        coverage: AuthoredCoverage::WholeRequest,
                    })
                } else {
                    None
                },
                ..Default::default()
            };
            let mut footer = ReadFooter::default();
            let projected = output::project(
                records::SelectedRecord::from(record),
                options,
                &evidence,
                false,
                &mut footer,
            )?;
            let projected = projected.unwrap_or(ReadRecord::Presentation { parts: vec![] });
            let value = run(
                projected, snapshot, header, format, budget, id, footer, &evidence,
            )?;
            reader.check()?;
            scope.check()?;
            Ok(super::super::execution::Completed {
                value,
                count: Some(1),
                exact: true,
                next: None,
            })
        },
    )
}

pub(super) fn page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &Request,
    operation: &Arc<crate::operation::Context>,
) -> Option<Answer<Value>> {
    let reference = match target(request)? {
        Ok(reference) => reference,
        Err(failure) => return Some(failure.answer()),
    };
    let Request::Read { options, .. } = request else {
        return None;
    };
    let result = execute(
        reader,
        d,
        request,
        operation,
        reference,
        |record, snapshot, header, format, budget, id, footer, evidence| {
            let mut budget = budget.ok_or_else(|| {
                Failure::new("bad_request", "streaming reads require stream delivery")
            })?;
            let framing_size = serde_json::to_vec(&ReadPageSnapshot::from(&snapshot))
                .map_err(|error| Failure::new("internal", error))?
                .len();
            budget.bytes =
                std::num::NonZeroUsize::new(budget.bytes.get().saturating_sub(framing_size))
                    .ok_or_else(|| {
                        Failure::new(
                            "bad_request",
                            "reading metadata exceeds page framing; use streaming delivery",
                        )
                    })?;
            let mut pager = PageBuilder::new(header, budget, None, ReadFraming::Object { run: id })
                .map_err(page_error)?;
            let at = SelectionLocator {
                row: 0,
                record: ReadRecordLocator {
                    kind: ReadRecordKind::Authored,
                    ordinal: 0,
                },
            };
            // The record is laid out exactly as a selected record is; an
            // object reading has no message attachments to open.
            let selected = |_: &Source| Ok(false);
            let ready_attachment = |_: &polylogue_model::MessageId, _, _, _: Option<&str>| {
                Err(Failure::new(
                    "not_yet_available",
                    "an object reading has no message attachments",
                ))
            };
            let layout = evidence::Layout {
                options,
                evidence,
                selected: &selected,
                ready_attachment: &ready_attachment,
                backwards: false,
                anchor: None,
            };
            evidence::visit_parts(reader, &record, &layout, |item, ordinal| {
                Ok(pager.push_at(at, ordinal, item).map_err(page_error)? != Push::Full)
            })?;
            let more = pager.position().is_some();
            let document = pager.finish(footer).map_err(page_error)?;
            if more {
                return Err(Failure::new(
                    "bad_request",
                    "this object exceeds the page budget; use streaming delivery for the complete retained record",
                ));
            }
            reader.check()?;
            Ok(ReadPage {
                format,
                snapshot: ReadPageSnapshot::from(&snapshot),
                records: vec![],
                document,
            })
        },
    );
    Some(match result {
        Ok(page) => match serde_json::to_value(&page) {
            Ok(value) => crate::protocol::read(value, &page.snapshot.currency),
            Err(problem) => error("internal", problem),
        },
        Err(failure) => failure.answer(),
    })
}

pub(super) fn produce(
    request: &Request,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    output: &mut dyn Write,
) -> Option<Result<(), Failure>> {
    let Request::Read {
        options,
        with_blobs,
        delivery: ReadDelivery::Stream,
        ..
    } = request
    else {
        return None;
    };
    let reference = match target(request)? {
        Ok(reference) => reference,
        Err(error) => return Some(Err(error)),
    };
    Some(super::super::accepting(d).and_then(|()| {
        let reader = d.reads.reader()?;
        execute(
            &reader,
            d,
            request,
            operation,
            reference,
            |record, snapshot, header, format, _, _, footer, _| {
                let check = || reader.check().map_err(Failure::from);
                let begin = ReadBegin {
                    format,
                    snapshot,
                    header,
                    options: options.clone(),
                };
                let mut stream = output::Stream::begin(output, &check, &begin)?;
                stream.record(&record)?;
                if *with_blobs {
                    blobs::Blobs::new(&reader)?.record(&record, &mut stream, &check)?;
                }
                check()?;
                stream.finish(footer)
            },
        )
    }))
}

#[cfg(test)]
mod tests;
