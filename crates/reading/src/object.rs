//! Accepted object reads share the stored begin and outcome records and the output codecs.
use super::ReadRequest;
use super::*;
use polylogue_model::{Reference, RunId, query_objects::*};

fn accepted(operation: &crate::operation::Context) -> Result<Option<serde_json::Value>, Error> {
    operation
        .accepted_request()
        .map(serde_json::to_value)
        .transpose()
        .map_err(Error::internal)
}

pub(super) fn target(request: &ReadRequest) -> Option<Result<Reference, Error>> {
    let ReadRequest {
        target, options, ..
    } = request;
    (target.starts_with("request:") || options.view == View::Raw && target.starts_with("run:"))
        .then(|| Reference::parse(target).map_err(Error::bad_request))
}

fn execute<T>(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    request: &ReadRequest,
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
        &LayoutFacts,
        polylogue_model::BlobHash,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    let (reading, _) = reading(request)?;
    let options = &reading.options;
    let format = reading.format;
    let budget = reading.budget;
    let delivery = request.delivery;
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
            delivery,
            format,
            page_budget: budget,
            direct_source: None,
        },
        accepted_request: accepted(operation)?,
    };
    binding.validate().map_err(Error::bad_request)?;
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
    let writer = d
        .writer()
        .ok_or_else(|| Error::not_yet_available("the read's run record is unavailable"))?;
    reader.check()?;
    scope.check()?;
    crate::execution::record(
        writer,
        operation.parent_lifetime(),
        begin,
        hash,
        || d.requests().now(),
        || {
            reader.check()?;
            scope.check()?;
            let run_reference = format!("run:{id}");
            let snapshot = ReadSnapshot {
                query: run_reference.clone(),
                query_text: reference.to_string(),
                ran_at: operation.ran_at,
                read_at: positions,
                freshness: d.freshness(reader)?,
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
                zone: Some(operation.clock.day.zone_name().to_owned()),
            };
            let facts = LayoutFacts {
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
                &facts,
                false,
                &mut footer,
            )?;
            let projected = projected.unwrap_or(ReadRecord::Presentation { parts: vec![] });
            let value = run(
                projected,
                snapshot,
                header,
                format,
                budget,
                id,
                footer,
                &facts,
                content_hash,
            )?;
            reader.check()?;
            scope.check()?;
            Ok(crate::execution::Completed {
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
    d: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
) -> Option<Result<ReadPage, Error>> {
    let reference = match target(request)? {
        Ok(reference) => reference,
        Err(error) => return Some(Err(error)),
    };
    Some(execute(
        reader,
        d,
        request,
        operation,
        reference,
        |record, snapshot, header, _, _, id, footer, facts, content_hash| {
            let (reading, _) = reading(request)?;
            if reading.budget.is_none() {
                return Err(Error::bad_request(
                    "streaming reads require stream delivery",
                ));
            }
            rows_page(
                reader,
                d,
                request,
                operation,
                &reading,
                &rows::Rows::Object(&record),
                snapshot,
                header,
                ReadFraming::Object { run: id },
                content_hash,
                None,
                Some(facts),
                footer,
            )
        },
    ))
}

pub(super) fn stream(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    output: &mut dyn Write,
) -> Option<Result<(), Error>> {
    let reference = match target(request)? {
        Ok(reference) => reference,
        Err(error) => return Some(Err(error)),
    };
    Some(execute(
        reader,
        d,
        request,
        operation,
        reference,
        |record, snapshot, header, _, _, _, footer, facts, _| {
            let (reading, _) = reading(request)?;
            rows_stream(
                reader,
                d,
                request,
                &reading,
                &rows::Rows::Object(&record),
                snapshot,
                header,
                None,
                Some(facts),
                footer,
                output,
            )
        },
    ))
}
