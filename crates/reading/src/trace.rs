//! A trace is one presentation of recorded source evidence, through normal read paging.
use super::*;
use polylogue_model::Reason;
use polylogue_model::{BlobHash, Reference};

fn not_built(target: &str) -> Error {
    let mut failure = command_surface::Failure::new(
        command_surface::FailureKind::NotYetAvailable,
        format!(
            "trace of {target} is not built yet; record-to-bytes tracing supports messages, blocks, tool calls and text spans"
        ),
    );
    failure.code = Some(polylogue_model::Code::NotBuilt.as_str().into());
    Error::Operation(failure)
}

fn prepare(
    reader: &Reader<'_>,
    request: &ReadRequest,
) -> Result<(Reading, ReadRecord, BlobHash), Error> {
    if request.target.starts_with("file:") {
        return Err(not_built("file-line lineage"));
    }
    let item = Reference::parse(&request.target).map_err(|error| Reason::BadReference {
        text: request.target.clone(),
        problem: error.to_string(),
    })?;
    let supported = match &item {
        Reference::Message(_) | Reference::Block { .. } | Reference::ToolCall(..) => true,
        Reference::Span { container, .. } => matches!(
            container.as_ref(),
            Reference::Message(_) | Reference::Block { .. }
        ),
        _ => false,
    };
    if !supported {
        return Err(not_built(&item.to_string()));
    }
    let card = reader
        .record_chain(&item)?
        .ok_or_else(|| Error::not_found(&item))?;
    let digest = BlobHash::of(&serde_json::to_vec(&card).map_err(Error::internal)?);
    let message = card
        .chain
        .iter()
        .find_map(|step| match step {
            ChainStep::Record {
                reference: Reference::Message(message),
            } => Some(message.clone()),
            _ => None,
        })
        .ok_or_else(|| Error::internal("a record trace has no holding message"))?;
    let (mut reading, _) = reading(request)?;
    reading.target = Reference::Message(message).to_string();
    reading.references = None;
    let record = ReadRecord::Presentation {
        parts: vec![ReadPart::Item(ReadItem {
            source: Source::record(item.to_string()),
            kind: ItemKind::Card {
                card: Box::new(Card::Trace(Box::new(card))),
            },
        })],
    };
    Ok((reading, record, digest))
}

/// The reading's header renamed for the traced record, once the message
/// holding that record is confirmed to be in the caller's selection.
fn trace_header(
    reader: &Reader<'_>,
    request: &ReadRequest,
    compiled: &Compiled<'_>,
    mut header: ReadHeader,
    near: Option<&NearFn<'_>>,
) -> Result<ReadHeader, Error> {
    let target = compiled
        .selection
        .direct_target()
        .ok_or_else(|| Error::internal("trace has no concrete message selection"))?;
    if !reader.selection_contains(compiled.selection, &target.reference(), near)? {
        return Err(Error::not_found(&request.target));
    }
    header.target = request.target.clone();
    header.kind = HeaderKind::Record {
        record_type: "trace".into(),
    };
    Ok(header)
}

pub(super) fn page(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    history: Option<HistorySlot>,
) -> Result<ReadPage, Error> {
    let (reading, record, digest) = prepare(reader, request)?;
    prepared(
        reader,
        services,
        operation,
        &reading,
        |compiled, snapshot, header, near| {
            let header = trace_header(reader, request, compiled, header, near)?;
            let page = rows_page(
                reader,
                services,
                request,
                operation,
                &reading,
                &rows::Rows::Object(&record, ReadRecordKind::Presentation),
                snapshot,
                header,
                ReadFraming::Query {
                    question: compiled.selection.question_id(),
                },
                digest,
                None,
                None,
                ReadFooter::default(),
            )?;
            if request.cursor.is_none() {
                record_compiled(
                    reader, services, operation, request, &reading, compiled, history,
                )?;
            }
            Ok(page)
        },
    )
}

pub(super) fn stream(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<Context>,
    output: &mut dyn Write,
) -> Result<ReadFooter, Error> {
    let (reading, record, _) = prepare(reader, request)?;
    prepared(
        reader,
        services,
        operation,
        &reading,
        |compiled, snapshot, header, near| {
            let header = trace_header(reader, request, compiled, header, near)?;
            rows_stream(
                reader,
                services,
                request,
                &reading,
                &rows::Rows::Object(&record, ReadRecordKind::Presentation),
                snapshot,
                header,
                None,
                None,
                ReadFooter::default(),
                output,
            )
        },
    )
}
