//! A whole export, stored, then split into records and ordered before its items are written.
use super::*;
use polylogue_model::batch::{ExportCursor, ExportDelivery, ExportDeliverySource};
use polylogue_model::{BlobHash, Output, RecordCounts};
use polylogue_sources::chain_names::{Formed, NoEarlierRows, settle};
use polylogue_sources::export::{self, ExportParser, ExportState};
use polylogue_store::{Batch, ParsePart, export_items::ExportItems};

struct Counted<F, R> {
    open: F,
    reader: std::io::BufReader<R>,
    count: u64,
    position: u64,
}
impl<F: FnMut() -> R, R: std::io::Read> Counted<F, R> {
    fn new(mut open: F) -> Self {
        Self {
            reader: std::io::BufReader::new(open()),
            open,
            count: 0,
            position: 0,
        }
    }
}
impl<F, R: std::io::Read> std::io::Read for Counted<F, R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(out)?;
        self.count += n as u64;
        self.position += n as u64;
        Ok(n)
    }
}
impl<F, R: std::io::Read> std::io::BufRead for Counted<F, R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        std::io::BufRead::fill_buf(&mut self.reader)
    }
    fn consume(&mut self, n: usize) {
        std::io::BufRead::consume(&mut self.reader, n);
        self.count += n as u64;
        self.position += n as u64;
    }
}
impl<F: FnMut() -> R, R: std::io::Read> export::Input for Counted<F, R> {
    fn rewind(&mut self) -> std::io::Result<()> {
        self.reader = std::io::BufReader::new((self.open)());
        self.position = 0;
        Ok(())
    }
}

struct ItemWriter<'a>(&'a mut ExportItems);
impl export::ItemSink for ItemWriter<'_> {
    fn push(
        &mut self,
        range: std::ops::Range<u64>,
        key: Vec<u8>,
        source_key: Option<&str>,
        raw: &[u8],
    ) -> Result<(), String> {
        self.0
            .push(range, key, source_key, raw)
            .map_err(|e| e.to_string())
    }
    fn begin(&mut self) -> Result<(), String> {
        self.0.begin_scan().map_err(|e| e.to_string())
    }
    fn commit(&mut self) -> Result<(), String> {
        self.0.commit_scan().map_err(|e| e.to_string())
    }
    fn rollback(&mut self) -> Result<(), String> {
        self.0.rollback_scan().map_err(|e| e.to_string())
    }
}

/// The scratch store behind an export's cross-item state. A failure here is
/// the import's own I/O, kept apart so it stops the export instead of being
/// mistaken for one unreadable item.
/// The parser's view of the export's scratch items. Its trait carries
/// errors as text, so the first failed read is kept typed beside it: a
/// failed read of the scratch file is never the export's content.
struct StateLookup<'a> {
    items: &'a mut ExportItems,
    failure: Option<polylogue_store::StoreError>,
}
impl StateLookup<'_> {
    fn note<T>(&mut self, result: polylogue_store::Result<T>) -> Result<T, String> {
        result.map_err(|error| {
            let text = error.to_string();
            self.failure.get_or_insert(error);
            text
        })
    }
}
impl ExportState for StateLookup<'_> {
    fn claim_native_id(
        &mut self,
        native_id: &str,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::SessionId>, String> {
        let result = self.items.claim_native_id(native_id, session);
        self.note(result)
    }
    fn last_message(
        &mut self,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::MessageId>, String> {
        let result = self.items.last_message(session);
        self.note(result)
    }
    fn set_last_message(
        &mut self,
        session: &polylogue_model::SessionId,
        message: &polylogue_model::MessageId,
    ) -> Result<(), String> {
        let result = self.items.set_last_message(session, message);
        self.note(result)
    }
}

impl Engine {
    pub(crate) fn execute_export(
        &mut self,
        plan: &mut Plan,
        parser: &mut dyn ExportParser,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if matches!(plan.source, pool::Bytes::File) {
            self.store_plan(plan, pass)?;
        }
        let mut pending = ExportBatch::default();
        // Items parsed after the last sent batch are not committed when the
        // parse fails; the next parse replays the committed prefix of
        // deliveries and continues after it.
        let read = parse_export(plan, parser, &self.blobs.clone(), |step| {
            self.write_export_step(plan, step, &mut pending, pass)
        })?;
        if !pending.steps.is_empty() {
            self.send_export_batch(plan, pending, pass)?;
        }
        self.export_batch_committed(pass)?;
        pass.bytes_read += read.bytes_read;
        if read.complete {
            plan.parse_pos = plan.size;
        } else {
            plan.failed = true;
            plan.failure_recorded = true;
        }
        Ok(())
    }

    /// A watched file's new bytes are stored before it is parsed: an
    /// append extends the chain, a rewrite starts a new one. Parsing then
    /// reads the stored chain.
    pub(crate) fn store_plan(
        &mut self,
        plan: &mut Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        use std::io::{Read, Seek, SeekFrom};
        let context = self.source_read_context(
            &plan.path,
            plan.origin.name,
            polylogue_store::capture_read::ReadRoute::NativeCapture,
        );
        let mut file = match &plan.frozen_file {
            Some(snapshot) => snapshot.open()?,
            None => std::fs::File::open(&plan.path)?,
        };
        if let (None, Some(expected)) = (&plan.frozen_file, plan.facts) {
            read::check_source(&file, &plan.path, expected)?;
        }
        file.seek(SeekFrom::Start(plan.stored_upto))?;
        let prepared = self
            .archive
            .prepare_bytes((&mut file).take(plan.size - plan.stored_upto))?;
        let bytes = prepared.len();
        pass.bytes_read += bytes;
        if let (None, Some(expected)) = (&plan.frozen_file, plan.facts) {
            read::check_source(&file, &plan.path, expected)?;
        }
        if bytes != plan.size - plan.stored_upto {
            return Err(polylogue_store::StoreError::SourceChanged(
                "source ended during export capture",
            )
            .into());
        }
        let source_read = context.finish(
            polylogue_store::capture_read::RawRange {
                start: plan.stored_upto,
                end: plan.size,
            },
            prepared.hash(),
        );
        self.writer.store_bytes(
            prepared,
            Batch {
                input: plan.input.clone(),
                change: plan.change.clone(),
                observation: Some(
                    plan.observation
                        .take()
                        .unwrap_or_else(|| self.observation(&plan.path, plan.facts)),
                ),
                continues: false,
                new_blob: None,
                blobs: Vec::new(),
                length: plan.size,
                parse: Vec::new(),
                capture_read: Some(source_read),
                release: Default::default(),
            },
        )?;
        let written = self.flush_writes()?;
        if let Some(error) = written.failed() {
            return Err(error.into());
        }
        pass.bytes_stored += bytes;
        let r = self.reads.reader()?;
        let progress = r
            .latest_chain(&plan.input.id)?
            .ok_or_else(|| EngineError::internal("a stored export has no revision"))?;
        let chain = r.stored_chain(&plan.input.id, progress.chain, progress.stored)?;
        if plan.tracked {
            let known = r.known_watched_input_at(
                &self.config.machine,
                plan.path.as_os_str().as_encoded_bytes(),
            )?;
            if let Some(known) = known
                && known.input_id == plan.input.id
                && known.chain == Some(progress.chain)
                && known.stored == plan.size
                && known.facts == plan.facts
                && let Some(state) = self.inputs.get_mut(&plan.path)
                && state.input_id == plan.input.id
            {
                // Stored bytes survive a parse refusal. It is not a parser
                // checkpoint, and cannot replace a newer source position.
                state.facts = known.facts;
                state.stored = known.stored;
                state.chain = known.chain;
            }
        }
        plan.frozen_file = None;
        plan.source = if matches!(
            (plan.origin.open)(&plan.path),
            Some(polylogue_sources::Parse::Export(_))
        ) {
            pool::Bytes::Stored(Arc::new(chain))
        } else {
            pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                chain,
                Arc::clone(&self.blobs),
            ))))
        };
        plan.change = Change::Reparse {
            chain: progress.chain,
        };
        plan.parser_chain = Some(progress.chain);
        plan.stored_upto = plan.size;
        plan.storing_started = true;
        plan.parse_pos = 0;
        Ok(())
    }

    /// Items join the pending batch until it covers `chunk` bytes of the
    /// export, the same unit a line input is read and committed in. One
    /// transaction per item would rewrite the input's reading state and sync
    /// once per record; each delivery is still committed in order inside the
    /// batch's transaction, with its cursor, so a commit never holds part of
    /// a record and a later parse resumes after the last committed one.
    fn write_export_step(
        &mut self,
        plan: &Plan,
        mut item: ExportStep,
        pending: &mut ExportBatch,
        pass: &mut Pass,
    ) -> Result<Option<ExportCursor>, EngineError> {
        for blob in std::mem::take(&mut item.output.blobs) {
            let (new_blob, file) = self.archive.prepare_blob(&blob.bytes)?;
            if new_blob.hash != blob.hash || new_blob.size != blob.size {
                return Err(polylogue_model::ChainFailure::bytes_differ().into());
            }
            pending.files.extend(file);
            pending.blobs.push(new_blob);
        }
        let done = item.done;
        pending.bytes += item.source_bytes();
        let (mut parse, cursor) = item.into_parse(plan, &self.blobs)?;
        if done && plan.origin.is_row_dump(&plan.path) {
            pass.bytes_read += plan.size;
        }
        parse.prepare();
        pending.steps.push(parse);
        if done || pending.bytes >= self.config.chunk {
            self.send_export_batch(plan, std::mem::take(pending), pass)?;
        }
        Ok(cursor)
    }

    /// Queues one batch of export items. The previous batch commits while
    /// this one is parsed; waiting for it first reports its errors before
    /// later deliveries, which extend its cursor, are queued.
    fn send_export_batch(
        &mut self,
        plan: &Plan,
        pending: ExportBatch,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        self.export_batch_committed(pass)?;
        let batch = Batch {
            input: plan.input.clone(),
            change: plan.change.clone(),
            observation: None,
            continues: false,
            new_blob: None,
            blobs: pending.blobs,
            length: plan.size,
            parse: pending.steps,
            capture_read: None,
            release: Default::default(),
        };
        if pending.files.is_empty() {
            self.writer.send(batch)?;
        } else {
            for result in
                self.writer
                    .store_group(vec![polylogue_store::staging::PreparedBatch {
                        batch,
                        files: pending.files,
                    }])?
            {
                result?;
            }
        }
        Ok(())
    }

    /// Waits until every queued export batch has committed.
    fn export_batch_committed(&mut self, pass: &mut Pass) -> Result<(), EngineError> {
        let written = self.flush_writes()?;
        pass.messages_written += written.messages;
        if let Some(error) = written.failed() {
            return Err(error.into());
        }
        Ok(())
    }
}

/// Export items parsed but not yet queued for the writer.
#[derive(Default)]
struct ExportBatch {
    steps: Vec<ParsePart>,
    blobs: Vec<polylogue_store::NewBlob>,
    files: Vec<polylogue_store::staging::PreparedBytes>,
    /// Bytes of the export the batch's items were read from.
    bytes: u64,
}

pub(crate) struct ExportStep {
    pub(crate) output: Output,
    pub(crate) counts: RecordCounts,
    pub(crate) order_from: u64,
    pub(crate) reset: bool,
    pub(crate) done: bool,
    pub(crate) repeats: bool,
    ordinal: u64,
    source: Option<ExportDeliverySource>,
    previous_prefix: BlobHash,
    /// The scanner declined the container, for this reason.
    refused: Option<String>,
}

impl ExportStep {
    /// Bytes of the stored export this delivery was read from.
    fn source_bytes(&self) -> u64 {
        match self.source {
            Some(ExportDeliverySource::Item { start, end }) => end.saturating_sub(start),
            _ => 0,
        }
    }

    /// Extracted blobs are prepared by the caller before sealing. Delivery
    /// metadata is identical for first parsing and archived replacement.
    /// The completing step of a row dump also commits the comparison of the
    /// whole stored chain, whether the rows were parsed on arrival or again
    /// for a replacement reading.
    pub(crate) fn into_parse(
        self,
        plan: &Plan,
        blobs: &Arc<polylogue_store::blobs::Blobs>,
    ) -> Result<(ParsePart, Option<ExportCursor>), EngineError> {
        let mut step = ParsePart::new(
            self.output,
            self.counts,
            0..if self.done { plan.size } else { 0 },
            self.done,
        );
        if self.done && plan.origin.is_row_dump(&plan.path) {
            let pool::Bytes::Stored(chain) = &plan.source else {
                return Err(EngineError::internal(
                    "a row dump must be stored before comparison",
                ));
            };
            step.compare_reader(std::io::BufReader::new(chain.reader(Arc::clone(blobs))))?;
        }
        step.order_from = Some(self.order_from);
        step.repeats_whole_conversations = self.repeats;
        step.reset = self.reset;
        step.parse_version = plan.origin.parse_version();
        if let Some(reason) = self.refused {
            step.failure = Some(polylogue_model::ChainFailure::refused(reason));
        }
        let cursor = self
            .source
            .map(|source| step.seal_export(self.ordinal, source, self.previous_prefix))
            .transpose()?;
        Ok((step, cursor))
    }
}

pub(crate) struct ExportRead {
    pub(crate) bytes_read: u64,
    pub(crate) complete: bool,
}

pub(crate) fn parse_export(
    plan: &Plan,
    parser: &mut dyn ExportParser,
    blobs: &Arc<polylogue_store::blobs::Blobs>,
    mut emit: impl FnMut(ExportStep) -> Result<Option<ExportCursor>, EngineError>,
) -> Result<ExportRead, EngineError> {
    let mut scratch = ExportItems::new()?;
    let pool::Bytes::Stored(chain) = &plan.source else {
        return Err(EngineError::internal(
            "an export must be stored before it is parsed",
        ));
    };
    let mut reader = Counted::new(|| chain.reader(Arc::clone(blobs)));
    let scanned = parser.scan(&plan.input.id, &mut reader, &mut ItemWriter(&mut scratch));
    let refusal = match scanned {
        Err(export::ScanError::Malformed(error)) => Some(error.to_string()),
        Err(error) => {
            // The stored bytes are read through the parser's io::Read, which
            // carries a stop only as text; a scan the stop ended is stopped.
            polylogue_store::cancellation::check_current()?;
            return Err(std::io::Error::other(error).into());
        }
        Ok(_) => {
            use std::io::BufRead;
            // Rewinds count as IO, but only the final logical position can
            // establish that the scanner consumed this immutable container.
            let eof = reader.fill_buf()?.is_empty();
            (chain.length != plan.size || reader.position != chain.length || !eof)
                .then(|| "export scanner returned before the immutable source EOF".to_owned())
        }
    };
    if let Some(error) = refusal {
        let step = export::refused(error.clone(), plan.size);
        // A refusal has no complete source range or parser checkpoint.
        // Staged prefix items are not delivered until the entire scan succeeds.
        delivered(
            emit(ExportStep {
                output: step.output,
                counts: step.counts,
                order_from: 0,
                reset: true,
                done: false,
                repeats: false,
                ordinal: 0,
                source: Some(ExportDeliverySource::ContainerHeader),
                previous_prefix: ExportDelivery::seed(),
                refused: Some(error),
            })?,
            0,
        )?;
        return Ok(ExportRead {
            bytes_read: reader.count,
            complete: false,
        });
    }
    let mut next = next_record(&mut scratch)?;
    let mut order_from = 0;
    let mut first = true;
    let mut repeats_whole_conversations = false;
    let mut ordinal = 0;
    let mut item_count = 0u64;
    let mut previous_prefix = ExportDelivery::seed();
    // An export is read whole in one run, so the rows its earlier items
    // settled are all the chain's earlier rows.
    let mut names = Formed::new(NoEarlierRows);
    if let Some(mut step) = parser
        .begin(
            plan.size,
            next.as_ref().map(|record| record.bytes.as_slice()),
        )
        .map_err(polylogue_model::ChainFailure::malformed)?
    {
        repeats_whole_conversations = step.repeats_whole_conversations;
        names
            .settle(&mut step.output)
            .map_err(|error| polylogue_model::ChainFailure::malformed(error.problem()))?;
        order_from = (step.output.messages.len() as u64).max(1);
        let cursor = delivered(
            emit(ExportStep {
                output: step.output,
                counts: step.counts,
                order_from: 0,
                reset: true,
                done: false,
                repeats: repeats_whole_conversations,
                ordinal,
                source: Some(ExportDeliverySource::ContainerHeader),
                previous_prefix,
                refused: None,
            })?,
            ordinal,
        )?;
        ordinal = cursor.next_ordinal().map_err(EngineError::internal)?;
        previous_prefix = cursor.prefix_hash;
        first = false;
    }
    while let Some(ExportRecord { range, bytes }) = next {
        // A stop ends a long export between items; what was delivered stays.
        polylogue_store::cancellation::check_current()?;
        polylogue_store::fault::check("parse", &plan.path.to_string_lossy())?;
        let mut state = StateLookup {
            items: &mut scratch,
            failure: None,
        };
        let parsed = export::parse_item(parser, &bytes, range.clone(), &mut state);
        if let Some(error) = state.failure {
            return Err(error.into());
        }
        // One element the parser cannot read is that element's outcome; the
        // export's other elements still parse. Its bytes stay stored.
        let mut step = parsed.unwrap_or_else(|error| export::unreadable_item(error, range.clone()));
        if repeats_whole_conversations {
            settle(&mut step.output, &mut NoEarlierRows).map(|_| ())
        } else {
            names.settle(&mut step.output)
        }
        .map_err(|error| polylogue_model::ChainFailure::malformed(error.problem()))?;
        let count = step.output.messages.len() as u64;
        let cursor = delivered(
            emit(ExportStep {
                output: step.output,
                counts: step.counts,
                order_from,
                reset: first,
                done: false,
                repeats: repeats_whole_conversations,
                ordinal,
                source: Some(ExportDeliverySource::Item {
                    start: range.start,
                    end: range.end,
                }),
                previous_prefix,
                refused: None,
            })?,
            ordinal,
        )?;
        ordinal = cursor.next_ordinal().map_err(EngineError::internal)?;
        previous_prefix = cursor.prefix_hash;
        item_count = item_count
            .checked_add(1)
            .ok_or_else(|| EngineError::internal("export item count overflow"))?;
        first = false;
        order_from = order_from
            .checked_add(count.max(1))
            .ok_or_else(|| EngineError::internal("export message order overflow"))?;
        next = next_record(&mut scratch)?;
    }
    delivered(
        emit(ExportStep {
            output: Output::default(),
            counts: RecordCounts::default(),
            order_from,
            reset: first,
            done: true,
            repeats: repeats_whole_conversations,
            ordinal,
            source: Some(ExportDeliverySource::Complete {
                length: plan.size,
                item_count,
                delivery_count: ordinal,
            }),
            previous_prefix,
            refused: None,
        })?,
        ordinal,
    )?;
    Ok(ExportRead {
        bytes_read: reader.count,
        complete: true,
    })
}

fn delivered(cursor: Option<ExportCursor>, ordinal: u64) -> Result<ExportCursor, EngineError> {
    let cursor = cursor
        .filter(|cursor| cursor.last_ordinal == ordinal)
        .ok_or_else(|| {
            EngineError::internal("export callback did not seal its delivery ordinal")
        })?;
    cursor.validate().map_err(EngineError::internal)?;
    Ok(cursor)
}

struct ExportRecord {
    range: std::ops::Range<u64>,
    bytes: Vec<u8>,
}

fn next_record(scratch: &mut ExportItems) -> Result<Option<ExportRecord>, EngineError> {
    scratch
        .next_item()?
        .map(|mut item| {
            use std::io::Read;
            let mut bytes = Vec::new();
            item.reader.read_to_end(&mut bytes)?;
            Ok(ExportRecord {
                range: item.range,
                bytes,
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests;
