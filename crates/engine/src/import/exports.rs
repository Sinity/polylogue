//! A whole export, stored, then split into records and ordered before its items are written.
use super::*;
use polylogue_model::{Output, RecordCounts};
use polylogue_sources::export::{self, ExportParser, ExportState};
use polylogue_store::{Batch, ParseStep, export_items::ExportItems};

struct Counted<F, R> {
    open: F,
    reader: std::io::BufReader<R>,
    count: u64,
}
impl<F: FnMut() -> R, R: std::io::Read> Counted<F, R> {
    fn new(mut open: F) -> Self {
        Self {
            reader: std::io::BufReader::new(open()),
            open,
            count: 0,
        }
    }
}
impl<F, R: std::io::Read> std::io::Read for Counted<F, R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(out)?;
        self.count += n as u64;
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
    }
}
impl<F: FnMut() -> R, R: std::io::Read> export::Input for Counted<F, R> {
    fn rewind(&mut self) -> std::io::Result<()> {
        self.reader = std::io::BufReader::new((self.open)());
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

struct StateLookup<'a>(&'a mut ExportItems);
impl ExportState for StateLookup<'_> {
    fn record_first_session(
        &mut self,
        native: &str,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::SessionId>, String> {
        self.0
            .claim_owner(native, session)
            .map_err(|e| e.to_string())
    }
    fn last_message(
        &mut self,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::MessageId>, String> {
        self.0.last_message(session).map_err(|e| e.to_string())
    }
    fn set_last_message(
        &mut self,
        session: &polylogue_model::SessionId,
        message: &polylogue_model::MessageId,
    ) -> Result<(), String> {
        self.0
            .set_last_message(session, message)
            .map_err(|e| e.to_string())
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
        let bytes_read = parse_export(plan, parser, &self.blobs.clone(), |step| {
            self.write_export_step(
                plan,
                step.output,
                step.counts,
                step.order_from,
                step.reset,
                step.done,
                step.repeats,
                pass,
            )
        })?;
        pass.bytes_read += bytes_read;
        plan.parse_pos = plan.size;
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
        let mut file = std::fs::File::open(&plan.path)?;
        let before = FileFacts::of(&file.metadata()?);
        if Some(before) != plan.facts || before.size != plan.size {
            return Err(std::io::Error::other("source changed before export capture").into());
        }
        let mut tail = vec![0; plan.size.min(read::TAIL as u64) as usize];
        file.seek(SeekFrom::Start(plan.size - tail.len() as u64))?;
        file.read_exact(&mut tail)?;
        file.seek(SeekFrom::Start(plan.stored_upto))?;
        let prepared = self
            .archive
            .prepare_bytes((&mut file).take(plan.size - plan.stored_upto))?;
        if FileFacts::of(&file.metadata()?) != before
            || prepared.len() != plan.size - plan.stored_upto
        {
            return Err(std::io::Error::other("source changed during export capture").into());
        }
        let bytes = prepared.len();
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
                release: Default::default(),
            },
        )?;
        let written = self.flush_writes()?;
        if !written.errors.is_empty() {
            return Err(std::io::Error::other(written.errors.join("; ")).into());
        }
        pass.bytes_read += bytes + tail.len() as u64;
        pass.bytes_stored += bytes;
        let r = self.reads.reader()?;
        let progress = r
            .input_progress(&plan.input.id)?
            .ok_or_else(|| std::io::Error::other("a stored export has no revision"))?;
        let chain = r.stored_chain(&plan.input.id, progress.chain, progress.length)?;
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
        plan.stored_upto = plan.size;
        plan.storing_started = true;
        plan.parse_pos = 0;
        plan.tail = Some(tail);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_export_step(
        &mut self,
        plan: &Plan,
        mut output: Output,
        counts: RecordCounts,
        order_from: u64,
        reset: bool,
        done: bool,
        repeats_whole_conversations: bool,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let mut blobs = Vec::new();
        let mut files = Vec::new();
        {
            for blob in std::mem::take(&mut output.blobs) {
                let (new_blob, file) = self.archive.prepare_decoded(&blob.bytes)?;
                if new_blob.hash != blob.hash || new_blob.size != blob.size {
                    return Err(std::io::Error::other(
                        "decoded export bytes differ from their declared hash or length",
                    )
                    .into());
                }
                files.extend(file);
                blobs.push(new_blob);
            }
        }
        let mut parse = ParseStep::new(output, counts, 0..if done { plan.size } else { 0 }, done);
        parse.repeats_whole_conversations = repeats_whole_conversations;
        parse.order_from = Some(order_from);
        parse.reset = reset;
        parse.reading_version = plan.origin.reading_version();
        if done && plan.origin.is_row_dump(&plan.path) {
            let pool::Bytes::Stored(chain) = &plan.source else {
                return Err(
                    std::io::Error::other("a row dump must be stored before comparison").into(),
                );
            };
            parse.compare_reader(std::io::BufReader::new(
                chain.reader(Arc::clone(&self.blobs)),
            ))?;
            pass.bytes_read += plan.size;
        }
        parse.prepare();
        let batch = Batch {
            input: plan.input.clone(),
            change: plan.change.clone(),
            observation: None,
            continues: false,
            new_blob: None,
            blobs,
            length: plan.size,
            parse: vec![parse],
            release: Default::default(),
        };
        if files.is_empty() {
            self.writer.send(batch)?;
        } else {
            for result in self.writer.sender()?.store_group(
                vec![polylogue_store::staging::PreparedBatch { batch, files }],
                Arc::clone(&self.blobs),
            )? {
                result?;
            }
        }
        let written = self.flush_writes()?;
        pass.messages_written += written.messages;
        if !written.errors.is_empty() {
            return Err(std::io::Error::other(written.errors.join("; ")).into());
        }
        Ok(())
    }
}

pub(crate) struct ExportStep {
    pub(crate) output: Output,
    pub(crate) counts: RecordCounts,
    pub(crate) order_from: u64,
    pub(crate) reset: bool,
    pub(crate) done: bool,
    pub(crate) repeats: bool,
}

pub(crate) fn parse_export(
    plan: &Plan,
    parser: &mut dyn ExportParser,
    blobs: &Arc<hashed_blobs::Store>,
    mut emit: impl FnMut(ExportStep) -> Result<(), EngineError>,
) -> Result<u64, EngineError> {
    let mut scratch = ExportItems::new()?;
    let pool::Bytes::Stored(chain) = &plan.source else {
        return Err(std::io::Error::other("an export must be stored before it is parsed").into());
    };
    let mut reader = Counted::new(|| chain.reader(Arc::clone(blobs)));
    let scanned = parser.scan(&plan.input.id, &mut reader, &mut ItemWriter(&mut scratch));
    if let Err(error) = scanned {
        let export::ScanError::Malformed(error) = error else {
            return Err(std::io::Error::other(error).into());
        };
        let step = export::refused(error.to_string(), plan.size);
        emit(ExportStep {
            output: step.output,
            counts: step.counts,
            order_from: 0,
            reset: true,
            done: true,
            repeats: false,
        })?;
        return Ok(reader.count);
    }
    let mut next = next_record(&mut scratch)?;
    let mut order_from = 0;
    let mut first = true;
    let mut repeats_whole_conversations = false;
    if let Some(step) = parser
        .begin(
            plan.size,
            next.as_ref().map(|record| record.bytes.as_slice()),
        )
        .map_err(EngineError::Parse)?
    {
        repeats_whole_conversations = step.repeats_whole_conversations;
        order_from = (step.output.messages.len() as u64).max(1);
        emit(ExportStep {
            output: step.output,
            counts: step.counts,
            order_from: 0,
            reset: true,
            done: false,
            repeats: repeats_whole_conversations,
        })?;
        first = false;
    }
    while let Some(ExportRecord { range, bytes }) = next {
        polylogue_store::fault::check("parse", &plan.path.to_string_lossy())?;
        let step = export::parse_item(
            parser,
            &bytes,
            range.clone(),
            &mut StateLookup(&mut scratch),
        )
        .map_err(EngineError::Parse)?;
        let count = step.output.messages.len() as u64;
        emit(ExportStep {
            output: step.output,
            counts: step.counts,
            order_from,
            reset: first,
            done: false,
            repeats: repeats_whole_conversations,
        })?;
        first = false;
        order_from += count.max(1);
        next = next_record(&mut scratch)?;
    }
    emit(ExportStep {
        output: Output::default(),
        counts: RecordCounts::default(),
        order_from,
        reset: first,
        done: true,
        repeats: repeats_whole_conversations,
    })?;
    Ok(reader.count)
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
