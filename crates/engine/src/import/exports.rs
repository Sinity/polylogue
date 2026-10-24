//! A complete retained export, framed and externally ordered before publication.
use super::*;
use polylogue_model::{Accounting, Output};
use polylogue_sources::export::{self, ExportParser, ExportState};
use polylogue_store::{Batch, ParseStep, Piece, export_items::ExportItems};

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
    fn claim(
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
            self.retain_export(plan, pass)?;
        }
        let mut scratch = ExportItems::new()?;
        let pool::Bytes::Stored(chain) = &plan.source else {
            return Err(
                std::io::Error::other("exports must have a retained chain before parsing").into(),
            );
        };
        let mut reader = Counted::new(|| chain.reader(Arc::clone(&self.blobs)));
        let scanned = parser.scan(&plan.input.id, &mut reader, &mut ItemWriter(&mut scratch));
        pass.bytes_read += reader.count;
        if let Err(error) = scanned {
            let export::ScanError::Malformed(error) = error else {
                return Err(std::io::Error::other(error).into());
            };
            let step = export::refused(error.to_string(), plan.size);
            self.write_export_step(plan, step.output, step.accounting, 0, true, true, pass)?;
            plan.parse_pos = plan.size;
            return Ok(());
        }
        let mut next = next_record(&mut scratch)?;
        let mut order_from = 0;
        let mut first = true;
        if let Some(step) = parser
            .begin(
                plan.size,
                next.as_ref().map(|record| record.bytes.as_slice()),
            )
            .map_err(std::io::Error::other)?
        {
            order_from = (step.output.messages.len() as u64).max(1);
            self.write_export_step(plan, step.output, step.accounting, 0, true, false, pass)?;
            first = false;
        }
        while let Some(ExportRecord { range, bytes }) = next {
            polylogue_store::fault::check("parse", &plan.path.to_string_lossy())?;
            let step = export::parse_item(parser, &bytes, range, &mut StateLookup(&mut scratch))
                .map_err(std::io::Error::other)?;
            let count = step.output.messages.len() as u64;
            // Flush is a commit acknowledgement. A second conversation is not
            // decoded while the writer still holds the preceding one's output.
            self.write_export_step(
                plan,
                step.output,
                step.accounting,
                order_from,
                first,
                false,
                pass,
            )?;
            first = false;
            order_from += count.max(1);
            next = next_record(&mut scratch)?;
        }
        self.write_export_step(
            plan,
            Output::default(),
            Accounting::default(),
            order_from,
            first,
            true,
            pass,
        )?;
        plan.parse_pos = plan.size;
        Ok(())
    }

    /// Native watched exports retain their next source extent before parsing.
    /// The existing revision chain assembles genuine appends; rewrites start
    /// a new chain. Parsing then uses that pinned representation.
    fn retain_export(&mut self, plan: &mut Plan, pass: &mut Pass) -> Result<(), EngineError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&plan.path)?;
        let before = FileFacts::of(&file.metadata()?);
        if Some(before) != plan.facts || before.size != plan.size {
            return Err(std::io::Error::other("source changed before export capture").into());
        }
        let mut tail = vec![0; plan.size.min(read::TAIL as u64) as usize];
        file.seek(SeekFrom::Start(plan.size - tail.len() as u64))?;
        file.read_exact(&mut tail)?;
        file.seek(SeekFrom::Start(plan.piece_pos))?;
        let captured = Captured::read(&self.blobs, (&mut file).take(plan.size - plan.piece_pos))?;
        if FileFacts::of(&file.metadata()?) != before
            || captured.piece.size != plan.size - plan.piece_pos
        {
            return Err(std::io::Error::other("source changed during export capture").into());
        }
        let bytes = captured.piece.size;
        self.writer.send(Batch {
            input: plan.input.clone(),
            change: plan.change.clone(),
            observation: plan.observation.take(),
            continues: false,
            piece: Some(captured.piece),
            blobs: Vec::new(),
            length: plan.size,
            parse: Vec::new(),
            release: Default::default(),
        })?;
        let written = self.flush_writes()?;
        if !written.errors.is_empty() {
            return Err(std::io::Error::other(written.errors.join("; ")).into());
        }
        pass.bytes_read += bytes + tail.len() as u64;
        pass.bytes_stored += bytes;
        let r = self.reads.reader()?;
        let retained = r
            .retained_input(&plan.input.id)?
            .ok_or_else(|| std::io::Error::other("export capture has no retained revision"))?;
        let chain = r.stored_chain(&plan.input.id, retained.chain, retained.length)?;
        plan.source = pool::Bytes::Stored(Arc::new(chain));
        plan.change = Change::Unchanged;
        plan.piece_pos = plan.size;
        plan.piece_started = true;
        plan.parse_pos = 0;
        plan.tail = Some(tail);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_export_step(
        &mut self,
        plan: &Plan,
        mut output: Output,
        accounting: Accounting,
        order_from: u64,
        reset: bool,
        done: bool,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let mut blobs = Vec::new();
        for blob in std::mem::take(&mut output.blobs) {
            blobs.push(Piece::prepare(&self.blobs, &blob.bytes)?);
        }
        let mut parse = ParseStep::new(
            output,
            accounting,
            0..if done { plan.size } else { 0 },
            plan.origin.parser_version,
            done,
        );
        parse.order_from = Some(order_from);
        parse.reset = reset;
        parse.prepare();
        self.writer.send(Batch {
            input: plan.input.clone(),
            change: plan.change.clone(),
            observation: None,
            continues: false,
            piece: None,
            blobs,
            length: plan.size,
            parse: vec![parse],
            release: Default::default(),
        })?;
        let written = self.flush_writes()?;
        pass.messages_written += written.messages;
        if !written.errors.is_empty() {
            return Err(std::io::Error::other(written.errors.join("; ")).into());
        }
        Ok(())
    }
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
