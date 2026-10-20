//! A complete retained export, framed and externally ordered before publication.
use super::*;
use polylogue_model::{Accounting, Output};
use polylogue_sources::array::{self, ArrayParser, Owners};
use polylogue_store::{Batch, ParseStep, Piece, export_array::ExportArray};

struct Counted<R> {
    reader: R,
    count: u64,
}
impl<R: std::io::Read> std::io::Read for Counted<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(out)?;
        self.count += n as u64;
        Ok(n)
    }
}

struct OwnerLookup<'a>(&'a mut ExportArray);
impl Owners for OwnerLookup<'_> {
    fn claim(
        &mut self,
        native: &str,
        session: &polylogue_model::SessionId,
    ) -> Result<polylogue_model::SessionId, String> {
        self.0
            .claim_message(native, session)
            .map_err(|e| e.to_string())
    }
}

impl Engine {
    pub(crate) fn execute_array(
        &mut self,
        plan: &mut Plan,
        parser: &dyn ArrayParser,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let mut scratch = ExportArray::new()?;
        let mut storage_error = None;
        let mut ingest = |range, raw: &serde_json::value::RawValue| {
            scratch
                .push(range, parser.order_key(raw), raw.get().as_bytes())
                .map_err(|e| {
                    storage_error = Some(e.to_string());
                    e.to_string()
                })
        };
        let scanned = match &plan.source {
            pool::Bytes::Import(source) => {
                let mut source = source.lock().expect("import lock");
                let mut reader = Counted {
                    reader: source.stream()?,
                    count: 0,
                };
                let result = array::scan(&mut reader, &mut ingest);
                pass.bytes_read += reader.count;
                result
            }
            pool::Bytes::Stored(chain) => {
                let mut reader = Counted {
                    reader: chain.reader(Arc::clone(&self.blobs)),
                    count: 0,
                };
                let result = array::scan(&mut reader, &mut ingest);
                pass.bytes_read += reader.count;
                result
            }
            pool::Bytes::File => {
                return Err(
                    std::io::Error::other("array exports must be retained before parsing").into(),
                );
            }
        };
        if let Some(error) = storage_error {
            return Err(std::io::Error::other(error).into());
        }
        if let Err(error) = scanned {
            if error.is_io() || error.to_string().contains("Serde element bytes differ") {
                return Err(std::io::Error::other(error).into());
            }
            let step = array::refused(error.to_string(), plan.size);
            self.write_array_step(plan, step.output, step.accounting, 0, true, true, pass)?;
            plan.parse_pos = plan.size;
            return Ok(());
        }
        let mut order_from = 0;
        let mut first = true;
        while let Some(item) = scratch.next_item()? {
            let value = serde_json::from_reader(item.reader)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            polylogue_store::fault::check("parse", &plan.path.to_string_lossy())?;
            let step =
                array::parse_item(parser, &value, item.range, &mut OwnerLookup(&mut scratch))
                    .map_err(std::io::Error::other)?;
            let count = step.output.messages.len() as u64;
            // Flush is a commit acknowledgement. A second conversation is not
            // decoded while the writer still holds the preceding one's output.
            self.write_array_step(
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
        }
        self.write_array_step(
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

    #[allow(clippy::too_many_arguments)]
    fn write_array_step(
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
        let mut parse = ParseStep {
            output,
            accounting,
            snapshot: None,
            start: 0,
            parsed_upto: if done { plan.size } else { 0 },
            parser_version: plan.origin.parser_version,
            at_end: done,
            order_from: Some(order_from),
            reset,
            hashes: Vec::new(),
            packed: Default::default(),
            bodies: Vec::new(),
            events: Vec::new(),
        };
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
        let written = self.writer.flush()?;
        pass.messages_written += written.messages;
        if !written.errors.is_empty() {
            return Err(std::io::Error::other(written.errors.join("; ")).into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
