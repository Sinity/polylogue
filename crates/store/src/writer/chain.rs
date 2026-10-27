//! A chain's parsed output waits on disk before its one replacement transaction.
use super::*;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};

/// Private scratch, deleted on drop. Only one parsed step is decoded at a time.
pub struct ChainReading {
    file: BufWriter<std::fs::File>,
    first: bool,
}

impl ChainReading {
    pub fn new() -> Result<Self> {
        Ok(Self {
            file: BufWriter::new(tempfile::tempfile()?),
            first: true,
        })
    }

    /// Append before preparing a step, while its structured values are complete.
    pub fn push(&mut self, step: &mut ParseStep) -> Result<()> {
        if step.prepared != Default::default() {
            return Err(StoreError::Invalid(
                "a chain reading needs complete parsed values".into(),
            ));
        }
        if !step.output.blobs.is_empty() {
            return Err(StoreError::Invalid(
                "decoded bytes must be stored before their chain reading".into(),
            ));
        }
        step.counts.check()?;
        step.reset = self.first;
        serde_json::to_writer(&mut self.file, step)?;
        self.file.write_all(b"\n")?;
        self.first = false;
        Ok(())
    }
}

impl Inner {
    pub(super) fn replace_chain(
        &mut self,
        input: InputKey,
        chain: i64,
        length: u64,
        mut reading: ChainReading,
    ) -> Result<u64> {
        let exists: bool = self.archive.query_row(
            "SELECT EXISTS(SELECT 1 FROM revision WHERE input=?1 AND chain=?2 AND length>=?3)",
            params![input.id, chain, length as i64],
            |r| r.get(0),
        )?;
        if !exists {
            return Err(StoreError::Invalid(
                "a chain replacement must name stored bytes".into(),
            ));
        }
        reading.file.flush()?;
        reading.file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(reading.file.into_inner().map_err(|e| e.into_error())?);
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&tx)?;
        let mut batch = Batch {
            input,
            change: Change::Reparse { chain },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length,
            parse: Vec::new(),
            release: Default::default(),
        };
        let mut line = String::new();
        let mut written = 0;
        let mut first = true;
        while reader.read_line(&mut line)? > 0 {
            let mut step: ParseStep = crate::encoding::read_json(line.as_bytes())?;
            if first && !step.reset {
                return Err(StoreError::Invalid(
                    "a chain replacement must start at its first record".into(),
                ));
            }
            if !first && step.reset {
                return Err(StoreError::Invalid(
                    "a chain replacement resets twice".into(),
                ));
            }
            first = false;
            step.prepare();
            written += write_rows(
                &tx,
                &batch,
                &step,
                chain,
                counter,
                self.first_build.as_ref(),
            )?;
            line.clear();
        }
        if first {
            return Err(StoreError::Invalid(
                "a chain replacement has no reading".into(),
            ));
        }
        rows::prune_sessions(&tx, &batch.input.id, chain)?;
        batch.parse.clear();
        crate::fault::check("write.index_commit", "")?;
        crate::fault::check("reparse.commit", &batch.input.id)?;
        tx.commit()?;
        self.derive_due = true;
        self.search.wake();
        self.report.transactions += 1;
        Ok(written)
    }
}
