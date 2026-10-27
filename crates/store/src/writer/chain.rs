//! A chain's parsed output waits on disk before its one replacement transaction.
use super::*;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Private scratch, deleted on drop. Only one parsed step is decoded at a time.
pub struct ChainReading {
    file: Option<zstd::stream::write::Encoder<'static, BufWriter<std::fs::File>>>,
    path: tempfile::TempPath,
    first: bool,
    decoded: Vec<crate::staging::PreparedBatch>,
}

impl ChainReading {
    pub fn new() -> Result<Self> {
        let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
        Ok(Self {
            file: Some(zstd::stream::write::Encoder::new(BufWriter::new(file), 1)?),
            path,
            first: true,
            decoded: Vec::new(),
        })
    }

    /// Pack a complete step on its parser thread, then write it to private scratch.
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
        step.prepare();
        let texts: Vec<_> = step.prepared.texts.iter().collect();
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| StoreError::Invalid("a completed reading cannot be extended".into()))?;
        serde_json::to_writer(
            &mut *file,
            &(
                &*step,
                &step.prepared.messages,
                &step.prepared.documents,
                &step.prepared.events,
                texts,
                step.prepared.released,
            ),
        )?;
        file.write_all(b"\n")?;
        self.first = false;
        Ok(())
    }

    /// Finish compression and close the file before waiting for the writer.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(file) = self.file.take() {
            let file = file
                .finish()?
                .into_inner()
                .map_err(|error| error.into_error())?;
            drop(file);
        }
        Ok(())
    }

    /// Decoded bytes stay private until this chain is ready to commit.
    pub fn decoded(&mut self, batch: crate::staging::PreparedBatch) {
        self.decoded.push(batch);
    }
}

#[derive(serde::Deserialize)]
struct StoredStep(
    ParseStep,
    Vec<PreparedMessage>,
    Vec<PreparedDocument>,
    Vec<(String, Vec<u8>)>,
    Vec<([u8; 32], Vec<u8>)>,
    bool,
);

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
        if !reading.decoded.is_empty() {
            let store = Arc::new(self.location.blobs()?);
            for result in self.store_group(std::mem::take(&mut reading.decoded), store) {
                result?;
            }
        }
        reading.finish()?;
        let file = std::fs::File::open(&reading.path)?;
        let mut reader = BufReader::new(zstd::stream::read::Decoder::new(file)?);
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
            let StoredStep(mut step, messages, documents, events, texts, released) =
                crate::encoding::read_json(line.as_bytes())?;
            step.prepared = batch::PreparedRows {
                messages,
                documents,
                events,
                texts: texts.into_iter().collect(),
                released,
            };
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
        crate::timing::time(crate::timing::Stage::Commit, || tx.commit())?;
        self.derive_due = true;
        self.search.wake();
        self.report.transactions += 1;
        Ok(written)
    }
}
