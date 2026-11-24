//! Parsed readings wait on disk before their single replacement transaction.
use super::*;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

/// Independently compressed parser steps bound memory before one replacement transaction.
pub struct ChainReading {
    file: Option<BufWriter<std::fs::File>>,
    path: tempfile::TempPath,
    first: bool,
    decoded: Vec<crate::staging::PreparedBatch>,
    /// In primary mode, its parsed events are in the producer spool.
    spooled: bool,
}

impl ChainReading {
    pub fn new() -> Result<Self> {
        let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
        Ok(Self {
            file: Some(BufWriter::new(file)),
            path,
            first: true,
            decoded: Vec::new(),
            spooled: false,
        })
    }
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
        write_step(
            self.file.as_mut().ok_or_else(|| {
                StoreError::Invalid("a completed reading cannot be extended".into())
            })?,
            step,
        )?;
        self.first = false;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<()> {
        if let Some(file) = self.file.take() {
            drop(file.into_inner().map_err(|error| error.into_error())?);
        }
        Ok(())
    }
    pub fn decoded(&mut self, batch: crate::staging::PreparedBatch) {
        self.decoded.push(batch);
    }

    pub(super) fn take_decoded(&mut self) -> Vec<crate::staging::PreparedBatch> {
        std::mem::take(&mut self.decoded)
    }

    /// Complete parse steps are read without resolving local cache keys first.
    pub(crate) fn visit_source(
        &mut self,
        mut apply: impl FnMut(&ParseStep) -> Result<()>,
    ) -> Result<()> {
        self.finish()?;
        let mut input = BufReader::new(std::fs::File::open(&self.path)?);
        while let Some(step) = read_step(&mut input)? {
            apply(&step)?;
        }
        Ok(())
    }

    /// Its parsed events were prepared in the producer spool.
    pub(super) fn mark_spooled(&mut self) {
        self.spooled = true;
    }

    /// A replacement names a valid chain and holds a reading whose events
    /// name their records.
    pub(super) fn check(&mut self, input: &str, chain: i64) -> Result<()> {
        if chain < 1 {
            return Err(StoreError::Invalid(
                "a reading must name a valid input chain".into(),
            ));
        }
        if self.first {
            return Err(StoreError::Invalid(
                "a chain replacement has no reading".into(),
            ));
        }
        self.visit_source(|step| step.check_events(input))?;
        Ok(())
    }
}

/// Writes one step record with its prepared rows.
fn write_step(file: &mut BufWriter<std::fs::File>, step: &ParseStep) -> Result<()> {
    let start = file.stream_position()?;
    file.write_all(&0u64.to_le_bytes())?;
    let mut encoder = zstd::stream::write::Encoder::new(&mut *file, 1)?;
    serde_json::to_writer(
        &mut encoder,
        &(
            step,
            &step.prepared.messages,
            &step.prepared.documents,
            &step.prepared.fragments,
            &step.prepared.events,
            &step.prepared.texts,
            step.prepared.released,
        ),
    )?;
    encoder.finish()?;
    let end = file.stream_position()?;
    file.seek(SeekFrom::Start(start))?;
    file.write_all(&(end - start - 8).to_le_bytes())?;
    file.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn read_step(reader: &mut BufReader<std::fs::File>) -> Result<Option<ParseStep>> {
    let mut size = [0u8; 8];
    let read = reader.read(&mut size[..1])?;
    if read == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut size[1..])?;
    let size = u64::from_le_bytes(size);
    let mut decoder = zstd::stream::read::Decoder::new(reader.take(size))?;
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    let StoredStep(mut step, messages, documents, fragments, events, texts, released) =
        crate::encoding::read_json(&bytes)?;
    step.prepared = batch::PreparedRows {
        messages,
        documents,
        fragments,
        events,
        texts,
        released,
    };
    Ok(Some(step))
}

#[derive(serde::Deserialize)]
struct StoredStep(
    ParseStep,
    Vec<PreparedMessage>,
    Vec<PreparedDocument>,
    crate::fragment::PreparedFragments,
    Vec<(String, Vec<u8>)>,
    crate::text_rows::PreparedTexts,
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
        if self.sinex.is_some() {
            if !reading.spooled {
                return Err(StoreError::Invalid(
                    "a primary replacement must spool its parsed events first".into(),
                ));
            }
            return self.replace_index(input, chain, length, reading);
        }
        let exists: bool = self.archive.exists(
            "SELECT EXISTS(SELECT 1 FROM revision WHERE input=?1 AND chain=?2 AND length>=?3)",
            params![input.id, chain, length as i64],
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
        let mut local_steps = false;
        let mut version = None;
        let mut complete = false;
        let mut refused_or_failed = false;
        reading.visit_source(|step| {
            if version
                .as_ref()
                .is_some_and(|version| version != &step.reading_version)
            {
                return Err(StoreError::Invalid(
                    "one replacement has several reading versions".into(),
                ));
            }
            version.get_or_insert_with(|| step.reading_version.clone());
            refused_or_failed |= matches!(
                step.progress,
                polylogue_model::batch::StepProgress::Refused { .. }
                    | polylogue_model::batch::StepProgress::Failed { .. }
            );
            local_steps |= step.delivery.is_none();
            complete = step.progress.reached_end() && step.parsed_upto == length;
            Ok(())
        })?;
        if local_steps && !complete {
            return Err(StoreError::Invalid(
                "a chain replacement with non-export steps must finish its stored source".into(),
            ));
        }
        if refused_or_failed {
            self.replace_index(input, chain, length, reading)
        } else {
            self.until_stopped(|inner| inner.replace_index(input, chain, length, reading))?
                .ok_or(StoreError::Stopped)
        }
    }

    pub(super) fn replace_index(
        &mut self,
        input: InputKey,
        chain: i64,
        length: u64,
        reading: ChainReading,
    ) -> Result<u64> {
        let file = std::fs::File::open(&reading.path)?;
        let mut reader = BufReader::new(file);
        let mut all_replayed = true;
        let mut has_step = false;
        while let Some(step) = read_step(&mut reader)? {
            has_step = true;
            if step.delivery.is_none()
                || crate::wire_shapes::admit_export(&self.index, &input.id, chain, length, &step)?
            {
                all_replayed = false;
                break;
            }
        }
        if has_step && all_replayed {
            return Ok(0);
        }
        drop(reader);
        let text_transaction = self.texts.transaction();
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::fault::check("write.replay_rows", "")?;
        let counter = bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        let mut touched = super::readings::Touched::default();
        write_reading(
            &tx,
            ReadingWrite {
                register: &self.archive,
                texts: &self.texts,
                counter,
                primary: self.sinex.is_some(),
                changes: &changes,
            },
            &input,
            chain,
            length,
            &reading,
            &mut touched,
        )?;
        let (written, read) = rows::finish(&tx, touched, &changes, &self.texts, false, counter)?;
        self.report.item_units_read += read;
        crate::fault::check("write.index_commit", "")?;
        crate::fault::check("reparse.commit", &input.id)?;
        crate::timing::time(crate::timing::Stage::Commit, || tx.commit())?;
        text_transaction.commit();
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.search.wake();
        self.report.transactions += 1;
        Ok(written)
    }
}

/// What a chain's reading is written with.
pub(super) struct ReadingWrite<'a> {
    /// Answers which attachment bytes the archive holds.
    pub register: &'a Connection,
    pub texts: &'a crate::text_rows::TextIds,
    pub counter: i64,
    pub primary: bool,
    pub changes: &'a crate::change_summary::ChangeSummary,
}

/// Writes a chain's reading, step by step from its first record, into the
/// reading tables of `tx`'s index, recording what it states in `touched`.
pub(super) fn write_reading(
    tx: &Transaction<'_>,
    w: ReadingWrite<'_>,
    input: &InputKey,
    chain: i64,
    length: u64,
    reading: &ChainReading,
    touched: &mut super::readings::Touched,
) -> Result<()> {
    let mut reader = BufReader::new(std::fs::File::open(&reading.path)?);
    // Attachment references name only bytes the archive holds, as of
    // this transaction.
    let mut known = HashMap::new();
    while let Some(step) = read_step(&mut reader)? {
        attachments::remember(&step, w.register, &HashMap::new(), &mut known)?;
    }
    reader.rewind()?;
    let batch = Batch {
        required_outcome: None,
        input: input.clone(),
        change: Change::Reparse { chain },
        observation: None,
        continues: false,
        new_blob: None,
        blobs: Vec::new(),
        length,
        parse: Vec::new(),
        acquisition: None,
        basis: None,
        release: Default::default(),
    };
    let mut first = true;
    while let Some(mut step) = read_step(&mut reader)? {
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
        attachments::normalize_known(&mut step, &known)?;
        let replacement = first && step.delivery.is_none();
        first = false;
        let source_commit =
            crate::source::Source::reading_commit(tx, &batch.input.id, chain, step.parsed_upto)?;
        rows::write_step(
            tx,
            &batch,
            &step,
            rows::StepWrite {
                texts: w.texts,
                counter: w.counter,
                chain,
                source_commit: Some(&source_commit),
                replacement,
                retraction: false,
                primary: w.primary,
                changes: w.changes,
            },
            touched,
        )?;
    }
    if first {
        return Err(StoreError::Invalid(
            "a chain replacement has no reading".into(),
        ));
    }
    Ok(())
}
