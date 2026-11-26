//! Parsed readings wait on disk before their single replacement transaction.
use super::*;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

/// Independently compressed parser steps bound memory before one replacement transaction.
pub struct ChainParse {
    file: Option<BufWriter<std::fs::File>>,
    path: tempfile::TempPath,
    first: bool,
    pub(crate) decoded: Vec<crate::staging::PreparedBatch>,
}

impl ChainParse {
    pub fn new() -> Result<Self> {
        let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
        Ok(Self {
            file: Some(BufWriter::new(file)),
            path,
            first: true,
            decoded: Vec::new(),
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

    /// A replacement names a valid chain and holds a reading whose events
    /// name their records.
    pub(crate) fn check(&mut self, input: &str, chain: i64) -> Result<()> {
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
            &step.prepared.content,
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
    let StoredStep(mut step, messages, documents, fragments, events, texts, content, released) =
        crate::encoding::read_json(&bytes)?;
    step.prepared = batch::PreparedRows {
        messages,
        documents,
        fragments,
        events,
        texts,
        content,
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
    Vec<crate::encoding::PreparedEvent>,
    crate::text_rows::PreparedTexts,
    crate::content::Groups,
    bool,
);
impl Inner {
    pub(crate) fn replace_chain(
        &mut self,
        input: InputKey,
        chain: i64,
        length: u64,
        mut reading: ChainParse,
    ) -> Result<u64> {
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
            return Err(StoreError::internal(
                "a replacement's decoded bytes are stored before its rows reach the index writer",
            ));
        }
        let mut local_steps = false;
        let mut version = None;
        let mut complete = false;
        let mut refused_or_failed = false;
        reading.visit_source(|step| {
            if version
                .as_ref()
                .is_some_and(|version| version != &step.parse_version)
            {
                return Err(StoreError::Invalid(
                    "one replacement has several parse versions".into(),
                ));
            }
            version.get_or_insert_with(|| step.parse_version.clone());
            refused_or_failed |= step.failure.is_some();
            local_steps |= step.delivery.is_none();
            complete = step.at_end && step.parsed_upto == length;
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

    pub(crate) fn replace_index(
        &mut self,
        input: InputKey,
        chain: i64,
        length: u64,
        reading: ChainParse,
    ) -> Result<u64> {
        let file = std::fs::File::open(&reading.path)?;
        let mut reader = BufReader::new(file);
        let mut all_replayed = true;
        let mut has_step = false;
        while let Some(step) = read_step(&mut reader)? {
            has_step = true;
            if step.delivery.is_none()
                || crate::wire_shapes::admit_export(
                    self.index.get()?,
                    &input.id,
                    chain,
                    length,
                    &step,
                )?
            {
                all_replayed = false;
                break;
            }
        }
        if has_step && all_replayed {
            return Ok(0);
        }
        drop(reader);
        let tx = self
            .index
            .get_mut()?
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::fault::check("write.replay_rows", "")?;
        let counter = bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        let mut touched = super::readings::Touched::default();
        write_reading(
            &tx,
            ReadingWrite {
                register: &self.archive,
                changes: &changes,
            },
            &input,
            chain,
            length,
            &reading,
            &mut touched,
        )?;
        let written = rows::finish(&tx, touched, &changes, counter)?;
        crate::fault::check("write.index_commit", "")?;
        crate::fault::check("reparse.commit", &input.id)?;
        crate::timing::time(crate::timing::Stage::Commit, || tx.commit())?;
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.search.wake();
        self.report.transactions += 1;
        Ok(written)
    }
}

/// What a chain's reading is written with.
pub(crate) struct ReadingWrite<'a> {
    /// Answers which attachment bytes the archive holds.
    pub register: &'a Connection,
    pub changes: &'a crate::change_summary::ChangeSummary,
}

/// Where a chain's steps go as they are parsed, in order from its first
/// record: a reading staged on disk for one replacement transaction
/// ([`ChainParse`]), or a build worker's scratch index
/// ([`super::build::OpenChain`]).
pub trait ChainSink {
    /// The next step of the chain.
    fn step(&mut self, step: &mut ParseStep) -> Result<()>;
    /// Bytes the parse decoded, given before the steps that name them.
    fn decoded(&mut self, batch: crate::staging::PreparedBatch) -> Result<()>;
}

impl ChainSink for ChainParse {
    fn step(&mut self, step: &mut ParseStep) -> Result<()> {
        self.push(step)
    }

    fn decoded(&mut self, batch: crate::staging::PreparedBatch) -> Result<()> {
        ChainParse::decoded(self, batch);
        Ok(())
    }
}

/// A chain's steps written in order into the reading tables of an index.
pub(crate) struct ChainSteps {
    batch: Batch,
    chain: i64,
    /// Attachment references name only bytes the archive holds, as of the
    /// transaction the steps are written in.
    known: HashMap<BlobHash, Option<u64>>,
    first: bool,
}

impl ChainSteps {
    pub(crate) fn new(input: &InputKey, chain: i64, length: u64) -> Self {
        Self {
            batch: Batch {
                input: input.clone(),
                change: Change::Reparse { chain },
                observation: None,
                continues: false,
                new_blob: None,
                blobs: Vec::new(),
                length,
                parse: Vec::new(),
                capture_read: None,
                release: Default::default(),
            },
            chain,
            known: HashMap::new(),
            first: true,
        }
    }

    /// Notes the bytes `step` names before any step is written, so every
    /// step is written against the same view of the archive.
    pub(crate) fn remember(&mut self, w: &ReadingWrite<'_>, step: &ParseStep) -> Result<()> {
        attachments::remember(step, w.register, &HashMap::new(), &mut self.known)
    }

    /// Writes the next step, which resets the chain's reading when it is the
    /// first.
    pub(crate) fn write(
        &mut self,
        tx: &Transaction<'_>,
        w: &ReadingWrite<'_>,
        step: &mut ParseStep,
        touched: &mut super::readings::Touched,
    ) -> Result<()> {
        if self.first && !step.reset {
            return Err(StoreError::Invalid(
                "a chain replacement must start at its first record".into(),
            ));
        }
        if !self.first && step.reset {
            return Err(StoreError::Invalid(
                "a chain replacement resets twice".into(),
            ));
        }
        self.remember(w, step)?;
        attachments::normalize_known(step, &self.known)?;
        let replacement = self.first && step.delivery.is_none();
        self.first = false;
        let source_commit = crate::source::Source::reading_commit(
            tx,
            &self.batch.input.id,
            self.chain,
            step.parsed_upto,
        )?;
        rows::write_step(
            tx,
            &self.batch,
            step,
            rows::StepWrite {
                chain: self.chain,
                source_commit: &source_commit,
                replacement,
                changes: w.changes,
            },
            touched,
        )
    }

    pub(crate) fn is_first(&self) -> bool {
        self.first
    }

    pub(crate) fn finish(&self) -> Result<()> {
        if self.first {
            return Err(StoreError::Invalid(
                "a chain replacement has no reading".into(),
            ));
        }
        Ok(())
    }
}

/// Writes a chain's reading, step by step from its first record, into the
/// reading tables of `tx`'s index, recording what it states in `touched`.
pub(crate) fn write_reading(
    tx: &Transaction<'_>,
    w: ReadingWrite<'_>,
    input: &InputKey,
    chain: i64,
    length: u64,
    reading: &ChainParse,
    touched: &mut super::readings::Touched,
) -> Result<()> {
    let mut steps = ChainSteps::new(input, chain, length);
    let mut reader = BufReader::new(std::fs::File::open(&reading.path)?);
    while let Some(step) = read_step(&mut reader)? {
        steps.remember(&w, &step)?;
    }
    reader.rewind()?;
    while let Some(mut step) = read_step(&mut reader)? {
        steps.write(tx, &w, &mut step, touched)?;
    }
    steps.finish()
}
