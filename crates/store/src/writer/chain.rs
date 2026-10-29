//! Parsed readings wait on disk before their single replacement transaction.
use super::*;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

/// Independently compressed steps keep payload memory bounded while workers
/// resolve all the chains of a confirmed position together.
pub struct ChainReading {
    file: Option<BufWriter<std::fs::File>>,
    path: tempfile::TempPath,
    first: bool,
    decoded: Vec<crate::staging::PreparedBatch>,
    keys: Option<Arc<resolve::ResolvedKeys>>,
    chain: Option<i64>,
    release: InFlight,
}
impl ChainReading {
    pub fn new() -> Result<Self> {
        let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
        Ok(Self {
            file: Some(BufWriter::new(file)),
            path,
            first: true,
            decoded: Vec::new(),
            keys: None,
            chain: None,
            release: InFlight::default(),
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

    /// Prepared steps keep the shared position lease alive until all readings
    /// have been dropped after the transaction result.
    pub(crate) fn visit(&mut self, mut apply: impl FnMut(&ParseStep) -> Result<()>) -> Result<()> {
        self.finish()?;
        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| StoreError::Invalid("a chain reading has no resolved keys".into()))?;
        let chain = self
            .chain
            .ok_or_else(|| StoreError::Invalid("a chain reading has no resolved chain".into()))?;
        let mut reader = BufReader::new(std::fs::File::open(&self.path)?);
        while let Some(mut step) = read_step(&mut reader)? {
            step.prepared.keys = Some(Arc::clone(keys));
            step.prepared.chain = Some(chain);
            apply(&step)?;
        }
        Ok(())
    }
    pub(super) fn resolve(
        &mut self,
        resolver: &Arc<resolve::Resolver>,
        input: &InputKey,
        chain: i64,
        length: u64,
    ) -> Result<()> {
        let archive = crate::archive::open_read(&resolver.location().archive_db())?;
        prepare_readings(
            resolver,
            &mut [Reading {
                input,
                chain,
                length,
                reading: self,
            }],
            ReadingLookup::Archive(&archive),
        )
    }
}

impl resolve::Resolver {
    /// One lease, one snapshot and one source overlay for the entire confirmed
    /// position, even when the position changes several overlapping chains.
    pub(crate) fn confirmed_readings(
        self: &Arc<Self>,
        readings: &mut [(InputKey, i64, u64, ChainReading)],
        available: &HashMap<BlobHash, Option<u64>>,
    ) -> Result<()> {
        let mut readings: Vec<_> = readings
            .iter_mut()
            .map(|(input, chain, length, reading)| Reading {
                input,
                chain: *chain,
                length: *length,
                reading,
            })
            .collect();
        prepare_readings(self, &mut readings, ReadingLookup::Confirmed(available))
    }
}
struct Reading<'a> {
    input: &'a InputKey,
    chain: i64,
    length: u64,
    reading: &'a mut ChainReading,
}
#[derive(Clone, Copy)]
enum ReadingLookup<'a> {
    Archive(&'a Connection),
    Confirmed(&'a HashMap<BlobHash, Option<u64>>),
}

fn prepare_readings(
    resolver: &Arc<resolve::Resolver>,
    readings: &mut [Reading<'_>],
    source: ReadingLookup<'_>,
) -> Result<()> {
    let mut distinct = BTreeSet::new();
    for reading in readings.iter_mut() {
        if reading.chain < 1 || !distinct.insert((reading.input.id.clone(), reading.chain)) {
            return Err(StoreError::Invalid(
                "a position must name each valid input chain once".into(),
            ));
        }
        reading.reading.finish()?;
        if reading.reading.first {
            return Err(StoreError::Invalid(
                "a chain replacement has no reading".into(),
            ));
        }
    }
    if readings.is_empty() {
        return Ok(());
    }
    let pending: HashMap<_, _> = readings
        .iter()
        .flat_map(|reading| &reading.reading.decoded)
        .flat_map(|item| item.batch.new_blob.iter().chain(&item.batch.blobs))
        .map(|blob| (blob.hash, blob.size))
        .collect();
    loop {
        let index = crate::archive::open_read(&resolver.location().index_db())?;
        let (keys, _) = gather(resolver, readings, &index, source, &pending)?;
        drop(index);
        let lease = resolver.lease(keys.clone());
        let index = crate::archive::open_read(&resolver.location().index_db())?;
        let snapshot = index.unchecked_transaction()?;
        let (fresh, history) = gather(resolver, readings, &snapshot, source, &pending)?;
        if keys != fresh {
            drop(snapshot);
            drop(lease);
            continue;
        }
        let mut resolved = resolve::ResolvedKeys::default();
        for key in keys {
            let row = resolver.resolve_key(&snapshot, &key)?;
            resolved.rows.insert(key, row);
        }
        let keys = Arc::new(resolved);
        let mut projection =
            resolve::versions::Projection::new(resolver, &snapshot, &keys, Some(&history))?;
        let mut prepared = Vec::new();
        for reading in readings.iter() {
            let input = keys
                .get(resolve::Kind::Input, reading.input.id.as_bytes())?
                .pk;
            let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
            let mut output = BufWriter::new(file);
            let location = history.add_path(&path);
            let mut reader = BufReader::new(std::fs::File::open(&reading.reading.path)?);
            while let Some(mut step) = read_step(&mut reader)? {
                attachments::normalize_known(&mut step, &history.available)?;
                projection.step(reading.input, reading.chain, &mut step)?;
                let offset = output.stream_position()?;
                write_step(&mut output, &step)?;
                output.flush()?;
                history.record(
                    input,
                    reading.chain,
                    &step,
                    Location {
                        path: location,
                        offset,
                    },
                );
            }
            drop(output.into_inner().map_err(|error| error.into_error())?);
            prepared.push(path);
        }
        drop(projection);
        drop(snapshot);
        let lease = Arc::new(lease);
        for (reading, path) in readings.iter_mut().zip(prepared) {
            reading.reading.path = path;
            reading.reading.keys = Some(Arc::clone(&keys));
            reading.reading.chain = Some(reading.chain);
            let lease = Arc::clone(&lease);
            reading.reading.release = InFlight::new(move || drop(lease));
        }
        return Ok(());
    }
}

fn gather<'a>(
    resolver: &resolve::Resolver,
    readings: &[Reading<'_>],
    index: &Connection,
    source: ReadingLookup<'a>,
    pending: &HashMap<BlobHash, u64>,
) -> Result<(BTreeSet<resolve::Key>, History<'a>)> {
    let mut keys = BTreeSet::new();
    let mut history = History::new(source);
    for reading in readings {
        let path = history.add_path(&reading.reading.path);
        let mut reader = BufReader::new(std::fs::File::open(&reading.reading.path)?);
        loop {
            let offset = reader.stream_position()?;
            let Some(mut step) = read_step(&mut reader)? else {
                break;
            };
            if let ReadingLookup::Archive(archive) = source {
                attachments::remember(&step, archive, pending, history.available.to_mut())?;
            }
            attachments::normalize_known(&mut step, &history.available)?;
            let location = Location { path, offset };
            for prepared in &step.prepared.messages {
                history.messages.insert(prepared.hash.to_vec(), location);
            }
            for prepared in &step.prepared.documents {
                history.documents.insert(prepared.hash.to_vec(), location);
            }
            let mut batch = Batch {
                input: reading.input.clone(),
                change: Change::Reparse {
                    chain: reading.chain,
                },
                observation: None,
                continues: false,
                new_blob: None,
                blobs: Vec::new(),
                length: reading.length,
                parse: vec![step],
                release: InFlight::default(),
            };
            let (step_keys, _) = resolver.gather(&mut [&mut batch], index, index)?;
            keys.extend(step_keys);
        }
    }
    Ok((keys, history))
}

fn write_step(file: &mut BufWriter<std::fs::File>, step: &ParseStep) -> Result<()> {
    let start = file.stream_position()?;
    file.write_all(&0u64.to_le_bytes())?;
    let mut encoder = zstd::stream::write::Encoder::new(&mut *file, 1)?;
    let texts: Vec<_> = step.prepared.texts.iter().collect();
    serde_json::to_writer(
        &mut encoder,
        &(
            step,
            &step.prepared.messages,
            &step.prepared.documents,
            &step.prepared.events,
            texts,
            step.prepared.released,
            &step.prepared.choices,
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
    let StoredStep(mut step, messages, documents, events, texts, released, choices) =
        crate::encoding::read_json(&bytes)?;
    step.prepared = batch::PreparedRows {
        messages,
        documents,
        events,
        texts: texts.into_iter().collect(),
        released,
        keys: None,
        chain: None,
        choices,
    };
    Ok(Some(step))
}

#[derive(serde::Deserialize)]
struct StoredStep(
    ParseStep,
    Vec<PreparedMessage>,
    Vec<PreparedDocument>,
    Vec<(String, Vec<u8>)>,
    Vec<([u8; 32], Vec<u8>)>,
    bool,
    Option<resolve::versions::Choices>,
);
#[derive(Clone, Copy)]
struct Location {
    path: usize,
    offset: u64,
}
struct History<'a> {
    paths: std::cell::RefCell<Vec<std::path::PathBuf>>,
    messages: BTreeMap<Vec<u8>, Location>,
    documents: BTreeMap<Vec<u8>, Location>,
    assertions: std::cell::RefCell<BTreeMap<(resolve::Kind, i64), Location>>,
    sessions: std::cell::RefCell<BTreeMap<(i64, i64, i64), Location>>,
    available: Cow<'a, HashMap<BlobHash, Option<u64>>>,
}
impl<'a> History<'a> {
    fn new(source: ReadingLookup<'a>) -> Self {
        Self {
            paths: std::cell::RefCell::new(Vec::new()),
            messages: BTreeMap::new(),
            documents: BTreeMap::new(),
            assertions: std::cell::RefCell::new(BTreeMap::new()),
            sessions: std::cell::RefCell::new(BTreeMap::new()),
            available: match source {
                ReadingLookup::Archive(_) => Cow::Owned(HashMap::new()),
                ReadingLookup::Confirmed(available) => Cow::Borrowed(available),
            },
        }
    }
    fn add_path(&self, path: &std::path::Path) -> usize {
        let mut paths = self.paths.borrow_mut();
        let index = paths.len();
        paths.push(path.to_path_buf());
        index
    }
    fn record(&self, input: i64, chain: i64, step: &ParseStep, location: Location) {
        if let Some(choices) = &step.prepared.choices {
            for pk in choices.sessions.keys() {
                self.sessions
                    .borrow_mut()
                    .insert((*pk, input, chain), location);
            }
            let mut assertions = self.assertions.borrow_mut();
            for value in &choices.event_values {
                assertions.insert((resolve::Kind::EventValue, value.key.pk), location);
            }
            for value in &choices.link_values {
                assertions.insert((resolve::Kind::LinkValue, value.key.pk), location);
            }
        }
    }
    fn step(&self, location: Location) -> Result<ParseStep> {
        let path = self.paths.borrow()[location.path].clone();
        let mut reader = BufReader::new(std::fs::File::open(path)?);
        reader.seek(SeekFrom::Start(location.offset))?;
        let mut step = read_step(&mut reader)?
            .ok_or_else(|| StoreError::Invalid("a scratch offset has no chain step".into()))?;
        attachments::normalize_known(&mut step, &self.available)?;
        Ok(step)
    }
}
impl resolve::versions::ValueSource for History<'_> {
    fn session(&self, pk: i64, input: i64, chain: i64) -> Result<Option<Session>> {
        let Some(location) = self.sessions.borrow().get(&(pk, input, chain)).copied() else {
            return Ok(None);
        };
        Ok(self
            .step(location)?
            .prepared
            .choices
            .and_then(|mut choices| choices.sessions.remove(&pk)))
    }
    fn assertion(&self, kind: resolve::Kind, pk: i64) -> Result<Option<Vec<u8>>> {
        let Some(location) = self.assertions.borrow().get(&(kind, pk)).copied() else {
            return Ok(None);
        };
        let choices =
            self.step(location)?.prepared.choices.ok_or_else(|| {
                StoreError::Invalid("a value offset has no prepared choices".into())
            })?;
        let values = if kind == resolve::Kind::EventValue {
            choices.event_values
        } else {
            choices.link_values
        };
        Ok(values
            .into_iter()
            .find(|value| value.key.pk == pk)
            .map(|value| value.body))
    }
    fn message(&self, hash: &[u8]) -> Result<Option<(Message, PreparedMessage)>> {
        let Some(location) = self.messages.get(hash).copied() else {
            return Ok(None);
        };
        let step = self.step(location)?;
        Ok(step
            .output
            .messages
            .into_iter()
            .zip(step.prepared.messages)
            .find(|(_, prepared)| prepared.hash.as_slice() == hash)
            .map(|(row, prepared)| (row.message, prepared)))
    }
    fn document(
        &self,
        hash: &[u8],
    ) -> Result<Option<(polylogue_model::Document, PreparedDocument)>> {
        let Some(location) = self.documents.get(hash).copied() else {
            return Ok(None);
        };
        let step = self.step(location)?;
        Ok(step
            .output
            .documents
            .into_iter()
            .zip(step.prepared.documents)
            .find(|(_, prepared)| prepared.hash.as_slice() == hash)
            .map(|(row, prepared)| (row.document, prepared)))
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
        if !reading.decoded.is_empty() {
            let store = Arc::new(self.location.blobs()?);
            for result in self.store_group(std::mem::take(&mut reading.decoded), store) {
                result?;
            }
        }
        reading.finish()?;
        let file = std::fs::File::open(&reading.path)?;
        let mut reader = BufReader::new(file);
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
        let mut written = 0;
        let mut first = true;
        while let Some(mut step) = read_step(&mut reader)? {
            step.prepared.keys = reading.keys.clone();
            step.prepared.chain = Some(chain);
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
                self.first_build.as_deref(),
            )?;
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
