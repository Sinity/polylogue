//! Parsed readings wait on disk before their single replacement transaction.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

/// Independently compressed parser steps bound memory before one replacement transaction.
pub struct ChainReading {
    file: Option<BufWriter<std::fs::File>>,
    path: tempfile::TempPath,
    first: bool,
    decoded: Vec<crate::staging::PreparedBatch>,
    resolved: Option<ResolvedChain>,
    release: InFlight,
    /// In primary mode, its parsed events are in the producer spool.
    spooled: bool,
}

/// The row identities a chain reading's steps were resolved with, and the
/// chain they were resolved for.
struct ResolvedChain {
    keys: Arc<resolve::ResolvedKeys>,
    chain: i64,
}
impl ChainReading {
    pub fn new() -> Result<Self> {
        let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
        Ok(Self {
            file: Some(BufWriter::new(file)),
            path,
            first: true,
            decoded: Vec::new(),
            resolved: None,
            release: InFlight::default(),
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
            None,
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
        while let Some((step, _)) = read_step(&mut input)? {
            apply(&step)?;
        }
        Ok(())
    }

    /// Its parsed events were prepared in the producer spool.
    pub(super) fn mark_spooled(&mut self) {
        self.spooled = true;
    }

    fn resolved_for(&self, chain: i64) -> Result<&ResolvedChain> {
        self.resolved
            .as_ref()
            .filter(|resolved| resolved.chain == chain)
            .ok_or_else(|| {
                StoreError::Invalid("a chain reading was not resolved for its chain".into())
            })
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
            &archive,
        )
    }
}

struct Reading<'a> {
    input: &'a InputKey,
    chain: i64,
    length: u64,
    reading: &'a mut ChainReading,
}

fn prepare_readings(
    resolver: &Arc<resolve::Resolver>,
    readings: &mut [Reading<'_>],
    archive: &Connection,
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
        let (keys, _, _) = gather(resolver, readings, &index, archive, &pending)?;
        drop(index);
        let key_lock = resolver.lock_keys(keys.clone());
        let index = crate::archive::open_read(&resolver.location().index_db())?;
        let snapshot = index.unchecked_transaction()?;
        let (fresh, history, recorded) = gather(resolver, readings, &snapshot, archive, &pending)?;
        if keys != fresh {
            drop(snapshot);
            drop(key_lock);
            continue;
        }
        let mut resolved = resolve::ResolvedKeys {
            identities: crate::session_alias::Identities::load(&snapshot, &recorded)?,
            ..resolve::ResolvedKeys::default()
        };
        for key in keys {
            let row = resolver.resolve_key(&snapshot, &key)?;
            resolved.rows.insert(key, row);
        }
        let keys = Arc::new(resolved);
        let mut membership_state =
            resolve::membership::MembershipState::new(resolver, &snapshot, &keys, Some(&history))?;
        let mut prepared = Vec::new();
        for reading in readings.iter() {
            let input = keys
                .get(resolve::Kind::Input, reading.input.id.as_bytes())?
                .pk;
            let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
            let mut output = BufWriter::new(file);
            let location = history.add_path(&path);
            let mut reader = BufReader::new(std::fs::File::open(&reading.reading.path)?);
            while let Some((mut step, _)) = read_step(&mut reader)? {
                attachments::normalize_known(&mut step, &history.available)?;
                let choices = membership_state.step(reading.input, reading.chain, &mut step)?;
                let offset = output.stream_position()?;
                write_step(&mut output, &step, Some(&choices))?;
                output.flush()?;
                history.record(
                    input,
                    reading.chain,
                    &choices,
                    Location {
                        path: location,
                        offset,
                    },
                );
            }
            drop(output.into_inner().map_err(|error| error.into_error())?);
            prepared.push(path);
        }
        drop(membership_state);
        drop(snapshot);
        let key_lock = Arc::new(key_lock);
        for (reading, path) in readings.iter_mut().zip(prepared) {
            reading.reading.path = path;
            reading.reading.resolved = Some(ResolvedChain {
                keys: Arc::clone(&keys),
                chain: reading.chain,
            });
            let key_lock = Arc::clone(&key_lock);
            reading.reading.release = InFlight::new(move || drop(key_lock));
        }
        return Ok(());
    }
}

fn gather(
    resolver: &resolve::Resolver,
    readings: &[Reading<'_>],
    index: &Connection,
    archive: &Connection,
    pending: &HashMap<BlobHash, u64>,
) -> Result<(
    BTreeSet<resolve::Key>,
    History,
    Vec<polylogue_model::Session>,
)> {
    let mut keys = BTreeSet::new();
    let mut recorded = Vec::new();
    let mut history = History::new();
    for reading in readings {
        let path = history.add_path(&reading.reading.path);
        let mut reader = BufReader::new(std::fs::File::open(&reading.reading.path)?);
        loop {
            let offset = reader.stream_position()?;
            let Some((mut step, _)) = read_step(&mut reader)? else {
                break;
            };
            attachments::remember(&step, archive, pending, &mut history.available)?;
            attachments::normalize_known(&mut step, &history.available)?;
            let location = Location { path, offset };
            recorded.extend(
                step.output
                    .sessions
                    .iter()
                    .filter(|s| !s.aliases.is_empty())
                    .cloned(),
            );
            for prepared in &step.prepared.messages {
                history
                    .messages
                    .borrow_mut()
                    .insert(prepared.hash.to_vec(), location);
            }
            for prepared in &step.prepared.documents {
                history.documents.insert(prepared.hash.to_vec(), location);
            }
            let mut batch = Batch {
                required_outcome: None,
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
                acquisition: None,
                basis: None,
                release: InFlight::default(),
            };
            let (step_keys, _) = resolver.gather(&mut [&mut batch], index, index)?;
            keys.extend(step_keys);
        }
    }
    Ok((keys, history, recorded))
}

/// Writes one step record; a resolved reading's records carry the versions
/// chosen for the step.
fn write_step(
    file: &mut BufWriter<std::fs::File>,
    step: &ParseStep,
    choices: Option<&resolve::membership::Choices>,
) -> Result<()> {
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
            choices,
        ),
    )?;
    encoder.finish()?;
    let end = file.stream_position()?;
    file.seek(SeekFrom::Start(start))?;
    file.write_all(&(end - start - 8).to_le_bytes())?;
    file.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn read_step(
    reader: &mut BufReader<std::fs::File>,
) -> Result<Option<(ParseStep, Option<resolve::membership::Choices>)>> {
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
    let StoredStep(mut step, messages, documents, fragments, events, texts, released, choices) =
        crate::encoding::read_json(&bytes)?;
    step.prepared = batch::PreparedRows {
        messages,
        documents,
        fragments,
        events,
        texts,
        released,
    };
    Ok(Some((step, choices)))
}

/// Reads one record of a resolved reading together with its resolution.
fn read_resolved_step(
    reader: &mut BufReader<std::fs::File>,
    resolved: &ResolvedChain,
) -> Result<Option<(ParseStep, resolve::Resolution)>> {
    let Some((step, choices)) = read_step(reader)? else {
        return Ok(None);
    };
    let choices = choices.ok_or_else(|| {
        StoreError::Invalid("a resolved chain step has no chosen versions".into())
    })?;
    Ok(Some((
        step,
        resolve::Resolution {
            keys: Arc::clone(&resolved.keys),
            chain: resolved.chain,
            choices,
        },
    )))
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
    Option<resolve::membership::Choices>,
);
#[derive(Clone, Copy)]
struct Location {
    path: usize,
    offset: u64,
}
struct History {
    paths: std::cell::RefCell<Vec<std::path::PathBuf>>,
    messages: std::cell::RefCell<BTreeMap<Vec<u8>, Location>>,
    documents: BTreeMap<Vec<u8>, Location>,
    assertions: std::cell::RefCell<BTreeMap<(resolve::Kind, i64), Location>>,
    sessions: std::cell::RefCell<BTreeMap<(i64, i64, i64), Location>>,
    event_sources: std::cell::RefCell<BTreeMap<(i64, i64, i64), Location>>,
    available: HashMap<BlobHash, Option<u64>>,
}
impl History {
    fn new() -> Self {
        Self {
            paths: std::cell::RefCell::new(Vec::new()),
            messages: std::cell::RefCell::new(BTreeMap::new()),
            documents: BTreeMap::new(),
            assertions: std::cell::RefCell::new(BTreeMap::new()),
            sessions: std::cell::RefCell::new(BTreeMap::new()),
            event_sources: std::cell::RefCell::new(BTreeMap::new()),
            available: HashMap::new(),
        }
    }
    fn add_path(&self, path: &std::path::Path) -> usize {
        let mut paths = self.paths.borrow_mut();
        let index = paths.len();
        paths.push(path.to_path_buf());
        index
    }
    fn record(
        &self,
        input: i64,
        chain: i64,
        choices: &resolve::membership::Choices,
        location: Location,
    ) {
        {
            for hash in choices.message_versions.keys() {
                self.messages.borrow_mut().insert(hash.0.to_vec(), location);
            }
            for pk in choices.sessions.keys() {
                self.sessions
                    .borrow_mut()
                    .insert((*pk, input, chain), location);
            }
            for (pk, _) in &choices.event_sources {
                self.event_sources
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
    fn step(
        &self,
        location: Location,
    ) -> Result<(ParseStep, Option<resolve::membership::Choices>)> {
        let path = self.paths.borrow()[location.path].clone();
        let mut reader = BufReader::new(std::fs::File::open(path)?);
        reader.seek(SeekFrom::Start(location.offset))?;
        let (mut step, choices) =
            read_step(&mut reader)?.or_invalid("a scratch offset has no chain step")?;
        attachments::normalize_known(&mut step, &self.available)?;
        Ok((step, choices))
    }
}
impl resolve::membership::ValueSource for History {
    fn session(&self, pk: i64, input: i64, chain: i64) -> Result<Option<Session>> {
        let Some(location) = self.sessions.borrow().get(&(pk, input, chain)).copied() else {
            return Ok(None);
        };
        Ok(self
            .step(location)?
            .1
            .and_then(|mut choices| choices.sessions.remove(&pk)))
    }
    fn event_source(
        &self,
        pk: i64,
        input: i64,
        chain: i64,
    ) -> Result<Option<resolve::membership::EventChoice>> {
        let Some(location) = self
            .event_sources
            .borrow()
            .get(&(pk, input, chain))
            .copied()
        else {
            return Ok(None);
        };
        Ok(self.step(location)?.1.and_then(|choices| {
            choices
                .event_sources
                .into_iter()
                .rev()
                .find(|(event, _)| *event == pk)
                .map(|(_, source)| source)
        }))
    }
    fn assertion(&self, kind: resolve::Kind, pk: i64) -> Result<Option<Vec<u8>>> {
        let Some(location) = self.assertions.borrow().get(&(kind, pk)).copied() else {
            return Ok(None);
        };
        let choices = self
            .step(location)?
            .1
            .or_invalid("a value offset has no prepared choices")?;
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
        let Some(location) = self.messages.borrow().get(hash).copied() else {
            return Ok(None);
        };
        let (step, choices) = self.step(location)?;
        if let Some(mut choices) = choices {
            let hash = BlobHash(hash.try_into().map_err(|_| {
                StoreError::Invalid("a recorded message version has an invalid hash".into())
            })?);
            if let Some(value) = choices.message_versions.remove(&hash) {
                let resolve::membership::MessageValue { message, prepared } =
                    Arc::unwrap_or_clone(value.value);
                return Ok(Some((message, prepared)));
            }
        }
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
        let (step, _) = self.step(location)?;
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
        reading.finish()?;
        let mut local_steps = false;
        let mut version = None;
        let mut complete = false;
        let mut diagnosed = false;
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
            diagnosed |= matches!(
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
        if diagnosed {
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
        while let Some((step, _)) = read_step(&mut reader)? {
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
        reader.rewind()?;
        let text_transaction = self.texts.transaction();
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::fault::check("write.replay_rows", "")?;
        let counter = bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        let mut batch = Batch {
            required_outcome: None,
            input,
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
        let resolved = reading.resolved_for(chain)?;
        let mut written = 0;
        let mut first = true;
        while let Some((step, resolution)) = read_resolved_step(&mut reader, resolved)? {
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
            let replace_wire = first && step.delivery.is_none();
            first = false;
            let source_commit = crate::source::Source::reading_commit(
                &tx,
                &batch.input.id,
                chain,
                step.parsed_upto,
            )?;
            let write = if replace_wire {
                rows::write_replacement_rows
            } else {
                write_rows
            };
            written += write(
                &tx,
                &batch,
                &step,
                rows::ReadingWrite {
                    texts: &self.texts,
                    resolution: &resolution,
                    counter,
                    first_build: self.first_build.as_deref(),
                    source_commit: Some(&source_commit),
                    changes: &changes,
                    retraction: false,
                    replacement: false,
                },
            )?;
        }
        if first {
            return Err(StoreError::Invalid(
                "a chain replacement has no reading".into(),
            ));
        }
        rows::prune_sessions(&tx, &batch.input.id, chain, counter)?;
        self.report.item_units_read += rows::finish(
            &tx,
            &changes,
            self.first_build.as_deref(),
            &self.texts,
            counter,
        )?;
        batch.parse.clear();
        crate::fault::check("write.index_commit", "")?;
        crate::fault::check("reparse.commit", &batch.input.id)?;
        crate::timing::time(crate::timing::Stage::Commit, || tx.commit())?;
        text_transaction.commit();
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.search.wake();
        self.report.transactions += 1;
        Ok(written)
    }
}
