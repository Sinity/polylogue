//! A chain's parsed output waits on disk before its one replacement transaction.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

/// Each compressed step is independently addressable. The worker retains only
/// hash-to-offset references when a later record selects an earlier reading.
pub struct ChainReading {
    file: Option<BufWriter<std::fs::File>>,
    path: tempfile::TempPath,
    first: bool,
    decoded: Vec<crate::staging::PreparedBatch>,
    keys: Option<Arc<resolve::ResolvedKeys>>,
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

    pub(super) fn resolve(
        &mut self,
        resolver: &Arc<resolve::Resolver>,
        input: &InputKey,
        chain: i64,
        length: u64,
    ) -> Result<()> {
        self.finish()?;
        loop {
            let index = crate::archive::open_read(&resolver.location().index_db())?;
            let archive = crate::archive::open_read(&resolver.location().archive_db())?;
            let (keys, _) = self.gather(resolver, input, chain, length, &index, &archive)?;
            drop(index);
            drop(archive);
            let lease = resolver.lease(keys.clone());
            let index = crate::archive::open_read(&resolver.location().index_db())?;
            let snapshot = index.unchecked_transaction()?;
            let archive = crate::archive::open_read(&resolver.location().archive_db())?;
            let (fresh, history) =
                self.gather(resolver, input, chain, length, &snapshot, &archive)?;
            if keys != fresh {
                drop(snapshot);
                drop(lease);
                continue;
            }
            let mut resolved = resolve::ResolvedKeys::default();
            for key in keys {
                let existing = resolver.find(&snapshot, &key)?;
                let row = resolve::RowKey {
                    pk: match existing {
                        Some(pk) => pk,
                        None => resolver.allocate(key.kind)?,
                    },
                    existing: existing.is_some(),
                };
                resolved.rows.insert(key, row);
            }
            let keys = Arc::new(resolved);
            let mut projection =
                resolve::versions::Projection::new(resolver, &snapshot, &keys, Some(&history))?;
            let (file, path) = tempfile::NamedTempFile::new()?.into_parts();
            let mut output = BufWriter::new(file);
            let mut reader = BufReader::new(std::fs::File::open(&self.path)?);
            let mut values = Vec::new();
            while let Some(mut step) = read_step(&mut reader)? {
                let pending:HashMap<_,_>=self.decoded.iter().flat_map(|item|item.batch.new_blob.iter().chain(&item.batch.blobs)).map(|blob|(blob.hash,blob.size)).collect();
                let mut batch=Batch {input:input.clone(),change:Change::Reparse{chain},observation:None,continues:false,new_blob:None,blobs:Vec::new(),length,parse:vec![step],release:InFlight::default()};
                attachments::normalize(&mut batch,&archive,&pending)?;
                step=batch.parse.pop().expect("one chain step");
                projection.step(input, chain, &mut step)?;
                if let Some(choices) = &step.prepared.choices {
                    values.extend(
                        choices
                            .event_values
                            .iter()
                            .chain(&choices.link_values)
                            .map(|value| Arc::clone(&value.key)),
                    );
                }
                write_step(&mut output, &step)?;
            }
            drop(output.into_inner().map_err(|error| error.into_error())?);
            drop(projection);
            drop(snapshot);
            self.path = path;
            self.keys = Some(keys);
            self.release = InFlight::new(move || {
                drop(values);
                drop(lease);
            });
            return Ok(());
        }
    }

    fn gather(
        &self,
        resolver: &resolve::Resolver,
        input: &InputKey,
        chain: i64,
        length: u64,
        index: &Connection,
        archive: &Connection,
    ) -> Result<(BTreeSet<resolve::Key>, History)> {
        let mut keys = BTreeSet::new();
        let mut history = History {
            path: self.path.to_path_buf(),
            messages: BTreeMap::new(),
            documents: BTreeMap::new(),
        };
        let mut reader = BufReader::new(std::fs::File::open(&self.path)?);
        loop {
            let offset = reader.stream_position()?;
            let Some(step) = read_step(&mut reader)? else {
                break;
            };
            for prepared in &step.prepared.messages {
                history.messages.insert(prepared.hash.to_vec(), offset);
            }
            for prepared in &step.prepared.documents {
                history.documents.insert(prepared.hash.to_vec(), offset);
            }
            let mut batch = Batch {
                input: input.clone(),
                change: Change::Reparse { chain },
                observation: None,
                continues: false,
                new_blob: None,
                blobs: Vec::new(),
                length,
                parse: vec![step],
                release: InFlight::default(),
            };
            let (step_keys, _) = resolver.gather(&mut [&mut batch], index, archive)?;
            keys.extend(step_keys);
        }
        Ok((keys, history))
    }
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
struct History {
    path: std::path::PathBuf,
    messages: BTreeMap<Vec<u8>, u64>,
    documents: BTreeMap<Vec<u8>, u64>,
}
impl History {
    fn step(&self, offset: u64) -> Result<ParseStep> {
        let mut reader = BufReader::new(std::fs::File::open(&self.path)?);
        reader.seek(SeekFrom::Start(offset))?;
        read_step(&mut reader)?
            .ok_or_else(|| StoreError::Invalid("a version offset has no chain step".into()))
    }
}
impl resolve::versions::ValueSource for History {
    fn message(&self, hash: &[u8]) -> Result<Option<(Message, PreparedMessage)>> {
        let Some(offset) = self.messages.get(hash) else {
            return Ok(None);
        };
        let step = self.step(*offset)?;
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
        let Some(offset) = self.documents.get(hash) else {
            return Ok(None);
        };
        let step = self.step(*offset)?;
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
