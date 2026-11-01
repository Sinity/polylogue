//! A compressed rollout keeps its original encoding as a byte input and
//! its decoded transcript at the same location identity as the plain file.
//! Re-encoding identical content adds no transcript revision.

use crate::plan::{InputState, Plan};
use crate::*;
use polylogue_store::staging::{PreparedBatch, PreparedBytes};
use std::io::{Read, Seek};

impl Engine {
    pub(crate) fn capture_compressed(
        &mut self,
        found: &collect::Found,
        facts: FileFacts,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        let path = &found.path;
        let decoded_path = path.with_extension("");
        if self.refused.contains(&decoded_path) {
            return Ok(None);
        }
        let mut file = std::fs::File::open(path)?;
        let original = self.archive.prepare_bytes(&mut file)?;
        file.rewind()?;
        let decoded = self
            .archive
            .prepare_bytes(zstd::stream::read::Decoder::new(&mut file)?)?;
        if FileFacts::of(&file.metadata()?) != facts || original.len() != facts.size {
            return Err(std::io::Error::other("source changed during compressed capture").into());
        }
        let decoded_id = self.location_id(&decoded_path);
        let reader = self.reads.reader()?;
        let previous = reader.input_progress(&decoded_id)?;
        let same = match &previous {
            Some(previous) if previous.length == decoded.len() => {
                let stored = reader.stored_chain(&decoded_id, previous.chain, previous.length)?;
                let mut bytes = stored.reader(Arc::clone(&self.blobs));
                let mut hasher = blake3::Hasher::new();
                let mut buffer = vec![0; 1 << 16];
                loop {
                    let n = bytes.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buffer[..n]);
                }
                hasher.finalize().as_bytes() == &decoded.hash().0
            }
            _ => false,
        };
        drop(reader);
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let change = if same {
            Change::Unchanged
        } else if previous.is_some() {
            Change::Rewritten
        } else {
            Change::First
        };
        let raw_change = if self.inputs.get(path).is_some_and(|s| s.stored > 0) {
            Change::Rewritten
        } else {
            Change::First
        };
        let raw_size = original.len();
        let decoded_size = decoded.len();
        let raw = self.compressed_batch(path, found.origin, raw_change, Some(facts), original);
        let decoded_facts = std::fs::metadata(&decoded_path).ok().map(|m| FileFacts::of(&m));
        let mut transcript =
            self.compressed_batch(&decoded_path, origin, change, decoded_facts, decoded);
        // The physical file is compressed. File facts are used only by its
        // byte input; the transcript is read from its stored decoded bytes.
        if same {
            transcript.batch.new_blob = None;
            transcript.files.clear();
        }
        let key = transcript.batch.input.clone();
        let outcomes = self
            .writer
            .sender()?
            .store_group(vec![raw, transcript], Arc::clone(&self.blobs))?;
        let mut outcomes = outcomes.into_iter();
        let raw = outcomes.next().expect("raw outcome")?;
        let transcript = outcomes.next().expect("transcript outcome")?;
        pass.bytes_read += raw_size;
        pass.bytes_stored += raw_size + if same { 0 } else { decoded_size };
        for (at, id, format, stored) in [
            (path, raw.input.clone(), found.origin, &raw),
            (&decoded_path, key.id.clone(), origin, &transcript),
        ] {
            self.inputs.insert(
                at.clone(),
                InputState {
                    input_id: id,
                    origin: format,
                    facts: if at == path { Some(facts) } else { decoded_facts },
                    stored: stored.length,
                    parsed_upto: if same && at == &decoded_path { previous.as_ref().map_or(0, |p| p.parsed_upto) } else { 0 },
                    parsed: format.support != Support::Parsed || same && previous.as_ref().is_some_and(|p| p.parsed),
                    waiting: same && previous.as_ref().is_some_and(|p| p.waiting),
                    parser_state: None,
                    tail: None,
                    chain: Some(stored.chain),
                    failed: None,
                    failed_history: HashMap::new(),
                },
            );
        }
        let plan = self.stored_plan(&decoded_path)?;
        Ok(plan)
    }

    fn compressed_batch(
        &self,
        path: &Path,
        origin: &'static OriginSpec,
        change: Change,
        facts: Option<FileFacts>,
        prepared: PreparedBytes,
    ) -> PreparedBatch {
        PreparedBatch {
            batch: polylogue_store::Batch {
                input: InputKey {
                    id: self.location_id(path),
                    format: origin.name.into(),
                    arrival_rank: 0,
                },
                change,
                observation: Some(self.observation(path, facts)),
                continues: false,
                length: prepared.len(),
                new_blob: Some(polylogue_store::NewBlob {
                    hash: prepared.hash(),
                    size: prepared.len(),
                    stored: polylogue_store::blobs::Stored::File { stored_size: 0 },
                }),
                blobs: Vec::new(),
                parse: Vec::new(),
                release: Default::default(),
            },
            files: vec![prepared],
        }
    }
}

#[cfg(test)]
mod tests;
