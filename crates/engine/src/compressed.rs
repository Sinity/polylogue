//! Rollouts share recorded identities across locations and encodings. Capturing
//! their bytes finishes before parsing, so an encoding never changes anchors.
use crate::plan::{InputState, Plan};
use crate::*;
use polylogue_store::staging::{PreparedBatch, PreparedBytes};
use std::io::{BufReader, Read, Seek, SeekFrom};

pub(crate) fn is_rollout(found: &collect::Found) -> bool {
    found.origin.name == "codex-compressed"
        || found.origin.name == "codex"
            && matches!(
                polylogue_sources::codex::File::from_path(&found.path),
                Some(polylogue_sources::codex::File::Rollout { .. })
            )
}

impl Engine {
    pub(crate) fn capture_rollout(
        &mut self,
        found: &collect::Found,
        facts: FileFacts,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        let path = &found.path;
        if self
            .inputs
            .get(path)
            .is_some_and(|s| s.facts == Some(facts))
        {
            return self.stored_plan(path);
        }
        std::fs::create_dir_all(self.archive.staging_dir())?;
        let mut source = std::fs::File::open(path)?;
        let compressed = found.origin.name == "codex-compressed";
        let raw = if compressed {
            Some(self.archive.prepare_bytes(&mut source)?)
        } else {
            None
        };
        source.rewind()?;
        let mut decoded = tempfile::tempfile_in(self.archive.staging_dir())?;
        if compressed {
            std::io::copy(
                &mut zstd::stream::read::Decoder::new(&mut source)?,
                &mut decoded,
            )?;
        } else {
            std::io::copy(&mut source, &mut decoded)?;
        }
        if FileFacts::of(&source.metadata()?) != facts || source.stream_position()? != facts.size {
            return Err(std::io::Error::other("source changed during rollout capture").into());
        }
        decoded.rewind()?;
        let native = polylogue_sources::codex::rollout_id(BufReader::new(&mut decoded))?;
        decoded.rewind()?;
        let bytes = self.archive.prepare_bytes(&mut decoded)?;
        let id = native
            .map(|native| format!("codex:rollout:{native}"))
            .unwrap_or_else(|| bytes.hash().to_string());
        self.capture_local_files(&mut decoded, &id, pass)?;
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let reader = self.reads.reader()?;
        let previous = reader.input_progress(&id)?;
        let same = if let Some(previous) = &previous
            && previous.length == bytes.len()
        {
            let chain = reader.stored_chain(&id, previous.chain, previous.length)?;
            let mut stored = chain.reader(Arc::clone(&self.blobs));
            let mut hash = blake3::Hasher::new();
            let mut buffer = [0; 65536];
            loop {
                let n = stored.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            hash.finalize().as_bytes() == &bytes.hash().0
        } else {
            false
        };
        let appended = if let Some(previous) = &previous
            && previous.length < bytes.len()
        {
            let chain = reader.stored_chain(&id, previous.chain, previous.length)?;
            let mut stored = chain.reader(Arc::clone(&self.blobs));
            decoded.rewind()?;
            let mut prefix = (&mut decoded).take(previous.length);
            let mut left = [0; 65536];
            let mut right = [0; 65536];
            loop {
                let n = stored.read(&mut left)?;
                if n == 0 {
                    break true;
                }
                prefix.read_exact(&mut right[..n])?;
                if left[..n] != right[..n] {
                    break false;
                }
            }
        } else {
            false
        };
        drop(reader);
        let mut observation = self.observation(path, Some(facts));
        let mut group = Vec::new();
        if let Some(raw) = raw {
            let raw_id = raw.hash().to_string();
            let exists = self.reads.reader()?.input_progress(&raw_id)?.is_some();
            let raw_obs = self.observation(path, Some(facts));
            let raw_len = raw.len();
            group.push(self.rollout_batch(
                raw_id.clone(),
                found.origin,
                if exists {
                    Change::Unchanged
                } else {
                    Change::First
                },
                raw_obs,
                raw,
                exists,
            ));
            observation.container = Some(polylogue_store::InContainer {
                input: raw_id,
                member_index: 0,
                kind: "zstd".into(),
                path_bytes: path
                    .with_extension("")
                    .file_name()
                    .unwrap_or_default()
                    .as_encoded_bytes()
                    .to_vec(),
            });
            pass.bytes_read += raw_len;
        } else {
            pass.bytes_read += facts.size;
        }
        let length = bytes.len();
        let change = if same {
            Change::Unchanged
        } else if appended {
            Change::Appended {
                from: previous.as_ref().expect("previous prefix").length,
            }
        } else if previous.is_some() {
            Change::Rewritten
        } else {
            Change::First
        };
        let bytes = if appended {
            decoded.seek(SeekFrom::Start(previous.as_ref().expect("prefix").length))?;
            self.archive.prepare_bytes(&mut decoded)?
        } else {
            bytes
        };
        let mut transcript =
            self.rollout_batch(id.clone(), origin, change, observation, bytes, same);
        transcript.batch.length = length;
        group.push(transcript);
        let results = self
            .writer
            .sender()?
            .store_group(group, Arc::clone(&self.blobs))?;
        let stored = results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .pop()
            .expect("transcript result");
        pass.bytes_stored += if same { 0 } else { length };
        self.inputs.insert(
            path.clone(),
            InputState {
                input_id: id,
                origin,
                facts: Some(facts),
                stored: stored.length,
                parsed_upto: if same {
                    previous.as_ref().map_or(0, |p| p.parsed_upto)
                } else {
                    0
                },
                parsed: same && previous.as_ref().is_some_and(|p| p.parsed),
                waiting: same && previous.as_ref().is_some_and(|p| p.waiting),
                parser_state: None,
                tail: None,
                chain: Some(stored.chain),
                failed: None,
                failed_history: HashMap::new(),
            },
        );
        self.stored_plan(path)
    }

    fn rollout_batch(
        &self,
        id: String,
        origin: &'static OriginSpec,
        change: Change,
        observation: Observation,
        bytes: PreparedBytes,
        same: bool,
    ) -> PreparedBatch {
        let length = bytes.len();
        PreparedBatch {
            batch: polylogue_store::Batch {
                input: InputKey {
                    id,
                    format: origin.name.into(),
                    arrival_rank: if origin.name == "codex-compressed" {
                        2
                    } else {
                        origin.arrival.rank()
                    },
                },
                change,
                observation: Some(observation),
                continues: false,
                length,
                new_blob: (!same).then(|| polylogue_store::NewBlob {
                    hash: bytes.hash(),
                    size: length,
                }),
                blobs: Vec::new(),
                parse: Vec::new(),
                release: Default::default(),
            },
            files: if same { Vec::new() } else { vec![bytes] },
        }
    }
}

#[cfg(test)]
mod tests;
