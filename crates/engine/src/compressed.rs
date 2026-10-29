//! Rollouts share recorded identities across locations and encodings. Decoding
//! streams into compressed preparation; no plaintext transcript is staged.
use crate::plan::{InputState, Plan};
use crate::*;
use polylogue_store::staging::{PreparedBatch, PreparedBytes};
use std::io::{BufRead, BufReader, Read, Seek};

pub(crate) fn is_rollout(found: &collect::Found) -> bool {
    found.origin.name == "codex-compressed"
        || found.origin.name == "codex"
            && matches!(
                polylogue_sources::codex::File::from_path(&found.path),
                Some(polylogue_sources::codex::File::Rollout { .. })
            )
}

fn decoded(source: &std::fs::File, compressed: bool) -> std::io::Result<Box<dyn BufRead>> {
    let mut file = source.try_clone()?;
    file.rewind()?;
    if compressed {
        Ok(Box::new(BufReader::new(zstd::stream::read::Decoder::new(
            file,
        )?)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
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
        let mut source = std::fs::File::open(path)?;
        let compressed = found.origin.name == "codex-compressed";
        let raw = if compressed {
            Some(self.archive.prepare_bytes(&mut source)?)
        } else {
            None
        };
        let native = polylogue_sources::codex::rollout_id(decoded(&source, compressed)?)?;
        let native_id = native.map(|native| format!("codex:rollout:{native}"));
        let reader = self.reads.reader()?;
        let mut previous = native_id
            .as_ref()
            .map(|id| reader.input_progress(id))
            .transpose()?
            .flatten();
        let mut stream = decoded(&source, compressed)?;
        let mut same = false;
        let mut appended = false;
        if let (Some(id), Some(previous)) = (&native_id, &previous) {
            let chain = reader.stored_chain(id, previous.chain, previous.length)?;
            let mut stored = chain.reader(Arc::clone(&self.blobs));
            let mut left = [0; 65536];
            let mut right = [0; 65536];
            let mut matches = true;
            loop {
                let n = stored.read(&mut left)?;
                if n == 0 {
                    break;
                }
                match stream.read_exact(&mut right[..n]) {
                    Ok(()) if left[..n] == right[..n] => {}
                    Ok(()) => {
                        matches = false;
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                        matches = false;
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            if matches {
                same = stream.fill_buf()?.is_empty();
                appended = !same;
            } else {
                stream = decoded(&source, compressed)?;
            }
        }
        let prepared = if same {
            None
        } else {
            Some(self.archive.prepare_bytes(stream)?)
        };
        let id = if let Some(id) = native_id {
            id
        } else {
            prepared
                .as_ref()
                .expect("unidentified bytes prepared")
                .hash()
                .to_string()
        };
        if previous.is_none() {
            previous = reader.input_progress(&id)?;
            if previous.is_some() {
                same = true;
            }
        }
        drop(reader);
        if FileFacts::of(&source.metadata()?) != facts
            || std::fs::metadata(path).map(|m| FileFacts::of(&m))? != facts
        {
            return Err(std::io::Error::other("source changed during rollout capture").into());
        }
        let length = if same {
            previous.as_ref().expect("same stored input").length
        } else {
            prepared.as_ref().expect("new bytes").len()
                + if appended {
                    previous.as_ref().expect("prefix").length
                } else {
                    0
                }
        };
        let added = if same {
            0
        } else {
            prepared.as_ref().expect("new bytes").len()
        };
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let mut observation = self.observation(path, Some(facts));
        let mut group = Vec::new();
        let mut raw_added = 0;
        if let Some(raw) = raw {
            let raw_id = raw.hash().to_string();
            let exists = self.reads.reader()?.input_progress(&raw_id)?.is_some();
            if !exists {
                raw_added = raw.len();
            }
            let raw_length = raw.len();
            group.push(self.rollout_batch(
                raw_id.clone(),
                found.origin,
                if exists {
                    Change::Unchanged
                } else {
                    Change::First
                },
                self.observation(path, Some(facts)),
                raw_length,
                (!exists).then_some(raw),
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
        }
        let change = if same {
            Change::Unchanged
        } else if appended {
            Change::Appended {
                from: previous.as_ref().expect("prefix").length,
            }
        } else if previous.is_some() {
            Change::Rewritten
        } else {
            Change::First
        };
        group.push(self.rollout_batch(
            id.clone(),
            origin,
            change,
            observation,
            length,
            if same { None } else { prepared },
        ));
        let stored = self
            .writer
            .sender()?
            .store_group(group, Arc::clone(&self.blobs))?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .pop()
            .expect("transcript result");
        pass.bytes_read += facts.size;
        pass.bytes_stored += raw_added + added;
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
        length: u64,
        bytes: Option<PreparedBytes>,
    ) -> PreparedBatch {
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
                new_blob: bytes.as_ref().map(|bytes| polylogue_store::NewBlob {
                    hash: bytes.hash(),
                    size: bytes.len(),
                }),
                blobs: Vec::new(),
                parse: Vec::new(),
                release: Default::default(),
            },
            files: bytes.into_iter().collect(),
        }
    }
}
#[cfg(test)]
mod tests;
