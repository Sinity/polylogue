//! Outside replay submits a complete parse, with the input bytes checked again
//! before its saved frame is published. The connection serializes publication.
use crate::{Error, Result, Sinex, client::Client};
use sinex_protocol::{
    Acknowledgment, DeclarationHash, Ingress, ProducerReplay, ReplayBatch, ReplayScopeQuery,
    ReplaySelection,
};
use std::sync::MutexGuard;

/// Holds publication while a caller selects materials and prepares their parse.
/// Dropping it allows the producer's other publication calls to proceed.
pub struct Replay<'a> {
    runtime: &'a tokio::runtime::Runtime,
    client: MutexGuard<'a, Option<Client>>,
}

impl Sinex {
    /// Other connection calls wait until the returned replay is dropped.
    pub fn replay(&self) -> Result<Replay<'_>> {
        let client = self
            .client
            .lock()
            .map_err(|_| Error::Invalid("the Sinex connection panicked".into()))?;
        if client.is_none() {
            return Err(Error::Invalid("the Sinex connection is closed".into()));
        }
        Ok(Replay {
            runtime: &self.runtime,
            client,
        })
    }
}

impl Replay<'_> {
    fn with<T>(&mut self, work: impl AsyncFnOnce(&mut Client) -> Result<T>) -> Result<T> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| Error::Invalid("the Sinex connection is closed".into()))?;
        self.runtime.block_on(work(client))
    }

    pub fn scope(&mut self, query: &ReplayScopeQuery) -> Result<Vec<ReplaySelection>> {
        self.with(async |client| client.replay_scope(query).await)
    }

    /// `batches` is the completed parse of every material in `selection`.
    /// An empty replacement is valid only when the completed parse has no
    /// readings in this scope. Only the confirmed consumer updates the index.
    /// A different parse or build needs a new operation.
    pub fn sequence(
        &mut self,
        operation: sinex_model::OperationId,
        selection: ReplaySelection,
        scope: ReplayScopeQuery,
        batches: Vec<ReplayBatch>,
    ) -> Result<Acknowledgment> {
        self.with(async |client| {
            let reader = hashed_blobs::Reader::new(client.store_root.clone());
            for material in &selection.materials {
                let hex = material
                    .blob
                    .strip_prefix("blake3:")
                    .ok_or_else(|| Error::Invalid("invalid material digest".into()))?;
                let fetched = client.fetch_material(material.id).await?;
                let extents = crate::material::decode_extents(&fetched.extents)?;
                if extents.is_empty()
                    || extents.iter().try_fold(0u64, |length, extent| {
                        length.checked_add(extent.decoded_len)
                    }) != Some(material.length)
                {
                    return Err(Error::Invalid(
                        "replay material descriptors differ from its length".into(),
                    ));
                }
                let mut bytes = reader.open(&extents, None)?;
                let mut digest = blake3::Hasher::new();
                let length = std::io::copy(&mut bytes, &mut digest)?;
                if length != material.length || digest.finalize().to_hex().as_str() != hex {
                    return Err(Error::Invalid("stored replay bytes changed".into()));
                }
            }
            let sequence = ProducerReplay {
                operation,
                root: selection.root,
                materials: selection.materials,
                event_types: scope.event_types,
                from: scope.from,
                until: scope.until,
                declarations: crate::kinds::declarations(client.family())
                    .iter()
                    .map(|d| DeclarationHash {
                        kind: d.kind.clone(),
                        hash: d.definition_hash(),
                    })
                    .collect(),
                parser_build: build_id()?,
                parser_version: crate::material::parser_version().to_string(),
                expected_live_hash: selection.expected_live_hash,
                batches,
            };
            let content = serde_json::to_vec(&sequence)?;
            let name = format!(
                "replay:{}:{}:{}",
                operation,
                sequence.root,
                blake3::hash(&content).to_hex()
            );
            let frame = Ingress::ReplaySequence {
                frame: name.clone(),
                sequence: Box::new(sequence),
            };
            let bytes = serde_json::to_vec(&frame)?;
            let blob = format!("blake3:{}", blake3::hash(&bytes).to_hex());
            client.write_bytes(&name, &blob, bytes.len() as u64, bytes.as_slice())?;
            client.send(frame).await
        })
    }
}

pub(crate) fn build_id() -> Result<String> {
    static BUILD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if let Some(build) = BUILD.get() {
        return Ok(build.clone());
    }
    let mut file = std::fs::File::open("/proc/self/exe")?;
    let mut hash = blake3::Hasher::new();
    std::io::copy(&mut file, &mut hash)?;
    let build = hash.finalize().to_hex().to_string();
    let _ = BUILD.set(build.clone());
    Ok(build)
}
