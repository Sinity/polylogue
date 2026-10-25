//! Sending Sinex the row batches and blobs it does not hold yet.
use crate::rows::{self, Encoded};
use crate::{Confirmed, Error, Result, Sinex, checked, content_name};
use polylogue_store::StoreError;
use polylogue_store::row_batches::{RowBatch, Snapshot};
use sinex_model::event_types::polylogue::PolylogueRows;
use sinex_model::{Class, RecordLocation};
use sinex_protocol::{Ingress, ProducerEvent};
use std::path::Path;

/// How many batches one page lists; `have` asks about them together.
const PAGE: usize = 500;

/// What one pass sent.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Sent {
    pub batches: u64,
    pub rows: u64,
    pub blobs: u64,
}

impl Sinex {
    /// Sends every row batch of the archive at `root` that Sinex does not
    /// hold yet, in write order, each after the blobs it cites. A batch whose
    /// bytes are still being written ends the pass; the next pass sends it.
    pub async fn send_missing(&mut self, root: &Path) -> Result<Sent> {
        let snapshot = Snapshot::open(root)?;
        self.finish_staged().await?;
        let archive = snapshot
            .archive_id()
            .map_err(|e| Error::Invalid(format!("the archive has no identity: {e}")))?;
        let (mut position, mut after) = match &self.confirmed {
            Some(confirmed) if confirmed.archive == archive => {
                (confirmed.position, Some(confirmed.batch.clone()))
            }
            _ => (0, None),
        };
        let mut sent = Sent::default();
        loop {
            let page = snapshot.batches(after.as_ref(), PAGE)?;
            if page.is_empty() {
                return Ok(sent);
            }
            let names: Vec<String> = (1..=page.len() as u64)
                .map(|offset| batch_name(&archive, position + offset))
                .collect();
            let held = self.have(&names).await?;
            for (batch, name) in page.into_iter().zip(names) {
                position += 1;
                match held.get(&name) {
                    Some(acknowledgment) => {
                        checked(acknowledgment.clone())?;
                    }
                    None => match self
                        .send_batch(&snapshot, &archive, position, &batch, name, &mut sent)
                        .await
                    {
                        Err(Error::Store(StoreError::InputPending { .. })) => return Ok(sent),
                        other => other?,
                    },
                }
                self.confirmed = Some(Confirmed {
                    archive: archive.clone(),
                    position,
                    batch: batch.clone(),
                });
                after = Some(batch);
            }
        }
    }

    async fn send_batch(
        &mut self,
        snapshot: &Snapshot,
        archive: &str,
        position: u64,
        batch: &RowBatch,
        name: String,
        sent: &mut Sent,
    ) -> Result<()> {
        let mut encoded = rows::encode(snapshot, batch)?;
        self.place_blobs(snapshot, archive, &encoded.blobs, sent)
            .await?;
        let material = self.place_rows(archive, &mut encoded).await?;
        let event = PolylogueRows {
            archive: archive.to_owned(),
            position,
            local_seq: batch.commit_seq,
            unit: batch.kind.clone(),
            key: batch.key.clone(),
            batch: content_name(&encoded.hash),
            rows: encoded.rows,
            blobs: encoded.blobs.iter().map(content_name).collect(),
        };
        let reading_order = i64::try_from(position)
            .map_err(|_| Error::Invalid("a batch position exceeds the wire order range".into()))?;
        self.send(&Ingress::Events {
            frame: name,
            failures: vec![],
            events: vec![ProducerEvent {
                event_type: "polylogue.rows".into(),
                payload: serde_json::to_value(&event)?,
                location: Some(RecordLocation {
                    material,
                    anchor: 0,
                }),
                ordinal: 0,
                parents: vec![],
                reading_order,
                class: Class::Open,
            }],
        })
        .await?;
        sent.batches += 1;
        sent.rows += encoded.rows;
        Ok(())
    }

    /// Places each cited blob Sinex does not hold yet.
    async fn place_blobs(
        &mut self,
        snapshot: &Snapshot,
        archive: &str,
        blobs: &[hashed_blobs::Hash],
        sent: &mut Sent,
    ) -> Result<()> {
        let names: Vec<String> = blobs
            .iter()
            .map(|hash| format!("blob:{}", hash.hex()))
            .collect();
        let held = self.have(&names).await?;
        for (hash, name) in blobs.iter().zip(names) {
            if let Some(acknowledgment) = held.get(&name) {
                checked(acknowledgment.clone())?;
                continue;
            }
            let (length, mut bytes) = snapshot.blob(&polylogue_model::BlobHash(hash.0))?;
            self.material(
                name,
                hash,
                length,
                &mut bytes,
                serde_json::json!({ "archive": archive }),
            )
            .await?;
            sent.blobs += 1;
        }
        Ok(())
    }

    /// Places a batch's rows as a material and returns its identity.
    async fn place_rows(
        &mut self,
        archive: &str,
        encoded: &mut Encoded,
    ) -> Result<sinex_model::MaterialId> {
        let name = format!("batch:{archive}:{}", encoded.hash.hex());
        let held = self.have(std::slice::from_ref(&name)).await?;
        if let Some(acknowledgment) = held.get(&name) {
            let acknowledgment = checked(acknowledgment.clone())?;
            return acknowledgment
                .material
                .ok_or_else(|| Error::Invalid(format!("{name} registered no material")));
        }
        self.material(
            name,
            &encoded.hash,
            encoded.length,
            &mut encoded.file,
            serde_json::json!({ "archive": archive, "media_type": "application/jsonl" }),
        )
        .await
    }
}

/// The name Sinex holds the batch at `position` of `archive` under.
fn batch_name(archive: &str, position: u64) -> String {
    format!("rows:{archive}:{position}")
}
