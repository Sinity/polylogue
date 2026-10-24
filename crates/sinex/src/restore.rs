//! Restoring an empty archive from the row batches Sinex holds.
use crate::{Error, Result, Sinex, parse_content_name, rows};
use async_nats::jetstream;
use futures_util::StreamExt;
use polylogue_store::Archive;
use polylogue_store::row_batches::Restore;
use sinex_model::event_types::polylogue::PolylogueRows;
use std::io::Read;

/// What a restore installed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Restored {
    pub batches: u64,
    /// Rows inserted; a batch received twice adds none the second time.
    pub rows: u64,
}

impl Sinex {
    /// Restores `archive` (the archive's identity) into the empty local
    /// archive from every batch Sinex has confirmed, in one transaction.
    /// Each batch and blob is checked against its content name. Sinex must
    /// hold positions 1 to the last without a gap. The loop then builds
    /// `index.db` from the restored rows and Sinex's content store.
    pub async fn restore(&self, local: &Archive, archive: &str) -> Result<Restored> {
        let mut restore = Restore::begin(local)?;
        let reader = hashed_blobs::Reader::new(self.store_root.clone(), vec![]);
        let mut positions = Positions::default();
        let subject = self.subjects.confirmed_event("polylogue.rows");
        let js = jetstream::new(self.client.clone());
        let stream = js
            .get_stream(self.subjects.confirmed_stream())
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        // Read up to the last event on the stream now; later ones are
        // sent after the restore.
        let last = match stream.get_last_raw_message_by_subject(&subject).await {
            Ok(message) => message.sequence,
            Err(e) if e.kind() == jetstream::stream::LastRawMessageErrorKind::NoMessageFound => 0,
            Err(e) => return Err(Error::Unreachable(e.to_string())),
        };
        if last > 0 {
            let consumer = stream
                .create_consumer(jetstream::consumer::pull::OrderedConfig {
                    filter_subject: subject,
                    deliver_policy: jetstream::consumer::DeliverPolicy::All,
                    ..Default::default()
                })
                .await
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let mut messages = consumer
                .messages()
                .await
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            loop {
                let message = tokio::time::timeout(crate::ACKNOWLEDGMENT_WAIT, messages.next())
                    .await
                    .map_err(|_| Error::Unreachable("the confirmed stream stopped".into()))?
                    .ok_or_else(|| Error::Unreachable("the confirmed stream closed".into()))?
                    .map_err(|e| Error::Unreachable(e.to_string()))?;
                let at = message
                    .info()
                    .map_err(|e| Error::Unreachable(e.to_string()))?
                    .stream_sequence;
                let confirmed: sinex_protocol::Confirmed =
                    serde_json::from_slice(&message.payload)?;
                if let sinex_protocol::Confirmed::Event(event) = confirmed {
                    let payload = event.payload.ok_or_else(|| {
                        Error::Invalid(format!("{} arrived without its payload", event.id))
                    })?;
                    let batch: PolylogueRows = serde_json::from_value(payload)?;
                    if batch.archive == archive {
                        positions.insert(batch.position)?;
                        install(&reader, &mut restore, &batch)?;
                    }
                }
                if at >= last {
                    break;
                }
            }
        }
        let batches = positions.complete()?;
        let rows = restore.finish(archive)?;
        Ok(Restored { batches, rows })
    }
}

/// Checks a batch's blobs and rows against their content names, then
/// inserts its rows.
fn install(
    reader: &hashed_blobs::Reader,
    restore: &mut Restore,
    batch: &PolylogueRows,
) -> Result<()> {
    for blob in &batch.blobs {
        let hash = parse_content_name(blob)?;
        let mut bytes = Checked::new(reader.read_plain(&hash, None, || Ok(()))?, hash);
        std::io::copy(&mut bytes, &mut std::io::sink())?;
        bytes.finish()?;
    }
    let hash = parse_content_name(&batch.batch)?;
    let mut bytes = Checked::new(reader.read_plain(&hash, None, || Ok(()))?, hash);
    let rows = rows::decode(&mut bytes, |row| Ok(restore.insert(row)?))?;
    bytes.finish()?;
    if rows != batch.rows {
        return Err(Error::Invalid(format!(
            "batch {} of archive {} has {rows} rows, not {}",
            batch.position, batch.archive, batch.rows
        )));
    }
    Ok(())
}

/// Hashes what it reads; `finish` compares the hash with the expected one.
struct Checked<R> {
    inner: R,
    digest: blake3::Hasher,
    expected: hashed_blobs::Hash,
}

impl<R: Read> Checked<R> {
    fn new(inner: R, expected: hashed_blobs::Hash) -> Self {
        Self {
            inner,
            digest: blake3::Hasher::new(),
            expected,
        }
    }

    fn finish(mut self) -> Result<()> {
        // Count any bytes after the last row too.
        std::io::copy(&mut self, &mut std::io::sink())?;
        if self.digest.finalize().as_bytes() != &self.expected.0 {
            return Err(Error::Invalid(format!(
                "Sinex's bytes for blake3:{} do not match their name",
                self.expected.hex()
            )));
        }
        Ok(())
    }
}

impl<R: Read> Read for Checked<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buffer)?;
        self.digest.update(&buffer[..n]);
        Ok(n)
    }
}

/// The batch positions received, as a bit set.
#[derive(Default)]
struct Positions {
    bits: Vec<u64>,
    count: u64,
}

impl Positions {
    fn insert(&mut self, position: u64) -> Result<()> {
        let index = position
            .checked_sub(1)
            .ok_or_else(|| Error::Invalid("a batch has position 0".into()))?;
        let word = usize::try_from(index / 64)
            .map_err(|_| Error::Invalid("a batch position is out of range".into()))?;
        if self.bits.len() <= word {
            self.bits.resize(word + 1, 0);
        }
        let bit = 1 << (index % 64);
        if self.bits[word] & bit == 0 {
            self.bits[word] |= bit;
            self.count += 1;
        }
        Ok(())
    }

    /// The number of batches, once positions 1 to the last are all present.
    fn complete(&self) -> Result<u64> {
        let last = self
            .bits
            .iter()
            .enumerate()
            .rev()
            .find(|(_, word)| **word != 0)
            .map_or(0, |(index, word)| {
                index as u64 * 64 + 64 - u64::from(word.leading_zeros())
            });
        if self.count != last {
            return Err(Error::Invalid(format!(
                "Sinex holds {} of batches 1 to {last}",
                self.count
            )));
        }
        Ok(last)
    }
}

#[cfg(test)]
mod tests;
