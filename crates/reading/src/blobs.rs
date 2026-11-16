//! Inline attachment bytes share one bounded transfer and disk-backed dedup.
use super::output::Stream;
use crate::Error;
use base64::Engine as _;
use polylogue_model::{Attachment, BlobHash, Block, bundle::*, read::*};
use polylogue_store::{Reader, export_items::ExportItems};
use std::io::Read;

pub(super) struct Blobs {
    seen: ExportItems,
    source: polylogue_store::blobs::Blobs,
}
impl Blobs {
    pub(super) fn new(reader: &Reader<'_>) -> Result<Self, Error> {
        Ok(Self {
            seen: ExportItems::new()?,
            source: reader.blob_reader()?,
        })
    }
    pub(super) fn record(
        &mut self,
        record: &ReadRecord,
        stream: &mut Stream<'_>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        let Some(record) = record.record() else {
            return Ok(());
        };
        match record {
            Record::Message(row) => self.blocks(&row.message.blocks, stream, check),
            Record::Document(row) => self.blocks(&row.document.blocks, stream, check),
            Record::Authored(row) => self.authored(&row.prepared.effect, stream, check),
            _ => Ok(()),
        }
    }
    fn authored(
        &mut self,
        effect: &polylogue_model::authored::AuthoredEffect,
        stream: &mut Stream<'_>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        for change in &effect.chats {
            for attachment in &change.action.attachments {
                self.blob(attachment.blob, Some(attachment.size), stream, check)?;
            }
        }
        Ok(())
    }
    fn blocks(
        &mut self,
        blocks: &[Block],
        stream: &mut Stream<'_>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        for block in blocks {
            for attachment in block.attachments() {
                self.attachment(attachment, stream, check)?;
            }
        }
        Ok(())
    }
    fn attachment(
        &mut self,
        attachment: &Attachment,
        stream: &mut Stream<'_>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        if let Some(hash) = attachment.blob {
            self.blob(hash, attachment.size, stream, check)?;
        }
        Ok(())
    }
    fn blob(
        &mut self,
        hash: BlobHash,
        size: Option<u64>,
        stream: &mut Stream<'_>,
        check: &dyn Fn() -> Result<(), Error>,
    ) -> Result<(), Error> {
        check()?;
        let key = hash.to_string();
        if self.seen.source_item(&key)?.is_some() {
            return Ok(());
        }
        let extent = self.source.extent(&hashed_blobs::Hash(hash.0))?;
        let size = size.unwrap_or(extent.decoded_len);
        let mut bytes = self
            .source
            .read_plain(&hashed_blobs::Hash(hash.0), Some(size), || Ok(()))?;
        let descriptor = Record::Blob(BlobRecord {
            reference: format!("blob:{hash}"),
            file: FileDescriptor {
                path: format!("blobs/{hash}"),
                hash,
                size,
            },
        });
        stream.record(&ReadRecord::Original {
            record: Box::new(descriptor),
            parts: vec![],
        })?;
        let mut digest = blake3::Hasher::new();
        let mut offset = 0_u64;
        let mut buffer = [0_u8; 16 << 10];
        loop {
            check()?;
            let length = bytes.read(&mut buffer).map_err(Error::io)?;
            if length == 0 {
                break;
            }
            digest.update(&buffer[..length]);
            stream.blob(&ReadBlobChunk {
                hash,
                offset,
                base64: base64::engine::general_purpose::STANDARD.encode(&buffer[..length]),
            })?;
            offset = offset.checked_add(length as u64).ok_or_else(|| {
                Error::internal("a stored file exceeds the supported byte offset")
            })?;
        }
        check()?;
        if offset != size || digest.finalize().as_bytes() != &hash.0 {
            return Err(Error::io(format!(
                "stored blob {hash} failed its size or content hash"
            )));
        }
        if size == 0 {
            stream.blob(&ReadBlobChunk {
                hash,
                offset: 0,
                base64: String::new(),
            })?;
        }
        self.seen.push(0..0, hash.0.to_vec(), Some(&key), b"")?;
        Ok(())
    }
}
