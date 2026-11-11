//! Attachment pages use stored file metadata. The byte operation checks bodies.
//! Canonical descriptors and their historical naming claims remain unchanged.
use super::Failure;
use polylogue_model::{Attachment, MessageId, read::*};
use polylogue_store::{Reader, StoreError, assets::AttachmentState};

pub(super) type ReadyAttachment<'a> =
    dyn Fn(&MessageId, u64, u64, Option<&str>) -> Result<ReadyCall, Failure> + 'a;

enum Availability {
    Present { input: Option<String> },
    Unknown(&'static str),
    Unavailable(String),
}

fn resolve(
    reader: &Reader<'_>,
    message: &MessageId,
    block: u64,
    index: u64,
    attachment: &Attachment,
) -> polylogue_store::Result<Availability> {
    reader.check()?;
    let block_index = usize::try_from(block)
        .map_err(|_| StoreError::Invalid("attachment block exceeds host address space".into()))?;
    let attachment_index = usize::try_from(index)
        .map_err(|_| StoreError::Invalid("attachment index exceeds host address space".into()))?;
    let resolution = reader.attachment(message, block_index, attachment_index)?;
    if resolution.source != *attachment {
        return Err(StoreError::Conflict(
            "attachment descriptor differs from the selected source".into(),
        ));
    }
    if let Some(hash) = attachment.blob {
        let extent = reader.blob_reader()?.extent(&hashed_blobs::Hash(hash.0))?;
        if attachment
            .size
            .is_some_and(|size| size != extent.decoded_len)
        {
            return Err(StoreError::Invalid(
                "attachment size differs from its stored size".into(),
            ));
        }
        return Ok(Availability::Present { input: None });
    }
    match resolution.state {
        AttachmentState::Unknown => Ok(Availability::Unknown("attachment bytes were not recorded")),
        AttachmentState::Ambiguous => Ok(Availability::Unknown(
            "several stored files match this attachment",
        )),
        AttachmentState::Unavailable => Ok(Availability::Unavailable(
            "the attachment's stored file is unavailable".into(),
        )),
        AttachmentState::Available => {
            let [asset] = resolution.matches.as_slice() else {
                return Ok(Availability::Unknown(
                    "several stored files match this attachment",
                ));
            };
            Ok(Availability::Present {
                input: Some(asset.input.clone()),
            })
        }
    }
}

pub(super) fn prepare(
    reader: &Reader<'_>,
    mut item: ReadItem,
    index: u64,
    ready: &ReadyAttachment<'_>,
) -> Result<ReadItem, Failure> {
    let ItemKind::Attachment {
        attachment,
        availability,
        open,
        inline_bytes,
        verified_hash,
        ..
    } = &mut item.kind
    else {
        return Ok(item);
    };
    let (Some(message), Some(block)) = (&item.source.message, item.source.block) else {
        *availability = ValueState::Unknown {
            reason: Some("attachment has no recorded message and block".into()),
        };
        *inline_bytes = None;
        *verified_hash = None;
        *open = None;
        return Ok(item);
    };
    reader.check()?;
    let resolved = match resolve(reader, message, block, index, attachment) {
        Ok(resolved) => resolved,
        Err(
            error @ (StoreError::Io(_)
            | StoreError::Blobs(_)
            | StoreError::BlobCorrupt(_)
            | StoreError::NotFound(_)
            | StoreError::Conflict(_)
            | StoreError::Invalid(_)),
        ) => {
            // A cancellation or scope change outranks a simultaneous byte error.
            reader.check()?;
            Availability::Unavailable(error.to_string())
        }
        Err(error) => return Err(Failure::store(&error)),
    };
    // A page never reads a body or claims to have verified its hash. The
    // attachment operation checks the bytes when the caller asks for them.
    *verified_hash = None;
    *inline_bytes = None;
    *open = None;
    match resolved {
        Availability::Present { input } => {
            *availability = ValueState::known("present".into());
            *open = Some(Box::new(ready(message, block, index, input.as_deref())?));
        }
        Availability::Unknown(reason) => {
            *availability = ValueState::Unknown {
                reason: Some(reason.into()),
            };
        }
        Availability::Unavailable(reason) => {
            *availability = ValueState::Unavailable { reason };
        }
    }
    reader.check()?;
    Ok(item)
}

#[cfg(test)]
mod tests;
