//! Attachment presentation follows actual receiving custody and checked bytes.
//! Canonical descriptors and their historical naming claims remain unchanged.
use super::Failure;
use polylogue_model::{Attachment, BlobHash, MessageId, read::*};
use polylogue_store::{
    Reader, StoreError,
    assets::{AttachmentState, bindings::AttachmentPin},
};

pub(super) type ReadyAttachment<'a> =
    dyn Fn(&MessageId, u64, u64, Option<&str>) -> Result<ReadyCall, Failure> + 'a;

enum Proof {
    Verified {
        hash: BlobHash,
        input: Option<String>,
    },
    Unknown(&'static str),
    Unavailable(String),
}

fn proof(
    reader: &Reader<'_>,
    message: &MessageId,
    block: u64,
    index: u64,
    attachment: &Attachment,
) -> polylogue_store::Result<Proof> {
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
        let blobs = reader.blob_reader()?;
        let extent = blobs.extent(&hashed_blobs::Hash(hash.0))?;
        if attachment
            .size
            .is_some_and(|size| size != extent.decoded_len)
        {
            return Err(StoreError::Invalid(
                "attachment size differs from its recorded body".into(),
            ));
        }
        blobs.verify_plain::<StoreError>(
            &hashed_blobs::Hash(hash.0),
            extent.decoded_len,
            |_| reader.check(),
        )?;
        reader.check()?;
        return Ok(Proof::Verified { hash, input: None });
    }
    match resolution.state {
        AttachmentState::Unknown => {
            return Ok(Proof::Unknown("attachment byte custody was not recorded"));
        }
        AttachmentState::Ambiguous => {
            return Ok(Proof::Unknown(
                "several receiving members or namespaces match this attachment",
            ));
        }
        AttachmentState::Unavailable => {
            return Ok(Proof::Unavailable(
                "the recorded attachment has no available receiving member".into(),
            ));
        }
        AttachmentState::Available => {}
    }
    let contexts = reader.attachment_binding_contexts(message, block, index)?;
    if contexts.namespace_count != 0 {
        let [context] = contexts.contexts.as_slice() else {
            return Ok(Proof::Unknown(
                "attachment receiving namespace is ambiguous",
            ));
        };
        return match reader.pin_attachment(&context.selection)? {
            AttachmentPin::Available(pin) => {
                let member = pin.identity();
                let hash = pin.content_hash()?;
                reader.check()?;
                Ok(Proof::Verified {
                    hash,
                    input: Some(member.input.clone()),
                })
            }
            AttachmentPin::Unknown => Ok(Proof::Unknown("attachment receiving custody is unknown")),
            AttachmentPin::Ambiguous => {
                Ok(Proof::Unknown("attachment receiving custody is ambiguous"))
            }
            AttachmentPin::Unavailable => Ok(Proof::Unavailable(
                "pinned attachment receiving bytes are unavailable".into(),
            )),
        };
    }
    // The native-import resolver proves actual import_member/source_observation
    // membership in the shown source's namespace. Source IDs alone never do.
    let [asset] = resolution.matches.as_slice() else {
        return Ok(Proof::Unknown("attachment receiving bytes are ambiguous"));
    };
    let hash = reader.attachment_hash(asset)?;
    reader.check()?;
    if let Some(expected) = asset.blob.filter(|expected| *expected != hash) {
        return Err(StoreError::BlobCorrupt(expected));
    }
    Ok(Proof::Verified {
        hash,
        input: Some(asset.input.clone()),
    })
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
            reason: Some("attachment source has no recorded message/slot byte operation".into()),
        };
        *inline_bytes = None;
        *verified_hash = None;
        *open = None;
        return Ok(item);
    };
    reader.check()?;
    let proof = match proof(reader, message, block, index, attachment) {
        Ok(proof) => proof,
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
            Proof::Unavailable(error.to_string())
        }
        Err(error) => return Err(Failure::store(&error)),
    };
    *verified_hash = None;
    *open = None;
    match proof {
        Proof::Verified { hash, input } => {
            *availability = ValueState::known("present".into());
            *verified_hash = Some(hash);
            if inline_bytes
                .as_ref()
                .is_some_and(|bytes| BlobHash::of(bytes) != hash)
            {
                *inline_bytes = None;
            }
            *open = Some(Box::new(ready(message, block, index, input.as_deref())?));
        }
        Proof::Unknown(reason) => {
            *availability = ValueState::Unknown {
                reason: Some(reason.into()),
            };
            *inline_bytes = None;
        }
        Proof::Unavailable(reason) => {
            *availability = ValueState::Unavailable { reason };
            *inline_bytes = None;
        }
    }
    reader.check()?;
    Ok(item)
}

#[cfg(test)]
mod tests;
