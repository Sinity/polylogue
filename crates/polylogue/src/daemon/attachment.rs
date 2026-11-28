//! Checked attachment bytes through the usual application operation.
use super::service::{Daemon, error, store_error};
use base64::Engine as _;
use command_surface::Answer;
use polylogue_model::read::{CursorError, CursorSeal};
use serde::{Deserialize, Serialize};

pub(crate) const PAGE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BytePage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<polylogue_store::assets::AttachmentResolution>,
    /// Receiving member input; inline bytes belong to the selected source slot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    pub blob: polylogue_model::BlobHash,
    pub size: u64,
    pub offset: u64,
    pub content_base64: String,
    pub next: Option<String>,
}

/// Names attachment cursors in the seal, so no other cursor opens as one.
const SEAL_DOMAIN: &[u8] = b"polylogue.attachment-cursor\0";

/// Sealed with the index instance's key ([`CursorSeal`]), so only a
/// cursor this archive issued can supply a byte position.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    binding: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    input: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<CursorSource>,
    offset: u64,
    blob: polylogue_model::BlobHash,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum CursorSource {
    Inline {
        message: polylogue_model::MessageId,
        block: usize,
        attachment: usize,
    },
}

/// One checked byte range of an attachment.
pub(crate) struct Range {
    pub message: polylogue_model::MessageId,
    pub block: usize,
    pub attachment: usize,
    pub input: Option<String>,
    pub length: usize,
    pub cursor: Option<String>,
}

pub(crate) fn read(range: Range, d: &Daemon) -> Answer<BytePage> {
    if let Err(failure) = super::service::accepting(d) {
        return failure.answer();
    }
    let Range {
        message,
        block,
        attachment,
        input,
        length,
        cursor,
    } = range;
    if length == 0 {
        return error(
            polylogue_model::Code::BadRequest,
            "attachment range length must be positive",
        );
    }
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    let scope = reader.scope_token();
    let resolution = match reader.attachment(&message, block, attachment) {
        Ok(value) => value,
        Err(e) => return store_error(&e),
    };
    let seal = match (|| {
        Ok::<_, polylogue_store::StoreError>(CursorSeal {
            key: reader.cursor_key()?,
            index_instance: reader.change_position()?.index_instance,
        })
    })() {
        Ok(seal) => seal,
        Err(e) => return store_error(&e),
    };
    const STALE: &str = "the selected attachment changed; read the attachment again from its start";
    let initial = cursor.is_none();
    let decoded = match cursor {
        Some(cursor) => match seal.open::<Cursor>(SEAL_DOMAIN, &cursor) {
            Ok(decoded) if decoded.binding == resolution.binding => Some(decoded),
            Ok(_) | Err(CursorError::Stale | CursorError::EarlierBuild) => {
                return error(polylogue_model::Code::StaleCursor, STALE);
            }
            Err(CursorError::Invalid | CursorError::OtherReading(_)) => {
                return error(
                    polylogue_model::Code::BadRequest,
                    "invalid attachment cursor",
                );
            }
        },
        None => None,
    };
    let inline = CursorSource::Inline {
        message,
        block,
        attachment,
    };
    if let Some(cursor) = &decoded {
        if let Some(source) = &cursor.source {
            if source != &inline
                || cursor.input.is_some()
                || resolution.source.blob != Some(cursor.blob)
            {
                return error(
                    polylogue_model::Code::StaleCursor,
                    "the inline attachment source changed; read the attachment again from its start",
                );
            }
            if input.is_some() {
                return error(
                    polylogue_model::Code::BadRequest,
                    "an inline attachment cursor does not select an input",
                );
            }
        } else if cursor.input.is_none() {
            return error(
                polylogue_model::Code::BadRequest,
                "attachment cursor has no source selection",
            );
        }
    }
    if input.is_none()
        && resolution.matches.is_empty()
        && resolution.source.blob.is_some()
        && decoded
            .as_ref()
            .is_none_or(|cursor| cursor.source.as_ref() == Some(&inline))
    {
        return inline_page(&reader, resolution, &seal, decoded, length, inline);
    }
    let chosen = input
        .as_deref()
        .or_else(|| decoded.as_ref().and_then(|cursor| cursor.input.as_deref()));
    let found = match chosen {
        Some(input) => resolution.matches.iter().find(|found| found.input == input),
        None if resolution.matches.len() == 1 => resolution.matches.first(),
        None => None,
    };
    let Some(found) = found else {
        let message = if resolution.matches.is_empty() {
            let availability = resolution.source.failure.map_or_else(
                || resolution.state.as_str().to_owned(),
                |failure| format!("failed: {}", failure.describe()),
            );
            format!("the attachment has no stored file (availability: {availability})")
        } else {
            "select one of the stored files with --input".to_string()
        };
        let failure =
            crate::protocol::failure(polylogue_model::Code::AttachmentUnavailable, message)
                .with_candidates(
                    resolution
                        .matches
                        .iter()
                        .map(|found| found.input.clone())
                        .collect(),
                );
        return Answer::error(failure);
    };
    if decoded
        .as_ref()
        .is_some_and(|cursor| cursor.input.as_deref() != Some(found.input.as_str()))
    {
        return error(
            polylogue_model::Code::BadRequest,
            "the cursor selects another input",
        );
    }
    let offset = decoded.as_ref().map_or(0, |cursor| cursor.offset);
    let blob = match decoded.as_ref().map(|cursor| cursor.blob).or(found.blob) {
        Some(hash) => hash,
        None => match reader.attachment_hash(found) {
            Ok(hash) => hash,
            Err(error) => return store_error(&error),
        },
    };
    if found.blob.is_some_and(|known| known != blob) {
        return error(
            polylogue_model::Code::StaleCursor,
            "the selected attachment hash changed; read the attachment again from its start",
        );
    }
    if offset > found.size || offset == found.size && offset != 0 {
        return error(
            polylogue_model::Code::BadRequest,
            "attachment cursor is outside the stored bytes",
        );
    }
    let end = offset
        .saturating_add(length.min(PAGE_BYTES) as u64)
        .min(found.size);
    let bytes = match reader.attachment_range(found, offset..end) {
        Ok(bytes) => bytes,
        Err(e) => return store_error(&e),
    };
    let next = (end < found.size).then(|| {
        seal.seal(
            SEAL_DOMAIN,
            &Cursor {
                binding: resolution.binding.clone(),
                input: Some(found.input.clone()),
                source: None,
                offset: end,
                blob,
            },
        )
    });
    let descriptor = initial.then(|| resolution.clone());
    let page = BytePage {
        descriptor,
        input: Some(found.input.clone()),
        blob,
        size: found.size,
        offset,
        content_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        next,
    };
    if let Err(error) = scope.check() {
        return store_error(&error);
    }
    Answer::ok(page)
}

fn inline_page(
    reader: &polylogue_store::Reader<'_>,
    mut resolution: polylogue_store::assets::AttachmentResolution,
    seal: &CursorSeal,
    cursor: Option<Cursor>,
    length: usize,
    source: CursorSource,
) -> Answer<BytePage> {
    use std::io::{Read, Seek, SeekFrom};
    let result = (|| -> polylogue_store::Result<BytePage> {
        reader.check()?;
        let hash = resolution.source.blob.expect("selected inline attachment");
        let blobs = reader.blob_reader()?;
        let extent = blobs.extent(&hashed_blobs::Hash(hash.0))?;
        let size = extent.decoded_len;
        if resolution
            .source
            .size
            .is_some_and(|expected| expected != size)
        {
            return Err(polylogue_store::StoreError::internal(
                "inline attachment size differs from its source",
            ));
        }
        let offset = cursor.as_ref().map_or(0, |cursor| cursor.offset);
        if offset > size || offset == size && offset != 0 {
            return Err(polylogue_store::StoreError::internal(
                "inline attachment cursor is outside its bytes",
            ));
        }
        // Initial availability requires the complete stored body, including
        // an encoded empty body. Continuations keep the same source and hash;
        // each selected range still passes the blob reader's integrity checks.
        if cursor.is_none() || size == 0 {
            blobs.verify_plain::<polylogue_store::StoreError>(
                &hashed_blobs::Hash(hash.0),
                size,
                |_| reader.check(),
            )?;
        }
        reader.check()?;
        let end = offset
            .saturating_add(length.min(PAGE_BYTES) as u64)
            .min(size);
        let scope = reader.scope_token();
        let limit = polylogue_store::read::limit::ReadLimit::current();
        let mut body = match blobs.read_plain(&hashed_blobs::Hash(hash.0), size, move || {
            scope
                .check()
                .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))?;
            if let Some(limit) = &limit {
                limit
                    .check()
                    .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))?;
            }
            Ok(())
        }) {
            Ok(body) => body,
            Err(error) => {
                reader.check()?;
                return Err(error);
            }
        };
        if let Err(error) = body.seek(SeekFrom::Start(offset)) {
            reader.check()?;
            return Err(error.into());
        }
        let mut bytes = Vec::new();
        if let Err(error) = body.take(end - offset).read_to_end(&mut bytes) {
            reader.check()?;
            return Err(error.into());
        }
        if bytes.len() as u64 != end - offset {
            return Err(polylogue_store::StoreError::internal(
                "inline attachment range is incomplete",
            ));
        }
        let next = (end < size).then(|| {
            seal.seal(
                SEAL_DOMAIN,
                &Cursor {
                    binding: resolution.binding.clone(),
                    input: None,
                    source: Some(source),
                    offset: end,
                    blob: hash,
                },
            )
        });
        let initial = cursor.is_none();
        resolution.state = polylogue_store::assets::AttachmentState::Available;
        reader.check()?;
        Ok(BytePage {
            descriptor: initial.then_some(resolution),
            input: None,
            blob: hash,
            size,
            offset,
            content_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
            next,
        })
    })();
    match result {
        Ok(page) => {
            if let Err(e) = reader.check() {
                return store_error(&e);
            }
            Answer::ok(page)
        }
        Err(e) => store_error(&e),
    }
}

#[cfg(test)]
mod tests;
