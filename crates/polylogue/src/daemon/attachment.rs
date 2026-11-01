//! Checked attachment bytes through the ordinary application operation.
use super::Request;
use super::service::{Daemon, error, store_error};
use base64::Engine as _;
use command_surface::Answer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const PAGE_BYTES: usize = 64 * 1024;

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub(crate) struct BytePage {
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

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    version: u8,
    binding: String,
    generation: String,
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

pub(super) fn read(request: Request, d: &Daemon) -> Answer<Value> {
    if let Err(failure) = super::service::accepting(d) {
        return failure.answer();
    }
    let Request::Attachment {
        message,
        block,
        attachment,
        input,
        length,
        cursor,
        ..
    } = request
    else {
        return error("bad_request", "not an attachment request");
    };
    if length == 0 {
        return error("bad_request", "attachment range length must be positive");
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
    let generation = match reader.change_position() {
        Ok(position) => position.generation,
        Err(e) => return store_error(&e),
    };
    let initial = cursor.is_none();
    let decoded = match cursor {
        Some(cursor) => {
            let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(cursor)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Cursor>(&bytes).ok());
            let Some(decoded) = decoded.filter(|cursor| cursor.version == 1) else {
                return error("bad_request", "invalid attachment cursor");
            };
            if decoded.binding != resolution.binding || decoded.generation != generation {
                return error(
                    "stale_cursor",
                    "the selected attachment changed; start again",
                );
            }
            Some(decoded)
        }
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
                return error("stale_cursor", "the inline attachment source changed");
            }
            if input.is_some() {
                return error(
                    "bad_request",
                    "an inline attachment cursor does not select an input",
                );
            }
        } else if cursor.input.is_none() {
            return error("bad_request", "attachment cursor has no source selection");
        }
    }
    if input.is_none()
        && resolution.matches.is_empty()
        && resolution.source.blob.is_some()
        && decoded
            .as_ref()
            .is_none_or(|cursor| cursor.source.as_ref() == Some(&inline))
    {
        return inline_page(&reader, resolution, generation, decoded, length, inline);
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
        let failure = crate::protocol::failure(
            "attachment_unavailable",
            "select one of the stored files with --input",
        )
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
        return error("bad_request", "the cursor selects another input");
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
        return error("stale_cursor", "the selected attachment hash changed");
    }
    if offset > found.size || offset == found.size && offset != 0 {
        return error(
            "bad_request",
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
    let next = if end < found.size {
        match serde_json::to_vec(&Cursor {
            version: 1,
            binding: resolution.binding.clone(),
            generation,
            input: Some(found.input.clone()),
            source: None,
            offset: end,
            blob,
        }) {
            Ok(bytes) => Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)),
            Err(e) => return error("internal", e),
        }
    } else {
        None
    };
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
    match serde_json::to_value(page) {
        Ok(data) => {
            if let Err(error) = scope.check() {
                return store_error(&error);
            }
            Answer::ok(data)
        }
        Err(e) => error("internal", e),
    }
}

fn inline_page(
    reader: &polylogue_store::Reader<'_>,
    mut resolution: polylogue_store::assets::AttachmentResolution,
    generation: String,
    cursor: Option<Cursor>,
    length: usize,
    source: CursorSource,
) -> Answer<Value> {
    use std::io::{Read, Seek, SeekFrom};
    let result = (|| -> polylogue_store::Result<BytePage> {
        reader.check()?;
        let hash = resolution
            .source
            .blob
            .expect("selected canonical inline attachment");
        let blobs = reader.blob_reader()?;
        let extent = blobs.extent(&hashed_blobs::Hash(hash.0))?;
        let size = extent.decoded_len;
        if resolution
            .source
            .size
            .is_some_and(|expected| expected != size)
        {
            return Err(polylogue_store::StoreError::Invalid(
                "inline attachment size differs from its source".into(),
            ));
        }
        let offset = cursor.as_ref().map_or(0, |cursor| cursor.offset);
        if offset > size || offset == size && offset != 0 {
            return Err(polylogue_store::StoreError::Invalid(
                "inline attachment cursor is outside its bytes".into(),
            ));
        }
        // Initial availability requires the complete physical body, including
        // an encoded empty body. Continuations retain the exact source/hash;
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
        let mut body = match blobs.read_plain(&hashed_blobs::Hash(hash.0), Some(size), move || {
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
            return Err(polylogue_store::StoreError::Invalid(
                "inline attachment range is incomplete".into(),
            ));
        }
        let next = if end < size {
            Some(
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(
                    &Cursor {
                        version: 1,
                        binding: resolution.binding.clone(),
                        generation,
                        input: None,
                        source: Some(source),
                        offset: end,
                        blob: hash,
                    },
                )?),
            )
        } else {
            None
        };
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
        Ok(page) => match serde_json::to_value(page) {
            Ok(data) => {
                if let Err(e) = reader.check() {
                    return store_error(&e);
                }
                Answer::ok(data)
            }
            Err(e) => error("internal", e),
        },
        Err(e) => store_error(&e),
    }
}

#[cfg(test)]
mod tests;
