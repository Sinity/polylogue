//! Checked attachment bytes through the ordinary application operation.
use super::service::{Daemon, error, store_error};
use super::{Request, Response};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

pub(crate) const PAGE_BYTES: usize = 64 * 1024;

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub(crate) struct BytePage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<polylogue_store::assets::AttachmentResolution>,
    pub input: String,
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
    input: String,
    offset: u64,
    blob: polylogue_model::BlobHash,
}

pub(super) fn read(request: Request, d: &Daemon) -> Response {
    if let Err(response) = super::service::accepting(d) {
        return response;
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
    let chosen = input
        .as_deref()
        .or_else(|| decoded.as_ref().map(|cursor| cursor.input.as_str()));
    let found = match chosen {
        Some(input) => resolution.matches.iter().find(|found| found.input == input),
        None if resolution.matches.len() == 1 => resolution.matches.first(),
        None => None,
    };
    let Some(found) = found else {
        return Response::Error {
            kind: "attachment_unavailable".into(),
            message: "select one of the stored files with --input".into(),
            detail: Some(serde_json::json!({"attachment": resolution})),
        };
    };
    if decoded
        .as_ref()
        .is_some_and(|cursor| cursor.input != found.input)
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
            input: found.input.clone(),
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
        input: found.input.clone(),
        blob,
        size: found.size,
        offset,
        content_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        next,
    };
    match serde_json::to_value(page) {
        Ok(data) => Response::Ok { data },
        Err(e) => error("internal", e),
    }
}
